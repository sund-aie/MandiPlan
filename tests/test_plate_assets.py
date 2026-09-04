"""The plate asset catalogue, its loaders, and rigid placement.

The load-bearing assertion in this file is the genus check. A watertight
solid's genus counts its handles, and a through-hole is a handle — so
``genus == hole_count`` is proof that the holes are real geometry and not
painted markers or a stand-in rectangle. A swept ribbon has genus 0 whatever
it is decorated with.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from mandiplan import plate_assets
from mandiplan.geometry.mesh_io import (
    Mesh,
    MeshLoadError,
    check_scale,
    load_mesh,
    scale_for_units,
)
from mandiplan.geometry.plate_fit import (
    hole_spacings_mm,
    kabsch,
    plate_targets,
    rigid_fit,
)
from mandiplan.plate_assets import (
    PlateAssetError,
    asset_by_id,
    assets_in_family,
    best_asset_for,
    families,
    load_asset_mesh,
    load_assets,
)


@pytest.fixture(scope="module")
def catalogue():
    assets = load_assets()
    assert assets, "no plate assets installed; run tools/make_plate_assets.py"
    return assets


# -- the catalogue --------------------------------------------------------


def test_every_asset_declares_provenance_and_licence(catalogue):
    for asset in catalogue:
        assert asset.provenance.strip(), asset.id
        assert asset.licence.strip(), asset.id


def test_every_shipped_asset_is_labelled_generic(catalogue):
    """Nothing shipped in this repository may claim to be a real implant."""
    for asset in catalogue:
        assert not asset.exact, asset.id
        assert asset.status_label == plate_assets.GENERIC_LABEL
        assert "Generic parametric approximation" in "\n".join(asset.summary_lines())


def test_an_exact_claim_without_provenance_is_refused():
    entry = {
        "id": "pretend-exact",
        "name": "Pretend exact plate",
        "mesh": "nothing.stl",
        "thickness_mm": 2.4,
        "hole_centres_mm": [[0, 0, 0], [9, 0, 0]],
        "exact": True,
    }
    with pytest.raises(PlateAssetError, match="provenance"):
        plate_assets._asset_from_entry(entry, Path("."))


def test_an_exact_claim_with_provenance_is_accepted():
    entry = {
        "id": "licensed-example",
        "name": "Licensed example plate",
        "mesh": "example.stl",
        "thickness_mm": 2.4,
        "hole_centres_mm": [[0, 0, 0], [9, 0, 0]],
        "exact": True,
        "provenance": "Vendor CAD supplied under agreement 1234",
        "licence": "Proprietary, licensed to this site",
    }
    asset = plate_assets._asset_from_entry(entry, Path("."))
    assert asset.exact
    assert asset.status_label == plate_assets.EXACT_LABEL


def test_a_miscounted_hole_list_is_refused():
    entry = {
        "id": "bad-count",
        "name": "Bad",
        "mesh": "x.stl",
        "thickness_mm": 2.0,
        "hole_count": 5,
        "hole_centres_mm": [[0, 0, 0], [9, 0, 0]],
    }
    with pytest.raises(PlateAssetError, match="hole centres"):
        plate_assets._asset_from_entry(entry, Path("."))


def test_families_and_lookup(catalogue):
    assert families()
    for family_id, _ in families():
        assert assets_in_family(family_id)
    assert asset_by_id(catalogue[0].id) is catalogue[0]
    with pytest.raises(KeyError):
        asset_by_id("no-such-plate")


def test_best_asset_covers_the_required_length():
    asset = best_asset_for(80.0)
    assert asset is not None and asset.length_mm >= 80.0
    assert best_asset_for(10_000.0) is None


# -- the meshes are real plates ------------------------------------------


def test_every_asset_mesh_is_watertight(catalogue):
    for asset in catalogue:
        mesh = load_asset_mesh(asset)
        assert mesh.is_watertight(), asset.id


def test_genus_equals_the_hole_count(catalogue):
    """The proof the holes are real: one topological handle per through-hole."""
    for asset in catalogue:
        mesh = load_asset_mesh(asset)
        assert mesh.genus() == asset.hole_count, (
            f"{asset.id}: genus {mesh.genus()} for {asset.hole_count} holes"
        )


def test_a_plate_is_not_a_rectangle(catalogue):
    """A scalloped silhouette: the width varies along the plate.

    A swept ribbon has one width everywhere. Sampling the mesh's cross-section
    along its own long axis has to show lobes and waists, or what is loaded is
    a strip.
    """
    asset = asset_by_id("generic-recon-2.4-16h")
    mesh = load_asset_mesh(asset)
    xs = mesh.points[:, 0]
    widths = []
    for centre in np.linspace(xs.min() + 6.0, xs.max() - 6.0, 60):
        near = np.abs(xs - centre) < 0.35
        if near.sum() > 6:
            widths.append(float(np.ptp(mesh.points[near, 1])))
    widths = np.array(widths)
    assert widths.max() - widths.min() > 2.0, (
        f"width varies by only {np.ptp(widths):.2f} mm: this is a strip, not a plate"
    )
    assert widths.max() == pytest.approx(asset.width_mm, abs=0.2)


def test_the_mesh_has_the_thickness_it_claims(catalogue):
    for asset in catalogue:
        mesh = load_asset_mesh(asset)
        assert mesh.extent_mm[2] == pytest.approx(asset.thickness_mm, abs=1e-3), asset.id


def test_hole_centres_are_a_pitch_apart(catalogue):
    for asset in catalogue:
        if asset.hole_count < 3:
            continue
        spacing = hole_spacings_mm(asset.hole_centres_mm)
        # Preformed plates measure a chord across each pitch of arc, so the
        # straight-line spacing is a shade under the nominal pitch.
        assert spacing.max() <= asset.hole_pitch_mm + 1e-6, asset.id
        assert spacing.min() > asset.hole_pitch_mm * 0.99, asset.id


# -- units ----------------------------------------------------------------


def _write_binary_stl(path: Path, mesh: Mesh, scale: float = 1.0) -> Path:
    points = mesh.points * scale
    with path.open("wb") as handle:
        handle.write(b" " * 80)
        handle.write(struct.pack("<I", len(mesh.triangles)))
        for triangle in mesh.triangles:
            handle.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for index in triangle:
                handle.write(struct.pack("<3f", *points[index]))
            handle.write(struct.pack("<H", 0))
    return path


def test_unit_names_convert_to_millimetres():
    assert scale_for_units("mm") == 1.0
    assert scale_for_units("m") == 1000.0
    assert scale_for_units("in") == 25.4
    with pytest.raises(MeshLoadError):
        scale_for_units("furlong")


def test_a_metre_authored_asset_loads_at_the_right_size(tmp_path, catalogue):
    """The fixture the units guard exists for."""
    asset = asset_by_id("generic-recon-2.4-8h")
    original = load_asset_mesh(asset)
    in_metres = _write_binary_stl(tmp_path / "metres.stl", original, scale=0.001)

    # Loaded as if it were millimetres, it is a thousand times too small.
    naive = load_mesh(in_metres)
    assert naive.extent_mm.max() < 0.2
    with pytest.raises(MeshLoadError, match="unit scale"):
        check_scale(naive, asset.length_mm)

    # Loaded with the scale its metadata should declare, it is correct.
    converted = load_mesh(in_metres, unit_scale=scale_for_units("m"))
    assert converted.extent_mm.max() == pytest.approx(
        original.extent_mm.max(), rel=1e-4
    )
    check_scale(converted, asset.length_mm)


def test_an_unsupported_format_is_refused(tmp_path):
    path = tmp_path / "plate.step"
    path.write_text("not a mesh")
    with pytest.raises(MeshLoadError, match="unsupported mesh format"):
        load_mesh(path)


def test_obj_and_ply_round_trip(tmp_path, catalogue):
    mesh = load_asset_mesh(asset_by_id("generic-recon-2.4-8h"))

    obj = tmp_path / "plate.obj"
    with obj.open("w") as handle:
        for point in mesh.points:
            handle.write(f"v {point[0]} {point[1]} {point[2]}\n")
        for triangle in mesh.triangles:
            handle.write(f"f {triangle[0] + 1} {triangle[1] + 1} {triangle[2] + 1}\n")
    from_obj = load_mesh(obj)
    assert len(from_obj.triangles) == len(mesh.triangles)
    assert from_obj.genus() == mesh.genus()

    ply = tmp_path / "plate.ply"
    with ply.open("w") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(mesh.points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write(f"element face {len(mesh.triangles)}\n")
        handle.write("property list uchar int vertex_indices\nend_header\n")
        for point in mesh.points:
            handle.write(f"{point[0]} {point[1]} {point[2]}\n")
        for triangle in mesh.triangles:
            handle.write(f"3 {triangle[0]} {triangle[1]} {triangle[2]}\n")
    from_ply = load_mesh(ply)
    assert len(from_ply.triangles) == len(mesh.triangles)
    assert from_ply.genus() == mesh.genus()


# -- rigid placement ------------------------------------------------------


def _rigid_case(asset_id: str, seed: int = 3):
    """Move an asset by a known rigid transform and build targets from it."""
    asset = asset_by_id(asset_id)
    mesh = load_asset_mesh(asset)
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation *= np.sign(np.linalg.det(rotation))
    translation = np.array([12.0, -7.0, 3.0])

    moved_holes = asset.hole_centres_mm @ rotation.T + translation
    axes = asset.hole_axes @ rotation.T
    plate_axis = asset.hole_centres_mm[-1] - asset.hole_centres_mm[0]
    plate_axis /= np.linalg.norm(plate_axis)
    binormals = np.cross(plate_axis @ rotation.T, axes)
    binormals /= np.linalg.norm(binormals, axis=1, keepdims=True)

    clearance = 0.5
    nodes = moved_holes - axes * (clearance + asset.thickness_mm / 2.0)
    targets = plate_targets(nodes, axes, binormals, clearance, asset.thickness_mm)
    fitted = rigid_fit(
        mesh.points,
        mesh.triangles,
        asset.hole_centres_mm,
        asset.hole_axes,
        targets,
        axes,
        binormals,
    )
    return asset, mesh, fitted, rotation, translation


@pytest.mark.parametrize(
    "asset_id",
    ["generic-recon-2.4-12h", "generic-body-curved-10h", "generic-angle-10h"],
)
def test_rigid_fit_recovers_a_known_transform(asset_id):
    _, _, fitted, rotation, translation = _rigid_case(asset_id)
    assert np.allclose(fitted.rotation, rotation, atol=1e-9)
    assert np.allclose(fitted.translation, translation, atol=1e-9)
    assert fitted.max_residual_mm < 1e-9


def test_rigid_placement_preserves_the_plate_exactly():
    asset, mesh, fitted, _, _ = _rigid_case("generic-recon-2.4-12h")

    before = hole_spacings_mm(asset.hole_centres_mm)
    after = hole_spacings_mm(fitted.hole_centres)
    assert np.allclose(before, after, atol=1e-9)

    placed = Mesh(fitted.points, fitted.triangles)
    assert placed.volume_mm3() == pytest.approx(mesh.volume_mm3(), rel=1e-9)
    assert placed.genus() == asset.hole_count
    assert placed.is_watertight()

    # Every internal distance survives, which is what "rigid" means. The
    # axis-aligned bounding box does not, because the plate has been rotated.
    sample = np.linspace(0, len(mesh.points) - 1, 200).astype(int)
    before = np.linalg.norm(
        mesh.points[sample][:, None] - mesh.points[sample][None], axis=-1
    )
    after = np.linalg.norm(
        placed.points[sample][:, None] - placed.points[sample][None], axis=-1
    )
    assert np.abs(before - after).max() < 1e-9


def test_the_placement_is_a_rotation_not_a_reflection():
    _, _, fitted, _, _ = _rigid_case("generic-recon-2.4-12h")
    assert np.allclose(fitted.rotation @ fitted.rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(fitted.rotation) == pytest.approx(1.0, abs=1e-12)


def test_screw_trajectories_start_at_the_fitted_hole_centres():
    asset, _, fitted, _, _ = _rigid_case("generic-recon-2.4-12h")
    segments = fitted.screw_trajectories(length_mm=14.0)
    assert segments.shape == (asset.hole_count, 2, 3)
    assert np.allclose(segments[:, 0], fitted.hole_centres)
    # They run into bone, i.e. against the outward hole axis.
    direction = segments[:, 1] - segments[:, 0]
    assert np.allclose(np.linalg.norm(direction, axis=1), 14.0)
    assert np.all(np.einsum("ij,ij->i", direction, fitted.hole_axes) < 0)


def test_the_clearance_lifts_the_plate_off_the_bone():
    nodes = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    binormals = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    targets = plate_targets(nodes, normals, binormals, 0.8, 2.4)
    assert np.allclose(targets[:, 2], 0.8 + 1.2)
    with pytest.raises(ValueError):
        plate_targets(nodes, normals, binormals, -1.0, 2.4)


def test_kabsch_refuses_too_few_points():
    with pytest.raises(ValueError):
        kabsch(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        kabsch(np.zeros((4, 3)), np.zeros((3, 3)))
