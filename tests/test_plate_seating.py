"""The plate rests on the jaw the right way up, on the right surface.

These measure the plate *mesh* against the bone, never the hole metadata. A
previous version reported perfectly aligned hole axes while the mesh itself
stood on its edge with half its width inside the bone: the bend mapped the
plate's width onto its thickness whenever the plate had already been rotated
onto the jaw, and the reported axes came from the target frame rather than
from the geometry. Nothing that only checked the metadata could see it.
"""

from __future__ import annotations

import numpy as np
import pytest

from mandiplan.geometry.plate_bend import bend_to_path
from mandiplan.plate_assets import asset_by_id, load_asset_mesh


@pytest.fixture(scope="module")
def seated(bone_surface, wide_arch_frames):
    """A plate drawn along the buccal cortex of the phantom, fitted and bent."""
    from mandiplan.render.surface import SurfaceProjector
    from mandiplan.ui.session import Session

    session = Session()
    session.frames = wide_arch_frames
    session.surface = bone_surface
    session.projector = SurfaceProjector(bone_surface)
    session.set_plate_asset("generic-recon-2.4-lp-8h")
    frames = wide_arch_frames
    for s_mm in np.linspace(frames.length_mm * 0.2, frames.length_mm * 0.8, 10):
        _, _, buccal = frames.frame_at(float(s_mm))
        session.add_plate_point(frames.point_at(float(s_mm)) + buccal * 12.0)
    assert session.bent_plate is not None
    return session


def _signed_to_bone(projector, points):
    out = np.empty(len(points))
    for i, q in enumerate(points):
        on_bone, outward = projector.project(q)
        out[i] = float(np.dot(q - on_bone, outward))
    return out


def _plate_axes(placed):
    idx = np.argmin(
        np.linalg.norm(placed.points[:, None] - placed.hole_centres[None], axis=2),
        axis=1,
    )
    rel = placed.points - placed.hole_centres[idx]
    normal = placed.hole_axes[idx]
    tangent = placed.hole_tangents[idx]
    return rel, normal, np.cross(tangent, normal)


def test_no_part_of_the_plate_is_inside_the_bone(seated):
    depth = _signed_to_bone(seated.projector, seated.bent_plate.points)
    assert depth.min() > -0.05, f"plate is {-depth.min():.2f} mm inside the bone"


def test_the_bone_face_is_the_one_against_the_bone(seated):
    placed, asset = seated.bent_plate, seated.plate_asset
    rel, normal, _ = _plate_axes(placed)
    height = np.einsum("ij,ij->i", rel, normal)
    bone_face = placed.points[height < -asset.thickness_mm * 0.45]
    outer_face = placed.points[height > asset.thickness_mm * 0.45]
    near = _signed_to_bone(seated.projector, bone_face[::3])
    far = _signed_to_bone(seated.projector, outer_face[::3])
    assert near.mean() < far.mean() - asset.thickness_mm * 0.8


def test_the_plate_lies_flat_not_on_its_edge(seated):
    """Width runs across the plate, thickness out of its face."""
    placed, asset = seated.bent_plate, seated.plate_asset
    rel, normal, across = _plate_axes(placed)
    width = np.ptp(np.einsum("ij,ij->i", rel, across))
    thickness = np.ptp(np.einsum("ij,ij->i", rel, normal))
    assert width == pytest.approx(asset.width_mm, abs=0.2)
    assert thickness < asset.thickness_mm + 0.6


def test_the_face_follows_the_bone_surface(seated):
    placed = seated.bent_plate
    for centre, axis in zip(placed.hole_centres, placed.hole_axes):
        _, outward = seated.projector.project(centre)
        assert float(np.dot(axis, outward)) > 0.98


def test_on_the_lateral_cortex_the_width_is_vertical(seated):
    placed = seated.bent_plate
    across = np.cross(placed.hole_tangents, placed.hole_axes)
    assert np.all(np.abs(across[:, 2]) > 0.95)


def test_the_clearance_is_the_one_asked_for(seated):
    placed, asset = seated.bent_plate, seated.plate_asset
    at_holes = _signed_to_bone(seated.projector, placed.hole_centres) - asset.thickness_mm / 2
    assert np.all(at_holes > 0.0)
    assert np.all(at_holes < seated.plate_clearance_mm + 0.3)


def test_bending_a_plate_that_is_already_rotated(seated):
    """The exact case that failed: the source plate is not lying in z.

    Rotate a flat plate so its face points along +x, then bend it onto a path
    whose bone normal is +y. Width must stay width and thickness thickness.
    """
    asset = asset_by_id("generic-recon-2.4-lp-8h")
    mesh = load_asset_mesh(asset)
    # Face normal z -> x, width y -> z, length x -> y.
    rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    points = mesh.points @ rotation.T
    holes = asset.hole_centres_mm @ rotation.T
    axes = asset.hole_axes @ rotation.T

    n = asset.hole_count
    radius = 60.0
    t = (np.arange(n) - (n - 1) / 2) * (asset.hole_pitch_mm / radius)
    path = np.column_stack([radius * np.sin(t), radius * np.cos(t), np.zeros(n)])
    bone_normal = np.column_stack([np.sin(t), np.cos(t), np.zeros(n)])

    bent = bend_to_path(points, mesh.triangles, holes, axes, path, bone_normal,
                        protected_radius_mm=asset.deformation.protected_radius_mm)
    idx = np.argmin(np.linalg.norm(bent.points[:, None] - bent.hole_centres[None], axis=2), axis=1)
    rel = bent.points - bent.hole_centres[idx]
    thickness = np.ptp(np.einsum("ij,ij->i", rel, bent.hole_axes[idx]))
    vertical = np.ptp(rel[:, 2])
    assert thickness < asset.thickness_mm + 0.6, "the plate was turned onto its edge"
    assert vertical == pytest.approx(asset.width_mm, abs=0.2)
    assert np.all(np.einsum("ij,ij->i", bent.hole_axes, bone_normal) > 0.99)


def test_the_etched_mark_is_in_the_outer_face_before_fitting():
    from mandiplan.render.marking import ETCH_DEPTH_MM, marking_in_plate_space

    asset = asset_by_id("generic-recon-2.4-lp-8h")
    points, triangles = marking_in_plate_space(asset)
    assert len(points) and len(triangles)
    top = asset.thickness_mm / 2.0
    assert points[:, 2].max() == pytest.approx(top, abs=1e-6)
    assert points[:, 2].min() == pytest.approx(top - ETCH_DEPTH_MM, abs=1e-6)


def test_the_etched_mark_reads_the_right_way_round():
    """Viewed from outside the plate, the lettering must not be mirrored."""
    from mandiplan.render.marking import build_marking

    along, outward = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    # "L": the foot runs toward +x from the stem, and the stem sits at the
    # left. Mirrored lettering puts the stem on the right.
    from vtkmodules.util.numpy_support import vtk_to_numpy

    glyph = build_marking("L", np.zeros(3), along, None, outward)
    pts = vtk_to_numpy(glyph.GetPoints().GetData())
    top_row = pts[pts[:, 1] > pts[:, 1].max() - 0.2]
    assert top_row[:, 0].mean() < pts[:, 0].mean(), "lettering is mirrored"


def test_the_etched_mark_travels_with_the_bent_plate(seated):
    points, _ = seated.plate_marking
    placed = seated.bent_plate
    mark = _signed_to_bone(seated.projector, points[::9])
    rel, normal, _ = _plate_axes(placed)
    height = np.einsum("ij,ij->i", rel, normal)
    bone_face = _signed_to_bone(seated.projector, placed.points[height < -1.0][::5])
    assert mark.min() > 0.0
    assert mark.mean() > bone_face.mean() + seated.plate_asset.thickness_mm * 0.7
