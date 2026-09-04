"""What gets exported is what gets displayed, and it is never a rectangle.

Two things are easy to get wrong and expensive to discover late: the viewport
and the file drifting apart, and a plan leaving the building without saying
what the plate in it actually is.
"""

from __future__ import annotations

import numpy as np
import pytest

from mandiplan.constants import ATTRIBUTION, DISCLAIMER
from mandiplan.exporting import (
    asset_provenance_lines,
    read_stl_attribution,
    write_bend_csv,
    write_plate_stl,
)
from mandiplan.geometry.mesh_io import Mesh, load_mesh
from mandiplan.plate_assets import asset_by_id, load_asset_mesh


@pytest.fixture(scope="module")
def planned(bone_surface, wide_arch_frames):
    """A session with a plate path drawn on the phantom and a plate fitted."""
    from mandiplan.ui.session import Session

    session = Session()
    session.frames = wide_arch_frames
    session.surface = bone_surface
    from mandiplan.render.surface import SurfaceProjector

    session.projector = SurfaceProjector(bone_surface)
    frames = wide_arch_frames
    for s_mm in np.linspace(frames.length_mm * 0.15, frames.length_mm * 0.85, 12):
        _, _, buccolingual = frames.frame_at(float(s_mm))
        session.add_plate_point(frames.point_at(float(s_mm)) + buccolingual * 12.0)
    assert session.plate_plan is not None
    return session


def test_a_plate_is_fitted_and_it_is_the_asset_mesh(planned):
    placed = planned.plate_mesh()
    assert placed is not None
    asset = planned.plate_asset
    source = load_asset_mesh(asset)
    # Same topology as the asset on disk: the placement moved vertices, it did
    # not substitute a different object.
    assert len(placed.points) == len(source.points)
    assert np.array_equal(placed.triangles, source.triangles)


def test_the_fitted_plate_is_not_a_rectangle(planned):
    """Genus survives placement, so the exported plate still has real holes."""
    placed = planned.plate_mesh()
    mesh = Mesh(placed.points, placed.triangles)
    assert mesh.is_watertight()
    assert mesh.genus() == planned.plate_asset.hole_count
    assert mesh.genus() > 0


def test_the_exported_stl_is_the_displayed_geometry(planned, tmp_path):
    placed = planned.plate_mesh()
    path = write_plate_stl(tmp_path / "plate.stl", placed)
    written = load_mesh(path)

    assert len(written.triangles) == len(placed.triangles)
    assert written.is_watertight()
    assert written.genus() == planned.plate_asset.hole_count
    # Binary STL stores float32, so the comparison is at that precision.
    assert np.allclose(
        np.sort(written.points, axis=0),
        np.sort(np.unique(placed.points, axis=0), axis=0),
        atol=1e-3,
    )


def test_the_exported_plate_still_carries_the_attribution(planned, tmp_path):
    path = write_plate_stl(tmp_path / "plate.stl", planned.plate_mesh())
    assert ATTRIBUTION in read_stl_attribution(path)


def test_the_bend_csv_names_the_plate_and_its_status(planned, tmp_path):
    path = write_bend_csv(
        tmp_path / "bends.csv",
        planned.plate_plan,
        planned.plate.width_mm,
        planned.plate.thickness_mm,
        asset=planned.plate_asset,
        fitted=planned.fitted_plate,
        bent=planned.bent_plate,
        contact=planned.plate_contact,
    )
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == f"# {DISCLAIMER}"
    assert ATTRIBUTION in text
    assert planned.plate_asset.id in text
    assert "Generic parametric approximation" in text
    assert "plate asset licence" in text
    assert "plate deformation" in text
    assert "plate-to-bone clearance" in text


def test_a_generic_plate_says_so_in_every_export():
    asset = asset_by_id("generic-recon-2.4-12h")
    lines = "\n".join(asset_provenance_lines(asset))
    assert "generic parametric approximation" in lines.lower()
    assert "must not be read as a device selection" in lines


def test_an_export_with_no_plate_selected_says_so():
    assert asset_provenance_lines(None) == ["plate asset: none selected"]


def test_no_export_claims_manufacturer_compatibility():
    """Nothing in the catalogue implies a real device or its clearance."""
    forbidden = ("fda", "ce mark", "ce-mark", "510(k)", "equivalent to", "compatible with")
    from mandiplan.plate_assets import load_assets

    for asset in load_assets():
        blob = " ".join(asset.summary_lines()).lower()
        for phrase in forbidden:
            assert phrase not in blob, (asset.id, phrase)
