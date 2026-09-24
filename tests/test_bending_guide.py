"""The clip-on bending guide: it stops each bend at its angle, and prints."""

from __future__ import annotations

import csv

import numpy as np
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy

from mandiplan.constants import ATTRIBUTION, DISCLAIMER
from mandiplan.exporting import read_stl_attribution, write_bending_guide
from mandiplan.geometry.bending_guide import (
    _rotation,
    joint_overlap,
    plan_bending_guide,
    saddle_blocks,
)
from mandiplan.geometry.mesh_io import Mesh
from mandiplan.render.bending_guide_mesh import block_polydata, guide_polydata

PITCH = 8.0


def _straight(n: int):
    centres = np.array([[i * PITCH, 0.0, 0.0] for i in range(n)])
    return centres, np.tile([0.0, 0.0, 1.0], (n, 1))


def _bent(n: int, bends: dict[int, np.ndarray]):
    """Hole frames of a plate bent by ``bends[j]`` at bridge j."""
    frames = [np.eye(3)]
    for j in range(n - 1):
        frames.append(frames[-1] @ bends.get(j, np.eye(3)))
    tangents = np.array([f[:, 0] for f in frames])
    axes = np.array([f[:, 2] for f in frames])
    return np.zeros((n, 3)), axes, tangents


@pytest.fixture(scope="module")
def guide():
    # Bridge 2-3 curls 15° into the bone (about +y turns the plate toward -z),
    # bridge 4-5 turns 12° counter-clockwise seen from the outer face.
    bends = {1: _rotation([0, 1, 0], np.radians(15.0)), 3: _rotation([0, 0, 1], np.radians(12.0))}
    centres, axes = _straight(6)
    bent = _bent(6, bends)
    return plan_bending_guide(
        centres, axes, *bent, width_mm=8.0, thickness_mm=2.4, seat_diameter_mm=4.0,
        overbend=lambda a: 1.05 * a, alloy_name="test alloy",
    )


def test_each_bridge_is_read_as_the_bend_table_reads_it(guide):
    curl, turn = guide.joints[1], guide.joints[3]
    assert curl.out_of_plane_deg == pytest.approx(-15.0, abs=0.05)
    assert curl.in_plane_deg == pytest.approx(0.0, abs=0.05)
    assert "into the bone" in curl.instruction()
    assert turn.in_plane_deg == pytest.approx(12.0, abs=0.05)
    assert "counter-clockwise" in turn.instruction()
    # Springback: the stop is past the target by the alloy's overbend.
    assert curl.stop_angle_deg == pytest.approx(1.05 * 15.0, abs=1e-6)
    for joint in (guide.joints[0], guide.joints[2], guide.joints[4]):
        assert not joint.has_stop
        assert "leave straight" in joint.instruction()


@pytest.mark.parametrize("bridge", [1, 3])
def test_the_saddles_meet_at_the_stop_angle_and_not_before(guide, bridge):
    """No contact up to the stop; contact just past it. That is the stop."""
    joint = guide.joints[bridge]
    assert joint_overlap(guide, joint, 0.0) == 0.0
    assert joint_overlap(guide, joint, 0.97) == 0.0
    assert joint_overlap(guide, joint, 1.12) > 0.2


def test_a_bridge_bent_the_other_way_opens_its_gap(guide):
    """Bending against the stop never closes it: the guide shows the way."""
    joint = guide.joints[1]
    assert joint_overlap(guide, joint, -1.5) == 0.0


def test_a_preformed_plate_already_in_shape_needs_no_bends():
    theta = np.radians(np.linspace(-40, 40, 7))
    centres = np.column_stack([40 * np.sin(theta), 40 * (1 - np.cos(theta)), np.zeros(7)])
    axes = np.column_stack([-np.sin(theta), np.cos(theta), np.zeros(7)])
    tangents = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(7)])
    guide = plan_bending_guide(centres, axes, centres + 5.0, axes, tangents, 8.0, 2.4, 4.0)
    assert all(joint.angle_deg < 0.5 for joint in guide.joints)
    assert not any(joint.has_stop for joint in guide.joints)


def test_every_block_is_a_closed_outward_solid(guide):
    for saddle in guide.saddles:
        for block in saddle_blocks(guide, saddle):
            mesh = block_polydata(block)
            points = vtk_to_numpy(mesh.GetPoints().GetData())
            triangles = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
            solid = Mesh(points, triangles)
            assert solid.is_consistently_oriented(), block.name
            assert solid.is_outward(), block.name
            assert solid.volume_mm3() > 0.05, block.name


def test_the_guide_lies_on_the_bed_and_spans_the_plate(guide):
    poly = guide_polydata(guide)
    b = poly.GetBounds()
    assert b[4] == pytest.approx(0.0, abs=1e-9)
    # Six saddles over five pitches, plus the half pitch past each end hole.
    assert b[1] - b[0] == pytest.approx(6 * PITCH + 1.0, abs=0.3)
    assert poly.GetNumberOfCells() < 20_000  # flat faces, not a voxel soup


def test_the_guide_export_carries_attribution_and_a_table(guide, tmp_path):
    stl, table = write_bending_guide(tmp_path / "guide.stl", guide)
    assert ATTRIBUTION[:40] in read_stl_attribution(stl)
    text = table.read_text(encoding="utf-8")
    assert DISCLAIMER in text
    rows = list(csv.reader(line for line in text.splitlines() if not line.startswith("#")))
    assert rows[0][0] == "bridge (holes)"
    assert len(rows) == 1 + len(guide.joints)
    assert rows[2][6] == "yes" and "until the saddles meet" in rows[2][7]
