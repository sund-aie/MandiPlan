"""The comb bending guide: made for one plate and one bender, and it prints."""

from __future__ import annotations

import csv

import numpy as np
import pytest
import vtk

from mandiplan.bending_steps import BendStep, steps_from_guide
from mandiplan.constants import ATTRIBUTION, DISCLAIMER
from mandiplan.exporting import read_stl_attribution, write_bending_guide, write_steps_csv
from mandiplan.geometry.bending_guide import _rotation, plan_bending_guide
from mandiplan.plate_catalog import kit_by_id
from mandiplan.render.bending_guide_mesh import guide_polydata

PITCH = 8.0


def _plate(n=6, bends=None):
    """A straight plate as supplied, and the same plate bent at some bridges."""
    centres = np.array([[i * PITCH, 0.0, 0.0] for i in range(n)])
    axes = np.tile([0.0, 0.0, 1.0], (n, 1))
    frames = [np.eye(3)]
    for j in range(n - 1):
        frames.append(frames[-1] @ (bends or {}).get(j, np.eye(3)))
    tangents = np.array([f[:, 0] for f in frames])
    up = np.array([f[:, 2] for f in frames])
    target = np.vstack([np.zeros(3), np.cumsum(PITCH * tangents[1:], axis=0)])
    return centres, axes, target, up, tangents


def _guide(kit="bending-pliers", bends=None, n=6, hole=2.7, **kwargs):
    stock, stock_axes, target, up, tangents = _plate(n, bends)
    return plan_bending_guide(stock, stock_axes, target, up, tangents, 8.0, 2.4, hole,
                              kit_by_id(kit), **kwargs)


#: The plate wrapped round a jaw: each bridge curls it towards the bone.
CURVED = {j: _rotation([0, 1, 0], np.radians(14.0)) for j in range(8)}


def _inside(guide, point):
    return any(p.field((point - p.centre) @ p.frame) < 0 for p in guide.parts)


def test_each_bridge_is_read_as_the_bend_table_reads_it():
    guide = _guide(bends={1: _rotation([0, 1, 0], np.radians(15.0)), 3: _rotation([0, 0, 1], np.radians(12.0))},
                   overbend=lambda a: 1.05 * a)
    curl, turn = guide.joints[1], guide.joints[3]
    assert curl.out_of_plane_deg == pytest.approx(-15.0, abs=0.05)
    assert "towards the bone" in guide.instruction(curl)
    assert turn.in_plane_deg == pytest.approx(12.0, abs=0.05)
    assert "counter-clockwise" in guide.instruction(turn)
    assert curl.stop_angle_deg == pytest.approx(15.75)
    assert "drops onto its pin" in guide.instruction(curl)
    assert "leave as supplied" in guide.instruction(guide.joints[0])


def test_every_pin_and_pad_is_where_the_finished_plate_has_that_hole():
    guide = _guide(bends=CURVED)
    placed = guide.to_print(guide.centres)
    pins = [p for p in guide.parts if p.name == "pin"]
    pads = [p for p in guide.parts if p.name == "pad"]
    assert len(pins) == len(pads) == 6
    out = np.einsum("ji,hjk->hik", guide.basis, guide.frames)[:, :, 2]
    for hole, normal, pin, pad in zip(placed, out, pins, pads):
        # The pad's face is the plate's bone face, a pin rises out of it along
        # the hole's axis, into the hole and not out of its far side.
        face = pad.centre + normal * 0.5 * guide.tooth_mm
        assert (face - hole) @ normal == pytest.approx(-1.2)
        assert np.linalg.norm(np.cross(pin.centre - hole, normal)) < 1e-9
        assert _inside(guide, hole - normal * 0.5)
        assert not _inside(guide, hole + normal * 1.1)


def test_the_pin_is_cut_to_this_plate_and_the_base_stands_back_for_this_bender():
    assert _guide(hole=2.7).pin_diameter_mm == pytest.approx(2.2)
    assert _guide(hole=1.8).pin_diameter_mm == pytest.approx(1.3)
    assert _guide("bar-bending-press").setback_mm > _guide("bending-pliers").setback_mm
    guide = _guide(bends=CURVED, n=9)
    assert sum(p.name == "tooth" for p in guide.parts) == 9


def test_the_bridges_are_free_for_the_bender():
    """Between teeth nothing touches the bridge, in front, behind or at its edges."""
    guide = _guide(bends=CURVED)
    placed = guide.to_print(guide.centres)
    frames = np.einsum("ji,hjk->hik", guide.basis, guide.frames)
    for i in range(len(placed) - 1):
        middle = 0.5 * (placed[i] + placed[i + 1])
        out = 0.5 * (frames[i, :, 2] + frames[i + 1, :, 2])
        across = 0.5 * (frames[i, :, 1] + frames[i + 1, :, 1])
        for depth in np.arange(-0.6 * guide.setback_mm, 4.0, 0.5):
            for side in (-4.0, 0.0, 4.0):
                assert not _inside(guide, middle + depth * out + side * across)


def test_it_says_when_this_benders_jaws_land_on_the_teeth():
    assert not _guide("bending-pliers").jaws_meet_teeth  # 14 mm jaws at 8 mm pitch
    assert _guide("bending-irons").jaws_meet_teeth  # 20 mm
    assert not _guide("template-wire").jaws_meet_teeth


def test_the_guide_is_one_closed_solid_flat_on_the_bed():
    poly = guide_polydata(_guide(bends=CURVED))
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    assert connectivity.GetNumberOfExtractedRegions() == 1
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(poly)
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    assert edges.GetOutput().GetNumberOfCells() == 0
    assert poly.GetBounds()[4] == pytest.approx(0.0)


def test_the_written_stl_is_still_closed_when_read_back(tmp_path):
    stl, _ = write_bending_guide(tmp_path / "guide.stl", _guide(bends=CURVED))
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl))
    reader.Update()
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(reader.GetOutput())
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()
    assert edges.GetOutput().GetNumberOfCells() == 0


def test_the_preview_stands_at_the_plate():
    guide = _guide(bends=CURVED)
    world = guide_polydata(guide, print_pose=False)
    bounds = np.array(world.GetBounds()).reshape(3, 2)
    assert np.all(guide.centres >= bounds[:, 0] - 1.0) and np.all(guide.centres <= bounds[:, 1] + 1.0)


def test_a_plate_curving_too_tightly_stands_the_base_nearer():
    tight = {j: _rotation([0, 1, 0], np.radians(60.0)) for j in range(4)}
    assert _guide("bar-bending-press", bends=tight, n=5).setback_mm < _guide("bar-bending-press").setback_mm


def test_impossible_sizes_are_refused():
    with pytest.raises(ValueError):
        _guide(tooth_mm=5.0)
    with pytest.raises(ValueError):
        _guide(hole=1.0, play_mm=0.3)


def test_the_export_carries_attribution_the_disclaimer_and_a_table(tmp_path):
    guide = _guide(bends={2: _rotation([1, 0, 0], np.radians(15.0))}, n=6)
    stl, table = write_bending_guide(tmp_path / "guide.stl", guide)
    assert ATTRIBUTION[:40] in read_stl_attribution(stl)
    text = table.read_text(encoding="utf-8")
    assert DISCLAIMER in text
    assert "drops over its pin" in text
    rows = list(csv.reader(line for line in text.splitlines() if not line.startswith("#")))
    assert rows[0][0] == "bridge (holes)" and len(rows) == 1 + len(guide.joints)
    joint = guide.joints[2]
    assert joint.twist_deg == pytest.approx(15.0)
    assert "this needs another tool" in guide.instruction(joint)
    assert guide.instruction(joint) in text


def test_bench_steps_count_from_hole_one_and_say_what_to_cut(tmp_path):
    from types import SimpleNamespace

    guide = _guide("bar-bending-press", bends={2: _rotation([0, 0, 1], np.radians(15.0))})
    steps = steps_from_guide(guide, stock_holes=9)
    assert "3 distal" in steps[0].text
    bend = next(s for s in steps if s.kind == "bend")
    # The bridge between the centres 16 and 24 mm from hole 1.
    assert bend.distance_mm == pytest.approx(20.0)
    assert bend.instrument == "bending press"
    assert guide.instruction(guide.joints[2]) in bend.text
    path = write_steps_csv(tmp_path / "new.csv", steps, SimpleNamespace(name="test"), guide.kit)
    rows = list(csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith("#")))
    assert float(next(r for r in rows if r["kind"] == "bend")["distance_from_hole_1_mm"]) == 20.0
    legacy = [BendStep(1, "bend", 24.0, 1, 15.0, "bending press", "24 mm from cut end")]
    old = write_steps_csv(tmp_path / "old.csv", legacy, SimpleNamespace(name="test"), guide.kit)
    assert "distance_from_cut_end_mm" in old.read_text()


def test_a_fresh_plate_starts_on_its_first_pin_and_ends_on_all_of_them():
    bends = {j: _rotation([0, 1, 0], np.radians(14.0)) for j in range(5)}
    stock, axes, target, up, tangents = _plate(6, bends)
    guide = plan_bending_guide(stock, axes, target, up, tangents, 8.0, 2.4, 2.7, kit_by_id("bending-pliers"))
    start, start_axes = guide.chain_after(stock, axes, 0)
    # Fresh: hole 1 on its pin, the rest running on straight, at the plate's spacing.
    assert np.linalg.norm(start[0] - guide.centres[0]) < 1e-9
    assert np.linalg.norm(start[-1] - guide.centres[-1]) > 10.0
    np.testing.assert_allclose(np.linalg.norm(np.diff(start, axis=0), axis=1), PITCH, atol=1e-6)
    np.testing.assert_allclose(start_axes, np.tile(start_axes[0], (6, 1)), atol=1e-6)
    # Halfway: three holes on their pins.
    middle, _ = guide.chain_after(stock, axes, 2)
    np.testing.assert_allclose(middle[:3], guide.centres[:3], atol=1e-9)
    assert np.linalg.norm(middle[-1] - guide.centres[-1]) > 1.0
    # Done: all of them.
    done, done_axes = guide.chain_after(stock, axes, 5)
    np.testing.assert_allclose(done, guide.centres, atol=1e-9)
    np.testing.assert_allclose(done_axes, guide.frames[:, :, 2], atol=1e-9)
    steps = guide.steps()
    assert steps[0][0] == 0 and "hole 1" in steps[0][1]
    assert steps[-1][0] == 5 and "every hole" in steps[-1][1]
    assert len(steps) == 7


def test_a_pin_takes_the_shape_of_an_oval_hole():
    guide = _guide(hole=3.1, hole_length_mm=3.95, bends=CURVED)
    assert guide.hole_stretch_mm == pytest.approx(0.85)
    pin = next(p for p in guide.parts if p.name == "pin")
    # At half height, away from the lead-in chamfer.
    middle = pin.centre
    along, across = pin.frame[:, 0], pin.frame[:, 1]
    long_end = 0.5 * 0.85 + 0.5 * 3.1 - 0.25
    assert pin.field((middle + along * (long_end - 0.1) - pin.centre) @ pin.frame) < 0
    assert pin.field((middle + along * (long_end + 0.1) - pin.centre) @ pin.frame) > 0
    assert pin.field((middle + across * (0.5 * 3.1 - 0.25 - 0.1) - pin.centre) @ pin.frame) < 0
    assert pin.field((middle + across * (0.5 * 3.1 - 0.25 + 0.1) - pin.centre) @ pin.frame) > 0


def test_a_ledge_carries_the_plates_lower_edge_clear_of_the_plate():
    guide = _guide(bends=CURVED)
    placed = guide.to_print(guide.centres)
    frames = np.einsum("ji,hjk->hik", guide.basis, guide.frames)
    ledges = [p for p in guide.parts if p.name == "ledge"]
    assert len(ledges) == len(placed)
    for hole, frame in zip(placed, frames):
        low = frame[:, 1] if frame[2, 1] < 0 else -frame[:, 1]
        edge = hole + low * 0.5 * guide.width_mm
        assert not _inside(guide, edge - low * 0.05)  # the plate itself is clear
        assert _inside(guide, edge + low * 0.3), "the ledge lies just under the edge"
    poly = guide_polydata(guide)
    assert poly.GetBounds()[4] == pytest.approx(0.0, abs=1e-6)


def test_every_hole_has_its_number_on_the_base():
    guide = _guide(n=12, bends={j: _rotation([0, 1, 0], np.radians(8.0)) for j in range(11)})
    digits = [p for p in guide.parts if p.name == "digit"]
    # 1 to 9, and two numerals each for 10, 11 and 12.
    assert len(digits) >= 2 * (9 + 6)
    assert min(p.centre[2] for p in digits) > 1.0


def test_the_export_includes_a_pin_test_piece_with_three_fits(tmp_path):
    from mandiplan.geometry.bending_guide import PIN_TEST_PLAYS_MM
    from mandiplan.render.bending_guide_mesh import pin_test_polydata

    guide = _guide(hole=2.7, bends=CURVED)
    pins = [p for p in guide.pin_test_parts() if p.name == "pin"]
    assert [round(0.5 * 2.7 - p.half[0], 3) for p in pins] == [round(v, 3) for v in PIN_TEST_PLAYS_MM]
    assert pin_test_polydata(guide).GetBounds()[4] == pytest.approx(0.0, abs=1e-6)
    stl, table = write_bending_guide(tmp_path / "guide.stl", guide)
    test_piece = tmp_path / "guide-pin-test.stl"
    assert test_piece.is_file() and ATTRIBUTION[:40] in read_stl_attribution(test_piece)
    assert "guide-pin-test.stl" in table.read_text(encoding="utf-8")


def test_a_tall_tooth_is_stiffer_and_keeps_its_outer_face():
    tall = {j: _rotation([1, 0, 0], np.radians(6.0)) @ _rotation([0, 1, 0], np.radians(10.0)) for j in range(9)}
    guide = _guide(n=10, bends=tall)
    teeth = [p for p in guide.parts if p.name == "tooth"]
    heights = np.array([2.0 * t.half[2] for t in teeth])
    thick = np.array([2.0 * t.half[1] for t in teeth])
    assert thick[np.argmax(heights)] >= thick[np.argmin(heights)]
    assert np.all(thick >= guide.tooth_mm - 1e-9)


def test_each_plates_oval_holes_are_read_from_its_own_mesh():
    from mandiplan.plate_assets import asset_by_id, hole_length_mm

    gold = asset_by_id("synthes-angled-square-left-24h")
    assert 3.7 < hole_length_mm(gold) < 4.05
    assert hole_length_mm(asset_by_id("generic-recon-2.4-lp-12h")) == pytest.approx(2.5, abs=0.02)

