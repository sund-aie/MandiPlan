"""The bend keys: made for one plate and one bender, each stopping one bend, and they print."""

from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from mandiplan.bending_steps import steps_from_guide
from mandiplan.constants import ATTRIBUTION, DISCLAIMER
from mandiplan.exporting import read_stl_attribution, write_bending_guide, write_steps_csv
from mandiplan.geometry.bending_guide import (
    FIT_TEST_ANGLE_DEG,
    MIN_SHOULDER_MM,
    PIN_TEST_PLAYS_MM,
    _rotation,
    die_height,
    fit_test_key,
    key_print_rotation,
    pin_tab_parts,
    plan_bending_guide,
    tool_parts,
)
from mandiplan.plate_catalog import kit_by_id, load_kits
from mandiplan.render.bending_guide_mesh import fit_test_ready_to_print, key_polydata, key_solids, keys_ready_to_print

PITCH = 8.0
WIDTH, THICK = 8.0, 2.4
BED = (119.0, 67.0, 75.0)


def _plate(n=6, bends=None):
    """A straight plate as supplied, and the same plate bent at some bridges."""
    centres = np.array([[i * PITCH, 0.0, 0.0] for i in range(n)])
    axes = np.tile([0.0, 0.0, 1.0], (n, 1))
    frames = [np.eye(3)]
    places = [np.zeros(3)]
    for j in range(n - 1):
        turn = (bends or {}).get(j, np.eye(3))
        # About the middle of the bridge, as a plate bends.
        places.append(places[j] + frames[j] @ (np.array([PITCH / 2, 0, 0]) + turn @ np.array([PITCH / 2, 0, 0])))
        frames.append(frames[-1] @ turn)
    tangents = np.array([f[:, 0] for f in frames])
    up = np.array([f[:, 2] for f in frames])
    return centres, axes, np.array(places), up, tangents


def _guide(kit="synthes-329-142", bends=None, n=6, hole=2.7, **kwargs):
    stock, stock_axes, target, up, tangents = _plate(n, bends)
    return plan_bending_guide(stock, stock_axes, target, up, tangents, WIDTH, THICK, hole, kit_by_id(kit), **kwargs)


def curl(degrees):
    """A bridge curving towards the bone: the outer face goes convex."""
    return _rotation([0, 1, 0], np.radians(degrees))


def turn(degrees):
    """A bridge turning in the plate's own plane."""
    return _rotation([0, 0, 1], np.radians(degrees))


#: The plate wrapped round a jaw.
CURVED = {j: curl(14.0) for j in range(5)}


def _closed(mesh):
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(mesh)
    edges.ManifoldEdgesOff()
    edges.FeatureEdgesOff()
    edges.Update()
    return edges.GetOutput().GetNumberOfCells() == 0


def _regions(mesh):
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(mesh)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    return connectivity.GetNumberOfExtractedRegions()


def _inside(parts, point):
    return any(p.field(tuple((np.asarray(point) - p.centre) @ p.frame)) < 0 for p in parts)


def test_each_bridge_is_read_as_the_bend_table_reads_it():
    guide = _guide(bends={1: curl(15.0), 3: turn(12.0)}, overbend=lambda a: 1.05 * a)
    curled, turned = guide.joints[1], guide.joints[3]
    assert curled.out_of_plane_deg == pytest.approx(-15.0, abs=0.05)
    assert turned.in_plane_deg == pytest.approx(12.0, abs=0.05)
    assert curled.stop_angle_deg == pytest.approx(15.75)
    assert [abs(j.angle_deg) < 0.01 for j in (guide.joints[0], guide.joints[2], guide.joints[4])] == [True] * 3


def test_a_key_for_each_bend_in_plane_first_and_none_for_a_bridge_left_alone():
    guide = _guide(bends={0: curl(10.0), 1: turn(25.0), 2: curl(0.4), 3: turn(25.0) @ curl(6.0)})
    assert [key.name for key in guide.keys] == ["A2", "A4", "B1", "B4"], "in the plate's plane first, as its maker says"
    assert [key.number for key in guide.keys] == [1, 2, 3, 4]
    assert all(key.bridge == int(key.name[1:]) - 1 for key in guide.keys)


def test_small_turns_are_gathered_into_few_keys_and_the_plate_stays_on_its_plan():
    from mandiplan.geometry.bending_guide import IN_PLANE_TOLERANCE_MM

    guide = _guide(n=14, bends={j: turn(3.0) for j in range(13)})
    made = [key for key in guide.keys if key.kind == "in_plane"]
    assert 1 <= len(made) <= 5, "not a key at each of 13 bridges"
    assert guide.deviation_mm < 1.5 * IN_PLANE_TOLERANCE_MM
    total = sum(j.in_plane_made_deg for j in guide.joints)
    assert 24.0 < abs(total) <= 39.0, "nearly the whole turn, the ends of the curve being a little flattened"
    assert not [w for w in guide.warnings if "lies up to" in w]


def test_what_lifts_the_plate_off_the_bone_is_held_tighter():
    gentle = _guide(n=14, bends={j: curl(3.0) for j in range(13)})
    sideways = _guide(n=14, bends={j: turn(3.0) for j in range(13)})
    assert len(gentle.keys) > len(sideways.keys)
    assert len(gentle.keys) < 13
    assert gentle.deviation_mm < 0.5


def test_a_sharp_turn_keeps_its_own_bridge():
    guide = _guide(n=8, bends={3: turn(30.0)})
    assert [key.name for key in guide.keys] == ["A4"]
    assert guide.keys[0].angle_deg == pytest.approx(30.0, abs=1.5)


def test_gathering_does_not_pile_up_more_than_a_plate_should_take():
    from mandiplan.geometry.bending_guide import MAX_GATHERED_DEG

    guide = _guide(n=20, bends={j: turn(4.0) for j in range(19)})
    assert max(key.angle_deg for key in guide.keys) <= MAX_GATHERED_DEG + 0.5


def test_twist_with_no_bend_to_ride_on_is_carried_to_the_next_key():
    # Small twists at each bridge, a real curve at the last: all the twist gets made.
    bends = {j: _rotation([1, 0, 0], np.radians(1.2)) for j in range(4)}
    bends[4] = curl(10.0)
    guide = _guide(n=6, bends=bends)
    assert sum(j.twist_made_deg for j in guide.joints) == pytest.approx(sum(j.twist_deg for j in guide.joints), abs=0.5)
    # Enough of it gets a key of its own.
    assert any(abs(j.out_of_plane_made_deg) < 1e-9 and abs(j.twist_made_deg) >= 3.0 for j in guide.joints)


def test_a_bridge_of_a_straight_plate_needs_no_keys():
    assert not _guide().keys


def test_the_keys_faces_are_the_plate_at_its_stop():
    guide = _guide(bends={1: curl(15.0)}, overbend=lambda a: a + 1.5)
    key = guide.keys[0]
    (qa, na), (qb, nb) = key.faces
    assert np.degrees(np.arccos(np.clip(na @ nb, -1.0, 1.0))) == pytest.approx(16.5, abs=0.05)
    assert key.stop_angle_deg == pytest.approx(16.5)
    assert key.angle_deg == pytest.approx(15.0, abs=0.05)
    # Each hole of that bridge at the stop has its plate's face on the key's face.
    for hole, (q, n) in ((key.hole_a, key.faces[0]), (key.hole_b, key.faces[1])):
        assert (hole + 0.5 * THICK * n - q) @ n == pytest.approx(0.0, abs=1e-6)


def test_at_its_stop_the_plate_is_where_the_key_says():
    guide = _guide(bends={j: curl(12.0) @ turn(3.0) for j in range(5)}, overbend=lambda a: 1.1 * a)
    for key in guide.keys:
        centres, _ = guide.stage(key)
        assert np.linalg.norm(centres[key.bridge] - key.hole_a) < 0.3
        assert np.linalg.norm(centres[key.bridge + 1] - key.hole_b) < 0.3


def test_the_keys_in_their_order_take_the_fresh_plate_to_the_planned_shape():
    guide = _guide(bends={0: curl(10.0), 1: turn(8.0), 3: curl(14.0) @ turn(5.0)})
    fresh, _ = guide.stage()
    np.testing.assert_allclose(np.linalg.norm(np.diff(fresh, axis=0), axis=1), PITCH, atol=1e-6)
    done, axes = guide.stage(done=True)
    # Compare shape, whatever the frame: every distance between holes.
    made = np.linalg.norm(done[:, None] - done[None, :], axis=2)
    planned = np.linalg.norm(guide.centres[:, None] - guide.centres[None, :], axis=2)
    assert np.abs(made - planned).max() < 2.0 * guide.deviation_mm + 0.05
    assert guide.deviation_mm < 1.0
    assert np.abs(made - np.linalg.norm(fresh[:, None] - fresh[None, :], axis=2)).max() > 0.3


def test_a_bend_made_earlier_is_released_at_the_next_keys_stop():
    guide = _guide(bends={0: curl(10.0), 3: curl(10.0)}, overbend=lambda a: 1.3 * a)
    second = guide.keys[1]
    centres, _ = guide.stage(second)
    # Hole 0 to hole 1 was made by the first key: it lies at the released angle.
    done, _ = guide.stage(done=True)
    near = np.linalg.norm(centres[0] - centres[2])
    assert near == pytest.approx(np.linalg.norm(done[0] - done[2]), abs=0.02)


def test_the_pins_stand_in_the_plates_own_holes():
    guide = _guide(hole=2.7, bends={1: curl(12.0)}, play_mm=0.1)
    key = guide.keys[0]
    pins = [p for p in key.parts if p.name == "pin"]
    assert len(pins) == 2
    for pin, hole in zip(pins, (key.hole_a, key.hole_b)):
        assert pin.half[0] == pytest.approx(0.5 * 2.7 - 0.1)
        assert np.linalg.norm(np.cross(pin.centre - hole, pin.frame[:, 2])) < 1e-6, "on the hole's axis"
    assert guide.pin_diameter_mm == pytest.approx(2.5)


def test_oval_holes_get_oval_pins():
    guide = _guide(hole=2.7, hole_length_mm=3.9, bends={1: curl(12.0)})
    assert guide.hole_stretch_mm == pytest.approx(1.2)
    pin = next(p for p in guide.keys[0].parts if p.name == "pin")
    assert pin.stretch == pytest.approx(1.2)
    # Longer along the plate than across it: at the base, by the stretch.
    base = pin.centre - pin.frame[:, 2] * (pin.half[2] - 0.01)
    assert pin.field(tuple((base + pin.frame[:, 0] * 1.8 - pin.centre) @ pin.frame)) < 0
    assert pin.field(tuple((base + pin.frame[:, 1] * 1.8 - pin.centre) @ pin.frame)) > 0


def test_the_pins_taper_so_they_stay_in_a_hole_that_is_still_turning():
    guide = _guide(bends={1: curl(12.0)})
    pin = next(p for p in guide.keys[0].parts if p.name == "pin")
    assert pin.tip_radius < 0.7 * pin.half[0]
    assert 2.0 * pin.half[2] < 1.4 * THICK


def test_a_bridges_key_rests_in_the_benders_head():
    guide = _guide(bends={1: curl(14.0)})
    key = guide.keys[0]
    head = guide.head
    assert head.kind == "three_point"
    a = 0.5 * head.lobe_spacing_mm
    body = key.parts[0]
    # Under each lobe, solid, its back just there for the lobe to bear on.
    for x in (-a, a):
        assert body.field((np.float64(x), np.float64(0.0), np.float64(key.lobes_z - 0.3))) < 0
        assert body.field((np.float64(x), np.float64(0.0), np.float64(key.lobes_z + 0.3))) > 0
    # Its back follows the arch with room, and never touches it.
    for x in np.linspace(-a, a, 41):
        over = key.lobes_z + float(die_height(head, x))
        assert body.field((np.float64(x), np.float64(0.0), np.float64(over + 0.01))) > 0
    # The post is under the bridge, in line with the middle of the key.
    assert key.post_top_z < min(key.lobes_z, 0.0)
    assert key.shoulder_mm >= MIN_SHOULDER_MM


def test_only_the_two_lobes_of_the_plate_touch_the_key_whatever_is_beyond_them():
    guide = _guide(bends={2: curl(14.0)})
    key = guide.keys[0]
    body = key.parts[0]
    (q, n), _ = key.faces
    # Lying on the face: on hole_a's side, past the hole there is a relief.
    at_hole = key.hole_a + 0.5 * THICK * n
    beyond = at_hole + np.array([-(2.0 * PITCH + 3.0), 0.0, 0.0])
    beyond = beyond - ((beyond - q) @ n) * n
    assert body.field(tuple(np.float64(v) for v in beyond + n * 0.05)) > 0, "the face has lifted away from the plate"
    # Over the bridge, the round channel: the plate curves there and does not fold.
    ridge_point, _ = body.ridge
    assert body.field(tuple(np.float64(v) for v in ridge_point)) > 0


def test_each_key_is_one_closed_solid_with_its_name_on_it():
    guide = _guide(bends={1: curl(14.0), 3: turn(25.0) @ curl(8.0)})
    assert [key.name for key in guide.keys] == ["A4", "B2", "B4"]
    for key in guide.keys:
        mesh = key_polydata(key)
        assert _closed(mesh) and _regions(mesh) == 1
        assert mesh.GetNumberOfCells() < 40_000, "a key is a small file"
        assert sum(1 for p in key.parts if p.name == "digit") >= 4


def test_a_key_prints_with_the_faces_the_plate_lies_on_standing_upright():
    guide = _guide(bends={1: curl(14.0)})
    key = guide.keys[0]
    rotation = key_print_rotation(key)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    for _, n in key.faces:
        assert abs((rotation @ n)[2]) < 1e-6


def test_an_in_plane_key_prints_with_its_pins_standing_up():
    guide = _guide(bends={1: turn(10.0)})
    key = guide.keys[0]
    assert key.kind == "in_plane"
    rotation = key_print_rotation(key)
    for pin in (p for p in key.parts if p.name == "pin"):
        tip_way = rotation @ pin.frame[:, 2]
        assert tip_way[2] > 0.99


def test_the_guide_is_made_for_the_plate_and_the_kit_chosen():
    bends = {1: curl(14.0)}
    large = _guide(bends=bends)
    small = _guide(kit="bending-pliers", bends=bends)
    low, high = np.array(key_polydata(large.keys[0]).GetBounds()), np.array(key_polydata(small.keys[0]).GetBounds())
    assert low[1] - low[0] > high[1] - high[0] + 5.0, "a key is as long as its bender's lobes are apart"
    narrower = plan_bending_guide(*_plate(6, bends), 6.0, 2.0, 2.0, kit_by_id("synthes-329-142"))
    assert narrower.pin_diameter_mm < large.pin_diameter_mm
    assert narrower.keys[0].faces[0][0][2] != large.keys[0].faces[0][0][2]


def test_a_measured_head_takes_the_place_of_the_kits():
    head = replace(kit_by_id("synthes-329-142").head, lobe_spacing_mm=40.0, source="measured")
    guide = _guide(bends={1: curl(14.0)}, head=head)
    assert guide.head.lobe_spacing_mm == 40.0
    bounds = np.array(key_polydata(guide.keys[0]).GetBounds())
    assert bounds[1] - bounds[0] > 40.0


def test_irons_lever_the_plate_down_onto_a_key_on_the_concave_side():
    guide = _guide(kit="bending-irons", bends={1: curl(12.0)})
    key = guide.keys[0]
    assert guide.head.kind == "lever"
    assert "Lever the plate" in guide.instruction(key)
    assert not tool_parts(guide, key)
    (_, n), _ = key.faces
    # The key is on the bridge's concave side: its faces meet in a valley for the plate.
    assert _closed(key_polydata(key))
    assert key.from_bone_side != _guide(bends={1: curl(12.0)}).keys[0].from_bone_side
    assert n is not None


def test_a_kit_with_nothing_to_stop_is_refused_in_plain_words():
    wire = next(kit for kit in load_kits() if kit.head.kind == "none")
    with pytest.raises(ValueError, match="by hand"):
        plan_bending_guide(*_plate(6, {1: curl(12.0)}), WIDTH, THICK, 2.7, wire)


def test_a_head_that_cannot_open_enough_is_said():
    head = replace(kit_by_id("synthes-329-142").head, opening_mm=2.5)
    guide = _guide(bends={1: curl(14.0)}, head=head)
    assert any("to load" in warning for warning in guide.warnings)
    guide = _guide(bends={1: turn(10.0)}, head=head)
    assert any("on its edge" in warning for warning in guide.warnings)
    assert not _guide(bends={1: curl(14.0)}).warnings


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError, match="hole counts"):
        stock, axes, target, up, tangents = _plate(6)
        plan_bending_guide(stock, axes, target[:-1], up[:-1], tangents[:-1], WIDTH, THICK, 2.7, kit_by_id("bending-pliers"))
    with pytest.raises(ValueError, match="play"):
        _guide(play_mm=0.8)
    with pytest.raises(ValueError, match="small"):
        _guide(hole=1.0, play_mm=0.3)


def test_the_steps_say_what_to_do_at_each_key():
    guide = _guide(bends=CURVED)
    steps = guide.steps()
    assert steps[0].key is None and "fresh plate" in steps[0].text
    assert [step.key.name for step in steps[1:-1]] == [key.name for key in guide.keys]
    assert steps[-1].key is None
    text = steps[1].text
    assert "holes 1 and 2" in text and "lies flat on the key" in text and "nest the key in the arch" in text


def test_the_bench_steps_follow_the_keys_and_say_what_to_cut(tmp_path):
    guide = _guide(bends={0: curl(10.0), 3: turn(25.0)})
    steps = steps_from_guide(guide, stock_holes=9, first=0)
    assert steps[0].kind == "cut" and "Cut off 3 holes" in steps[0].text and "holes 1-6" in steps[0].text
    assert [s.node_index for s in steps if s.kind == "bend"] == [3, 0]
    assert steps[1].distance_mm == pytest.approx(3.5 * PITCH)
    assert steps[-1].kind == "check"
    assert steps_from_guide(guide)[0].kind == "check"
    assert "holes 3-8" in steps_from_guide(guide, stock_holes=9, first=2)[0].text
    from mandiplan.plate_catalog import load_systems

    path = write_steps_csv(tmp_path / "steps.csv", steps, load_systems()[0], guide.kit)
    assert "Cut off 3 holes" in path.read_text(encoding="utf-8")


def test_the_keys_come_ready_to_print_on_rods_and_a_raft():
    guide = _guide(n=12, bends={j: curl(10.0) for j in range(11)})
    beds = keys_ready_to_print(guide, BED)
    assert beds
    solids = dict(key_solids(guide))
    assert sorted(solids) == sorted(key.name for key in guide.keys)
    for mesh in beds:
        bounds = np.array(mesh.GetBounds()).reshape(3, 2)
        assert bounds[2, 0] == pytest.approx(0.0, abs=1e-6), "the raft is on the bed"
        assert np.all(bounds[:, 0] >= 0.0) and np.all(bounds[:, 1] <= np.array(BED))
    solid = next(iter(solids.values()))
    assert _closed(solid) and solid.GetBounds()[4] == pytest.approx(0.0, abs=1e-6)


def test_rods_touch_a_key_only_on_its_flat_side():
    from mandiplan.render import supports as rods
    from mandiplan.render.bending_guide_mesh import _flat_faces_only, _lay_out

    guide = _guide(bends={1: curl(14.0)})
    plate = _lay_out([solid for _, solid in key_solids(guide)], BED)[0][0]
    held = rods.support_structure(plate, keep_off=_flat_faces_only, spacing=6.0)
    assert held.rods >= 3
    assert np.abs(held.contacts[:, 2] - rods.lift_mm()).max() < 0.05, "all on the flat side, turned to the bed"


def test_keys_that_cannot_fit_the_printer_are_refused():
    guide = _guide(bends={1: curl(14.0)})
    with pytest.raises(ValueError, match="larger printer"):
        keys_ready_to_print(guide, (30.0, 30.0, 30.0))


def test_the_fit_test_has_a_known_bend_its_gauge_and_three_fits_of_pin():
    guide = _guide(bends=CURVED)
    key = fit_test_key(guide)
    assert key.angle_deg == FIT_TEST_ANGLE_DEG
    assert key.name == str(int(FIT_TEST_ANGLE_DEG))
    pins = [p for p in pin_tab_parts(guide) if p.name == "pin"]
    assert [round(0.5 * 2.7 - p.half[0], 3) for p in pins] == [round(v, 3) for v in PIN_TEST_PLAYS_MM]
    mesh = fit_test_ready_to_print(guide, BED)
    bounds = np.array(mesh.GetBounds()).reshape(3, 2)
    assert np.all(bounds[:, 1] <= np.array(BED)) and bounds[2, 0] == pytest.approx(0.0, abs=1e-6)


def test_the_export_writes_every_bed_the_fit_test_each_key_and_the_table(tmp_path):
    guide = _guide(n=12, bends={j: curl(10.0) for j in range(11)}, overbend=lambda a: 1.08 * a)
    stl, table = write_bending_guide(tmp_path / "guide.stl", guide, bed=BED, printer_name="Asiga MAX")
    assert stl.is_file() and ATTRIBUTION[:40] in read_stl_attribution(stl)
    test_piece = tmp_path / "guide-fit-test.stl"
    assert test_piece.is_file() and ATTRIBUTION[:40] in read_stl_attribution(test_piece)
    keys = sorted((tmp_path / "guide-keys").iterdir())
    assert [path.name for path in keys] == sorted(f"key-{key.name}.stl" for key in guide.keys)
    text = table.read_text(encoding="utf-8")
    assert text.splitlines()[0] == f"# {DISCLAIMER}"
    assert "guide-fit-test.stl" in text and "Asiga MAX" in text and "lies flat on the key" in text
    rows = list(csv.reader(line for line in text.splitlines() if not line.startswith("#")))
    assert rows[0][:3] == ["order", "key", "holes"]
    assert [row[1] for row in rows[1:]] == [key.name for key in guide.keys]
    # A key bends more than the shape needs, to spring back to it.
    assert all(float(row[5]) > float(row[4]) for row in rows[1:])


def test_each_plates_oval_holes_are_read_from_its_own_mesh():
    from mandiplan.plate_assets import asset_by_id, hole_length_mm

    gold = asset_by_id("synthes-angled-square-left-24h")
    assert 3.7 < hole_length_mm(gold) < 4.05
    assert hole_length_mm(asset_by_id("generic-recon-2.4-lp-12h")) == pytest.approx(2.5, abs=0.02)


def test_the_guide_for_every_catalogue_bender_builds():
    for kit in load_kits():
        if kit.head.kind == "none" or not kit.can_make("out_of_plane"):
            continue
        guide = _guide(kit=kit.id, bends={1: curl(12.0)})
        assert guide.keys and _closed(key_polydata(guide.keys[0]))


_ = vtk_to_numpy
