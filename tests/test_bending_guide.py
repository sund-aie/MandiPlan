"""The bend keys: made for one plate and one bender, each one followed through its press, and they print."""

from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest
import vtk

from mandiplan.bending_steps import steps_from_guide
from mandiplan.constants import ATTRIBUTION, DISCLAIMER
from mandiplan.exporting import read_stl_attribution, write_bending_guide, write_steps_csv
from mandiplan.geometry.bending_guide import (
    DESIGN_FORCE_N,
    DIAMOND_RELIEF_MM,
    FIT_TEST_ANGLE_DEG,
    KEY_STRESS_LIMIT_MPA,
    LOBE_GAP_MM,
    PIN_TEST_PLAYS_MM,
    TOP_CLEARANCE_MM,
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
        bend = (bends or {}).get(j, np.eye(3))
        # About the middle of the bridge, as a plate bends.
        places.append(places[j] + frames[j] @ (np.array([PITCH / 2, 0, 0]) + bend @ np.array([PITCH / 2, 0, 0])))
        frames.append(frames[-1] @ bend)
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


def twist(degrees):
    return _rotation([1, 0, 0], np.radians(degrees))


#: The plate wrapped round a jaw.
CURVED = {j: curl(14.0) for j in range(5)}
#: Bends either way, sharp and gentle, side by side: the hardest neighbours a key has.
MIXED = {0: curl(22.0), 1: curl(-18.0), 2: curl(9.0) @ twist(5.0), 3: curl(25.0), 4: curl(-12.0), 5: curl(16.0),
         6: turn(14.0) @ curl(10.0)}


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


def _press_keys(guide):
    return [key for key in guide.keys if not key.check]


def test_each_bridge_is_read_as_the_bend_table_reads_it():
    guide = _guide(bends={1: curl(15.0), 3: turn(12.0)}, overbend=lambda a: 1.05 * a)
    curled, turned = guide.joints[1], guide.joints[3]
    assert curled.out_of_plane_deg == pytest.approx(-15.0, abs=0.05)
    assert turned.in_plane_deg == pytest.approx(12.0, abs=0.05)
    assert curled.stop_angle_deg == pytest.approx(15.75)
    assert [abs(j.angle_deg) < 0.01 for j in (guide.joints[0], guide.joints[2], guide.joints[4])] == [True] * 3


def test_in_plane_bends_are_checked_on_trays_first_then_out_of_plane_ones_pressed():
    guide = _guide(bends={0: curl(10.0), 1: turn(25.0), 2: curl(0.4), 3: turn(25.0) @ curl(6.0)})
    assert [key.name for key in guide.keys] == ["A2", "A4", "B1", "B4"], "in the plate's plane first, as its maker says"
    assert [key.check for key in guide.keys] == [True, True, False, False]
    assert [key.number for key in guide.keys] == [1, 2, 3, 4]
    assert all(key.bridge == int(key.name[1:]) - 1 for key in guide.keys)


def test_a_bridge_of_a_straight_plate_needs_no_keys():
    assert not _guide().keys


def test_small_turns_are_gathered_into_few_keys_and_the_plate_stays_on_its_plan():
    from mandiplan.geometry.bending_guide import IN_PLANE_TOLERANCE_MM

    guide = _guide(n=14, bends={j: turn(3.0) for j in range(13)})
    assert 1 <= len(guide.keys) <= 5, "not a key at each of 13 bridges"
    assert guide.deviation_mm < 1.5 * IN_PLANE_TOLERANCE_MM
    total = sum(j.in_plane_made_deg for j in guide.joints)
    assert 24.0 < abs(total) <= 39.0, "nearly the whole turn, the ends of the curve being a little flattened"


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


def test_twist_with_no_bend_to_ride_on_is_carried_to_the_next_key():
    bends = {j: twist(1.2) for j in range(4)}
    bends[4] = curl(10.0)
    guide = _guide(n=6, bends=bends)
    assert sum(j.twist_made_deg for j in guide.joints) == pytest.approx(sum(j.twist_deg for j in guide.joints), abs=0.5)
    assert any(abs(j.out_of_plane_made_deg) < 1e-9 and abs(j.twist_made_deg) >= 3.0 for j in guide.joints)


def test_the_keys_in_their_order_take_the_fresh_plate_to_the_planned_shape():
    guide = _guide(bends={0: curl(10.0), 1: turn(8.0), 3: curl(14.0) @ turn(5.0)})
    fresh, _ = guide.stage()
    np.testing.assert_allclose(np.linalg.norm(np.diff(fresh, axis=0), axis=1), PITCH, atol=1e-6)
    done, _ = guide.stage(done=True)
    made = np.linalg.norm(done[:, None] - done[None, :], axis=2)
    planned = np.linalg.norm(guide.centres[:, None] - guide.centres[None, :], axis=2)
    assert np.abs(made - planned).max() < 2.0 * guide.deviation_mm + 0.05
    assert guide.deviation_mm < 1.0


def test_the_press_bends_the_plate_up_into_the_key_and_no_further():
    """The post pushes the plate's middle into the key's V: its two lobes swing down
    either side of the bridge, more and more, until they lie on the key's faces."""
    guide = _guide(bends={2: curl(18.0)}, overbend=lambda a: a + 1.5)
    key = guide.keys[0]
    angles, heights = [], []
    for fraction in np.linspace(0.0, 1.0, 6):
        places, frames = guide.pose(key, fraction)
        a, b = frames[key.bridge][:, 0], frames[key.bridge + 1][:, 0]
        angles.append(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))
        heights.append(0.5 * (places[key.bridge][2] + places[key.bridge + 1][2]))
        # A roof, never a valley: the holes either side of the bridge lie below it.
        bridge = 0.5 * (places[key.bridge] + places[key.bridge + 1])
        assert places[key.bridge - 1][2] <= bridge[2] + 1e-6 and places[key.bridge + 2][2] <= bridge[2] + 1e-6
    assert angles[0] == pytest.approx(0.0, abs=0.01), "it starts from the plate as it reached the key"
    assert all(b > a for a, b in zip(angles, angles[1:]))
    assert angles[-1] == pytest.approx(19.5, abs=0.05), "and stops at the key's angle"
    assert all(b >= a - 1e-6 for a, b in zip(heights, heights[1:])), "the post pushes it up into the key"
    check = guide.check(key)
    assert check.lobes_flat


def test_every_key_passes_its_press_whatever_its_neighbours_did():
    guide = _guide(n=9, bends=MIXED, overbend=lambda a: 1.08 * a + 0.5)
    assert len(_press_keys(guide)) >= 6
    for key in guide.keys:
        result = guide.check(key)
        assert result.ok, (key.name, result)
        if not key.check:
            assert result.plate_clear_of_key_mm >= -0.02, "the rest of the plate never touches the key"
            assert result.plate_clear_of_head_mm >= LOBE_GAP_MM - 0.05, "nor the bender's head"
            assert result.pin_clear_mm > 0.0, "and no pin binds in its hole on the way"
    assert not guide.check_warnings()


def test_a_twisted_bend_still_lies_flat_on_its_key():
    guide = _guide(bends={2: curl(12.0) @ twist(7.0)})
    key = guide.keys[0]
    assert guide.joints[2].twist_made_deg == pytest.approx(7.0, abs=0.5)
    assert guide.check(key).lobes_flat and guide.check(key).ok
    # The channel along the fold is deep enough for the step a twist leaves there.
    assert key.parts[0].channel_radius > _guide(bends={2: curl(12.0)}).keys[0].parts[0].channel_radius


def test_the_key_sits_up_in_the_arch_and_the_lobes_never_touch_anything():
    guide = _guide(bends={2: curl(16.0)})
    head = guide.head
    key = guide.keys[0]
    body = key.parts[0]
    assert head.kind == "three_point"
    # Shorter than the arch: the head bears on the key in its arch, not on its lobes.
    a = 0.5 * head.lobe_spacing_mm
    assert body.x_reach < a - 0.5 * head.lobe_radius_mm
    xs = np.linspace(-body.x_reach, body.x_reach, 41)
    np.testing.assert_allclose(body.top(xs), key.lobes_z + die_height(head, xs) - TOP_CLEARANCE_MM)
    assert guide.check(key).plate_clear_of_head_mm >= LOBE_GAP_MM - 0.05
    assert key.loading_depth_mm < head.opening_mm


def test_a_key_is_strong_enough_for_a_hard_squeeze():
    guide = _guide(n=8, bends={1: curl(8.0), 3: curl(18.0), 5: curl(26.0)})
    for key in guide.keys:
        assert 5.0 < guide.check(key).stress_mpa <= KEY_STRESS_LIMIT_MPA, key.name
    assert DESIGN_FORCE_N >= 2000.0


def test_the_pins_stand_in_the_plates_own_holes_the_second_flattened():
    guide = _guide(hole=2.7, bends={1: curl(12.0)}, play_mm=0.1)
    key = guide.keys[0]
    pins = [p for p in key.parts if p.name == "pin"]
    assert len(pins) == 2
    for pin, hole in zip(pins, (key.hole_a, key.hole_b)):
        assert pin.half[0] == pytest.approx(0.5 * 2.7 - 0.1)
        assert np.linalg.norm(np.cross(pin.centre - hole, pin.frame[:, 2])) < 1e-6, "on the hole's axis"
    assert pins[0].relief == 0.0 and pins[1].relief == pytest.approx(DIAMOND_RELIEF_MM)
    # Flattened along the plate, full across it.
    pin = pins[1]
    base = pin.centre - pin.frame[:, 2] * (pin.half[2] - 0.01)
    assert pin.field(tuple((base + pin.frame[:, 0] * (pin.half[0] - 0.2) - pin.centre) @ pin.frame)) > 0
    assert pin.field(tuple((base + pin.frame[:, 1] * (pin.half[0] - 0.2) - pin.centre) @ pin.frame)) < 0
    assert guide.pin_diameter_mm == pytest.approx(2.5)


def test_oval_holes_get_oval_pins():
    guide = _guide(hole=2.7, hole_length_mm=3.9, bends={1: curl(12.0)})
    assert guide.hole_stretch_mm == pytest.approx(1.2)
    pin = next(p for p in guide.keys[0].parts if p.name == "pin")
    assert pin.stretch == pytest.approx(1.2)
    base = pin.centre - pin.frame[:, 2] * (pin.half[2] - 0.01)
    assert pin.field(tuple((base + pin.frame[:, 0] * 1.8 - pin.centre) @ pin.frame)) < 0
    assert pin.field(tuple((base + pin.frame[:, 1] * 1.8 - pin.centre) @ pin.frame)) > 0
    assert guide.check(guide.keys[0]).pin_clear_mm > 0.0


def test_a_check_key_holds_four_holes_where_the_turn_puts_them():
    guide = _guide(n=8, bends={3: turn(20.0)})
    key = guide.keys[0]
    assert key.check and key.pinned == (2, 3, 4, 5)
    pins = [p for p in key.parts if p.name == "pin"]
    assert len(pins) == 4 and pins[0].relief == 0.0 and all(p.relief > 0.0 for p in pins[1:])
    places, frames = guide.pose(key, 1.0)
    # The plate as it should come out lies on the tray, its pins in its holes.
    for pin, i in zip(pins, key.pinned):
        assert np.linalg.norm(np.cross(pin.centre - places[i], pin.frame[:, 2])) < 1e-6
    check = guide.check(key)
    assert check.ok and check.pin_clear_mm > 0.0 and abs(check.plate_clear_of_key_mm) < 0.02
    # Turned the wrong amount, its third and fourth holes would miss their pins.
    wrong = _guide(n=8, bends={3: turn(16.0)}).pose(_guide(n=8, bends={3: turn(16.0)}).keys[0], 1.0)[0]
    assert np.linalg.norm(wrong[5][:2] - places[5][:2]) > 0.3
    assert not tool_parts(guide, key)


def test_a_check_key_at_the_end_of_the_plate_holds_the_holes_there_are():
    guide = _guide(n=6, bends={0: turn(15.0)})
    assert guide.keys[0].pinned == (0, 1, 2)


def test_each_key_is_one_closed_solid_with_its_name_on_it():
    guide = _guide(bends={1: curl(14.0), 3: turn(25.0) @ curl(8.0)})
    assert [key.name for key in guide.keys] == ["A4", "B2", "B4"]
    for key in guide.keys:
        mesh = key_polydata(key)
        assert _closed(mesh) and _regions(mesh) == 1
        assert mesh.GetNumberOfCells() < 40_000, "a key is a small file"
        assert sum(1 for p in key.parts if p.name == "digit") >= 4


def test_a_press_key_prints_with_the_faces_the_plate_lies_on_standing_upright():
    guide = _guide(bends={1: curl(14.0)})
    key = guide.keys[0]
    rotation = key_print_rotation(key)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    for _, n in key.faces:
        assert abs((rotation @ n)[2]) < 1e-6


def test_a_check_key_prints_as_it_is_used_pins_up():
    guide = _guide(bends={1: turn(16.0)})
    key = guide.keys[0]
    assert key.check
    np.testing.assert_allclose(key_print_rotation(key), np.eye(3))
    for pin in (p for p in key.parts if p.name == "pin"):
        assert pin.frame[2, 2] > 0.99


def test_the_guide_is_made_for_the_plate_and_the_kit_chosen():
    bends = {1: curl(14.0)}
    large = _guide(bends=bends)
    small = _guide(kit="bending-pliers", bends=bends)
    assert large.keys[0].lobes_z != small.keys[0].lobes_z, "each nests in its own bender's head"
    narrower = plan_bending_guide(*_plate(6, bends), 6.0, 2.0, 2.0, kit_by_id("synthes-329-142"))
    assert narrower.pin_diameter_mm < large.pin_diameter_mm
    assert narrower.keys[0].faces[0][0][2] != large.keys[0].faces[0][0][2]


def test_a_measured_head_takes_the_place_of_the_kits():
    head = replace(kit_by_id("synthes-329-142").head, arch_rise_mm=12.0, source="measured")
    guide = _guide(bends={1: curl(14.0)}, head=head)
    assert guide.head.arch_rise_mm == 12.0
    standard = _guide(bends={1: curl(14.0)})
    assert guide.keys[0].parts[0].top(0.0) > standard.keys[0].parts[0].top(0.0) + 2.0, "it fills the deeper arch"


def test_irons_lever_the_plate_down_onto_a_key_on_the_concave_side():
    guide = _guide(kit="bending-irons", bends={1: curl(12.0)})
    key = guide.keys[0]
    assert guide.head.kind == "lever"
    assert "Lever the plate" in guide.instruction(key)
    assert not tool_parts(guide, key)
    assert _closed(key_polydata(key))
    assert key.from_bone_side != _guide(bends={1: curl(12.0)}).keys[0].from_bone_side


def test_a_kit_with_nothing_to_stop_is_refused_in_plain_words():
    wire = next(kit for kit in load_kits() if kit.head.kind == "none")
    with pytest.raises(ValueError, match="by hand"):
        plan_bending_guide(*_plate(6, {1: curl(12.0)}), WIDTH, THICK, 2.7, wire)


def test_a_head_that_cannot_open_enough_is_said():
    head = replace(kit_by_id("synthes-329-142").head, opening_mm=2.0)
    guide = _guide(bends={1: curl(14.0)}, head=head)
    assert any("open to go in" in warning for warning in guide.warnings)
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
    guide = _guide(bends={**CURVED, 4: turn(18.0) @ curl(14.0)})
    steps = guide.steps()
    assert steps[0].key is None and "fresh plate" in steps[0].text
    assert [step.key.name for step in steps[1:-1]] == [key.name for key in guide.keys]
    assert steps[-1].key is None
    check = next(step.text for step in steps if step.key is not None and step.key.check)
    assert "in-plane bender" in check and "lies flat on the key" in check
    press = next(step.text for step in steps if step.key is not None and not step.key.check)
    assert "flattened pin" in press and "arch" in press and "stops there" in press


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
        keys_ready_to_print(guide, (14.0, 14.0, 14.0))


def test_the_fit_test_has_a_known_bend_its_gauge_and_three_fits_of_pin():
    guide = _guide(bends=CURVED, overbend=lambda a: 1.1 * a)
    key = fit_test_key(guide)
    assert key.angle_deg == pytest.approx(FIT_TEST_ANGLE_DEG, abs=0.05)
    assert key.stop_angle_deg == pytest.approx(1.1 * FIT_TEST_ANGLE_DEG, abs=0.05), "springback allowed as for the rest"
    assert key.name == str(int(FIT_TEST_ANGLE_DEG))
    pins = [p for p in pin_tab_parts(guide) if p.name == "pin"]
    assert [round(0.5 * 2.7 - p.half[0], 3) for p in pins] == [round(v, 3) for v in PIN_TEST_PLAYS_MM]
    mesh = fit_test_ready_to_print(guide, BED)
    bounds = np.array(mesh.GetBounds()).reshape(3, 2)
    assert np.all(bounds[:, 1] <= np.array(BED)) and bounds[2, 0] == pytest.approx(0.0, abs=1e-6)


def test_the_export_writes_every_bed_the_fit_test_each_key_and_the_checks(tmp_path):
    guide = _guide(n=12, bends={**{j: curl(10.0) for j in range(11)}, 5: turn(18.0) @ curl(10.0)},
                   overbend=lambda a: 1.08 * a)
    stl, table = write_bending_guide(tmp_path / "guide.stl", guide, bed=BED, printer_name="Asiga MAX")
    assert stl.is_file() and ATTRIBUTION[:40] in read_stl_attribution(stl)
    test_piece = tmp_path / "guide-fit-test.stl"
    assert test_piece.is_file() and ATTRIBUTION[:40] in read_stl_attribution(test_piece)
    keys = sorted((tmp_path / "guide-keys").iterdir())
    assert [path.name for path in keys] == sorted(f"key-{key.name}.stl" for key in guide.keys)
    text = table.read_text(encoding="utf-8")
    assert text.splitlines()[0] == f"# {DISCLAIMER}"
    assert "guide-fit-test.stl" in text and "Asiga MAX" in text and "lies flat on the key" in text
    assert "checked: each B key's press was followed" in text
    rows = list(csv.reader(line for line in text.splitlines() if not line.startswith("#")))
    assert rows[0][:3] == ["order", "key", "type"]
    assert [row[1] for row in rows[1:]] == [key.name for key in guide.keys]
    assert {row[2] for row in rows[1:]} == {"check", "press"}
    assert all(row[15] == "yes" for row in rows[1:])
    # A key bends more than the shape needs, to spring back to it.
    assert all(float(row[7]) > float(row[6]) for row in rows[1:] if row[2] == "press")


def test_each_plates_oval_holes_are_read_from_its_own_mesh():
    from mandiplan.plate_assets import asset_by_id, hole_length_mm

    gold = asset_by_id("synthes-angled-square-left-24h")
    assert 3.7 < hole_length_mm(gold) < 4.05
    assert hole_length_mm(asset_by_id("generic-recon-2.4-lp-12h")) == pytest.approx(2.5, abs=0.02)


def test_the_guide_for_every_catalogue_bender_builds_and_passes():
    for kit in load_kits():
        if kit.head.kind == "none" or not kit.can_make("out_of_plane"):
            continue
        guide = _guide(kit=kit.id, bends={1: curl(12.0), 3: curl(-10.0)})
        assert guide.keys and all(_closed(key_polydata(key)) for key in guide.keys)
        assert all(guide.check(key).plate_clear_of_key_mm >= -0.02 for key in guide.keys), kit.id
