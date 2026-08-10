"""Plate catalogue, fit checking and bench instructions."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import circular_arc_path
from mandiplan.bending_steps import anatomical_direction, generate_steps
from mandiplan.geometry.plate import compute_plate_plan
from mandiplan.plate_catalog import (
    fit_check,
    kit_by_id,
    load_kits,
    load_systems,
    system_by_id,
)

RADIUS = 30.0


def _plan(system, span_rad: float = 2.0, normals_axis=(0.0, 0.0, 1.0)):
    path = circular_arc_path(RADIUS, span_rad)
    normals = np.tile(normals_axis, (len(path), 1))
    return compute_plate_plan(path, normals, system.hole_pitch_mm)


def test_every_system_has_usable_dimensions():
    systems = load_systems()
    assert len(systems) >= 3
    for system in systems:
        assert system.hole_pitch_mm > 0
        assert system.width_mm > 0
        assert system.thickness_mm > 0
        assert system.min_holes_per_side >= 1
        assert all(h > 0 for h in system.hole_counts)


def test_plate_lengths_grow_with_the_hole_count():
    system = system_by_id("recon-2.4-bar")
    lengths = [option.length_mm for option in system.options()]
    assert lengths == sorted(lengths)
    assert system.length_for(2) == pytest.approx(system.hole_pitch_mm + 2 * system.end_margin_mm)


def test_the_smallest_sufficient_length_is_chosen():
    system = system_by_id("recon-2.4-bar")
    option = system.smallest_option_for(70.0)
    assert option is not None
    assert option.length_mm >= 70.0
    smaller = [o for o in system.options() if o.length_mm < option.length_mm]
    assert all(o.length_mm < 70.0 for o in smaller)


def test_a_path_that_fits_reports_the_length_to_take():
    system = system_by_id("recon-2.4-bar")
    plan = _plan(system)
    report = fit_check(plan, system)
    assert report.option is not None
    assert report.fits
    assert f"{report.option.holes}-hole" in report.verdict
    assert "mm" in report.verdict


def test_a_path_longer_than_the_system_demands_a_custom_plate():
    system = system_by_id("recon-2.7-bar")
    longest = system.longest_option().length_mm
    plan = _plan(system, span_rad=(longest + 60.0) / RADIUS)
    report = fit_check(plan, system)
    assert report.option is None
    assert not report.fits
    assert report.uncovered_mm > 0
    assert "custom plate" in report.verdict


def test_a_custom_plate_always_fits():
    system = system_by_id("custom-psi")
    plan = _plan(system, span_rad=3.0)
    report = fit_check(plan, system)
    assert report.fits
    assert "Custom plate" in report.verdict


def test_too_few_screw_holes_on_one_side_is_a_problem(arch_frames, spec):
    """Nearly the whole path lies over the defect, so there is nothing to fix to."""
    system = system_by_id("recon-2.4-bar")
    path = np.array(
        [
            spec.point_at_arc(s) + np.array([0.0, 0.0, 0.0])
            for s in np.linspace(6.0, spec.arc_length_mm - 6.0, 60)
        ]
    )
    normals = np.tile([0.0, 0.0, 1.0], (len(path), 1))
    plan = compute_plate_plan(path, normals, system.hole_pitch_mm)

    whole = (4.0, spec.arc_length_mm - 20.0)
    report = fit_check(plan, system, defect_span=whole, frames=arch_frames)
    assert report.holes_over_defect > 0
    assert any("retained bone proximal" in p for p in report.problems)


def test_holes_are_counted_either_side_of_a_small_defect(arch_frames, spec):
    system = system_by_id("recon-2.4-bar")
    path = np.array(
        [spec.point_at_arc(s) for s in np.linspace(4.0, spec.arc_length_mm - 4.0, 80)]
    )
    normals = np.tile([0.0, 0.0, 1.0], (len(path), 1))
    plan = compute_plate_plan(path, normals, system.hole_pitch_mm)

    mid = spec.arc_length_mm / 2.0
    report = fit_check(plan, system, defect_span=(mid - 8.0, mid + 8.0), frames=arch_frames)
    assert report.holes_proximal >= 3
    assert report.holes_distal >= 3
    assert not report.problems


def test_steep_bends_raise_a_warning():
    system = system_by_id("recon-2.7-bar")
    # A tight arch turns far more per node than the working limit allows.
    path = circular_arc_path(12.0, 2.4)
    normals = np.tile([0.0, 0.0, 1.0], (len(path), 1))
    plan = compute_plate_plan(path, normals, system.hole_pitch_mm)
    report = fit_check(plan, system)
    assert any("working limit" in w for w in report.warnings)


def test_anatomical_directions_are_named_in_patient_axes():
    assert anatomical_direction([1, 0, 0]) == "toward the patient's left"
    assert anatomical_direction([-1, 0, 0]) == "toward the patient's right"
    assert anatomical_direction([0, 1, 0]) == "posteriorly"
    assert anatomical_direction([0, -0.2, 0]) == "anteriorly"
    assert anatomical_direction([0, 0, 3]) == "superiorly"
    assert anatomical_direction([0.1, 0, -3]) == "inferiorly"


def test_bending_steps_cover_every_bend_and_measure_from_the_cut_end():
    system = system_by_id("recon-2.4-bar")
    kit = kit_by_id("bending-irons")
    plan = _plan(system)
    report = fit_check(plan, system)
    steps = generate_steps(plan, system, kit, report)

    assert steps[0].kind == "cut"
    assert steps[-1].kind == "check"
    bends = [s for s in steps if s.kind == "bend"]
    interior = [b for b in plan.bends if np.isfinite(b.in_plane_deg)]
    assert len(bends) == len(interior)
    # Distances are measured from the plate's cut end, not from the first hole.
    first_bend = bends[0]
    node = interior[0]
    assert first_bend.distance_mm == pytest.approx(
        node.cumulative_mm + system.end_margin_mm
    )
    assert "Mark at" in first_bend.text
    assert kit.instruments[0] in first_bend.instrument


def test_a_kit_that_cannot_twist_says_so():
    system = system_by_id("recon-2.4-bar")
    irons = kit_by_id("bending-irons")
    # A path whose surface normal rolls along it needs a twist at every node.
    s = np.arange(0.0, 70.0, 0.2)
    path = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
    roll = np.radians(2.0) * s
    normals = np.column_stack([np.zeros_like(s), np.sin(roll), np.cos(roll)])
    plan = compute_plate_plan(path, normals, system.hole_pitch_mm)

    steps = generate_steps(plan, system, irons, fit_check(plan, system))
    twists = [s for s in steps if s.kind == "twist"]
    assert twists
    assert all("cannot twist" in step.text for step in twists)

    forceps = kit_by_id("twisting-forceps")
    steps = generate_steps(plan, system, forceps, fit_check(plan, system))
    twists = [s for s in steps if s.kind == "twist"]
    assert twists
    assert all("twist" in step.text and "cannot" not in step.text for step in twists)


def test_a_custom_plate_gets_no_bending_steps():
    system = system_by_id("custom-psi")
    kit = kit_by_id("bending-irons")
    plan = _plan(system)
    steps = generate_steps(plan, system, kit, fit_check(plan, system))
    assert len(steps) == 1
    assert "no bending steps" in steps[0].text


def test_a_pre_bent_bar_tells_you_it_is_pre_bent():
    system = system_by_id("angle-2.4-prebent")
    kit = kit_by_id("bending-irons")
    plan = _plan(system)
    steps = generate_steps(plan, system, kit, fit_check(plan, system))
    assert any("pre-formed" in step.text for step in steps)


def test_bending_passes_respect_the_kit_limit():
    system = system_by_id("recon-2.0-plate")
    kit = kit_by_id("bending-pliers")
    path = circular_arc_path(14.0, 2.0)
    normals = np.tile([0.0, 0.0, 1.0], (len(path), 1))
    plan = compute_plate_plan(path, normals, system.hole_pitch_mm)
    steps = generate_steps(plan, system, kit, fit_check(plan, system))
    multi = [s for s in steps if "passes of about" in s.text]
    assert multi, "a 32 deg bend should be split into passes for these pliers"


def test_every_kit_declares_what_it_can_do():
    for kit in load_kits():
        assert kit.instruments
        assert kit.handles
        assert kit.can_twist == ("twist" in kit.handles)
