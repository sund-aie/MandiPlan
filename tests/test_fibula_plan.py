"""Planning a fibula graft: the course through the defect, the pieces, the cuts."""

from __future__ import annotations

import numpy as np
import pytest

from mandiplan.geometry import cpr
from mandiplan.geometry.fibula import Fibula
from mandiplan.geometry.fibula_plan import (
    choose_joints,
    defect_course,
    fit_joints,
    plan_fibula,
)
from mandiplan.geometry.resection import CutPlane
from mandiplan.geometry.spline import ArchCurve

HEIGHT, WIDTH, RADIUS = 30.0, 12.0, 45.0
FIBULA_R = 7.0


def _horseshoe():
    """A U of a jaw, a body of elliptical section round a circular arch."""
    theta = np.radians(np.linspace(-100.0, 100.0, 21))
    seeds = np.column_stack([RADIUS * np.sin(theta), -RADIUS * np.cos(theta), np.zeros_like(theta)])
    frames = cpr.build_frames(ArchCurve(seeds), 0.5)
    rings = []
    angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    for s in np.arange(0.0, frames.length_mm, 0.5):
        at = frames.point_at(float(s))
        _, up, side = frames.frame_at(float(s))
        rings.append(at + 0.5 * HEIGHT * np.cos(angles)[:, None] * up + 0.5 * WIDTH * np.sin(angles)[:, None] * side)
    return frames, np.vstack(rings)


@pytest.fixture(scope="module")
def jaw():
    return _horseshoe()


def _cuts(frames, a, b):
    ta, _, _ = frames.frame_at(a)
    tb, _, _ = frames.frame_at(b)
    # A cut's normal points into the piece taken out.
    return [CutPlane(frames.point_at(a), ta), CutPlane(frames.point_at(b), -tb)]


def _fibula(length=340.0, radius=FIBULA_R):
    z = np.arange(0.0, length + 1e-9, 1.0)
    points = np.column_stack([np.full_like(z, -22.0), np.full_like(z, 8.0), z])
    return Fibula("right", points, z.copy(), np.full_like(z, radius), np.array([-1.0, 0.0, 0.0]))


def _plan(jaw, a=40.0, b=120.0, **kwargs):
    frames, vertices = jaw
    course = defect_course(vertices, frames, _cuts(frames, a, b), FIBULA_R)
    joints, deviation = choose_joints(course.points)
    return course, plan_fibula(course, _fibula(), joints, deviation, **kwargs)


def test_the_course_runs_a_fibula_above_the_lower_border_and_in_from_the_outer_face(jaw):
    frames, vertices = jaw
    planes = _cuts(frames, 40.0, 120.0)
    course = defect_course(vertices, frames, planes, FIBULA_R)
    assert course is not None
    # From cut to cut.
    assert abs(planes[0].signed_distance(course.points[0])) < 1e-6
    assert abs(planes[1].signed_distance(course.points[-1])) < 1e-6
    for point in course.points[5:-5:7]:
        s = float(frames.s[np.argmin(np.linalg.norm(frames.points - point, axis=1))])
        at = frames.point_at(s)
        _, up, side = frames.frame_at(s)
        # Out of the jaw: away from the middle of the U.
        out = side if float(at[:2] @ side[:2]) >= 0.0 else -side
        u = float((point - at) @ up)
        b = float((point - at) @ out)
        # The section is an ellipse: its lower border at -HEIGHT / 2.
        assert u == pytest.approx(-0.5 * HEIGHT + FIBULA_R, abs=0.6)
        # The outer face at that height, less the fibula's radius: here the
        # fibula is wider than the jaw, so its middle is inside the arch line.
        face = 0.5 * WIDTH * np.sqrt(1.0 - (u / (0.5 * HEIGHT)) ** 2)
        assert b == pytest.approx(face - FIBULA_R, abs=0.6)


def test_raising_the_fibula_lifts_the_whole_course(jaw):
    frames, vertices = jaw
    planes = _cuts(frames, 40.0, 120.0)
    low = defect_course(vertices, frames, planes, FIBULA_R)
    high = defect_course(vertices, frames, planes, FIBULA_R, raise_mm=8.0)
    middle = len(low.points) // 2
    lift = (high.points[middle] - low.points[middle]) @ low.up[middle]
    assert lift == pytest.approx(8.0, abs=0.7)


def test_a_cut_through_one_limb_is_not_taken_for_the_other(jaw):
    frames, vertices = jaw
    # Near the end of the U a cut's plane also crosses the far limb.
    course = defect_course(vertices, frames, _cuts(frames, 30.0, 150.0), FIBULA_R)
    assert course is not None
    assert course.length_mm == pytest.approx(117.0, abs=4.0)


def test_chords_along_an_arc_are_even_and_stray_by_the_sagitta():
    radius, sweep = 50.0, np.radians(120.0)
    t = np.linspace(0.0, sweep, 211)
    arc = np.column_stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)])
    joints, deviation = fit_joints(arc, 3, 20.0)
    chords = np.linalg.norm(np.diff(arc[joints], axis=0), axis=1)
    assert np.ptp(chords) < 1.5
    # Each chord spans a third of the sweep: its sagitta is the largest gap.
    assert deviation == pytest.approx(radius * (1.0 - np.cos(sweep / 6.0)), abs=0.05)


def test_no_piece_is_shorter_than_the_least_a_piece_may_be():
    t = np.linspace(0.0, np.radians(90.0), 120)
    arc = np.column_stack([40.0 * np.cos(t), 40.0 * np.sin(t), np.zeros_like(t)])
    # 63 mm of arc: four pieces of at least 20 mm cannot be had, three can.
    assert fit_joints(arc, 4, 20.0) is None
    joints, _ = fit_joints(arc, 3, 20.0)
    assert np.linalg.norm(np.diff(arc[joints], axis=0), axis=1).min() >= 20.0 - 1e-9


def test_the_fewest_pieces_that_keep_within_the_tolerance_are_chosen():
    straight = np.column_stack([np.linspace(0.0, 60.0, 61), np.zeros(61), np.zeros(61)])
    joints, deviation = choose_joints(straight)
    assert len(joints) == 2 and deviation < 1e-9
    t = np.linspace(0.0, np.radians(120.0), 160)
    arc = np.column_stack([45.0 * np.cos(t), 45.0 * np.sin(t), np.zeros_like(t)])
    joints, deviation = choose_joints(arc, tolerance_mm=2.0)
    assert deviation <= 2.0
    fewer = fit_joints(arc, len(joints) - 2, 20.0)
    assert fewer is None or fewer[1] > 2.0


def test_each_piece_carries_its_stretch_of_fibula_onto_its_chord(jaw):
    _, plan = _plan(jaw)
    fibula = _fibula()
    assert len(plan.pieces) == 3
    for piece in plan.pieces:
        ends = piece.to_jaw.apply(np.vstack([fibula.point_at(piece.s_distal), fibula.point_at(piece.s_proximal)]))
        # The vessels leave from the second cut: the proximal end is towards it.
        assert np.allclose(ends, [piece.start, piece.end], atol=1e-6)
        assert piece.s_proximal - piece.s_distal == pytest.approx(piece.length_mm, abs=1e-3)


def test_the_vessels_end_decides_which_piece_comes_from_highest_up_the_leg(jaw):
    _, second = _plan(jaw, pedicle_end="second")
    _, first = _plan(jaw, pedicle_end="first")
    assert max(second.pieces, key=lambda p: p.s_proximal).number == 3
    assert max(first.pieces, key=lambda p: p.s_proximal).number == 1
    fibula = _fibula()
    piece = first.pieces[0]
    proximal = piece.to_jaw.apply(fibula.point_at(piece.s_proximal))
    assert np.allclose(proximal, piece.start, atol=1e-6)


def test_at_each_joint_the_two_ends_are_one_plane_in_the_jaw_each_leaning_half_the_angle(jaw):
    _, plan = _plan(jaw)
    for k in range(len(plan.pieces) - 1):
        lower, upper = plan.pieces[k], plan.pieces[k + 1]  # course order: k ends where k + 1 starts
        end = lower.proximal_face
        start = upper.distal_face
        n_end = lower.to_jaw.rotation @ end.normal
        n_start = upper.to_jaw.rotation @ start.normal
        assert float(n_end @ n_start) == pytest.approx(-1.0, abs=1e-9)
        assert np.allclose(lower.to_jaw.apply(end.point), upper.to_jaw.apply(start.point), atol=1e-6)
        half = 0.5 * plan.joint_angles_deg[k]
        assert end.lean_deg == pytest.approx(half, abs=1e-6)
        assert start.lean_deg == pytest.approx(half, abs=1e-6)


def test_no_piece_is_twisted_on_the_next(jaw):
    _, plan = _plan(jaw)
    fibula = _fibula()
    for k in range(len(plan.pieces) - 1):
        lower, upper = plan.pieces[k], plan.pieces[k + 1]
        # The fibula's lateral side, as each piece carries it into the jaw ...
        side_lower = lower.to_jaw.rotation @ fibula.lateral_at(lower.s_distal)
        side_upper = upper.to_jaw.rotation @ fibula.lateral_at(upper.s_distal)
        d0 = (lower.end - lower.start) / lower.length_mm
        d1 = (upper.end - upper.start) / upper.length_mm
        # ... is the lower piece's carried by the least turn from one to the next.
        axis = np.cross(d0, d1)
        angle = np.arctan2(np.linalg.norm(axis), d0 @ d1)
        axis /= np.linalg.norm(axis)
        carried = (
            side_lower * np.cos(angle)
            + np.cross(axis, side_lower) * np.sin(angle)
            + axis * (axis @ side_lower) * (1.0 - np.cos(angle))
        )
        assert np.allclose(carried, side_upper, atol=1e-9)
    # Round a flat arch the lateral side faces straight out in every piece.
    assert max(plan.facing_deg) < 1.0


def test_on_the_leg_the_cuts_of_a_joint_never_meet_inside_the_bone(jaw):
    _, plan = _plan(jaw)
    fibula = _fibula()
    up_the_leg = plan.pieces_up_the_leg()
    angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    for below, above in zip(up_the_leg, up_the_leg[1:]):
        p, q = below.proximal_face, above.distal_face
        axis = np.array([0.0, 0.0, 1.0])
        # Across the bone's section, a ring a third wider than it.
        rim = (4.0 / 3.0) * FIBULA_R * np.column_stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)])
        tp = p.point[2] - (rim @ p.normal) / (axis @ p.normal)
        tq = q.point[2] - (rim @ q.normal) / (axis @ q.normal)
        assert (tq - tp).min() >= 1.0, "the two cuts keep apart all the way across"


def test_the_graft_starts_where_the_ankle_keeps_its_fibula(jaw):
    _, plan = _plan(jaw, distal_keep_mm=60.0)
    assert plan.harvest_mm[0] == pytest.approx(60.0, abs=0.1)
    assert plan.harvest_mm[1] < 340.0 - 60.0
    assert not plan.warnings


def test_a_graft_longer_than_the_fibula_can_give_is_said(jaw):
    frames, vertices = jaw
    course = defect_course(vertices, frames, _cuts(frames, 20.0, 140.0), FIBULA_R)
    joints, deviation = choose_joints(course.points)
    plan = plan_fibula(course, _fibula(length=220.0), joints, deviation)
    assert any("can be taken" in w for w in plan.warnings)
