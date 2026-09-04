"""Resection: which fragment goes, how big it is, how long the gap is."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import rel_error
from mandiplan.geometry.resection import (
    CutPlane,
    PlanePlacement,
    build_report,
    plane_from_frame,
    plane_from_placement,
    resected_mask,
)
from mandiplan.render.surface import clip_closed, mesh_volume_mm3

# Cut angles measured from the middle of the arch (the symphysis).
THETA_A_DEG = -20.0
THETA_B_DEG = 20.0


def _radial_plane(spec, offset_deg: float, into_positive_theta: bool) -> CutPlane:
    """Plane containing the arch axis, cutting the sweep ``offset_deg`` from
    the middle of the arch.

    Its normal is the tangential direction, pointing into the fragment that is
    being removed.
    """
    theta_deg = spec.arch_centre_deg + offset_deg
    t = np.radians(theta_deg)
    tangential = np.array([-np.sin(t), np.cos(t), 0.0])
    normal = tangential if into_positive_theta else -tangential
    origin = np.array([spec.centre_xy[0], spec.centre_xy[1], spec.centre_z])
    return CutPlane(origin=origin, normal=normal, label=f"cut@{offset_deg:+.0f}deg")


@pytest.fixture(scope="module")
def cut_planes(spec):
    return [
        _radial_plane(spec, THETA_A_DEG, into_positive_theta=True),
        _radial_plane(spec, THETA_B_DEG, into_positive_theta=False),
    ]


def test_resected_fragment_volume_matches_the_analytic_wedge(
    bone_surface, cut_planes, spec
):
    fragment = clip_closed(bone_surface, cut_planes, keep_resected=True)
    measured = mesh_volume_mm3(fragment)
    span = np.radians(THETA_B_DEG - THETA_A_DEG)
    analytic = (
        np.pi * spec.semi_axis_bl_mm * spec.semi_axis_si_mm * spec.radius_mm * span
    )
    assert rel_error(measured, analytic) < 0.03


def test_whole_bone_volume_matches_the_analytic_solid(bone_surface, spec):
    assert rel_error(mesh_volume_mm3(bone_surface), spec.volume_mm3) < 0.03


def test_retained_bone_is_the_rest_of_the_mandible(bone_surface, cut_planes):
    whole = mesh_volume_mm3(bone_surface)
    fragment = mesh_volume_mm3(clip_closed(bone_surface, cut_planes, keep_resected=True))
    retained = mesh_volume_mm3(clip_closed(bone_surface, cut_planes, keep_resected=False))
    assert rel_error(fragment + retained, whole) < 0.01


def test_resection_span_along_the_arch_curve(arch_frames, cut_planes, spec):
    report = build_report(arch_frames, cut_planes)
    analytic = spec.radius_mm * np.radians(THETA_B_DEG - THETA_A_DEG)
    assert rel_error(report.arc_length_mm, analytic) < 0.02
    # Across a curved mandible the chord is meaningfully shorter than the arc.
    assert report.straight_length_mm < report.arc_length_mm
    assert rel_error(report.straight_length_mm, 2 * spec.radius_mm * np.sin(analytic / (2 * spec.radius_mm))) < 0.02


def test_single_plane_removes_one_side(arch_frames, spec):
    plane = _radial_plane(spec, 0.0, into_positive_theta=True)
    mask = resected_mask([plane], arch_frames.points)
    assert mask.any() and not mask.all()
    # Everything kept is on one side, everything removed on the other.
    d = plane.signed_distance(arch_frames.points)
    assert np.all(d[mask] >= 0) and np.all(d[~mask] < 0)


def test_margins_report_signed_distance_to_each_plane(arch_frames, cut_planes, spec):
    lesion = spec.point_at_arc(spec.arc_length_mm / 2.0)
    report = build_report(arch_frames, cut_planes, landmarks=[("lesion", lesion)])
    assert len(report.margins_mm) == 2
    for _, name, distance in report.margins_mm:
        assert name == "lesion"
        assert distance > 0  # the lesion sits inside the resected fragment


def test_no_planes_means_nothing_is_resected(arch_frames):
    report = build_report(arch_frames, [])
    assert not np.isfinite(report.arc_length_mm)
    assert not resected_mask([], arch_frames.points).any()


# -- placing a cut by numbers rather than by dragging it ---------------------


def test_a_plane_placed_by_numbers_sits_on_the_curve(arch_frames):
    origin, normal = plane_from_frame(arch_frames, 30.0)
    index = arch_frames.index_of(30.0)
    assert np.allclose(origin, arch_frames.points[index])
    assert np.allclose(normal, arch_frames.tangents[index])
    assert np.linalg.norm(normal) == pytest.approx(1.0)


def test_yaw_turns_the_cut_about_the_superior_axis(arch_frames):
    index = arch_frames.index_of(30.0)
    tangent = arch_frames.tangents[index]
    _, normal = plane_from_frame(arch_frames, 30.0, yaw_deg=25.0)
    assert normal[2] == pytest.approx(0.0, abs=1e-9)  # still horizontal
    turned = np.degrees(np.arccos(np.clip(np.dot(normal, tangent), -1, 1)))
    assert turned == pytest.approx(25.0, abs=1e-6)


def test_tilt_lifts_the_cut_out_of_the_axial_plane(arch_frames):
    _, normal = plane_from_frame(arch_frames, 30.0, tilt_deg=20.0)
    assert normal[2] == pytest.approx(np.sin(np.radians(20.0)), abs=1e-6)
    _, opposite = plane_from_frame(arch_frames, 30.0, tilt_deg=-20.0)
    assert opposite[2] == pytest.approx(-np.sin(np.radians(20.0)), abs=1e-6)


def test_the_offset_moves_the_cut_in_patient_axes(arch_frames):
    base, base_normal = plane_from_frame(arch_frames, 30.0)
    moved, moved_normal = plane_from_frame(arch_frames, 30.0, offset_mm=(2.0, -3.0, 4.0))
    assert np.allclose(moved - base, [2.0, -3.0, 4.0])
    assert np.allclose(moved_normal, base_normal)


def _obliquity_deg(frames, session, index: int) -> float:
    """Angle between a cut's normal and the arch tangent where the cut sits.

    This — not the world-space normal — is the quantity a surgeon dials in and
    expects to survive a move along the jaw.
    """
    s_mm = session.plane_arc_position(index)
    tangent, _, _ = frames.frame_at(s_mm)
    normal = session.planes[index].normal
    return float(
        np.degrees(np.arccos(abs(np.clip(np.dot(tangent, normal), -1.0, 1.0))))
    )


def _seeded_session(frames, s_mm: float):
    from mandiplan.ui.session import Session

    session = Session()
    session.frames = frames
    index = frames.index_of(s_mm)
    session.add_plane(frames.points[index], frames.tangents[index])
    return session


def test_translating_a_cut_keeps_its_angulation_relative_to_the_jaw(arch_frames):
    """A cut travels along the mandible carrying its obliquity, not its normal.

    Set 30 degrees of yaw at one point and slide the cut: at every new
    position the plane is re-derived from the local mandibular frame there, so
    it stays 30 degrees oblique *to the jaw* while its world-space normal
    changes with the curve.
    """
    mid = arch_frames.length_mm / 2.0
    session = _seeded_session(arch_frames, mid)

    session.rotate_plane(0, yaw_deg=30.0, tilt_deg=10.0)
    reference = _obliquity_deg(arch_frames, session, 0)
    world_normal = session.planes[0].normal.copy()

    for position in (mid - 15.0, mid + 5.0, mid + 20.0):
        session.translate_plane(0, position)
        assert session.plane_arc_position(0) == pytest.approx(position, abs=0.3)
        # The offsets survive the move...
        placement = session.placement(0)
        assert placement.yaw_deg == pytest.approx(30.0)
        assert placement.tilt_deg == pytest.approx(10.0)
        # ...and so does the angle to the jaw at the new position.
        assert _obliquity_deg(arch_frames, session, 0) == pytest.approx(
            reference, abs=0.5
        ), position

    # The world normal genuinely followed the curve; that is the whole point.
    session.translate_plane(0, mid + 20.0)
    assert not np.allclose(session.planes[0].normal, world_normal)


def test_translation_leaves_the_angular_offsets_untouched(arch_frames):
    mid = arch_frames.length_mm / 2.0
    session = _seeded_session(arch_frames, mid)
    session.rotate_plane(0, yaw_deg=18.0, tilt_deg=-7.0, roll_deg=5.0)
    before = session.placement(0)

    session.translate_plane(0, mid + 12.0, offset_mm=(1.0, 0.0, -2.0))
    after = session.placement(0)
    assert (after.yaw_deg, after.tilt_deg, after.roll_deg) == (
        before.yaw_deg,
        before.tilt_deg,
        before.roll_deg,
    )
    assert after.s_mm != before.s_mm


def test_rotating_a_cut_never_moves_it(arch_frames):
    mid = arch_frames.length_mm / 2.0
    session = _seeded_session(arch_frames, mid)
    session.translate_plane(0, mid, offset_mm=(2.0, 1.0, 0.0))
    origin = session.planes[0].origin.copy()
    s_before = session.plane_arc_position(0)

    for yaw in (10.0, -25.0, 40.0):
        session.rotate_plane(0, yaw_deg=yaw, tilt_deg=0.0)
        assert np.allclose(session.planes[0].origin, origin), yaw
        assert session.plane_arc_position(0) == pytest.approx(s_before), yaw


def test_dragging_the_plane_in_3d_is_a_translation(arch_frames):
    """A widget drag moves the cut and re-derives its angulation there.

    The drag delivers a world origin. It sets the arc position and the
    off-curve offset; it never sets the normal, and it never clears the
    operator's yaw and tilt.
    """
    mid = arch_frames.length_mm / 2.0
    session = _seeded_session(arch_frames, mid)
    session.rotate_plane(0, yaw_deg=35.0, tilt_deg=12.0)
    reference = _obliquity_deg(arch_frames, session, 0)

    index = arch_frames.index_of(mid)
    target = arch_frames.points[index] + np.array([4.0, -2.0, 1.0])
    session.set_plane_origin(0, target)

    assert np.allclose(session.planes[0].origin, target)
    placement = session.placement(0)
    assert placement.yaw_deg == pytest.approx(35.0)
    assert placement.tilt_deg == pytest.approx(12.0)
    assert _obliquity_deg(arch_frames, session, 0) == pytest.approx(reference, abs=0.5)


def test_the_default_cut_is_perpendicular_to_the_local_arch(arch_frames):
    """The documented convention: with no offsets, the normal is the tangent."""
    for s_mm in np.linspace(2.0, arch_frames.length_mm - 2.0, 12):
        origin, normal = plane_from_placement(
            arch_frames, PlanePlacement(s_mm=float(s_mm))
        )
        tangent, _, _ = arch_frames.frame_at(float(s_mm))
        assert np.allclose(normal, tangent, atol=1e-9), s_mm
        assert np.allclose(origin, arch_frames.point_at(float(s_mm)))


def test_the_normal_turns_smoothly_along_the_arch(arch_frames):
    """No jump, flip or sudden 180 degree reversal between adjacent positions."""
    positions = np.arange(1.0, arch_frames.length_mm - 1.0, 0.5)
    normals = np.array(
        [
            plane_from_placement(
                arch_frames, PlanePlacement(s_mm=float(s), yaw_deg=15.0, tilt_deg=8.0)
            )[1]
            for s in positions
        ]
    )
    dots = np.einsum("ij,ij->i", normals[:-1], normals[1:])
    # Never inverted: adjacent normals always face the same way.
    assert dots.min() > 0.99, dots.min()
    steps = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    assert steps.max() < 5.0, steps.max()


def test_the_cut_angulates_through_body_angle_and_ramus(wide_arch_frames):
    """A cut at one end of a curved mandible is not parallel to one at the other.

    If it were, the plane would be sliding in global coordinates rather than
    following the anatomy.
    """
    frames = wide_arch_frames
    near, far = 5.0, frames.length_mm - 5.0
    _, first = plane_from_placement(frames, PlanePlacement(s_mm=near))
    _, last = plane_from_placement(frames, PlanePlacement(s_mm=far))
    between = np.degrees(np.arccos(abs(np.clip(np.dot(first, last), -1.0, 1.0))))

    # The two cuts differ by exactly the turn the arch itself makes between
    # them. Asserting against the curve rather than a fixed number keeps this
    # honest for any phantom: it is the definition of following the anatomy.
    near_tangent, _, _ = frames.frame_at(near)
    far_tangent, _, _ = frames.frame_at(far)
    arch_turn = np.degrees(
        np.arccos(abs(np.clip(np.dot(near_tangent, far_tangent), -1.0, 1.0)))
    )
    assert between == pytest.approx(arch_turn, abs=1e-6)
    assert between > 20.0, f"phantom arch only turns {between:.1f} degrees"


def test_a_curve_that_climbs_the_ramus_keeps_a_stable_frame(spec):
    """Parallel transport, not a global up-axis, is what survives the ramus.

    The old frame construction refused any curve leaving the axial plane.
    """
    from mandiplan.geometry import cpr
    from mandiplan.geometry.spline import ArchCurve

    points = np.asarray(spec.centre_line(9, extend_deg=15.0), dtype=float).copy()
    points[0, 2] += 25.0
    points[1, 2] += 12.0
    points[-2, 2] += 12.0
    points[-1, 2] += 25.0
    frames = cpr.build_frames(ArchCurve(points), step_mm=0.2)

    # Frame stays orthonormal and never spins about the tangent.
    assert np.abs(np.einsum("ij,ij->i", frames.tangents, frames.ups)).max() < 1e-9
    carried = np.einsum("ij,ij->i", frames.ups[:-1], frames.ups[1:])
    assert carried.min() > 0.999, carried.min()

    normals = np.array(
        [
            plane_from_placement(frames, PlanePlacement(s_mm=float(s)))[1]
            for s in np.arange(1.0, frames.length_mm - 1.0, 0.5)
        ]
    )
    dots = np.einsum("ij,ij->i", normals[:-1], normals[1:])
    assert dots.min() > 0.99, dots.min()


def test_roll_selects_the_axes_yaw_and_tilt_act_about(arch_frames):
    """Roll alone leaves the cut alone; a plane is invariant about its normal."""
    mid = arch_frames.length_mm / 2.0
    _, plain = plane_from_placement(arch_frames, PlanePlacement(s_mm=mid))
    _, rolled = plane_from_placement(
        arch_frames, PlanePlacement(s_mm=mid, roll_deg=40.0)
    )
    assert np.allclose(plain, rolled, atol=1e-9)

    # With a yaw applied, roll changes which plane the yaw swings in.
    _, yawed = plane_from_placement(
        arch_frames, PlanePlacement(s_mm=mid, yaw_deg=20.0)
    )
    _, rolled_yaw = plane_from_placement(
        arch_frames, PlanePlacement(s_mm=mid, roll_deg=40.0, yaw_deg=20.0)
    )
    assert not np.allclose(yawed, rolled_yaw, atol=1e-6)


def test_a_placement_is_clamped_to_the_curve(arch_frames):
    session = _seeded_session(arch_frames, arch_frames.length_mm / 2.0)
    session.translate_plane(0, arch_frames.length_mm + 500.0)
    assert session.plane_arc_position(0) == pytest.approx(arch_frames.length_mm)
    session.translate_plane(0, -50.0)
    assert session.plane_arc_position(0) == pytest.approx(0.0)


def test_placing_a_cut_by_numbers_keeps_the_side_it_removes(arch_frames, spec):
    from mandiplan.ui.session import Session

    session = Session()
    session.frames = arch_frames
    mid = arch_frames.length_mm / 2.0
    for s, sign in ((mid - 10.0, +1.0), (mid + 10.0, -1.0)):
        index = arch_frames.index_of(s)
        session.add_plane(
            arch_frames.points[index], sign * arch_frames.tangents[index]
        )
    before = [plane.normal.copy() for plane in session.planes]

    session.translate_plane(1, mid + 14.0)
    # The side removed is held on the placement, so it survives the move even
    # though the world normal has followed the curve.
    assert session.placement(1).flipped
    tangent, _, _ = arch_frames.frame_at(session.plane_arc_position(1))
    assert np.dot(session.planes[1].normal, tangent) < 0
    assert np.dot(session.planes[1].normal, before[1]) > 0
    assert session.plane_arc_position(1) == pytest.approx(mid + 14.0, abs=0.3)
    assert np.isfinite(session.report.arc_length_mm)
