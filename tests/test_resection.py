"""Resection: which fragment goes, how big it is, how long the gap is."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import rel_error
from mandiplan.geometry.resection import (
    CutPlane,
    build_report,
    plane_from_frame,
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


def test_translating_a_cut_never_changes_its_angulation(arch_frames):
    """Sliding a cut along the jaw is not a request to re-angle it."""
    from mandiplan.ui.session import Session

    session = Session()
    session.frames = arch_frames
    mid = arch_frames.length_mm / 2.0
    index = arch_frames.index_of(mid)
    session.add_plane(arch_frames.points[index], arch_frames.tangents[index])

    session.rotate_plane(0, yaw_deg=30.0, tilt_deg=10.0)
    angled = session.planes[0].normal.copy()

    for position in (mid - 15.0, mid + 5.0, mid + 20.0):
        session.translate_plane(0, position)
        assert np.allclose(session.planes[0].normal, angled), position
        assert session.plane_arc_position(0) == pytest.approx(position, abs=0.3)

    session.translate_plane(0, mid, offset_mm=(3.0, 0.0, -2.0))
    assert np.allclose(session.planes[0].normal, angled)


def test_rotating_a_cut_never_moves_it(arch_frames):
    from mandiplan.ui.session import Session

    session = Session()
    session.frames = arch_frames
    mid = arch_frames.length_mm / 2.0
    index = arch_frames.index_of(mid)
    session.add_plane(arch_frames.points[index], arch_frames.tangents[index])
    session.translate_plane(0, mid, offset_mm=(2.0, 1.0, 0.0))
    origin = session.planes[0].origin.copy()

    for yaw in (10.0, -25.0, 40.0):
        session.rotate_plane(0, yaw_deg=yaw, tilt_deg=0.0)
        assert np.allclose(session.planes[0].origin, origin), yaw


def test_dragging_the_plane_in_3d_only_moves_it(arch_frames):
    """The 3-D widget reports an origin; the angulation is not sourced from it."""
    from mandiplan.ui.session import Session

    session = Session()
    session.frames = arch_frames
    index = arch_frames.index_of(arch_frames.length_mm / 2.0)
    session.add_plane(arch_frames.points[index], arch_frames.tangents[index])
    session.rotate_plane(0, yaw_deg=35.0, tilt_deg=12.0)
    angled = session.planes[0].normal.copy()

    # This is what View3D.plane_translated delivers on a widget drag.
    session.set_plane_origin(0, arch_frames.points[index] + np.array([4.0, -2.0, 1.0]))
    assert np.allclose(session.planes[0].normal, angled)
    assert np.allclose(
        session.planes[0].origin, arch_frames.points[index] + np.array([4.0, -2.0, 1.0])
    )


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
    assert np.dot(session.planes[1].normal, before[1]) > 0
    assert session.plane_arc_position(1) == pytest.approx(mid + 14.0, abs=0.3)
    assert np.isfinite(session.report.arc_length_mm)
