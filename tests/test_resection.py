"""Resection: which fragment goes, how big it is, how long the gap is."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import rel_error
from mandiplan.geometry.resection import CutPlane, build_report, resected_mask
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
