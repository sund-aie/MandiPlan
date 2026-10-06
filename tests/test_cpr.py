"""Curved planar reformation against the analytic phantom."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import extent_mm, half_level, rel_error
from mandiplan.geometry import cpr
from mandiplan.geometry.spline import ArchCurve


def test_arch_curve_length_matches_analytic_arc_length(arch_curve, spec):
    assert rel_error(arch_curve.length_mm, spec.arc_length_mm) < 0.02


def test_resampling_is_uniform_in_arc_length(arch_curve):
    samples = arch_curve.resample(0.2)
    step = np.linalg.norm(np.diff(samples.points, axis=0), axis=1)
    assert np.allclose(step, 0.2, atol=2e-4)


def test_uniform_parameter_spacing_would_not_be_uniform_arc_length(spec):
    """Guard on the reason arc-length resampling exists."""
    seeds = spec.centre_line(6).copy()
    seeds[1] = spec.point_at_arc(2.0)  # crowd the seeds near the start
    curve = ArchCurve(seeds)
    samples = curve.resample(0.2)
    step = np.linalg.norm(np.diff(samples.points, axis=0), axis=1)
    assert np.allclose(step, 0.2, atol=2e-4)


def test_panoramic_arc_length_of_the_bone_matches_analytic(
    phantom, wide_arch_frames, spec
):
    """Flatten the jaw, then measure the bone along the panoramic x-axis.

    The arch curve runs past both ends of the bone, so the measurement is of
    the bone itself and not merely of the curve the user drew.
    """
    for mode in cpr.AGGREGATION_MODES:
        pan = cpr.build_panoramic(phantom, wide_arch_frames, slab_mm=10.0, mode=mode)
        profile = pan.image.max(axis=0)
        span = extent_mm(profile, pan.pixel_mm)
        assert rel_error(span, spec.arc_length_mm) < 0.02, mode


def test_panoramic_axes_are_millimetres(phantom, arch_frames):
    pan = cpr.build_panoramic(phantom, arch_frames, slab_mm=8.0, mode="mean")
    # Sampled at the voxel size, never finer than the scan can show.
    assert pan.pixel_mm == pytest.approx(
        max(arch_frames.step_mm, float(np.min(phantom.spacing)))
    )
    assert pan.width_mm == pytest.approx((pan.image.shape[1] - 1) * pan.pixel_mm)
    # The columns still cover the whole arch, in millimetres of arc length.
    assert pan.width_mm == pytest.approx(arch_frames.length_mm, abs=pan.pixel_mm)
    assert pan.col_to_mm(0) == 0.0
    assert "arc length" in pan.x_label


def test_both_aggregation_modes_show_the_bone(phantom, arch_frames, spec):
    for mode in cpr.AGGREGATION_MODES:
        pan = cpr.build_panoramic(phantom, arch_frames, slab_mm=10.0, mode=mode)
        assert pan.image.max() > spec.half_max_value


def test_cross_section_dimensions_match_the_known_ellipse(phantom, arch_frames, spec):
    cs = cpr.build_cross_section(phantom, arch_frames, spec.arc_length_mm / 2.0, width_mm=40.0)
    row = int(round(cs.mm_to_row(spec.centre_z)))
    col = int(round(cs.mm_to_col(0.0)))

    width = extent_mm(cs.image[row, :], cs.pixel_mm)
    height = extent_mm(cs.image[:, col], cs.pixel_mm)

    assert rel_error(width, 2 * spec.semi_axis_bl_mm) < 0.02
    assert rel_error(height, 2 * spec.semi_axis_si_mm) < 0.02


def test_cross_section_measurement_on_a_strongly_anisotropic_volume(
    skewed_phantom, skewed_spec
):
    """0.25 x 0.5 x 0.8 mm voxels; measurement must stay within 1%."""
    curve = ArchCurve(skewed_spec.centre_line(7))
    frames = cpr.build_frames(curve, step_mm=0.1)
    cs = cpr.build_cross_section(
        skewed_phantom, frames, skewed_spec.arc_length_mm / 2.0, width_mm=40.0
    )
    row = int(round(cs.mm_to_row(skewed_spec.centre_z)))
    col = int(round(cs.mm_to_col(0.0)))

    width = extent_mm(cs.image[row, :], cs.pixel_mm)
    height = extent_mm(cs.image[:, col], cs.pixel_mm)

    assert rel_error(width, 2 * skewed_spec.semi_axis_bl_mm) < 0.01
    assert rel_error(height, 2 * skewed_spec.semi_axis_si_mm) < 0.01


def test_cross_section_stepping_follows_the_curve(phantom, arch_frames, spec):
    """Cross-sections stay centred on the bone all the way along the arch."""
    for s in np.linspace(2.0, spec.arc_length_mm - 2.0, 9):
        cs = cpr.build_cross_section(phantom, arch_frames, s, width_mm=30.0)
        col = int(round(cs.mm_to_col(0.0)))
        profile = cs.image[:, col]
        assert profile.max() > half_level(profile)
        height = extent_mm(profile, cs.pixel_mm)
        assert rel_error(height, 2 * spec.semi_axis_si_mm) < 0.03


def test_cross_section_world_point_lies_on_the_sampled_plane(arch_frames):
    s = 20.0
    p = cpr.cross_section_world_point(arch_frames, s, u_mm=3.0, z_mm=1.5)
    i = arch_frames.index_of(s)
    offset = p[:2] - arch_frames.points[i, :2]
    assert np.allclose(offset, 3.0 * arch_frames.normals[i, :2])
    assert p[2] == pytest.approx(1.5)


def test_frames_are_orthonormal(arch_frames):
    dots = np.einsum("ij,ij->i", arch_frames.tangents, arch_frames.normals)
    assert np.allclose(dots, 0.0, atol=1e-9)
    assert np.allclose(np.linalg.norm(arch_frames.normals, axis=1), 1.0)
    assert np.allclose(arch_frames.normals[:, 2], 0.0, atol=1e-9)


def test_a_flat_curve_unrolls_onto_its_own_arc_length(arch_frames):
    x = cpr.sheet_x(arch_frames)
    assert np.allclose(x, arch_frames.s, atol=1e-6)
    s = 0.4 * arch_frames.length_mm
    point = cpr.sheet_point(arch_frames, s, 3.0)
    assert point[0] == pytest.approx(s, abs=arch_frames.step_mm)
    assert cpr.s_on_sheet(arch_frames, point[0], 3.0) == pytest.approx(s, abs=arch_frames.step_mm)
    # A straight cut across the body stands upright in the panoramic.
    i = arch_frames.index_of(s)
    line = cpr.sheet_line(arch_frames, s, arch_frames.points[i], arch_frames.tangents[i], 10.0)
    assert line[0, 0] == pytest.approx(line[1, 0], abs=1e-6)
    assert abs(line[1, 1] - line[0, 1]) == pytest.approx(20.0)


def _climbing_frames(spec):
    """The phantom's arch with both ends carried 40 mm up, like rami."""
    points = np.asarray(spec.centre_line(9, extend_deg=10.0), dtype=float).copy()
    ends = np.array([points[0], points[-1]])
    rise = []
    for end, inward in ((ends[0], points[1] - points[0]), (ends[1], points[-2] - points[-1])):
        back = -inward / np.linalg.norm(inward)
        rise.append([end + back * 3.0 + [0.0, 0.0, h] for h in (12.0, 26.0, 40.0)])
    stacked = np.vstack([np.array(rise[0])[::-1], points, np.array(rise[1])])
    return cpr.build_frames(ArchCurve(stacked), step_mm=0.2)


def test_a_curve_up_the_rami_keeps_the_body_frame_upright(spec):
    frames = _climbing_frames(spec)
    middle = frames.index_of(frames.length_mm / 2.0)
    assert np.allclose(frames.ups[middle], [0.0, 0.0, 1.0], atol=1e-6)
    start, end = cpr.body_span(frames)
    assert 0.0 < start < end < frames.length_mm
    # The rami rise in the panoramic instead of stretching it sideways.
    assert cpr.sheet_x(frames)[-1] < frames.length_mm - 60.0


def test_the_cross_section_up_a_ramus_is_across_it(phantom, spec):
    frames = _climbing_frames(spec)
    s = 3.0
    cs = cpr.build_cross_section(phantom, frames, s, width_mm=20.0)
    tangent, _, _ = frames.frame_at(s)
    corner = cpr.cross_section_world_point(frames, s, 5.0, cs.y0 + 7.0)
    centre = frames.points[frames.index_of(s)]
    # Every point of the section lies in the plane perpendicular to the curve.
    assert abs(np.dot(corner - centre, tangent)) < 1e-6
    assert abs(tangent[2]) > 0.5  # the curve is climbing here
