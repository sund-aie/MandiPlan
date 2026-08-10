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
    assert pan.pixel_mm == pytest.approx(arch_frames.step_mm)
    assert pan.width_mm == pytest.approx(
        (pan.image.shape[1] - 1) * arch_frames.step_mm
    )
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
