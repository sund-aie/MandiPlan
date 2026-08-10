"""Plate bend decomposition and bending-template geometry."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import circular_arc_path, rel_error
from mandiplan.geometry.plate import compute_plate_plan, ribbon_mesh

PITCH = 9.0
RADIUS = 30.0


@pytest.fixture(scope="module")
def flat_arc():
    """Circular arc in the axial plane with the surface normal along +z.

    The plate lies flat against an inferior border, so the whole turn is a
    contour bend in the plate's own plane.
    """
    path = circular_arc_path(RADIUS, 2.0)
    normals = np.tile([0.0, 0.0, 1.0], (len(path), 1))
    return compute_plate_plan(path, normals, PITCH)


@pytest.fixture(scope="module")
def wrapped_arc():
    """Same arc, but the plate wraps a buccal surface: normals point radially."""
    path = circular_arc_path(RADIUS, 2.0)
    normals = path / np.linalg.norm(path[:, :2], axis=1)[:, None]
    normals[:, 2] = 0.0
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return compute_plate_plan(path, normals, PITCH)


def _interior(values):
    return np.array([v for v in values if np.isfinite(v)])


def test_in_plane_bend_matches_the_analytic_turn_angle(flat_arc):
    expected = np.degrees(PITCH / RADIUS)
    angles = _interior([b.in_plane_deg for b in flat_arc.bends])
    assert len(angles) >= 4
    assert np.max(np.abs(angles - expected)) < 1.0


def test_out_of_plane_and_twist_vanish_for_the_flat_arc(flat_arc):
    out = _interior([b.out_of_plane_deg for b in flat_arc.bends])
    twist = _interior([b.twist_deg for b in flat_arc.bends])
    assert np.max(np.abs(out)) < 1e-6
    assert np.max(np.abs(twist)) < 1e-6


def test_a_wrapped_plate_turns_out_of_plane_instead(wrapped_arc):
    expected = np.degrees(PITCH / RADIUS)
    out = _interior([b.out_of_plane_deg for b in wrapped_arc.bends])
    in_plane = _interior([b.in_plane_deg for b in wrapped_arc.bends])
    # The path curves away from the outward normal, so the sign is negative.
    assert np.max(np.abs(np.abs(out) - expected)) < 1.0
    assert np.all(out < 0)
    assert np.max(np.abs(in_plane)) < 1e-6


def test_twist_is_zero_on_a_planar_path(flat_arc, wrapped_arc):
    for plan in (flat_arc, wrapped_arc):
        twist = _interior([b.twist_deg for b in plan.bends])
        assert np.max(np.abs(twist)) < 1e-6


def test_twist_tracks_a_rotating_surface_normal():
    """Straight path whose surface normal spirals about the direction of travel."""
    k = np.radians(2.0)  # degrees of roll per millimetre
    s = np.arange(0.0, 60.0, 0.1)
    path = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
    normals = np.column_stack([np.zeros_like(s), np.sin(k * s), np.cos(k * s)])
    plan = compute_plate_plan(path, normals, PITCH)
    twist = _interior([b.twist_deg for b in plan.bends])
    expected = -np.degrees(k * PITCH)  # rotating +z toward +y is left-handed about +x
    assert np.max(np.abs(twist - expected)) < 0.1


def test_total_plate_length_matches_the_analytic_arc_length(flat_arc):
    spanned = flat_arc.bends[-1].cumulative_mm
    analytic = PITCH * (len(flat_arc.nodes) - 1)
    assert rel_error(flat_arc.total_length_mm, analytic) < 0.02
    assert spanned == pytest.approx(flat_arc.total_length_mm)


def test_segments_are_one_screw_hole_pitch_apart(flat_arc):
    lengths = flat_arc.segment_lengths_mm
    # Chords of a circle are slightly shorter than the arc they subtend.
    assert np.all(lengths <= PITCH + 1e-9)
    assert rel_error(float(lengths.mean()), PITCH) < 0.01


def test_ribbon_template_is_closed_and_the_right_size(flat_arc):
    points, triangles = ribbon_mesh(flat_arc, width_mm=12.0, thickness_mm=2.0)
    assert len(points) == 4 * len(flat_arc.nodes)

    edges = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(a, b), max(a, b))
            edges[key] = edges.get(key, 0) + 1
    assert all(count == 2 for count in edges.values()), "template mesh is not closed"

    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    assert volume > 0, "template triangles are wound inwards"
    assert rel_error(volume, 12.0 * 2.0 * flat_arc.total_length_mm) < 0.03


def test_pitch_is_not_hardcoded():
    path = circular_arc_path(RADIUS, 2.0)
    normals = np.tile([0.0, 0.0, 1.0], (len(path), 1))
    for pitch in (5.0, 7.5, 12.0):
        plan = compute_plate_plan(path, normals, pitch)
        expected = np.degrees(pitch / RADIUS)
        angles = _interior([b.in_plane_deg for b in plan.bends])
        assert np.max(np.abs(angles - expected)) < 1.0
        assert rel_error(float(plan.segment_lengths_mm.mean()), pitch) < 0.01
