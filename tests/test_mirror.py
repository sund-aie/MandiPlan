"""Mirror reconstruction: the plane, what it covers, and what it cannot."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import rel_error
from make_phantom import PhantomSpec, make_phantom
from mandiplan.geometry import cpr
from mandiplan.geometry.mirror import (
    bridge_mesh,
    estimate_midsagittal_plane,
    estimate_profile,
    mirror_coverage,
)
from mandiplan.geometry.resection import CutPlane
from mandiplan.geometry.spline import ArchCurve
from mandiplan.render.surface import (
    clip_closed,
    extract_isosurface,
    mesh_volume_mm3,
    reflect_polydata,
)


@pytest.fixture(scope="module")
def plane(phantom, spec):
    return estimate_midsagittal_plane(phantom, spec.half_max_value)


def test_the_midsagittal_plane_is_found_on_a_symmetric_mandible(plane, spec):
    # The phantom is swept symmetrically about x = 0 with the chin anterior.
    assert abs(plane.point[0] - spec.centre_xy[0]) < 0.6
    assert plane.tilt_deg < 1.5
    assert plane.symmetry > 0.9


def test_the_plane_follows_a_laterally_shifted_mandible():
    spec = PhantomSpec(centre_xy=(7.5, 0.0), spacing=(0.4, 0.4, 0.8), seed=4)
    volume = make_phantom(spec)
    plane = estimate_midsagittal_plane(volume, spec.half_max_value)
    assert abs(plane.point[0] - 7.5) < 0.8
    assert plane.symmetry > 0.9


def test_a_lateral_defect_mirrors_completely(arch_frames, plane, spec):
    """A defect on one side has a healthy counterpart for all of it."""
    mid = spec.arc_length_mm / 2.0
    coverage = mirror_coverage(arch_frames, plane, mid + 12.0, mid + 32.0)
    assert not coverage.crosses_midline
    assert coverage.uncovered_mm == pytest.approx(0.0, abs=1.0)
    assert coverage.covered_fraction > 0.95
    assert "one side of the midline" in coverage.summary()


def test_a_defect_across_the_midline_cannot_be_fully_mirrored(arch_frames, plane, spec):
    """The bone that would be mirrored into the crossing part is resected too.

    For a symmetric arch the uncovered span is exactly twice the distance from
    the midline to the nearer cut.
    """
    mid = spec.arc_length_mm / 2.0
    near, far = 10.0, 25.0
    coverage = mirror_coverage(arch_frames, plane, mid - near, mid + far)

    assert coverage.crosses_midline
    assert coverage.midline_s_mm == pytest.approx(mid, abs=1.0)
    assert coverage.uncovered_mm == pytest.approx(2 * near, abs=1.5)
    assert coverage.covered_mm == pytest.approx((near + far) - 2 * near, abs=1.5)
    assert "no healthy counterpart" in coverage.summary()
    assert len(coverage.uncovered_spans) == 1


def test_mirroring_the_retained_bone_reproduces_the_removed_fragment(
    phantom, spec, plane, arch_frames
):
    """Cut a lateral segment, mirror the healthy side, compare with the truth."""
    surface = extract_isosurface(phantom, spec.half_max_value)
    cuts = _radial_cuts(spec, offset_a_deg=20.0, offset_b_deg=45.0)

    fragment = clip_closed(surface, cuts, keep_resected=True)
    retained = clip_closed(surface, cuts, keep_resected=False)
    graft = clip_closed(reflect_polydata(retained, plane), cuts, keep_resected=True)

    truth = mesh_volume_mm3(fragment)
    assert rel_error(mesh_volume_mm3(graft), truth) < 0.05


def _radial_cuts(spec, offset_a_deg: float, offset_b_deg: float) -> list[CutPlane]:
    origin = np.array([spec.centre_xy[0], spec.centre_xy[1], spec.centre_z])
    cuts = []
    for offset, into_positive in ((offset_a_deg, True), (offset_b_deg, False)):
        t = np.radians(spec.arch_centre_deg + offset)
        tangential = np.array([-np.sin(t), np.cos(t), 0.0])
        cuts.append(
            CutPlane(
                origin=origin,
                normal=tangential if into_positive else -tangential,
                label=f"cut{offset:+.0f}",
            )
        )
    return cuts


def test_profile_estimation_recovers_the_known_ellipse(phantom, arch_frames, spec):
    cross = cpr.build_cross_section(
        phantom, arch_frames, spec.arc_length_mm / 2.0, width_mm=40.0
    )
    profile = estimate_profile(cross, spec.half_max_value, spec.arc_length_mm / 2.0)
    assert profile is not None
    assert rel_error(profile.semi_bl_mm, spec.semi_axis_bl_mm) < 0.03
    assert rel_error(profile.semi_si_mm, spec.semi_axis_si_mm) < 0.03
    assert abs(profile.centre_u_mm) < 0.5


def test_the_bridge_reproduces_the_missing_volume(phantom, arch_frames, spec):
    """Estimate the missing bone across a gap and compare with the truth."""
    s_a, s_b = 25.0, 50.0
    profiles = []
    for s in (s_a, s_b):
        cross = cpr.build_cross_section(phantom, arch_frames, s, width_mm=40.0)
        profiles.append(estimate_profile(cross, spec.half_max_value, s))

    points, triangles = bridge_mesh(arch_frames, profiles[0], profiles[1], station_step_mm=0.5)
    volume = _mesh_volume(points, triangles)
    analytic = np.pi * spec.semi_axis_bl_mm * spec.semi_axis_si_mm * (s_b - s_a)
    assert rel_error(volume, analytic) < 0.03


def test_the_bridge_mesh_is_closed(arch_frames, phantom, spec):
    profiles = [
        estimate_profile(
            cpr.build_cross_section(phantom, arch_frames, s, width_mm=40.0),
            spec.half_max_value,
            s,
        )
        for s in (20.0, 40.0)
    ]
    _, triangles = bridge_mesh(arch_frames, profiles[0], profiles[1])
    edges: dict[tuple[int, int], int] = {}
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(a, b), max(a, b))
            edges[key] = edges.get(key, 0) + 1
    assert all(count == 2 for count in edges.values())


def test_reflection_is_its_own_inverse(plane):
    rng = np.random.default_rng(2)
    points = rng.normal(0.0, 30.0, size=(64, 3))
    assert np.allclose(plane.reflect(plane.reflect(points)), points, atol=1e-9)
    on_plane = plane.reflect(points) * 0.5 + points * 0.5
    assert np.allclose(plane.signed_distance(on_plane), 0.0, atol=1e-9)


def _mesh_volume(points: np.ndarray, triangles: np.ndarray) -> float:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))
