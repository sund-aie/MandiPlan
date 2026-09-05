"""Controlled bending, and the checks that say whether the result is usable.

The claim under test is narrow and specific: bending changes the shape of the
bridges between screw holes and changes nothing about the holes themselves. A
hole bent out of round is a hole no screw passes through, so "hole diameter is
preserved" is asserted numerically at several bend radii rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mandiplan.geometry.mesh_io import Mesh
from mandiplan.geometry.plate_bend import (
    bend_to_path,
    clearance_report,
    transported_frames,
)
from mandiplan.geometry.plate_fit import extend_targets
from mandiplan.plate_assets import asset_by_id, load_asset_mesh

ASSET_ID = "generic-recon-2.4-12h"


@pytest.fixture(scope="module")
def asset():
    return asset_by_id(ASSET_ID)


@pytest.fixture(scope="module")
def mesh(asset):
    return load_asset_mesh(asset)


def arc_path(asset, radius_mm: float, holes: int | None = None):
    """A circular bone path whose stations are one plate pitch apart."""
    n = holes or asset.hole_count
    t = (np.arange(n) - (n - 1) / 2) * (asset.hole_pitch_mm / radius_mm)
    points = np.column_stack(
        [radius_mm * np.sin(t), radius_mm * np.cos(t) - radius_mm, np.zeros(n)]
    )
    normals = np.column_stack([np.sin(t), np.cos(t), np.zeros(n)])
    return points, normals


def bend(asset, mesh, radius_mm, **kwargs):
    points, normals = arc_path(asset, radius_mm)
    return bend_to_path(
        mesh.points,
        mesh.triangles,
        asset.hole_centres_mm,
        asset.hole_axes,
        points,
        normals,
        protected_radius_mm=asset.deformation.protected_radius_mm,
        **kwargs,
    )


def hole_wall_radii(asset, mesh, bent, hole: int):
    """Radii of the vertices on one hole's wall, before and after bending."""
    centre, axis = asset.hole_centres_mm[hole], asset.hole_axes[hole]
    nominal = asset.hole_diameter_mm / 2.0
    offset = mesh.points - centre
    radial = np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1)
    wall = np.abs(radial - nominal) < 0.03
    if wall.sum() < 4:
        return None, None
    moved = bent.points[wall] - bent.hole_centres[hole]
    axis_after = bent.hole_axes[hole]
    after = np.linalg.norm(
        moved - np.outer(moved @ axis_after, axis_after), axis=1
    )
    return radial[wall], after


# -- what bending must not change ----------------------------------------


@pytest.mark.parametrize("radius_mm", [400.0, 200.0, 90.0, 60.0])
def test_the_geometric_bend_alone_does_not_deform_a_screw_hole(asset, mesh, radius_mm):
    """The idealised bend holds the holes rigid; only the bridges are swept.

    This is the geometry stage, not the whole story. Real contouring does take
    holes out of round, and that is applied on top of this by
    ``hole_distortion`` according to the alloy and the bending kit — see
    tests/test_hole_distortion.py. Separating them is deliberate: the shape
    the plate is being bent *to* is one question, and what the instrument does
    to it on the way is another.
    """
    bent = bend(asset, mesh, radius_mm)
    nominal = asset.hole_diameter_mm / 2.0
    for hole in range(asset.hole_count):
        before, after = hole_wall_radii(asset, mesh, bent, hole)
        if after is None:
            continue
        # Measured against the same vertices before bending, so the asset's
        # own float32 STL storage error cancels and what is left is purely
        # what bending did to the hole.
        assert np.abs(after - before).max() < 1e-9, (
            f"hole {hole} at bend radius {radius_mm} mm moved by "
            f"{np.abs(after - before).max() * 1e6:.3f} nm"
        )
        # And the hole was round to begin with, to the precision a binary STL
        # can store.
        assert np.abs(before - nominal).max() < 1e-5


@pytest.mark.parametrize("radius_mm", [200.0, 60.0])
def test_bending_preserves_thickness_at_the_holes(asset, mesh, radius_mm):
    bent = bend(asset, mesh, radius_mm)
    for hole in range(asset.hole_count):
        centre, axis = asset.hole_centres_mm[hole], asset.hole_axes[hole]
        offset = mesh.points - centre
        radial = np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1)
        wall = np.abs(radial - asset.hole_diameter_mm / 2.0) < 0.03
        if wall.sum() < 4:
            continue
        moved = bent.points[wall] - bent.hole_centres[hole]
        along = moved @ bent.hole_axes[hole]
        assert np.ptp(along) == pytest.approx(asset.thickness_mm, abs=1e-5)


@pytest.mark.parametrize("radius_mm", [200.0, 90.0, 60.0])
def test_a_bent_plate_is_still_a_watertight_plate(asset, mesh, radius_mm):
    bent = bend(asset, mesh, radius_mm)
    result = Mesh(bent.points, bent.triangles)
    assert result.is_watertight()
    assert result.genus() == asset.hole_count


def test_bending_barely_changes_the_volume(asset, mesh):
    """Bending redistributes material; it does not create or destroy it."""
    bent = bend(asset, mesh, 60.0)
    result = Mesh(bent.points, bent.triangles)
    assert result.volume_mm3() == pytest.approx(mesh.volume_mm3(), rel=0.02)


def test_the_holes_land_on_the_planned_path(asset, mesh):
    points, _ = arc_path(asset, 90.0)
    bent = bend(asset, mesh, 90.0)
    assert np.abs(bent.hole_centres - points).max() < 0.25


def test_nothing_is_scaled_along_the_plate(asset, mesh):
    """Hole-to-hole spacing follows the path's chords, not a stretch factor."""
    points, _ = arc_path(asset, 90.0)
    bent = bend(asset, mesh, 90.0)
    expected = np.linalg.norm(np.diff(points, axis=0), axis=1)
    actual = np.linalg.norm(np.diff(bent.hole_centres, axis=0), axis=1)
    assert np.allclose(actual, expected, atol=0.25)


# -- the limits ----------------------------------------------------------


def test_a_bend_past_the_working_limit_is_a_problem(asset, mesh):
    bent = bend(asset, mesh, 30.0, max_bend_deg_per_node=5.0)
    assert not bent.is_usable
    assert any("working limit" in p for p in bent.problems)


def test_too_tight_a_radius_is_a_problem(asset, mesh):
    bent = bend(asset, mesh, 30.0, min_bend_radius_mm=60.0)
    assert not bent.is_usable
    assert any("radius" in p for p in bent.problems)


def test_a_gentle_path_raises_nothing(asset, mesh):
    bent = bend(asset, mesh, 200.0)
    assert bent.is_usable, bent.problems


def test_a_path_that_folds_back_is_refused(asset, mesh):
    """A hairpin puts non-neighbouring holes on top of each other."""
    n = asset.hole_count
    half = n // 2
    forward = np.column_stack(
        [np.arange(half) * asset.hole_pitch_mm, np.zeros(half), np.zeros(half)]
    )
    back = forward[::-1].copy()
    back[:, 1] = 1.0
    points = np.vstack([forward, back])[:n]
    normals = np.tile([0.0, 0.0, 1.0], (n, 1))
    bent = bend_to_path(
        mesh.points,
        mesh.triangles,
        asset.hole_centres_mm,
        asset.hole_axes,
        points,
        normals,
        protected_radius_mm=asset.deformation.protected_radius_mm,
        max_bend_deg_per_node=180.0,
        min_bend_radius_mm=0.0,
    )
    assert not bent.is_usable
    assert any("folds onto itself" in p for p in bent.problems)


def test_mismatched_pairing_is_refused(asset, mesh):
    points, normals = arc_path(asset, 90.0, holes=asset.hole_count - 3)
    with pytest.raises(ValueError, match="path points"):
        bend_to_path(
            mesh.points,
            mesh.triangles,
            asset.hole_centres_mm,
            asset.hole_axes,
            points,
            normals,
        )


# -- frames --------------------------------------------------------------


def test_transported_frames_stay_orthonormal_and_do_not_spin():
    t = np.linspace(0, 2.2, 90)
    points = np.column_stack([40 * np.sin(t), 40 * np.cos(t), 6 * t])
    s, tangents, ups, binormals = transported_frames(points, np.array([0.0, 0.0, 1.0]))

    assert np.all(np.diff(s) > 0)
    assert np.abs(np.einsum("ij,ij->i", tangents, ups)).max() < 1e-9
    assert np.abs(np.einsum("ij,ij->i", ups, binormals)).max() < 1e-9
    assert np.allclose(np.linalg.norm(ups, axis=1), 1.0)
    carried = np.einsum("ij,ij->i", ups[:-1], ups[1:])
    assert carried.min() > 0.99


# -- pairing -------------------------------------------------------------


def test_extend_targets_trims_when_the_path_is_longer():
    targets = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
    normals = np.tile([0.0, 0.0, 1.0], (10, 1))
    binormals = np.tile([0.0, 1.0, 0.0], (10, 1))
    points, n, b = extend_targets(6, targets, normals, binormals)
    assert len(points) == len(n) == len(b) == 6
    assert points[0][0] == pytest.approx(2.0)


def test_extend_targets_continues_straight_when_the_plate_is_longer():
    targets = np.column_stack([np.arange(4.0), np.zeros(4), np.zeros(4)])
    normals = np.tile([0.0, 0.0, 1.0], (4, 1))
    binormals = np.tile([0.0, 1.0, 0.0], (4, 1))
    points, n, b = extend_targets(8, targets, normals, binormals)
    assert len(points) == len(n) == len(b) == 8
    steps = np.diff(points[:, 0])
    assert np.allclose(steps, 1.0), steps


# -- contact -------------------------------------------------------------


def _flat_bone(n=8, spacing=9.0):
    bone = np.column_stack([np.arange(n) * spacing, np.zeros(n), np.zeros(n)])
    normals = np.tile([0.0, 0.0, 1.0], (n, 1))
    return bone, normals


def test_clearance_is_measured_from_the_inner_face():
    bone, normals = _flat_bone()
    thickness = 2.4
    holes = bone + normals * (0.5 + thickness / 2.0)
    report = clearance_report(holes, normals, bone, normals, thickness)
    assert np.allclose(report.clearance_mm, 0.5)
    assert not report.collisions.any()
    assert not report.problems


def test_a_plate_inside_bone_is_a_collision():
    bone, normals = _flat_bone()
    thickness = 2.4
    holes = bone + normals * (thickness / 2.0)
    holes[3] -= normals[3] * 1.5
    report = clearance_report(holes, normals, bone, normals, thickness)
    assert report.collisions[3]
    assert any("cuts into bone" in p for p in report.problems)


def test_an_excessive_gap_is_a_warning_not_a_refusal():
    bone, normals = _flat_bone()
    thickness = 2.4
    holes = bone + normals * (0.5 + thickness / 2.0)
    holes[5] += normals[5] * 4.0
    report = clearance_report(holes, normals, bone, normals, thickness)
    assert report.excessive_gaps[5]
    assert any("stands off" in w for w in report.warnings)
    assert not report.problems


def test_a_screw_pointing_across_the_bone_is_flagged():
    bone, normals = _flat_bone()
    thickness = 2.4
    holes = bone + normals * (0.5 + thickness / 2.0)
    axes = np.tile([0.0, 0.0, 1.0], (len(bone), 1)).astype(float)
    axes[2] = [0.9, 0.0, 0.44]
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    report = clearance_report(holes, axes, bone, normals, thickness)
    assert report.poor_screw_angles[2]
    assert any("skive" in w for w in report.warnings)
