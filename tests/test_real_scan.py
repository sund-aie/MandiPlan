"""The planning chain on a real head CBCT, not a phantom.

The bundled scan (``mandiplan/reference_cases.py``) has what a phantom does
not: soft tissue in the histogram, teeth in occlusion joining the mandible
to the maxilla, fillings, condyles in their fossae, and a jaw that is not
perfectly symmetric. Each test here failed on real anatomy before the
matching fix.
"""

from __future__ import annotations

import numpy as np
import pytest
import vtk

from mandiplan.geometry import cpr
from mandiplan.geometry.mandible import isolate_mandible, masked_bone_volume
from mandiplan.geometry.mirror import estimate_midsagittal_plane
from mandiplan.geometry.plate_profile import point_in_polygon
from mandiplan.geometry.resection import CutPlane
from mandiplan.geometry.spline import ArchCurve
from mandiplan.geometry.threshold import estimate_bone_threshold, otsu_threshold
from mandiplan.reference_cases import load_sample
from mandiplan.render.reconstruct import junction_steps, reconstruct
from mandiplan.render.surface import SurfaceExtractor, clip_closed


@pytest.fixture(scope="module")
def scan():
    volume, _, _, arch_points = load_sample()
    threshold = estimate_bone_threshold(volume)
    frames = cpr.build_frames(ArchCurve(arch_points), 0.25)
    return volume, threshold, frames


@pytest.fixture(scope="module")
def isolation(scan):
    volume, threshold, frames = scan
    return isolate_mandible(volume, threshold, frames)


@pytest.fixture(scope="module")
def mandible(scan, isolation):
    volume, threshold, _ = scan
    bone = masked_bone_volume(volume, isolation)
    return bone, SurfaceExtractor(bone).update(threshold)


def test_the_threshold_seed_draws_bone_not_skin(scan):
    """Two-class Otsu splits air from skin on a real scan; the seed must not."""
    volume, threshold, _ = scan
    counts, edges = np.histogram(volume.array, bins=256)
    skin = otsu_threshold(counts, edges)
    assert threshold > skin + 300.0
    # About the bone fraction of a jaw-and-skull crop, not of the whole head.
    fraction = float(np.mean(volume.array >= threshold))
    assert 0.03 < fraction < 0.35


def test_the_mandible_is_separated_from_the_maxilla_and_skull(scan, isolation):
    volume, threshold, frames = scan
    assert isolation.separated
    assert "at the bite" in isolation.summary()
    # An adult mandible with its teeth, as thresholded bone.
    assert 35_000 < isolation.volume_mm3 < 75_000

    bite = isolation.bite
    assert bite is not None and bite.has_teeth.mean() > 0.5
    top_of_bite = float(np.max(bite.heights_mm[bite.has_teeth]))
    kk, jj, ii = np.nonzero(isolation.mask)
    world = volume.origin + np.column_stack([ii, jj, kk]) * volume.spacing
    # Nothing of the skull: no mandible voxel above the condyles (this head
    # is tipped chin-down, so they sit high above the bite) ...
    assert world[:, 2].max() < top_of_bite + 55.0
    # ... but both rami and condyles: the mandible rises well above the bite
    # on each side, behind the teeth.
    midline_x = float(np.mean(frames.points[:, 0]))
    for side in (world[:, 0] < midline_x, world[:, 0] > midline_x):
        assert world[side, 2].max() > top_of_bite + 12.0


def test_the_maxilla_is_not_in_the_mandible(scan, isolation):
    """Above the bite and inside the arch is upper jaw and palate, never mandible."""
    volume, threshold, frames = scan
    bite = isolation.bite
    z_bite = float(np.median(bite.heights_mm[bite.has_teeth]))
    zs = volume.origin[2] + volume.spacing[2] * np.arange(volume.array.shape[0])
    levels = np.flatnonzero((zs > z_bite + 8.0) & (zs < z_bite + 25.0))
    ny, nx = volume.array.shape[1:]
    X, Y = np.meshgrid(
        volume.origin[0] + volume.spacing[0] * np.arange(nx),
        volume.origin[1] + volume.spacing[1] * np.arange(ny),
    )
    # The arch outline, closed across the open end of the U.
    outline = frames.points[::8, :2]
    inside = point_in_polygon(np.column_stack([X.ravel(), Y.ravel()]), outline).reshape(ny, nx)
    bone = (volume.array[levels] >= threshold) & inside[None]
    assert bone.sum() > 500  # the maxilla and palate are there ...
    assert not (isolation.mask[levels] & inside[None]).any()  # ... and none is mandible


def _lateral_cuts(frames, first=0.22, second=0.40):
    planes = []
    for fraction, sign in ((first, 1.0), (second, -1.0)):
        i = frames.index_of(fraction * frames.length_mm)
        planes.append(CutPlane(frames.points[i], sign * frames.tangents[i]))
    return planes


def test_a_cut_removes_mandible_only(scan, isolation, mandible):
    """Cut planes are infinite; on the whole skull they would slice the maxilla."""
    volume, threshold, frames = scan
    _, surface = mandible
    fragment = clip_closed(surface, _lateral_cuts(frames), keep_resected=True)
    top_of_bite = float(np.max(isolation.bite.heights_mm[isolation.bite.has_teeth]))
    assert fragment.GetNumberOfPoints() > 0
    assert fragment.GetBounds()[5] < top_of_bite + 3.0


def _pieces(polydata) -> int:
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(polydata)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    return connectivity.GetNumberOfExtractedRegions()


def test_the_reconstruction_of_a_real_jaw_is_flush(scan, mandible):
    """Registered, blended mirror against the plain mirror, on real bone."""
    volume, threshold, frames = scan
    bone, surface = mandible
    planes = _lateral_cuts(frames)
    symmetry = estimate_midsagittal_plane(bone, threshold)

    rebuilt = reconstruct(bone, threshold, surface, planes, symmetry)
    plain = reconstruct(bone, threshold, surface, planes, symmetry, register=False, band_mm=0.05)
    steps = junction_steps(rebuilt.volume, threshold, planes)
    plain_steps = junction_steps(plain.volume, threshold, planes)

    assert _pieces(rebuilt.surface) == 1
    assert max(steps) < 0.15
    assert sum(steps) < sum(plain_steps)
    assert all(reg.points_used > 100 for reg in rebuilt.report.registrations)


def test_the_session_separates_the_sample_mandible_and_cuts_only_it(qt_app):
    """File › Open sample scan, draw the arch, cut: the path a user takes."""
    from mandiplan.ui.session import Session

    session = Session()
    session.load_sample_scan()
    whole = session.surface_volume_mm3
    _, _, _, arch_points = load_sample()
    for point in arch_points:
        session.add_arch_seed(point)
    # Separation waits for the clicks to stop; a cut does not wait for it.
    assert session.mandible is None
    for fraction, sign in ((0.22, 1.0), (0.40, -1.0)):
        i = session.frames.index_of(fraction * session.frames.length_mm)
        session.add_plane(session.frames.points[i], sign * session.frames.tangents[i])
    assert session.mandible is not None and session.mandible.separated
    assert session.surface_volume_mm3 < 0.6 * whole
    session.execute_cut()
    bite = session.mandible.bite
    assert session.fragment_surface.GetBounds()[5] < np.max(bite.heights_mm[bite.has_teeth]) + 3.0
    # Turning separation off gives the whole bone back.
    session.set_separate_mandible(False)
    assert session.mandible is None
    assert session.surface_volume_mm3 == pytest.approx(whole, rel=1e-6)
