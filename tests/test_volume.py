"""Voxel spacing and world-space sampling."""

from __future__ import annotations

import numpy as np
import pytest

from mandiplan.geometry.measure import distance_mm
from mandiplan.geometry.volume import Volume


@pytest.fixture
def ramp() -> Volume:
    """A volume whose value is an exact linear function of world position."""
    spacing = np.array([0.3, 0.5, 0.8])
    origin = np.array([-4.0, 7.0, 2.5])
    nx, ny, nz = 11, 9, 7
    i = np.arange(nx)[None, None, :]
    j = np.arange(ny)[None, :, None]
    k = np.arange(nz)[:, None, None]
    x = origin[0] + i * spacing[0]
    y = origin[1] + j * spacing[1]
    z = origin[2] + k * spacing[2]
    array = 2.0 * x + 3.0 * y - 1.5 * z + 11.0
    return Volume(array=np.broadcast_to(array, (nz, ny, nx)).copy(), spacing=spacing, origin=origin)


def test_index_to_world_uses_per_axis_spacing(ramp):
    world = ramp.index_to_world([2, 3, 4])
    expected = ramp.origin + np.array([2 * 0.3, 3 * 0.5, 4 * 0.8])
    assert np.allclose(world, expected)
    assert np.allclose(ramp.world_to_index(world), [2, 3, 4])


def test_distance_between_voxels_is_anisotropic(ramp):
    a = ramp.index_to_world([0, 0, 0])
    b = ramp.index_to_world([10, 10, 10])
    truth = np.linalg.norm([10 * 0.3, 10 * 0.5, 10 * 0.8])
    assert distance_mm(a, b) == pytest.approx(truth, rel=1e-12)
    # The naive "index distance times one scalar" answer is wrong by a lot.
    naive = np.linalg.norm([10, 10, 10]) * 0.3
    assert abs(naive - truth) > 0.1 * truth


def test_trilinear_sampling_is_exact_on_a_linear_field(ramp):
    rng = np.random.default_rng(0)
    lo, hi = ramp.bounds_mm
    pts = lo + rng.random((200, 3)) * (hi - lo)
    truth = 2.0 * pts[:, 0] + 3.0 * pts[:, 1] - 1.5 * pts[:, 2] + 11.0
    assert np.allclose(ramp.sample(pts), truth, atol=1e-4)


def test_samples_outside_the_volume_take_the_fill_value(ramp):
    lo, _ = ramp.bounds_mm
    outside = np.array([[lo[0] - 5.0, lo[1], lo[2]]])
    assert ramp.sample(outside, fill=-999.0)[0] == -999.0


def test_extent_and_bounds(ramp):
    assert np.allclose(ramp.size_xyz, [11, 9, 7])
    assert np.allclose(ramp.extent_mm, [10 * 0.3, 8 * 0.5, 6 * 0.8])


def test_rejects_non_positive_spacing():
    with pytest.raises(ValueError):
        Volume(array=np.zeros((2, 2, 2)), spacing=[1.0, 0.0, 1.0])
