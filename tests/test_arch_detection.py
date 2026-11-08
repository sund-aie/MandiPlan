"""Arch selection must follow anatomical height before head orientation."""

import numpy as np
import pytest

from mandiplan.geometry.mandible import find_arch
from mandiplan.geometry.volume import Volume


def _two_arch_volume(lower_turn_deg):
    """A lower turned jaw and a higher straight distractor, in millimetres."""
    x, y = np.meshgrid(np.arange(-65.0, 66.0), np.arange(-65.0, 66.0))

    def arch_section(turn_deg):
        angle = np.radians(turn_deg)
        u = x * np.cos(angle) + y * np.sin(angle)
        v = -x * np.sin(angle) + y * np.cos(angle)
        front = (v <= 0.0) & (np.abs(np.hypot(u, v) - 32.0) <= 3.0)
        arms = (v > 0.0) & (v <= 18.0) & (np.abs(np.abs(u) - 32.0) <= 3.0)
        return front | arms

    array = np.zeros((80, 131, 131), dtype=np.float32)
    # Both arches persist for more than the required 10 mm body height.
    array[4:21, arch_section(lower_turn_deg)] = 1200.0
    array[32:49, arch_section(0.0)] = 1200.0
    return Volume(array, spacing=np.ones(3), origin=np.array([-65.0, -65.0, 0.0]))


@pytest.mark.parametrize("lower_turn_deg", [-36.0, 36.0])
def test_lowest_rotated_arch_wins_over_higher_straight_arch(lower_turn_deg):
    volume = _two_arch_volume(lower_turn_deg)

    # Establish the ambiguity: a straight-only search finds the upper arch.
    straight = find_arch(volume, threshold=500.0, turns=(0.0,))
    assert straight is not None
    np.testing.assert_allclose(straight[:, 2], 42.0)

    # The default must try every supported orientation at the lower levels
    # before accepting the higher arch; the body curve belongs at 4 + 10 mm.
    found = find_arch(volume, threshold=500.0)
    assert found is not None
    np.testing.assert_allclose(found[:, 2], 14.0)


@pytest.mark.parametrize("crop_slices", [0, 1, 2, 3])
def test_short_body_run_is_found_independently_of_crop_phase(crop_slices):
    """A half-millimetre crop shift must not exchange jaw for upper jaw."""
    section = _two_arch_volume(0.0).array[10] > 0
    z = np.arange(160) * 0.5
    array = np.zeros((len(z), *section.shape), dtype=np.float32)
    # Nine millimetres of valid sections meet the existing 80% persistence
    # criterion. A 2 mm search misses the run at one crop phase (4/6 hits).
    array[(z >= 4.5) & (z <= 13.5)] = 1200.0 * section
    array[(z >= 32.0) & (z <= 48.0)] = 1200.0 * section
    volume = Volume(
        array[crop_slices:], np.array([1.0, 1.0, 0.5]),
        np.array([-65.0, -65.0, crop_slices * 0.5]),
    )
    found = find_arch(volume, 500.0)
    assert found is not None
    assert np.all((found[:, 2] >= 12.5) & (found[:, 2] <= 13.5))


def test_jaw_localization_resolves_a_short_lower_arch_before_the_maxilla():
    from mandiplan.geometry.head_region import coarse_view, find_jaw_region

    section = _two_arch_volume(0.0).array[10] > 0
    x, y = np.meshgrid(np.arange(-65.0, 66.0), np.arange(-65.0, 66.0))
    tissue = x * x + y * y < 60.0**2
    z = np.arange(160) * 0.5
    array = np.broadcast_to(np.where(tissue, 40.0, -1000.0), (len(z), *section.shape)).copy()
    for levels in ((z >= 4.0) & (z <= 12.5), (z >= 32.0) & (z <= 48.0)):
        array[levels] = np.where(section, 1200.0, array[levels])
    volume = Volume(array.astype(np.float32), [1.0, 1.0, 0.5], [-65.0, -65.0, 0.0])
    region = find_jaw_region(coarse_view(volume.array, volume.spacing, volume.origin))
    assert region is not None
    assert 11.0 < region.arch_level_mm < 15.0
    assert region.lo[2] <= 4.0
