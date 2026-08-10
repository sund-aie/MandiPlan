"""Measurement helpers shared by the tests.

These deliberately measure the way the application does: a sub-pixel
half-maximum crossing along a profile, converted to millimetres through the
image's own pixel size.
"""

from __future__ import annotations

import numpy as np


def half_level(profile: np.ndarray) -> float:
    """Mid-point between the profile's air and bone plateaus.

    Robust percentiles rather than min/max, so a single noisy voxel cannot
    move the level.
    """
    lo = float(np.percentile(profile, 5))
    hi = float(np.percentile(profile, 95))
    return 0.5 * (lo + hi)


def crossing_positions(profile: np.ndarray, level: float) -> tuple[float, float]:
    """First and last crossings of ``level``, in fractional sample units."""
    idx = np.where(profile >= level)[0]
    if len(idx) == 0:
        raise AssertionError("profile never reaches the requested level")
    first, last = int(idx[0]), int(idx[-1])

    def interp(lo: int, hi: int) -> float:
        a, b = profile[lo], profile[hi]
        if a == b:
            return float(lo)
        return lo + (level - a) / (b - a) * (hi - lo)

    start = interp(first - 1, first) if first > 0 else float(first)
    end = interp(last + 1, last) if last < len(profile) - 1 else float(last)
    return start, end


def extent_mm(profile: np.ndarray, pixel_mm: float, level: float | None = None) -> float:
    """Width of a bone profile at half maximum, in millimetres."""
    level = half_level(profile) if level is None else level
    start, end = crossing_positions(profile, level)
    return (end - start) * pixel_mm


def rel_error(measured: float, truth: float) -> float:
    return abs(measured - truth) / abs(truth)


def circular_arc_path(
    radius_mm: float, theta_span_rad: float, step_mm: float = 0.1, z: float = 0.0
) -> np.ndarray:
    """Dense polyline along a circular arc in the axial plane."""
    n = int(radius_mm * theta_span_rad / step_mm) + 1
    theta = np.linspace(0.0, theta_span_rad, n)
    return np.column_stack(
        [radius_mm * np.cos(theta), radius_mm * np.sin(theta), np.full(n, z)]
    )
