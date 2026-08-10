"""Bone threshold estimation.

CBCT gray values are not Hounsfield units: they depend on the scanner, the
field of view and the exposure, so a fixed "bone = 226 HU" threshold is
meaningless here.  Otsu's method gives a data-driven starting point, which the
user then adjusts on a live slider.
"""

from __future__ import annotations

import numpy as np


def otsu_threshold(counts: np.ndarray, edges: np.ndarray) -> float:
    """Otsu's between-class-variance threshold from a histogram."""
    counts = np.asarray(counts, dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    total = counts.sum()
    if total <= 0:
        return float(centres[len(centres) // 2])

    w0 = np.cumsum(counts)
    w1 = total - w0
    sum_all = np.cumsum(counts * centres)
    mean_total = sum_all[-1] / total

    valid = (w0 > 0) & (w1 > 0)
    mu0 = np.divide(sum_all, w0, out=np.zeros_like(sum_all), where=w0 > 0)
    mu1 = np.divide(
        sum_all[-1] - sum_all, w1, out=np.zeros_like(sum_all), where=w1 > 0
    )
    between = w0 * w1 * (mu0 - mu1) ** 2
    between[~valid] = -1.0

    # When the two populations are well separated the criterion is flat across
    # the whole empty gap between them, and argmax would return the very edge
    # of the noise floor.  Take the middle of the near-optimal run instead, so
    # the isosurface starts in the middle of the gap.
    best = between.max()
    plateau = np.where(between >= 0.995 * best)[0]
    k = int(round(float(np.median(plateau))))
    _ = mean_total
    return float(centres[k])


def estimate_bone_threshold(volume, bins: int = 256) -> float:
    """Seed value for the bone isosurface threshold, in native gray values."""
    counts, edges = volume.histogram(bins=bins)
    return otsu_threshold(counts, edges)
