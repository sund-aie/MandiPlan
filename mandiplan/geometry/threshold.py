"""Bone threshold estimation.

CBCT gray values are not Hounsfield units: they depend on the scanner, the
field of view and the exposure, so a fixed "bone = 226 HU" threshold is
meaningless here.  Otsu's method gives a data-driven starting point, which the
user then adjusts on a live slider.

A phantom holds two populations, air and bone, and the two-class split is the
answer. A real head scan holds three — air, soft tissue, bone — and the
two-class split then falls between air and skin, which draws the patient's
face instead of the jaw. So the histogram is tested for a soft-tissue
population of its own, and when there is one the upper split of a three-class
Otsu is used. Metal restorations and brackets reach many times the brightest
bone and would compress every tissue into a few bins, so the histogram range
stops at the 99.9th percentile.
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


def otsu_three_class(counts: np.ndarray, edges: np.ndarray) -> tuple[float, float]:
    """The two thresholds of a three-class Otsu split, from a histogram."""
    counts = np.asarray(counts, dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    n = len(counts)
    p = counts / max(counts.sum(), 1.0)
    w = np.concatenate([[0.0], np.cumsum(p)])
    m = np.concatenate([[0.0], np.cumsum(p * centres)])
    # Class boundaries a < b split bins [0, a), [a, b), [b, n).
    a = np.arange(1, n - 1)[:, None]
    b = np.arange(2, n)[None, :]
    w0, m0 = w[a], m[a]
    w1, m1 = w[b] - w[a], m[b] - m[a]
    w2, m2 = w[n] - w[b], m[n] - m[b]
    with np.errstate(divide="ignore", invalid="ignore"):
        score = m0 * m0 / w0 + m1 * m1 / w1 + m2 * m2 / w2
    score[~((b > a) & (w0 > 0) & (w1 > 0) & (w2 > 0))] = -np.inf
    ia, ib = np.unravel_index(int(np.argmax(score)), score.shape)
    # As in the two-class case, the criterion is flat across an empty gap;
    # take the middle of each near-optimal run rather than its first bin.
    best = score[ia, ib]
    near = 0.995 * best if best > 0 else best
    ia = int(round(float(np.median(np.flatnonzero(score[:, ib] >= near)))))
    ib = int(round(float(np.median(np.flatnonzero(score[ia, :] >= near)))))
    return float(edges[ia + 1]), float(edges[ib + 2])


def _has_middle_population(counts: np.ndarray, edges: np.ndarray, t1: float, t2: float) -> bool:
    """Whether the bins between ``t1`` and ``t2`` hold a peak of their own.

    Soft tissue is a peak; the flank of the air peak, or partial-volume
    voxels between air and bone, only fall away from one end.
    """
    centres = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(np.asarray(counts, dtype=float), np.ones(5) / 5.0, mode="same")
    inside = np.flatnonzero((centres > t1) & (centres < t2))
    if len(inside) < 5 or smooth.sum() <= 0:
        return False
    peak = inside[int(np.argmax(smooth[inside]))]
    if peak - inside[0] < 2 or inside[-1] - peak < 2:
        return False
    at_bone = smooth[min(inside[-1] + 1, len(smooth) - 1)]
    share = smooth[inside].sum() / smooth.sum()
    return share > 0.05 and smooth[peak] > 2.0 * max(at_bone, 1.0)


def _sample(array: np.ndarray, limit: int = 4_000_000) -> np.ndarray:
    stride = max(int(np.ceil((array.size / limit) ** (1.0 / 3.0))), 1)
    return np.asarray(array[::stride, ::stride, ::stride], dtype=np.float32).ravel()


def estimate_bone_threshold(volume, bins: int = 256) -> float:
    """Seed value for the bone isosurface threshold, in native gray values."""
    values = _sample(volume.array)
    lo, hi = (float(v) for v in np.percentile(values, [0.5, 99.9]))
    if hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return lo
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    t1, t2 = otsu_three_class(counts, edges)
    if _has_middle_population(counts, edges, t1, t2):
        return t2
    return otsu_threshold(counts, edges)
