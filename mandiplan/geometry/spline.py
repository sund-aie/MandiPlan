"""Catmull-Rom arch curve with arc-length-uniform resampling.

Uniform *parameter* spacing along a spline is not uniform *arc length*
spacing, and every measurement downstream of the curve (panoramic x-axis,
plate segment lengths, resection span) is an arc length.  So the curve is
densely evaluated once, its cumulative chord length is built, and all public
resampling is done by inverting that table.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Dense evaluation step used to build the arc-length table.  Chord error of a
# polyline through a curve of radius R with step h is h^2 / (24 R^2) relative,
# i.e. below 1e-7 for h = 0.05 mm and R = 5 mm.
_DENSE_SAMPLES_PER_SEGMENT = 256


@dataclass
class CurveSamples:
    """Points sampled at uniform arc length along a curve."""

    s: np.ndarray  # (M,) cumulative arc length in mm, s[0] == 0
    points: np.ndarray  # (M, 3) world mm
    tangents: np.ndarray  # (M, 3) unit
    step_mm: float


def _drop_duplicates(points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    keep = [0]
    for i in range(1, len(points)):
        if np.linalg.norm(points[i] - points[keep[-1]]) > tol:
            keep.append(i)
    return points[keep]


def _centripetal_knots(points: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    d = np.linalg.norm(np.diff(points, axis=0), axis=1) ** alpha
    return np.concatenate([[0.0], np.cumsum(d)])


def _circumcentre(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray | None:
    """Centre of the circle through three points, or None if they are collinear."""
    ab, ac = b - a, c - a
    cross = np.cross(ab, ac)
    denom = 2.0 * float(np.dot(cross, cross))
    if denom < 1e-18:
        return None
    offset = (
        np.cross(cross, ab) * float(np.dot(ac, ac))
        + np.cross(ac, cross) * float(np.dot(ab, ab))
    ) / denom
    return a + offset


def _curvature_ghost(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Extrapolate one step beyond ``p1``, continuing the circle through p1-p3.

    Falls back to a straight reflection when the three points are collinear.
    """
    centre = _circumcentre(p1, p2, p3)
    if centre is None:
        return 2 * p1 - p2
    radius = np.linalg.norm(p1 - centre)
    if radius > 1e4 * np.linalg.norm(p2 - p1):
        return 2 * p1 - p2
    u = (p1 - centre) / radius
    w = p2 - centre
    along = float(np.dot(w, u)) * u
    return centre + along - (w - along)


def _barry_goldman(p, t, u):
    """Non-uniform Catmull-Rom via the Barry-Goldman pyramid.

    ``p`` is (4, 3) control points, ``t`` is (4,) knots, ``u`` is (N,)
    parameter values inside ``[t[1], t[2]]``.  Returns (N, 3).
    """
    u = u[:, None]
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    t0, t1, t2, t3 = t
    a1 = (t1 - u) / (t1 - t0) * p0 + (u - t0) / (t1 - t0) * p1
    a2 = (t2 - u) / (t2 - t1) * p1 + (u - t1) / (t2 - t1) * p2
    a3 = (t3 - u) / (t3 - t2) * p2 + (u - t2) / (t3 - t2) * p3
    b1 = (t2 - u) / (t2 - t0) * a1 + (u - t0) / (t2 - t0) * a2
    b2 = (t3 - u) / (t3 - t1) * a2 + (u - t1) / (t3 - t1) * a3
    return (t2 - u) / (t2 - t1) * b1 + (u - t1) / (t2 - t1) * b2


class ArchCurve:
    """A cubic (centripetal Catmull-Rom) spline through user seed points."""

    def __init__(self, seeds, samples_per_segment: int = _DENSE_SAMPLES_PER_SEGMENT):
        seeds = np.asarray(seeds, dtype=float).reshape(-1, 3)
        seeds = _drop_duplicates(seeds)
        if len(seeds) < 2:
            raise ValueError("an arch curve needs at least 2 distinct seed points")
        self.seeds = seeds
        self._dense_points = self._evaluate_dense(seeds, samples_per_segment)
        step = np.linalg.norm(np.diff(self._dense_points, axis=0), axis=1)
        self._dense_s = np.concatenate([[0.0], np.cumsum(step)])
        self._dense_tangents = self._dense_tangent_field(self._dense_points)

    @staticmethod
    def _evaluate_dense(seeds: np.ndarray, per_segment: int) -> np.ndarray:
        if len(seeds) == 2:
            w = np.linspace(0.0, 1.0, per_segment + 1)[:, None]
            return seeds[0] * (1 - w) + seeds[1] * w
        # Ghost control points beyond each end, so every segment has 4
        # controls.  They continue the local curvature rather than the local
        # chord: a straight reflection makes the end tangent follow the first
        # chord, which on an arch is half a seed spacing off the true tangent.
        ext = np.vstack(
            [
                _curvature_ghost(seeds[0], seeds[1], seeds[2]),
                seeds,
                _curvature_ghost(seeds[-1], seeds[-2], seeds[-3]),
            ]
        )
        knots = _centripetal_knots(ext)
        pieces = []
        for i in range(len(ext) - 3):
            p = ext[i : i + 4]
            t = knots[i : i + 4]
            u = np.linspace(t[1], t[2], per_segment + 1)
            seg = _barry_goldman(p, t, u)
            pieces.append(seg if i == 0 else seg[1:])
        return np.vstack(pieces)

    @staticmethod
    def _dense_tangent_field(points: np.ndarray) -> np.ndarray:
        tan = np.gradient(points, axis=0)
        norm = np.linalg.norm(tan, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return tan / norm

    @property
    def length_mm(self) -> float:
        """Arc length of the whole curve, in millimetres."""
        return float(self._dense_s[-1])

    def sample_at(self, s) -> tuple[np.ndarray, np.ndarray]:
        """Point and unit tangent at arc positions ``s`` (mm from the start)."""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        pts = np.column_stack(
            [np.interp(s, self._dense_s, self._dense_points[:, c]) for c in range(3)]
        )
        tan = np.column_stack(
            [np.interp(s, self._dense_s, self._dense_tangents[:, c]) for c in range(3)]
        )
        norm = np.linalg.norm(tan, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return pts, tan / norm

    def resample(self, step_mm: float) -> CurveSamples:
        """Sample the curve at uniform arc-length intervals of ``step_mm``."""
        if step_mm <= 0:
            raise ValueError("resampling step must be positive")
        n = max(int(np.floor(self.length_mm / step_mm)) + 1, 2)
        s = np.arange(n) * step_mm
        pts, tan = self.sample_at(s)
        return CurveSamples(s=s, points=pts, tangents=tan, step_mm=float(step_mm))

    def arc_position_of(self, point) -> float:
        """Arc length (mm) of the point on the curve closest to ``point``."""
        d = np.linalg.norm(self._dense_points - np.asarray(point, dtype=float), axis=1)
        return float(self._dense_s[int(np.argmin(d))])

    @property
    def dense_points(self) -> np.ndarray:
        """Dense polyline for drawing the curve."""
        return self._dense_points


def resample_polyline(points, step_mm: float) -> np.ndarray:
    """Resample an open polyline at uniform arc length (mm) along its chords."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    pts = _drop_duplicates(pts)
    if len(pts) < 2:
        raise ValueError("need at least 2 distinct points to resample")
    if step_mm <= 0:
        raise ValueError("resampling step must be positive")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(np.floor(s[-1] / step_mm)) + 1
    target = np.arange(max(n, 2)) * step_mm
    target = target[target <= s[-1] + 1e-9]
    return np.column_stack([np.interp(target, s, pts[:, c]) for c in range(3)])


def interpolate_along_polyline(points, values, step_mm: float) -> np.ndarray:
    """Interpolate per-vertex ``values`` at the same stations as ``resample_polyline``."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    vals = np.asarray(values, dtype=float).reshape(len(pts), -1)
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 1e-6:
            keep.append(i)
    pts, vals = pts[keep], vals[keep]
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = int(np.floor(s[-1] / step_mm)) + 1
    target = np.arange(max(n, 2)) * step_mm
    target = target[target <= s[-1] + 1e-9]
    return np.column_stack(
        [np.interp(target, s, vals[:, c]) for c in range(vals.shape[1])]
    )
