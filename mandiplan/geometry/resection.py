"""Segmental resection planning geometry.

A :class:`CutPlane` normal points **into the fragment that will be removed**.
With one plane the resected side is everything on the positive side of that
plane; with two planes the resected fragment is the intersection of the two
positive half-spaces, i.e. the bone between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CutPlane:
    origin: np.ndarray
    normal: np.ndarray  # unit, pointing into the resected fragment
    label: str = ""

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)
        n = np.asarray(self.normal, dtype=float).reshape(3)
        length = np.linalg.norm(n)
        if length == 0:
            raise ValueError("cut plane normal must be non-zero")
        self.normal = n / length

    def signed_distance(self, points) -> np.ndarray:
        """Distance in mm; positive on the resected side."""
        pts = np.asarray(points, dtype=float)
        return (pts - self.origin) @ self.normal


@dataclass
class ResectionReport:
    """Everything the resection readout shows.  All lengths in millimetres."""

    arc_length_mm: float = float("nan")
    straight_length_mm: float = float("nan")
    entry_s_mm: float = float("nan")
    exit_s_mm: float = float("nan")
    entry_point: np.ndarray | None = None
    exit_point: np.ndarray | None = None
    margins_mm: list[tuple[str, str, float]] = field(default_factory=list)
    fragment_volume_mm3: float = float("nan")


def resected_mask(planes: list[CutPlane], points) -> np.ndarray:
    """Boolean mask: which points fall inside the resected fragment."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if not planes:
        return np.zeros(len(pts), dtype=bool)
    mask = np.ones(len(pts), dtype=bool)
    for plane in planes:
        mask &= plane.signed_distance(pts) >= 0
    return mask


def curve_crossings(frames, plane: CutPlane) -> list[float]:
    """Arc positions (mm) where the arch curve crosses ``plane``."""
    d = plane.signed_distance(frames.points)
    sign_change = np.where(np.sign(d[:-1]) * np.sign(d[1:]) < 0)[0]
    crossings = []
    for i in sign_change:
        w = d[i] / (d[i] - d[i + 1])
        crossings.append(float(frames.s[i] + w * (frames.s[i + 1] - frames.s[i])))
    return crossings


def _crossing_point(frames, s_mm: float) -> np.ndarray:
    i = frames.index_of(s_mm)
    return frames.points[i].copy()


def build_report(
    frames,
    planes: list[CutPlane],
    landmarks: list[tuple[str, np.ndarray]] | None = None,
) -> ResectionReport:
    """Segment length along the arch curve plus margin distances to landmarks.

    The arc length is measured along the arch curve, which is the correct
    quantity for a plate that will follow the bone; the straight-line distance
    between the two cuts is reported alongside it because the two differ
    substantially across a curved mandible.
    """
    report = ResectionReport()
    inside = (
        resected_mask(planes, frames.points)
        if frames is not None
        else np.zeros(0, dtype=bool)
    )
    if np.any(inside):
        idx = np.where(inside)[0]
        s_lo, s_hi = float(frames.s[idx[0]]), float(frames.s[idx[-1]])
        # Refine the ends with the exact plane crossings when available.
        crossings = sorted(c for p in planes for c in curve_crossings(frames, p))
        inner = [c for c in crossings if s_lo - frames.step_mm <= c <= s_hi + frames.step_mm]
        if len(inner) >= 2:
            s_lo, s_hi = inner[0], inner[-1]
        report.entry_s_mm = s_lo
        report.exit_s_mm = s_hi
        report.entry_point = _crossing_point(frames, s_lo)
        report.exit_point = _crossing_point(frames, s_hi)
        report.arc_length_mm = abs(s_hi - s_lo)
        report.straight_length_mm = float(
            np.linalg.norm(report.exit_point - report.entry_point)
        )

    for plane in planes:
        for name, point in landmarks or []:
            d = float(plane.signed_distance(np.asarray(point).reshape(1, 3))[0])
            report.margins_mm.append((plane.label or "cut", name, d))
    return report
