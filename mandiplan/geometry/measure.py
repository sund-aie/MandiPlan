"""Calibrated measurement helpers.

Everything here consumes and returns millimetres.  There is deliberately no
function that takes pixels: the 2-D views convert to millimetres before they
measure anything, so a pixel-space measurement path cannot creep in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Measurement:
    """A recorded measurement, always carrying its kind and its unit."""

    kind: str  # "straight" or "arc"
    value_mm: float
    label: str

    @property
    def text(self) -> str:
        return f"{self.label}: {self.value_mm:.2f} mm"


def distance_mm(p, q) -> float:
    """Straight-line distance between two world points, in millimetres."""
    return float(np.linalg.norm(np.asarray(q, dtype=float) - np.asarray(p, dtype=float)))


def polyline_length_mm(points) -> float:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def arc_length_between(frames, s_a: float, s_b: float) -> float:
    """Arc length along the arch curve between two arc positions, in mm."""
    i = frames.index_of(s_a)
    j = frames.index_of(s_b)
    return float(abs(frames.s[j] - frames.s[i]))


def format_mm(value: float, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f} mm"


def format_deg(value: float, decimals: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.{decimals}f}°"
