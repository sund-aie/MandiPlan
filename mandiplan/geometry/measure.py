"""Calibrated measurement helpers.

Everything here consumes and returns millimetres.  There is deliberately no
function that takes pixels: the 2-D views convert to millimetres before they
measure anything, so a pixel-space measurement path cannot creep in.
"""

from __future__ import annotations

import numpy as np


def distance_mm(p, q) -> float:
    """Straight-line distance between two world points, in millimetres."""
    return float(np.linalg.norm(np.asarray(q, dtype=float) - np.asarray(p, dtype=float)))


def format_mm(value: float, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f} mm"


def format_deg(value: float, decimals: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.{decimals}f}°"
