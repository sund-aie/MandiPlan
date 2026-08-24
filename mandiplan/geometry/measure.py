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


def angle_deg(a, b, c) -> float:
    """Angle at ``b`` in the corner a-b-c, in degrees.

    Both planning products list angle measurement beside distance, and it is
    the natural way to record a gonial angle or check a bend against the plan.
    """
    u = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    w = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    lengths = np.linalg.norm(u) * np.linalg.norm(w)
    if lengths == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, w) / lengths, -1.0, 1.0))))


def format_mm(value: float, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{decimals}f} mm"


def format_deg(value: float, decimals: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.{decimals}f}°"
