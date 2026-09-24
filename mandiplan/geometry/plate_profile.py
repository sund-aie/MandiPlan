"""The flat shape of a bone plate: outline, notches, and screw-hole sections.

Pure geometry, numpy only. ``tools/make_plate_assets.py`` meshes these; the
application never needs to, but keeping the shape here means it is testable
and the same numbers drive the mesh, the catalogue and the marking layout.

Real reconstruction plates are not chains of washers. They are bars with
straight parallel edges, rounded ends, and small circular notches cut into
both edges between the screw holes — the notches reduce the section where the
plate is meant to bend, so it bends there and not through a hole. Miniplates
are the same construction with the notches so deep the plate becomes a row of
eyelets joined by thin bars. Both are produced by :func:`outline`.

Every loop is sampled with a fixed number of points per segment regardless of
how far it is inset. That is what lets the edge fillet be built as a stack of
rings with matching vertices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


#: Thinnest wall of solid metal allowed between a screw seat and an edge or
#: a notch. Miniplate eyelets run close to this; reconstruction bars far above.
MIN_WALL_MM = 0.3


@dataclass(frozen=True)
class NotchSpec:
    """A circular notch cut into both edges between two neighbouring holes."""

    depth_mm: float
    #: Length of the opening along the edge.
    opening_mm: float

    @property
    def radius_mm(self) -> float:
        half = self.opening_mm / 2.0
        return (half * half + self.depth_mm * self.depth_mm) / (2.0 * self.depth_mm)


@dataclass(frozen=True)
class HoleSpec:
    """The section of one screw hole, top face to bone face.

    ``seat_diameter_mm`` is where the screw head sits, at the outer face.
    Non-locking holes are a plain conical countersink down to the bore;
    locking holes carry a conical thread in that section instead, drawn as
    concentric rings at the thread pitch rather than a helix.
    """

    bore_diameter_mm: float
    seat_diameter_mm: float
    #: Depth of the conical seat below the outer face, as a fraction of
    #: the plate thickness.
    seat_depth_fraction: float = 0.55
    locking: bool = False
    thread_pitch_mm: float = 0.35
    thread_depth_mm: float = 0.08

    def section(self, thickness_mm: float) -> list[tuple[float, float]]:
        """``(radius, z)`` pairs from the outer face (z = +t/2) down to bone."""
        top = thickness_mm / 2.0
        bottom = -thickness_mm / 2.0
        r_seat = self.seat_diameter_mm / 2.0
        r_bore = self.bore_diameter_mm / 2.0
        seat_bottom = top - thickness_mm * self.seat_depth_fraction
        rings: list[tuple[float, float]] = []
        if self.locking:
            # A conical thread: crest and root rings alternating every half
            # pitch down the cone.
            steps = max(2, int(round((top - seat_bottom) / (self.thread_pitch_mm / 2.0))))
            for k in range(steps + 1):
                f = k / steps
                z = top + (seat_bottom - top) * f
                r = r_seat + (r_bore - r_seat) * f
                if 0 < k < steps and k % 2 == 1:
                    r += self.thread_depth_mm
                rings.append((r, z))
        else:
            rings.append((r_seat, top))
            rings.append((r_bore, seat_bottom))
        rings.append((r_bore, bottom))
        return rings


@dataclass(frozen=True)
class PlateShape:
    """Everything needed to draw one flat plate."""

    hole_x_mm: tuple[float, ...]
    width_mm: float
    thickness_mm: float
    hole: HoleSpec
    notch: NotchSpec
    edge_radius_mm: float = 0.3
    #: Gaps (index i means between hole i and i+1) that carry no notch.
    unnotched_gaps: tuple[int, ...] = field(default_factory=tuple)
    #: Gaps whose notch differs from the default, as ``(index, NotchSpec)``
    #: pairs — the long bridge of a bridged miniplate, for instance.
    gap_notches: tuple[tuple[int, NotchSpec], ...] = field(default_factory=tuple)

    def notch_for(self, gap: int) -> NotchSpec:
        for index, spec in self.gap_notches:
            if index == gap:
                return spec
        return self.notch

    @property
    def half_width_mm(self) -> float:
        return self.width_mm / 2.0

    @property
    def length_mm(self) -> float:
        return float(self.hole_x_mm[-1] - self.hole_x_mm[0] + self.width_mm)

    def validate(self) -> None:
        """Refuse a shape that could not be machined as drawn."""
        xs = np.asarray(self.hole_x_mm, dtype=float)
        if len(xs) < 1:
            raise ValueError("a plate needs at least one hole")
        if np.any(np.diff(xs) <= 0):
            raise ValueError("hole positions must increase along the plate")
        r_seat = self.hole.seat_diameter_mm / 2.0
        if r_seat + MIN_WALL_MM > self.half_width_mm - self.edge_radius_mm:
            raise ValueError(
                f"a {self.hole.seat_diameter_mm} mm screw seat leaves less than "
                f"{MIN_WALL_MM} mm of wall on a {self.width_mm} mm plate"
            )
        if self.hole.bore_diameter_mm >= self.hole.seat_diameter_mm:
            raise ValueError("the bore must be narrower than the screw seat")
        for i, (a, b) in enumerate(zip(xs[:-1], xs[1:])):
            if i in self.unnotched_gaps:
                continue
            clearance = _notch_clearance(self, a, b, self.notch_for(i))
            if clearance < MIN_WALL_MM:
                raise ValueError(
                    f"the notch between holes {i + 1} and {i + 2} comes within "
                    f"{clearance:.2f} mm of a screw seat"
                )


def _notch_centre(shape: PlateShape, a: float, b: float, n: NotchSpec) -> tuple[float, float]:
    return (a + b) / 2.0, shape.half_width_mm + n.radius_mm - n.depth_mm


def _notch_clearance(shape: PlateShape, a: float, b: float, n: NotchSpec) -> float:
    """Material left between a notch and the nearest screw seat."""
    cx, cy = _notch_centre(shape, a, b, n)
    r_seat = shape.hole.seat_diameter_mm / 2.0
    gap = math.hypot(cx - a, cy) - n.radius_mm - r_seat
    return float(gap)


# Samples per segment, fixed so that inset loops stay vertex-for-vertex
# aligned with each other.
_CAP_SAMPLES = 14
_NOTCH_SAMPLES = 14
_STRAIGHT_STEP_MM = 0.45


def outline(shape: PlateShape, inset_mm: float = 0.0) -> np.ndarray:
    """The plate outline, inset by ``inset_mm``, as a counter-clockwise loop.

    Returns (N, 2). The same shape always yields the same N whatever the
    inset, and point k of one inset corresponds to point k of another.
    """
    xs = np.asarray(shape.hole_x_mm, dtype=float)
    h = shape.half_width_mm - inset_mm
    if h <= 0:
        raise ValueError("inset is larger than half the plate width")
    upper: list[tuple[float, float]] = []

    def straight(p0, p1, count):
        for t in np.linspace(0.0, 1.0, count, endpoint=False):
            upper.append((p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t))

    # Left end: quarter circle from the leftmost point up to the top edge.
    for a in np.linspace(math.pi, math.pi / 2.0, _CAP_SAMPLES, endpoint=False):
        upper.append((xs[0] + h * math.cos(a), h * math.sin(a)))
    cursor = (xs[0], h)

    for i, (a, b) in enumerate(zip(xs[:-1], xs[1:])):
        if i in shape.unnotched_gaps:
            continue
        n = shape.notch_for(i)
        r_notch = n.radius_mm + inset_mm
        cx, cy = _notch_centre(shape, a, b, n)
        # Where the (inset) edge line meets the (grown) notch circle.
        dy = h - cy
        dx = math.sqrt(max(r_notch * r_notch - dy * dy, 0.0))
        start = (cx - dx, h)
        count = _straight_samples(shape, a, b, "before", i)
        straight(cursor, start, count)
        a_left = math.atan2(dy, -dx)
        a_right = math.atan2(dy, dx)
        if a_left > 0:
            a_left -= 2.0 * math.pi
        for ang in np.linspace(a_left, a_right, _NOTCH_SAMPLES, endpoint=False):
            upper.append((cx + r_notch * math.cos(ang), cy + r_notch * math.sin(ang)))
        cursor = (cx + dx, h)

    count = _straight_samples(shape, xs[-1], xs[-1], "tail", len(xs))
    straight(cursor, (xs[-1], h), count)
    for a in np.linspace(math.pi / 2.0, 0.0, _CAP_SAMPLES, endpoint=False):
        upper.append((xs[-1] + h * math.cos(a), h * math.sin(a)))
    upper.append((xs[-1] + h, 0.0))

    # The lower edge mirrors the upper, walked back from right to left; the
    # two points on the centreline are shared.
    lower = [(x, -y) for x, y in reversed(upper[1:-1])]
    loop = np.array(upper + lower, dtype=float)
    # Upper edge left to right then lower edge back is clockwise; flip it.
    return loop[::-1].copy()


def _straight_samples(shape: PlateShape, a: float, b: float, where: str, index: int) -> int:
    """Samples on a straight run, fixed from the un-inset geometry."""
    xs = np.asarray(shape.hole_x_mm, dtype=float)
    h = shape.half_width_mm
    notched = [i for i in range(len(xs) - 1) if i not in shape.unnotched_gaps]
    # Reconstruct the un-inset run lengths so the count never depends on inset.
    runs = []
    cursor = xs[0]
    for i in notched:
        n = shape.notch_for(i)
        cx, cy = _notch_centre(shape, xs[i], xs[i + 1], n)
        dx = math.sqrt(max(n.radius_mm ** 2 - (h - cy) ** 2, 0.0))
        runs.append((i, cursor, cx - dx))
        cursor = cx + dx
    tail = xs[-1] - cursor
    if where == "tail":
        return max(2, int(math.ceil(abs(tail) / _STRAIGHT_STEP_MM)))
    for i, lo, hi in runs:
        if i == index:
            return max(2, int(math.ceil(abs(hi - lo) / _STRAIGHT_STEP_MM)))
    return 2


def hole_loop(centre_x: float, radius: float, samples: int) -> np.ndarray:
    """A screw-hole circle, counter-clockwise, starting on +x."""
    angles = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    return np.column_stack([centre_x + radius * np.cos(angles), radius * np.sin(angles)])


def point_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd rule, vectorised over ``points``."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    x0, y0 = polygon[:, 0][None, :], polygon[:, 1][None, :]
    x1, y1 = np.roll(polygon[:, 0], -1)[None, :], np.roll(polygon[:, 1], -1)[None, :]
    crosses = ((y0 > y) != (y1 > y)) & (
        x < (x1 - x0) * (y - y0) / np.where(y1 - y0 == 0, 1e-30, y1 - y0) + x0
    )
    return np.count_nonzero(crosses, axis=1) % 2 == 1


def marking_slots(shape: PlateShape, text_height_mm: float) -> list[tuple[float, float, float]]:
    """Where lettering fits on the outer face: ``(x_start, x_end, y)`` runs.

    Lettering goes along the band between the screw seats and one edge,
    in the straight runs between notches, where there is solid face under it.
    """
    xs = np.asarray(shape.hole_x_mm, dtype=float)
    r_seat = shape.hole.seat_diameter_mm / 2.0
    face_edge = shape.half_width_mm - shape.edge_radius_mm
    band_low = r_seat + 0.15
    band_high = face_edge - 0.1
    if band_high - band_low < text_height_mm:
        return []
    y = -(band_low + band_high) / 2.0 - text_height_mm / 2.0
    slots = []
    for i, (a, b) in enumerate(zip(xs[:-1], xs[1:])):
        if i in shape.unnotched_gaps:
            slots.append((a + r_seat * 0.2, b - r_seat * 0.2, y))
            continue
        n = shape.notch_for(i)
        cx, cy = _notch_centre(shape, a, b, n)
        # The notch eats the band near its centre; use the two runs beside it.
        reach = math.sqrt(max(n.radius_mm ** 2 - (cy - band_high) ** 2, 0.0))
        left, right = cx - reach - 0.1, cx + reach + 0.1
        if left - a > 1.0:
            slots.append((a, left, y))
        if b - right > 1.0:
            slots.append((right, b, y))
    return slots
