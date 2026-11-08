"""Segmental resection planning geometry.

A :class:`CutPlane` normal points **into the fragment that will be removed**.
With one plane the resected side is everything on the positive side of that
plane; with two planes the resected fragment is the intersection of the two
positive half-spaces, i.e. the bone between them.

Planes are infinite, but a saw is not: it cuts the jaw where it is held, and
the far side of the arch, which the same infinite plane also crosses, is
untouched. So the resection is also resolved along the jaw
(:class:`SegmentReach`), by the nearest point of the arch curve to each point
of bone. Close to a cut the plane decides. Elsewhere only the position along
the curve does: one cut takes everything from there on round the jaw, through
the far ramus and its condyle wherever they lie relative to the plane, and
two take the stretch between them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

SUPERIOR = np.array([0.0, 0.0, 1.0])


#: How far along the jaw from a cut its plane decides, mm: enough for a
#: steeply oblique cut.
SEGMENT_MARGIN_MM = 25.0
#: Depth reported for bone far along the jaw from a cut, mm: well inside or
#: well outside the resection, whatever side of the infinite plane it is on.
FAR_ALONG_MM = 50.0


@dataclass
class SegmentReach:
    """Where along the arch curve each plane of a resection cuts.

    ``stations`` are points of the whole curve about a millimetre apart, at
    arc positions ``s``. ``cuts[i]`` is the arc position plane ``i`` cuts at
    and ``sides[i]`` the direction along the curve it removes (+1 or -1).
    """

    stations: np.ndarray
    s: np.ndarray
    cuts: list[float]
    sides: list[float]

    @property
    def lo(self) -> float:
        """The arc position the removed stretch starts at; ``s[0]`` if open."""
        ends = [c for c, side in zip(self.cuts, self.sides) if side > 0]
        return max(ends) if ends else float(self.s[0])

    @property
    def hi(self) -> float:
        """The arc position the removed stretch ends at; ``s[-1]`` if open."""
        ends = [c for c, side in zip(self.cuts, self.sides) if side < 0]
        return min(ends) if ends else float(self.s[-1])

    def nearest_s(self, points) -> np.ndarray:
        """Arc position of the station nearest each point."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        nearest = np.empty(len(pts), dtype=float)
        chunk = max(1, 2_000_000 // max(len(self.stations), 1))
        for start in range(0, len(pts), chunk):
            part = pts[start : start + chunk]
            d2 = ((part[:, None, :] - self.stations[None, :, :]) ** 2).sum(axis=2)
            nearest[start : start + chunk] = self.s[np.argmin(d2, axis=1)]
        return nearest


@dataclass
class CutPlane:
    origin: np.ndarray
    normal: np.ndarray  # unit, pointing into the resected fragment
    label: str = ""
    #: The stretch of jaw the cuts this plane belongs to take out; shared by
    #: the planes of one resection. None: unbounded (see the module notes).
    reach: SegmentReach | None = None

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
class PlanePlacement:
    """Where a cut sits on the jaw and how it is angled there.

    This, not a world-space normal, is the authoritative state of a cutting
    plane. Keeping the position and the angulation as separate numbers is what
    lets a cut travel along the mandible while holding the obliquity the
    surgeon dialled in: the base orientation is re-derived from the local
    mandibular frame at the new position, and the offsets are re-applied to
    that new frame.

    ``s_mm``
        Arc length along the arch curve. This is the only translation state.
    ``yaw_deg``, ``tilt_deg``, ``roll_deg``
        Angular offsets **relative to the local mandibular frame**, never to
        global axes. See :func:`plane_from_placement` for the convention.
    ``offset_mm``
        A further translation of the origin in patient axes
        (x = left, y = posterior, z = superior), for fine adjustment off the
        curve itself.
    ``flipped``
        Which side the cut removes. A plane normal points into the fragment
        being resected; with two cuts the second one faces back down the
        curve, so it carries ``flipped=True``. Held here rather than as a sign
        on the normal so it survives translation.
    ``anchor_mm``
        The point of the curve the cut was put at, in world mm. Arc length
        means nothing on a different curve, so when the curve is replaced
        the cut is found again at the new curve's point nearest this one.
    """

    s_mm: float = 0.0
    yaw_deg: float = 0.0
    tilt_deg: float = 0.0
    roll_deg: float = 0.0
    offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    flipped: bool = False
    anchor_mm: tuple[float, float, float] | None = None

    def replace(self, **changes) -> "PlanePlacement":
        """A copy with some fields changed."""
        return dataclasses.replace(self, **changes)


def plane_from_placement(frames, placement: PlanePlacement):
    """Resolve a placement into a world-space ``(origin, normal)``.

    Local mandibular frame
    ----------------------
    At arc position ``s_mm`` the arch curve gives a right-handed triad:

    ``T``  the arch tangent, direction of travel toward increasing arc length
    ``U``  patient superior, parallel-transported along the curve so it never
           spins or flips as the curve climbs into the angle and ramus
    ``B``  ``T x U``, the buccolingual direction

    Convention for the cut
    ----------------------
    **The plane normal is T**: the default cut is perpendicular to the local
    arch. That is the osteotomy a saw makes held square to the bone at that
    point, and it is what makes the same plane look right in the anterior
    body, at the angle and in the ramus without being re-dialled.

    The offsets are then applied to that frame, in this order, each about an
    axis of the frame as already rotated:

    1. ``roll_deg`` about ``T`` — spins U and B about the direction of travel.
       A plane is invariant under rotation about its own normal, so roll does
       not move the cut by itself; it chooses the axes that yaw and tilt then
       act about, which is how a surgeon reaches an oblique that is neither
       purely axial nor purely vertical.
    2. ``yaw_deg`` about the rolled ``U`` — swings the cut in the axial sense,
       positive counter-clockwise seen from superior.
    3. ``tilt_deg`` about the twice-rotated ``B`` — tips the cut superiorly or
       inferiorly, positive tipping the normal toward superior.

    Returns ``(origin, normal)``, the normal pointing into the fragment that
    will be removed.
    """
    s_mm = frames.clamp(placement.s_mm)
    tangent, up, binormal = frames.frame_at(s_mm)

    roll = np.radians(placement.roll_deg)
    up = _rotate(up, tangent, roll)
    binormal = _rotate(binormal, tangent, roll)

    yaw = np.radians(placement.yaw_deg)
    normal = _rotate(tangent, up, yaw)
    binormal = _rotate(binormal, up, yaw)

    normal = _rotate(normal, binormal, np.radians(placement.tilt_deg))
    normal = normal / np.linalg.norm(normal)
    if placement.flipped:
        normal = -normal

    origin = frames.point_at(s_mm) + np.asarray(
        placement.offset_mm, dtype=float
    ).reshape(3)
    return origin, normal


def plane_from_frame(
    frames,
    s_mm: float,
    yaw_deg: float = 0.0,
    tilt_deg: float = 0.0,
    offset_mm=(0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Backwards-compatible wrapper over :func:`plane_from_placement`."""
    return plane_from_placement(
        frames,
        PlanePlacement(
            s_mm=s_mm,
            yaw_deg=yaw_deg,
            tilt_deg=tilt_deg,
            offset_mm=tuple(np.asarray(offset_mm, dtype=float).reshape(3)),
        ),
    )


def _rotate(vector, axis, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation of ``vector`` about a unit ``axis``."""
    v = np.asarray(vector, dtype=float)
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    return (
        v * np.cos(angle_rad)
        + np.cross(k, v) * np.sin(angle_rad)
        + k * np.dot(k, v) * (1.0 - np.cos(angle_rad))
    )


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
    return plane_depths(planes, pts).min(axis=1) >= 0


def plane_depths(planes: list[CutPlane], points) -> np.ndarray:
    """How far each point is into the resected side of each plane, mm.

    Shape (len(points), len(planes)); positive means on the side the plane
    removes. Far along the jaw from a cut, that side is decided by the position
    along the jaw, and the depth is at least ``FAR_ALONG_MM`` in or out.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    depths = np.column_stack([plane.signed_distance(pts) for plane in planes])
    reach = planes[0].reach if planes else None
    if reach is None:
        return depths
    along = reach.nearest_s(pts)
    for column, (cut, side) in enumerate(zip(reach.cuts, reach.sides)):
        far = np.abs(along - cut) > SEGMENT_MARGIN_MM
        removed = far & (side * (along - cut) > 0)
        kept = far & ~removed
        depths[removed, column] = np.maximum(depths[removed, column], FAR_ALONG_MM)
        depths[kept, column] = np.minimum(depths[kept, column], -FAR_ALONG_MM)
    return depths


def bound_resection(frames, planes: list[CutPlane]) -> None:
    """Resolve ``planes`` along the arch curve (see the module notes).

    Each plane cuts the curve at the crossing nearest its origin, and removes
    the direction along the curve its normal points. Sets ``reach`` on each
    plane; None, leaving plain half-spaces, when there is no curve or the cuts
    take nothing between them (both facing away from each other).
    """
    for plane in planes:
        plane.reach = None
    if frames is None or not planes:
        return
    stride = max(int(round(1.0 / frames.step_mm)), 1)
    stations = np.asarray(frames.points[::stride], dtype=float)
    s = np.asarray(frames.s[::stride], dtype=float)
    cuts, sides = [], []
    for plane in planes:
        cut = float(frames.s[np.argmin(np.linalg.norm(frames.points - plane.origin, axis=1))])
        crossings = curve_crossings(frames, plane)
        if crossings:
            nearest = min(crossings, key=lambda c: abs(c - cut))
            if abs(nearest - cut) <= SEGMENT_MARGIN_MM:
                cut = nearest
        tangent = frames.tangents[frames.index_of(cut)]
        cuts.append(cut)
        sides.append(1.0 if float(tangent @ plane.normal) >= 0 else -1.0)
    reach = SegmentReach(stations, s, cuts, sides)
    if reach.lo >= reach.hi:
        return
    for plane in planes:
        plane.reach = reach


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
    reach = planes[0].reach if planes else None
    if reach is not None:
        report.entry_s_mm = reach.lo
        report.exit_s_mm = reach.hi
        report.entry_point = _crossing_point(frames, reach.lo)
        report.exit_point = _crossing_point(frames, reach.hi)
        report.arc_length_mm = reach.hi - reach.lo
        report.straight_length_mm = float(
            np.linalg.norm(report.exit_point - report.entry_point)
        )
    inside = (
        resected_mask(planes, frames.points)
        if frames is not None
        else np.zeros(0, dtype=bool)
    )
    if reach is None and np.any(inside):
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
