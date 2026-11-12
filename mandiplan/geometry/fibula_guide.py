"""A printable guide for cutting the fibula into the planned pieces.

It sits on the fibula, on the face the surgeon comes to it from (lateral by
default; DePuy lets the surgeon have it anterior too), over the whole stretch
the graft is cut from, and carries a guide for the saw at every cut.

The body. A shell standing ``GAP_MM`` off the bone, ``BASE_MM`` thick, wrapped
``WRAP_DEG`` round it centred on that face, so it seats on the bone's own shape
and lifts off the way it went on (BoneReconstructionPlanner: 0.3 mm off the
bone, 120 degrees, 4 mm; Materialise's recipe 0.3 mm; 2 to 4 mm in use).

The saw. Each cut gets a slot ``SLOT_MM`` wide (DePuy's 1.0 mm, for blades of
0.38 to 0.6 mm), in a tower of walls ``WALL_MM`` thick standing ``TOWER_MM`` above
the shell, so the blade is held over some depth. A slot's wall on the kept
piece's side lies on the planned cut, and the slot opens towards the bone that
is thrown away: wherever the blade wanders in it, the saw takes bone from the
wedge and never from a piece (guided pieces otherwise come out short).

At a joint the two cuts of a wedge lean towards each other. Where they stay far
enough apart through the guide for two slots with plastic between them, they
get two; where they do not, the joint gets one window between the two cuts,
whose two walls guide the blade for each (a flange, which held cuts better
than closed slots in Lohn's test). The planner keeps the cuts apart through the
guide's depth as well as through the bone (``REACH_MM``), so a window never
closes.

Screw holes for 2.0 mm monocortical screws hold it, in each piece clear of the
towers; the pieces' numbers are raised on it, and a P marks its proximal end.

Everything here is in the leg's own millimetres. No Qt or VTK: the guide is a
signed field (negative inside), contoured in ``render/fibula_guide_mesh.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .bending_guide import DIGIT_HEIGHT_MM, digit_parts
from .fibula import fibula_volume

GAP_MM = 0.3
BASE_MM = 3.0
WRAP_DEG = 120.0
SLOT_MM = 1.0
#: The thickest blade the slots are checked with, mm.
BLADE_MM = 0.6
#: The least plastic between two slots side by side, mm.
RIB_MM = 1.5
TOWER_MM = 8.0
WALL_MM = 3.0
#: A slot runs on past the bone this far at each side, mm.
SLOT_REACH_MM = 1.5
SCREW_HOLE_MM = 2.2
#: A screw hole keeps this far from a tower, mm.
SCREW_CLEAR_MM = 2.0
#: The shell runs on past the end towers this far, mm: room for the P.
END_MM = 6.0
#: Contouring grid for printing, and for looking at, mm.
GRID_MM = 0.25
PREVIEW_GRID_MM = 0.5
#: How far out from the bone the guide reaches on its face, mm: two cuts of a
#: joint are kept apart through this as through the bone.
REACH_MM = GAP_MM + BASE_MM + TOWER_MM
#: A screw is two this far apart, at least, mm, when a piece has room for two.
TWO_SCREWS_MM = 16.0
#: A screw's path into the bone, mm: as deep as a monocortical screw goes.
SCREW_DEPTH_MM = 8.0
#: Bone of its own piece kept round a screw's path, mm.
SCREW_BONE_MM = 2.0


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


# -- the bone, as a distance ------------------------------------------------------


@dataclass(eq=False)
class Grid:
    """A field sampled on a regular grid: ``values[k, j, i]`` at ``origin + (i, j, k) * step``."""

    origin: np.ndarray
    step: float
    values: np.ndarray

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        nz, ny, nx = self.values.shape
        return nx, ny, nz

    def points(self, k0: int = 0, k1: int | None = None) -> np.ndarray:
        """The grid's points in slabs ``k0 .. k1 - 1``, x fastest, as (n, 3)."""
        nx, ny, nz = self.shape_xyz
        k1 = nz if k1 is None else k1
        x = self.origin[0] + np.arange(nx) * self.step
        y = self.origin[1] + np.arange(ny) * self.step
        z = self.origin[2] + np.arange(k0, k1) * self.step
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    def sample(self, points: np.ndarray) -> np.ndarray:
        """Trilinear; outside the grid, the largest value it holds."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        nx, ny, nz = self.shape_xyz
        f = (points - self.origin) / self.step
        inside = np.all((f >= 0.0) & (f <= np.array([nx - 1, ny - 1, nz - 1])), axis=1)
        f = np.clip(f, 0.0, np.array([nx - 1.000001, ny - 1.000001, nz - 1.000001]))
        i = np.floor(f).astype(int)
        t = f - i
        v = self.values
        out = np.zeros(len(points))
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = (
                        (t[:, 0] if dx else 1.0 - t[:, 0])
                        * (t[:, 1] if dy else 1.0 - t[:, 1])
                        * (t[:, 2] if dz else 1.0 - t[:, 2])
                    )
                    out += w * v[i[:, 2] + dz, i[:, 1] + dy, i[:, 0] + dx]
        out[~inside] = float(v.max())
        return out


def bone_distance(volume, fibula, threshold: float, lo, hi, step: float) -> Grid:
    """Signed distance to the fibula's surface over the box ``lo``..``hi``, mm:
    negative inside the bone. The scan is resampled to ``step`` first, so the
    surface is the scan's own at ``threshold``, to a fraction of a step."""
    import SimpleITK as sitk

    bone = fibula_volume(volume, fibula)
    image = sitk.GetImageFromArray(np.ascontiguousarray(bone.array, dtype=np.float32))
    image.SetSpacing(tuple(float(v) for v in bone.spacing))
    image.SetOrigin(tuple(float(v) for v in bone.origin))
    lo = np.asarray(lo, dtype=float)
    size = np.ceil((np.asarray(hi, dtype=float) - lo) / step).astype(int) + 1
    resample = sitk.ResampleImageFilter()
    resample.SetOutputOrigin(tuple(lo))
    resample.SetOutputSpacing((float(step),) * 3)
    resample.SetSize([int(v) for v in size])
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(float(np.min(bone.array)))
    fine = resample.Execute(image)
    mask = sitk.BinaryThreshold(fine, lowerThreshold=float(threshold), upperThreshold=1e12, insideValue=1, outsideValue=0)
    distance = sitk.SignedMaurerDistanceMap(
        mask, insideIsPositive=False, squaredDistance=False, useImageSpacing=True
    )
    # Maurer measures to the centres of the bone's outermost voxels; the surface
    # lies half a step out from them.
    values = sitk.GetArrayFromImage(distance).astype(np.float32) - np.float32(0.5 * step)
    return Grid(lo, float(step), values)


# -- the guide ---------------------------------------------------------------------


@dataclass(eq=False)
class Cut:
    """One cut through the guide: a piece's end, and where its slot runs."""

    point: np.ndarray  # on the fibula's line
    normal: np.ndarray  # out of the piece it ends: into the bone thrown away
    piece: int  # the number of the piece it ends
    across: np.ndarray  # in its plane, square to the guide's face
    travel: np.ndarray  # in its plane, out from the bone through the guide
    centre: float  # the middle of the bone across, from ``point`` along ``across``
    half_span: float  # half the slot's length across, mm
    top: float  # how far out along ``travel`` its tower stands, from the line, mm

    def local(self, points: np.ndarray):
        rel = points - self.point
        return rel @ self.normal, rel @ self.across - self.centre, rel @ self.travel

    def tower(self, points: np.ndarray) -> np.ndarray:
        n, y, t = self.local(points)
        return np.maximum.reduce(
            (-WALL_MM - n, n - (SLOT_MM + WALL_MM), np.abs(y) - (self.half_span + WALL_MM), -t, t - self.top)
        )

    def slot(self, points: np.ndarray) -> np.ndarray:
        n, y, _ = self.local(points)
        return np.maximum.reduce((-n, n - SLOT_MM, np.abs(y) - self.half_span))

    def ball(self) -> tuple[np.ndarray, float]:
        """A ball holding its tower, and the shell and bone under it."""
        centre = self.point + self.centre * self.across + 0.5 * self.top * self.travel
        radius = np.sqrt((self.half_span + WALL_MM) ** 2 + (0.5 * self.top) ** 2 + (SLOT_MM + WALL_MM) ** 2)
        return centre, float(radius) + 1.0


@dataclass(eq=False)
class Opening:
    """Where the saw goes through: one cut's slot, or a joint's window between two."""

    cuts: tuple

    @property
    def is_window(self) -> bool:
        return len(self.cuts) == 2

    def _frame(self):
        """A window's own measure across: one for both its cuts (each cut's slot
        length along its own plane would cut the other's short at the ends,
        where the two planes turn from each other); its middle; its way out."""
        a, b = self.cuts
        # The two cuts face opposite ways, so their measures across run opposite.
        sign = 1.0 if float(a.across @ b.across) >= 0.0 else -1.0
        across = _unit(a.across + sign * b.across)
        middle = 0.5 * ((a.point + a.centre * a.across) + (b.point + b.centre * b.across))
        travel = _unit(a.travel + b.travel)
        return across, middle, travel

    def half_span(self) -> float:
        """Half its length across: far enough for its cut's slot, or both."""
        if not self.is_window:
            return self.cuts[0].half_span
        across, middle, _ = self._frame()
        reach = 0.0
        for cut in self.cuts:
            for sign in (-1.0, 1.0):
                end = cut.point + (cut.centre + sign * cut.half_span) * cut.across
                reach = max(reach, abs(float((end - middle) @ across)))
        return reach + 0.5

    def field(self, points: np.ndarray) -> np.ndarray:
        """Negative inside the opening: the saw's way through the guide."""
        if not self.is_window:
            return self.cuts[0].slot(points)
        a, b = self.cuts
        across, middle, _ = self._frame()
        na = (points - a.point) @ a.normal
        nb = (points - b.point) @ b.normal
        y = (points - middle) @ across
        return np.maximum.reduce((-na, -nb, np.abs(y) - self.half_span()))

    def tower(self, points: np.ndarray) -> np.ndarray:
        """The walls standing round the opening, negative inside: ``WALL_MM`` past
        each cut on its kept piece's side, and past the opening's ends."""
        if not self.is_window:
            return self.cuts[0].tower(points)
        a, b = self.cuts
        across, middle, travel = self._frame()
        na = (points - a.point) @ a.normal
        nb = (points - b.point) @ b.normal
        y = (points - middle) @ across
        t = (points - middle) @ travel
        top = max(a.top, b.top)
        return np.maximum.reduce(
            (-WALL_MM - na, -WALL_MM - nb, np.abs(y) - (self.half_span() + WALL_MM), -t, t - top)
        )

    def ball(self) -> tuple[np.ndarray, float]:
        """A ball holding its walls, and the shell and bone under them."""
        if not self.is_window:
            return self.cuts[0].ball()
        a, b = self.cuts
        across, middle, travel = self._frame()
        top = max(a.top, b.top)
        wide = float(np.linalg.norm(a.point - b.point)) + top + 2.0 * WALL_MM + SLOT_MM
        radius = np.sqrt((self.half_span() + WALL_MM) ** 2 + (0.5 * top) ** 2 + wide**2)
        return middle + 0.5 * top * travel, float(radius) + 2.0


@dataclass(eq=False)
class Screw:
    point: np.ndarray  # on the fibula's line
    direction: np.ndarray  # out from the bone, through the guide
    piece: int

    def hole(self, points: np.ndarray) -> np.ndarray:
        rel = points - self.point
        along = rel @ self.direction
        return np.linalg.norm(rel - along[:, None] * self.direction, axis=1) - 0.5 * SCREW_HOLE_MM


@dataclass(eq=False)
class FibulaGuide:
    origin: np.ndarray  # on the line at the lowest cut
    axis: np.ndarray  # up the fibula
    side: np.ndarray  # out of the bone on the guide's face
    across: np.ndarray  # axis x side
    face: str  # "lateral" or "anterior"
    u_range: tuple[float, float]  # the shell's run along the axis, from the origin
    cuts: list[Cut]
    openings: list[Opening]
    screws: list[Screw]
    labels: list = field(default_factory=list)
    #: The least plastic between two slots side by side, as planned, mm; inf if none.
    rib_mm: float = float("inf")
    #: Pieces too short for a screw clear of their towers.
    unscrewed: list[int] = field(default_factory=list)
    #: The bone's top, seen from the guide's face: see ``TopMap``.
    tops: object = None

    def field(self, points: np.ndarray, bone: np.ndarray) -> np.ndarray:
        """The guide, negative inside, at ``points`` whose distance to the bone is ``bone``.

        Each tower, label, opening and hole is worked out only near itself: far
        from it, it changes nothing. ``labels`` are ``(centre, radius, parts)``:
        a word's strokes and a ball holding them.
        """
        points = np.asarray(points, dtype=float)
        bone = np.asarray(bone, dtype=float)
        # Away from the shell's band round the bone and from every tower there is
        # nothing: the shell's own bound says so, and nothing else need be worked out.
        solid = np.maximum(bone - (GAP_MM + BASE_MM), GAP_MM - bone)
        balls = [opening.ball() for opening in self.openings]
        near_opening = [np.linalg.norm(points - c, axis=1) < r for c, r in balls]
        active = bone < GAP_MM + BASE_MM + 1.0
        for near in near_opening:
            active |= near
        if not active.any():
            return solid
        at = points[active]
        d = bone[active]
        rel = at - self.origin
        u, x, y = rel @ self.axis, rel @ self.side, rel @ self.across
        half = np.radians(0.5 * WRAP_DEG)
        sector = np.maximum(-np.sin(half) * x + np.cos(half) * y, -np.sin(half) * x - np.cos(half) * y)
        part = np.maximum.reduce(
            (GAP_MM - d, d - (GAP_MM + BASE_MM), sector, self.u_range[0] - u, u - self.u_range[1])
        )
        nears = [near[active] for near in near_opening]
        for opening, near in zip(self.openings, nears):
            if near.any():
                part[near] = np.minimum(part[near], opening.tower(at[near]))
        for centre, radius, parts in self.labels:
            near = np.linalg.norm(at - centre, axis=1) < radius
            if not near.any():
                continue
            for glyph in parts:
                local = (at[near] - glyph.centre) @ glyph.frame
                part[near] = np.minimum(part[near], glyph.field((local[:, 0], local[:, 1], local[:, 2])))
        # Nothing within the gap of the bone, wherever the towers reach; and
        # nothing under the bone's top seen from the guide's face, so it lifts
        # straight off that face however its towers lean.
        part = np.maximum(part, GAP_MM - d)
        if self.tops is not None:
            top = self.tops.at(u, y)
            under = np.isfinite(top)
            part[under] = np.maximum(part[under], top[under] + GAP_MM - x[under])
        for opening, near in zip(self.openings, nears):
            if near.any():
                part[near] = np.maximum(part[near], -opening.field(at[near]))
        for screw in self.screws:
            near = np.abs((at - screw.point) @ self.axis) < SCREW_HOLE_MM
            if near.any():
                part[near] = np.maximum(part[near], -screw.hole(at[near]))
        solid[active] = part
        return solid

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """A box, in the leg's axes, that holds the guide and the bone under it."""
        reach = max(c.top for c in self.cuts) + 2.0
        wide = max(o.half_span() for o in self.openings) + max(abs(c.centre) for c in self.cuts) + WALL_MM + 2.0
        corners = [
            self.origin + u * self.axis + x * self.side + y * self.across
            for u in (self.u_range[0] - 1.0, self.u_range[1] + 1.0)
            for x in (-wide, reach)
            for y in (-wide, wide)
        ]
        return np.min(corners, axis=0), np.max(corners, axis=0)


@dataclass(eq=False)
class TopMap:
    """How far out on the guide's face the bone reaches, over a grid along and
    across it: ``values[i, j]`` at ``u0 + i * step`` along, ``y0 + j * step``
    across; -inf where there is no bone under that spot."""

    u0: float
    y0: float
    step: float
    values: np.ndarray

    STEP_MM = 0.5

    @classmethod
    def of(cls, bone_points, origin, axis, side, across) -> "TopMap":
        rel = np.asarray(bone_points, dtype=float) - origin
        u, x, y = rel @ axis, rel @ side, rel @ across
        step = cls.STEP_MM
        u0, y0 = float(u.min()) - step, float(y.min()) - step
        shape = (int(np.ceil((u.max() - u0) / step)) + 3, int(np.ceil((y.max() - y0) / step)) + 3)
        values = np.full(shape, -np.inf)
        np.maximum.at(values, (np.rint((u - u0) / step).astype(int), np.rint((y - y0) / step).astype(int)), x)
        # The surface's points are about a voxel apart: fill the cells between them,
        # each with the highest of its neighbours (the bone's top rounded up).
        for _ in range(2):
            padded = np.pad(values, 1, constant_values=-np.inf)
            values = np.max(
                [padded[1 + di : 1 + di + shape[0], 1 + dj : 1 + dj + shape[1]] for di in (-1, 0, 1) for dj in (-1, 0, 1)],
                axis=0,
            )
        return cls(u0, y0, step, values)

    def at(self, u: np.ndarray, y: np.ndarray) -> np.ndarray:
        i = np.rint((u - self.u0) / self.step).astype(int)
        j = np.rint((y - self.y0) / self.step).astype(int)
        inside = (i >= 0) & (i < self.values.shape[0]) & (j >= 0) & (j < self.values.shape[1])
        out = np.full(np.shape(u), -np.inf)
        out[inside] = self.values[i[inside], j[inside]]
        return out


def _faces_up_the_leg(plan):
    """Each cut, lowest first, with the number of the piece it ends."""
    out = []
    for piece in plan.pieces_up_the_leg():
        out.append((piece.distal_face, piece.number))
        out.append((piece.proximal_face, piece.number))
    return out


def _axial_gap(axis, a, b, points) -> np.ndarray:
    """How far apart, along ``axis``, the planes of cut ``a`` (lower) and ``b`` lie
    at each of ``points``: positive while ``b`` is above ``a``."""
    def height(cut):
        return ((cut.point - points) @ cut.normal) / float(cut.normal @ axis)

    return height(b) - height(a)


def plan_guide(plan, fibula, bone_points: np.ndarray, face: str = "lateral") -> FibulaGuide:
    """The guide for cutting ``plan``'s pieces from ``fibula``, seated on ``face``.

    ``bone_points`` are points of the fibula's surface, in the leg.
    """
    bone_points = np.asarray(bone_points, dtype=float)
    faces = _faces_up_the_leg(plan)
    low, high = plan.harvest_mm
    origin = fibula.point_at(low)
    axis = _unit(fibula.point_at(high) - origin)
    side = fibula.side_at(0.5 * (low + high), face)
    side = _unit(side - float(side @ axis) * axis)
    across = np.cross(axis, side)
    radius = float(np.max(fibula.radius_mm))
    cuts: list[Cut] = []
    for face_, number in faces:
        normal = _unit(face_.normal)
        point = np.asarray(face_.point, dtype=float)
        cut_across = _unit(np.cross(normal, side))
        travel = _unit(side - float(side @ normal) * normal)
        near = bone_points[np.abs((bone_points - point) @ normal) < 1.0]
        if len(near) >= 12:
            ys = (near - point) @ cut_across
            centre = 0.5 * float(ys.max() + ys.min())
            half_span = 0.5 * float(ys.max() - ys.min()) + SLOT_REACH_MM
        else:
            centre, half_span = 0.0, radius + SLOT_REACH_MM
        around = bone_points[np.abs((bone_points - point) @ axis) < 4.0]
        outer = float(((around - point) @ side).max()) if len(around) else radius
        top = (outer + GAP_MM + BASE_MM + TOWER_MM) / max(float(travel @ side), 0.5)
        cuts.append(Cut(point, normal, number, cut_across, travel, centre, half_span, top))
    openings: list[Opening] = [Opening((cuts[0],))]
    rib = float("inf")
    for k in range(1, len(cuts) - 1, 2):
        a, b = cuts[k], cuts[k + 1]
        # The room the guide takes over this joint: the bone under it and the
        # guide's depth above, on its face.
        region = bone_points[np.abs((bone_points - 0.5 * (a.point + b.point)) @ axis) < 12.0]
        facing = region[((region - a.point) @ side) >= 0.0] if len(region) else region
        sample = np.vstack([facing + d * side for d in np.linspace(0.0, REACH_MM, 5)]) if len(facing) else facing
        gap = float(_axial_gap(axis, a, b, sample).min()) if len(sample) else float(
            _axial_gap(axis, a, b, a.point[None, :])[0]
        )
        if gap >= 2.0 * SLOT_MM + RIB_MM:
            openings.append(Opening((a,)))
            openings.append(Opening((b,)))
            rib = min(rib, gap - 2.0 * SLOT_MM)
        else:
            openings.append(Opening((a, b)))
    openings.append(Opening((cuts[-1],)))
    # Along the axis: from below the lowest tower to above the highest.
    ends = [float((c.point - origin) @ axis) for c in cuts]
    u_range = (min(ends) - (SLOT_MM + WALL_MM + END_MM), max(ends) + (SLOT_MM + WALL_MM + END_MM))
    guide = FibulaGuide(origin, axis, side, across, face, u_range, cuts, openings, [], [], rib)
    guide.tops = TopMap.of(bone_points, origin, axis, side, across)
    _place_screws_and_labels(guide, plan, fibula, bone_points)
    return guide


def _outer_height(bone_points, origin, axis, side, across, u: float, y: float) -> float:
    """How far out on the guide's face the bone's surface is, at ``u`` along and ``y`` across."""
    rel = bone_points - origin
    near = (np.abs(rel @ axis - u) < 1.5) & (np.abs(rel @ across - y) < 1.0) & ((rel @ side) > 0.0)
    if not near.any():
        return float((rel @ side).max()) if len(rel) else 0.0
    return float((rel[near] @ side).max())


def _radius_at(bone_points, origin, axis, side, across, u: float, angle: float) -> float:
    """How far from the line the bone's surface is at ``u`` along, ``angle`` round
    from the guide's face (radians)."""
    rel = bone_points - origin
    along = rel @ axis
    x, y = rel @ side, rel @ across
    near = (np.abs(along - u) < 1.5) & (np.abs(np.arctan2(y, x) - angle) < np.radians(4.0))
    if not near.any():
        return float(np.hypot(x, y).max()) if len(rel) else 0.0
    return float(np.hypot(x[near], y[near]).max())


def _place_screws_and_labels(guide: FibulaGuide, plan, fibula, bone_points) -> None:
    axis, side, across, origin = guide.axis, guide.side, guide.across, guide.origin
    radius = 0.5 * SCREW_HOLE_MM
    clear_of_towers = WALL_MM + SCREW_CLEAR_MM + radius
    in_bone = SCREW_BONE_MM + radius
    for piece in plan.pieces_up_the_leg():
        own = [c for c in guide.cuts if c.piece == piece.number]
        lower, upper = sorted(own, key=lambda c: float((c.point - origin) @ axis))

        def at_level(cut: Cut, height: float) -> float:
            # Where the cut's plane is along the axis, ``height`` out on the guide's face.
            return float((cut.point - origin) @ axis) - height * float(side @ cut.normal) / float(axis @ cut.normal)

        # A cut leans, and the bone's surface rises and falls: try each place along
        # the piece, with the bone's surface where the screw goes in there, and keep
        # those whose whole path is clear of the towers at the guide's surface and
        # inside the piece from the bone's surface to the screw's tip.
        ends = [float((c.point - origin) @ axis) for c in (lower, upper)]
        candidates = np.arange(min(ends) - 15.0, max(ends) + 15.0, 0.5)
        fits = []
        for u in candidates:
            entry = _outer_height(bone_points, origin, axis, side, across, float(u), 0.0)
            checks = [(entry + GAP_MM + BASE_MM, clear_of_towers), (entry, in_bone), (entry - SCREW_DEPTH_MM, in_bone)]
            fits.append(all(at_level(lower, h) + m <= u <= at_level(upper, h) - m for h, m in checks))
        fits = np.array(fits)
        run = _longest_run(fits)
        text = str(piece.number)
        label_half = 0.5 * len(text) * (0.6 * DIGIT_HEIGHT_MM + 0.8)
        middle = 0.5 * sum(ends)
        if run is None:
            guide.unscrewed.append(piece.number)
            places, label_at = [], (middle, 0.0)
        else:
            lo, hi = float(candidates[run[0]]), float(candidates[run[1]])
            if hi - lo >= 2.0 * (label_half + SCREW_CLEAR_MM + radius) and hi - lo >= TWO_SCREWS_MM:
                places = [lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo)]
                label_at = (0.5 * (lo + hi), 0.0)
            else:
                places = [0.5 * (lo + hi)]
                label_at = (0.5 * (lo + hi), radius + SCREW_CLEAR_MM + 0.5 * DIGIT_HEIGHT_MM)
        for u in places:
            guide.screws.append(Screw(origin + u * axis, side.copy(), piece.number))
        u, y = label_at
        guide.labels.append(_word(_label(text, guide, bone_points, u, y)))
    # The proximal end: a P beyond the highest tower.
    top = max(float((c.point - origin) @ axis) for c in guide.cuts) + SLOT_MM + WALL_MM + 0.5 * END_MM
    guide.labels.append(_word(_label("P", guide, bone_points, top, 0.0)))


def _longest_run(flags: np.ndarray) -> tuple[int, int] | None:
    """The first and last index of the longest run of True in ``flags``."""
    best, start = None, None
    for i, flag in enumerate(np.append(flags, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if best is None or i - 1 - start > best[1] - best[0]:
                best = (start, i - 1)
            start = None
    return best


def _word(parts: list) -> tuple[np.ndarray, float, list]:
    """A word's strokes, with a ball that holds them all."""
    centres = np.array([p.centre for p in parts])
    middle = centres.mean(axis=0)
    radius = max(float(np.linalg.norm(p.centre - middle) + np.linalg.norm(p.half)) for p in parts) + 0.5
    return middle, radius, parts


def _label(text: str, guide: FibulaGuide, bone_points, u: float, y: float) -> list:
    """Raised characters on the shell at ``u`` along and about ``y`` across, reading
    up the leg, flat on the shell where they stand: sunk 0.3 mm into it."""
    reach = _radius_at(bone_points, guide.origin, guide.axis, guide.side, guide.across, u, 0.0)
    angle = y / max(reach + GAP_MM + BASE_MM, 1.0)
    radius = _radius_at(bone_points, guide.origin, guide.axis, guide.side, guide.across, u, angle)
    normal = np.cos(angle) * guide.side + np.sin(angle) * guide.across
    up = _unit(np.cross(normal, guide.axis))
    centre = guide.origin + u * guide.axis + (radius + GAP_MM + BASE_MM) * normal
    return digit_parts(text, centre, guide.axis, up, normal)


# -- following its use ------------------------------------------------------------------


@dataclass
class GuideCheck:
    """What following the guide's use found, in mm."""

    bone_clear_mm: float  # the least room between guide and bone, seated
    lift_off_mm: float  # the deepest it runs into bone being lifted off its face: 0 is clean
    blade_clear_mm: float  # the least room round a blade pressed to any slot's kept wall
    uncut_share: float  # of the bone in any cut's plane, the share the blade cannot reach
    kerf_into_piece_mm: float  # how far any slot lets the saw into a kept piece
    rib_mm: float  # the thinnest plastic between two slots side by side
    screw_margin_mm: float  # the least bone of its own piece round where a screw goes in
    unscrewed: list[int] = field(default_factory=list)
    #: The graft's joints, and how many of them have one window for both cuts.
    joints: int = 0
    windows: int = 0

    @property
    def ok(self) -> bool:
        return (
            self.bone_clear_mm >= GAP_MM - 0.05
            and self.lift_off_mm <= 0.05
            and self.blade_clear_mm > 0.0
            and self.uncut_share <= 0.001
            and self.kerf_into_piece_mm <= 0.05
            and self.rib_mm >= RIB_MM - 1e-6
            and self.screw_margin_mm >= 1.0
            and not self.unscrewed
        )

    def lines(self) -> list[str]:
        return [
            f"Seated, it stands {self.bone_clear_mm:.2f} mm off the bone at its closest.",
            "It lifts straight off its face." if self.lift_off_mm <= 0.05 else
            f"Lifting it off its face it would catch the bone by {self.lift_off_mm:.2f} mm.",
            f"A {BLADE_MM:.1f} mm blade runs freely down every slot, with "
            f"{SLOT_MM - BLADE_MM:.1f} mm of play." if self.blade_clear_mm > 0.0 else
            f"A {BLADE_MM:.1f} mm blade catches on the guide in a slot.",
            "Every cut's slot reaches across the whole bone." if self.uncut_share <= 0.001 else
            f"{self.uncut_share:.1%} of a cut's bone is out of the blade's reach.",
            "No slot lets the saw into a piece that is kept." if self.kerf_into_piece_mm <= 0.05 else
            f"A slot lets the saw {self.kerf_into_piece_mm:.2f} mm into a kept piece.",
            self._joint_line(),
            f"Each screw goes into its own piece, {self.screw_margin_mm:.1f} mm or more from its cuts."
            if not self.unscrewed else
            "Piece " + ", ".join(str(i) for i in self.unscrewed) + " is too short for a screw clear of its slots.",
        ]

    def _joint_line(self) -> str:
        if self.joints == 0:
            return "One piece: no joint to cut."
        if self.windows == self.joints:
            return "Each joint has one window, its two walls guiding its two cuts."
        line = f"The thinnest plastic between two slots is {self.rib_mm:.1f} mm."
        if self.windows:
            line += f" {self.windows} of the {self.joints} joints have one window for both cuts."
        return line


def _in_piece_depth(piece, points: np.ndarray) -> np.ndarray:
    """How deep each point is inside ``piece``'s stretch, between its two cuts: >0 inside."""
    d1 = -((points - piece.distal_face.point) @ piece.distal_face.normal)
    d2 = -((points - piece.proximal_face.point) @ piece.proximal_face.normal)
    return np.minimum(d1, d2)


def check_guide(
    guide: FibulaGuide, plan, bone: Grid, solid: np.ndarray | None = None, printed_clear_mm: float | None = None
) -> GuideCheck:
    """Seat the guide, lift it off, run a blade down every slot, and drive every
    screw, on the bone as the scan has it. ``solid`` is the guide's field on the
    grid of ``bone``, if it has been worked out already. ``printed_clear_mm`` is
    how far the printed guide's surface stands from the bone's, measured on the
    two meshes; without it, the grid's own figure is used."""
    step = bone.step
    if solid is None:
        solid = guide_field_on(guide, bone)
    inside = solid.ravel() < 0.0
    distance = bone.values.ravel()
    clear = float(distance[inside].min()) if inside.any() else float("inf")
    if printed_clear_mm is not None:
        clear = float(printed_clear_mm)
    # Lift it off: every bit of it near the bone moved straight out along its face.
    near = inside & (distance < GAP_MM + 2.0)
    points = _grid_points_where(bone, near)
    worst = 0.0
    for t in np.arange(step, 6.0 + 1e-9, step):
        worst = max(worst, float(-bone.sample(points + t * guide.side).min()))
    # The blades, pressed to each slot's kept wall, through the guide and the bone.
    blade_clear = float("inf")
    uncut = 0.0
    bone_in = _grid_points_where(bone, distance < 0.0)
    for cut in guide.cuts:
        n = np.array([0.05, 0.5 * BLADE_MM, BLADE_MM - 0.05])
        y = np.linspace(-cut.half_span + 0.1, cut.half_span - 0.1, int(2.0 * cut.half_span / 0.25) + 1)
        t = np.arange(-cut.top, cut.top + 2.0, 0.25)
        nn, yy, tt = np.meshgrid(n, y, t, indexing="ij")
        samples = (
            cut.point
            + nn.reshape(-1, 1) * cut.normal
            + (yy.reshape(-1, 1) + cut.centre) * cut.across
            + tt.reshape(-1, 1) * cut.travel
        )
        blade_clear = min(blade_clear, float(guide.field(samples, bone.sample(samples)).min()))
        # The bone in the cut's plane, and how much of it the slot's length covers.
        local_n, local_y, _ = cut.local(bone_in)
        in_plane = (local_n >= 0.0) & (local_n <= BLADE_MM)
        if in_plane.any():
            uncut = max(uncut, float(np.mean(np.abs(local_y[in_plane]) > cut.half_span)))
    # The saw anywhere in a slot or a window: does it reach into a kept piece?
    into = 0.0
    for opening in guide.openings:
        cutting = bone_in[opening.field(bone_in) < 0.0]
        for piece in plan.pieces:
            depth = _in_piece_depth(piece, cutting)
            if len(depth):
                into = max(into, float(depth.max()))
    # The screws: each driven from the guide into the bone, as deep as a
    # monocortical screw goes; how far its path stays inside its own piece.
    margin = float("inf")
    by_number = {p.number: p for p in plan.pieces}
    for screw in guide.screws:
        out = np.arange(25.0, -25.0, -0.1)  # from beyond the guide, inwards
        line = screw.point + np.outer(out, screw.direction)
        d = bone.sample(line)
        entering = np.flatnonzero((d[:-1] >= 0.0) & (d[1:] < 0.0))
        if not len(entering):
            margin = -1.0
            continue
        entry = out[entering[0] + 1]
        path = screw.point + np.outer(np.arange(entry, entry - SCREW_DEPTH_MM - 1e-9, -0.25), screw.direction)
        margin = min(margin, float(_in_piece_depth(by_number[screw.piece], path).min()))
    return GuideCheck(
        bone_clear_mm=clear,
        lift_off_mm=worst,
        blade_clear_mm=blade_clear,
        uncut_share=uncut,
        kerf_into_piece_mm=into,
        rib_mm=guide.rib_mm,
        screw_margin_mm=margin,
        unscrewed=list(guide.unscrewed),
        joints=len(plan.pieces) - 1,
        windows=sum(1 for o in guide.openings if o.is_window),
    )


def _grid_points_where(grid: Grid, mask: np.ndarray) -> np.ndarray:
    index = np.flatnonzero(mask)
    nx, ny, _ = grid.shape_xyz
    i = index % nx
    j = (index // nx) % ny
    k = index // (nx * ny)
    return grid.origin + np.column_stack([i, j, k]) * grid.step


def guide_field_on(guide: FibulaGuide, bone: Grid) -> np.ndarray:
    """The guide's field over the whole of ``bone``'s grid, (nz, ny, nx)."""
    nx, ny, nz = bone.shape_xyz
    out = np.empty((nz, ny, nx), dtype=np.float32)
    slab = max(1, 600_000 // (nx * ny))
    for k in range(0, nz, slab):
        last = min(nz, k + slab)
        points = bone.points(k, last)
        out[k:last] = guide.field(points, bone.values[k:last].reshape(-1)).reshape(last - k, ny, nx)
    return out
