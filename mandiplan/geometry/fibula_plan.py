"""Planning a fibula graft for the resected mandible.

The fibula is cut into straight pieces which, set end to end, follow the jaw
through the defect. The plan has three parts.

The course. Along the reconstructed jaw (the mirror of the healthy side, or the
jaw as it was), section by section, where the fibula's line has to run for the
bone to restore the jaw's lower border and outer face: its centre a radius above
the border and a radius in from the outer face. From one cut to the other.

The pieces. The course is replaced by a chain of straight chords, the joints on
the course, chosen so the chain strays from the course as little as it can
(the largest gap, the sagitta of the worst chord, as small as it can be) with no
piece shorter than a vascularised piece of fibula can be. Fewer pieces are
preferred while they keep within the tolerance.

The fibula. Each piece is a stretch of the fibula, laid along its chord: the
fibula's own line onto the chord, its proximal end towards the end of the defect
the vessels are to leave from, and the fibula turned about its line so a chosen
side of it faces a chosen way. Between two pieces the jaw's joint is one plane,
half-way between the two chords; each piece's end is that plane carried back
onto the fibula, so on the leg the two cuts of a joint lean towards each other
and take out a wedge. The pieces follow one another up the fibula, as far down
it as the ankle allows, and the bone between each pair is just wide enough that
the two cuts never meet inside the bone.

All in patient millimetres: the jaw's own (head scan) and the leg's own (leg
scan) are different spaces, and each piece carries the rigid transform from the
leg to the jaw. No Qt or VTK here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .reconstruction import RigidTransform
from .resection import CutPlane

#: The course is read and resampled at this spacing, mm.
COURSE_STEP_MM = 1.0
#: A section of the jaw is the surface within this of the station's plane, mm.
SECTION_HALF_MM = 0.6
#: How far either side of the arch a section reaches, (across, up), mm.
SECTION_REACH_MM = (25.0, 45.0)
#: The outer face is read in a band this tall round the fibula's centre, mm.
FACE_BAND_MM = 1.5
#: The course is smoothed along the jaw over this much, mm.
COURSE_SMOOTH_MM = 3.0
#: A cut's crossing of the arch is looked for this far from its origin's place on it, mm.
CUT_SEARCH_MM = 15.0

# The clinical defaults. Every one is a setting; these are where planners and
# the literature put them (sources in DECISIONS.md and the research report).
#: No piece shorter than this, mm. A 1 cm piece has no periosteal vessel a third
#: of the time, a piece over 2 cm seldom (Fry 2016); 20 mm is the usual floor.
MIN_PIECE_MM = 20.0
#: The automatic choice goes up to this many pieces; KLS Martin's order form
#: offers 1 to 5, 3D Systems' 1 to 6.
MAX_PIECES = 5
#: Fewer pieces are used while the chain stays this close to the course, mm:
#: inside the 0.7 to 2 mm a guided cut lands from its plan.
COURSE_TOLERANCE_MM = 2.0
#: Fibula left at the ankle, mm from the malleolus' tip: Materialise keeps the
#: graft 70 mm clear (KLS Martin 80, the surgical atlases 50 to 60).
DISTAL_KEEP_MM = 70.0
#: Fibula left at the knee, mm from the head's tip: no figure is published; the
#: common peroneal nerve winds round the neck just below the head.
PROXIMAL_KEEP_MM = 60.0
#: Bone left between neighbouring cuts at their closest, mm: as
#: BoneReconstructionPlanner leaves; a saw's kerf (0.5 to 1 mm) fits inside it.
CUT_CLEARANCE_MM = 1.5
#: An end of the graft leaning more than this from square to the fibula is said:
#: the jaw's cut is far from square to the course there, and the fibula's end
#: would be a long oblique face, hard to saw and to seat.
MAX_END_LEAN_DEG = 45.0

@dataclass(eq=False)
class Course:
    """Where the fibula's line has to run through the defect, in the jaw."""

    points: np.ndarray  # (M, 3), from the first cut to the second, COURSE_STEP_MM apart
    outward: np.ndarray  # (M, 3) unit, away from the jaw's middle, square to the course
    up: np.ndarray  # (M, 3) unit, towards the alveolar crest, square to the course
    border: np.ndarray  # (M, 3) the lower border under each point
    first_plane: CutPlane  # the cut the course starts at
    second_plane: CutPlane

    @property
    def length_mm(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


@dataclass(eq=False)
class Face:
    """One end of a piece: a plane through the piece's line."""

    point: np.ndarray  # where it crosses the piece's line
    normal: np.ndarray  # unit, pointing out of the piece
    #: How far it leans from square to the piece's line, degrees.
    lean_deg: float = 0.0


@dataclass(eq=False)
class Piece:
    """One straight piece of fibula and where it goes."""

    number: int  # 1 at the first cut of the defect, counting along the jaw
    start: np.ndarray  # its line's ends in the jaw, in course order
    end: np.ndarray
    #: Where it is taken from: its line's distal and proximal ends, mm from
    #: the fibula's distal end.
    s_distal: float
    s_proximal: float
    #: Its two ends on the fibula, in the leg's space: the distal, the proximal.
    distal_face: Face
    proximal_face: Face
    to_jaw: RigidTransform  # leg space to jaw space

    @property
    def length_mm(self) -> float:
        return float(np.linalg.norm(self.end - self.start))


@dataclass(eq=False)
class FibulaPlan:
    pieces: list[Piece]  # in course order
    joints: np.ndarray  # (n + 1, 3) the chain's vertices in the jaw
    #: The angle between neighbouring pieces at each inner joint, degrees.
    joint_angles_deg: list[float]
    deviation_mm: float  # the chain's largest gap from the course
    #: The bone left between neighbouring pieces on the fibula, at their lines, mm.
    gaps_mm: list[float]
    #: The stretch of fibula used, mm from its distal end: first cut to last cut.
    harvest_mm: tuple[float, float]
    #: Fibula above the graft, below the proximal limit: freed for the vessels, mm.
    above_graft_mm: float
    pedicle_end: str  # "first" or "second": the end of the defect the proximal piece is at
    #: How far each piece's chosen side turns from facing straight out of the jaw,
    #: degrees: no piece is twisted on the next, so on a curving jaw they cannot all.
    facing_deg: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def harvest_length_mm(self) -> float:
        return self.harvest_mm[1] - self.harvest_mm[0]

    def pieces_up_the_leg(self) -> list[Piece]:
        return sorted(self.pieces, key=lambda p: p.s_distal)


# -- the course --------------------------------------------------------------


def smooth_polyline(points: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along a polyline, its ends held by repetition."""
    points = np.asarray(points, dtype=float)
    radius = int(np.ceil(3.0 * sigma))
    if radius < 1 or len(points) < 3:
        return points.copy()
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    padded = np.concatenate(
        [np.repeat(points[:1], radius, axis=0), points, np.repeat(points[-1:], radius, axis=0)]
    )
    out = np.zeros_like(points, dtype=float)
    for w, o in zip(weights, offsets):
        out += w * padded[radius + o : radius + o + len(points)]
    return out / weights.sum()


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def _cut_position(frames, plane: CutPlane) -> float:
    """Where along the arch a cut crosses it: the crossing nearest its origin.

    A crossing exactly on a station counts. A plane through one limb of the U
    can cross the other limb as well; a crossing far from the cut's own place
    on the arch is not its own.
    """
    near = float(frames.s[np.argmin(np.linalg.norm(frames.points - plane.origin, axis=1))])
    d = plane.signed_distance(frames.points)
    found = []
    for i in np.flatnonzero(d[:-1] * d[1:] <= 0.0):
        if d[i] == d[i + 1]:
            continue
        w = float(d[i] / (d[i] - d[i + 1]))
        found.append(float(frames.s[i] + w * (frames.s[i + 1] - frames.s[i])))
    found = [c for c in found if abs(c - near) <= CUT_SEARCH_MM]
    return min(found, key=lambda c: abs(c - near)) if found else near


def defect_course(
    vertices: np.ndarray,
    frames,
    planes: list[CutPlane],
    fibula_radius_mm: float,
    raise_mm: float = 0.0,
    inset_mm: float = 0.0,
) -> Course | None:
    """The course of the fibula's line through the defect between two cuts.

    ``vertices`` are the reconstructed jaw's surface. Section by section the
    fibula's centre is put ``fibula_radius_mm + raise_mm`` above the lower border
    and ``fibula_radius_mm + inset_mm`` in from the outer face at that height.
    None when the defect is not two cuts or the jaw cannot be read through it.
    """
    if len(planes) != 2 or frames is None:
        return None
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    r = float(fibula_radius_mm)
    cuts = sorted((_cut_position(frames, p), i) for i, p in enumerate(planes))
    (s_a, i_a), (s_b, i_b) = cuts
    first, second = planes[i_a], planes[i_b]
    if s_b - s_a < 2.0 * COURSE_STEP_MM:
        return None
    reach = 6.0
    stations = np.arange(max(s_a - reach, float(frames.s[0])), min(s_b + reach, float(frames.s[-1])) + 1e-9, COURSE_STEP_MM)
    centre = frames.points.mean(axis=0)
    rows = []
    previous = None
    for s in stations:
        tangent, up, binormal = frames.frame_at(float(s))
        at = frames.point_at(float(s))
        side = binormal if float((at - centre) @ binormal) >= 0.0 else -binormal
        if previous is not None and float(side @ previous) < 0.0:
            side = -side
        previous = side
        rel = vertices - at
        slab = np.abs(rel @ tangent) < SECTION_HALF_MM
        if slab.sum() < 12:
            continue
        rel = rel[slab]
        u, b = rel @ up, rel @ side
        near = (np.abs(b) < SECTION_REACH_MM[0]) & (np.abs(u) < SECTION_REACH_MM[1])
        if near.sum() < 12:
            continue
        u, b = u[near], b[near]
        low = float(np.percentile(u, 1.0))
        level = low + r + float(raise_mm)
        band = np.abs(u - level) < FACE_BAND_MM
        if band.sum() < 3:
            continue
        face = float(b[band].max())
        across = face - r - float(inset_mm)
        rows.append((at + level * up + across * side, side, up, at + low * up + across * side))
    if len(rows) < 3:
        return None
    points = smooth_polyline(np.array([row[0] for row in rows]), COURSE_SMOOTH_MM / COURSE_STEP_MM)
    outward = np.array([row[1] for row in rows])
    ups = np.array([row[2] for row in rows])
    border = smooth_polyline(np.array([row[3] for row in rows]), COURSE_SMOOTH_MM / COURSE_STEP_MM)
    # From where the course crosses the first cut to where it crosses the second.
    run = _between_cuts(points, first, second)
    if run is None:
        return None
    lo, hi, t_lo, t_hi = run
    # The ends exactly on the cuts, then evenly spaced.
    def blend(a, i, t):
        return a[i] + t * (a[i + 1] - a[i])

    ends = [blend(points, lo, t_lo), blend(points, hi, t_hi)]
    inner = points[lo + 1 : hi + 1]
    chain = np.vstack([ends[0], inner, ends[1]])
    steps = np.linalg.norm(np.diff(chain, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    count = max(int(round(arc[-1] / COURSE_STEP_MM)), 2) + 1
    target = np.linspace(0.0, arc[-1], count)

    def resample(values):
        values = np.vstack([blend(values, lo, t_lo), values[lo + 1 : hi + 1], blend(values, hi, t_hi)])
        return np.column_stack([np.interp(target, arc, values[:, c]) for c in range(3)])

    course_points = resample(points)
    tangent = _unit(np.gradient(course_points, axis=0))
    out = resample(outward)
    out = _unit(out - np.einsum("ij,ij->i", out, tangent)[:, None] * tangent)
    up = resample(ups)
    up = _unit(up - np.einsum("ij,ij->i", up, tangent)[:, None] * tangent)
    return Course(course_points, out, up, resample(border), first, second)


def _between_cuts(points: np.ndarray, first: CutPlane, second: CutPlane):
    """The run of ``points`` inside both cuts: ``(lo, hi, t_lo, t_hi)``.

    The run starts on the segment ``lo``-``lo + 1`` at fraction ``t_lo`` (where
    it crosses ``first``) and ends on ``hi``-``hi + 1`` at ``t_hi``.
    """
    d1 = first.signed_distance(points)
    d2 = second.signed_distance(points)
    inside = (d1 > 0.0) & (d2 > 0.0)
    if inside.sum() < 2:
        return None
    # The longest run inside.
    best, start = None, None
    for i, flag in enumerate(np.append(inside, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    a, b = best  # inside for a .. b - 1
    if a == 0 or b == len(points):
        return None  # the course never reached past a cut: not read far enough

    def crossing(d, i):
        return float(d[i] / (d[i] - d[i + 1])) if d[i] != d[i + 1] else 0.5

    # Entering at a - 1 -> a, through whichever cut is crossed there.
    d_in = d1 if d1[a - 1] <= 0.0 else d2
    d_out = d2 if d2[b] <= 0.0 else d1
    return a - 1, b - 1, crossing(d_in, a - 1), crossing(d_out, b - 1)


# -- the pieces -----------------------------------------------------------------


def _chord_deviation(points: np.ndarray) -> np.ndarray:
    """``out[i, j]``: the largest distance of ``points[i..j]`` from the chord i-j."""
    n = len(points)
    out = np.zeros((n, n))
    for i in range(n - 1):
        p = points[i:]
        a = p[0]
        d = p[1:] - a  # chords to each later point
        length2 = np.einsum("ij,ij->i", d, d)
        rel = p - a  # every point from a
        # t of each point m on chord k: clipped projection.
        t = np.clip((rel @ d.T) / np.maximum(length2, 1e-12), 0.0, 1.0)  # (m, k)
        closest = t[:, :, None] * d[None, :, :]
        dist = np.linalg.norm(rel[:, None, :] - closest, axis=2)  # (m, k)
        # Only points up to the chord's end count: m <= k + 1.
        m = np.arange(len(p))[:, None]
        k = np.arange(1, len(p))[None, :]
        dist = np.where(m <= k, dist, 0.0)
        out[i, i + 1 :] = dist.max(axis=0)
    return out


def fit_joints(points: np.ndarray, count: int, min_piece_mm: float) -> tuple[list[int], float] | None:
    """The joints of the best chain of ``count`` chords along ``points``.

    Returns the indices of the chain's vertices (first and last point included)
    and its largest gap from the points, or None if no chain of that many
    chords, none shorter than ``min_piece_mm``, fits.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if count < 1 or n < 2:
        return None
    deviation = _chord_deviation(points)
    chord = np.linalg.norm(points[None, :, :] - points[:, None, :], axis=2)
    allowed = chord >= float(min_piece_mm) - 1e-9
    inf = np.inf
    # best[c][j]: the smallest worst gap of c chords from point 0 to point j.
    best = np.full((count + 1, n), inf)
    total = np.full((count + 1, n), inf)  # the summed squared gaps, to break ties
    back = np.zeros((count + 1, n), dtype=int)
    best[0, 0] = 0.0
    total[0, 0] = 0.0
    for c in range(1, count + 1):
        for j in range(1, n):
            i = np.arange(j)
            ok = allowed[i, j] & np.isfinite(best[c - 1, i])
            if not ok.any():
                continue
            worst = np.maximum(best[c - 1, i], deviation[i, j])
            summed = total[c - 1, i] + deviation[i, j] ** 2
            worst = np.where(ok, worst, inf)
            key = worst + 1e-6 * np.where(ok, summed, inf)
            k = int(np.argmin(key))
            if np.isfinite(worst[k]):
                best[c, j], total[c, j], back[c, j] = worst[k], summed[k], k
    if not np.isfinite(best[count, n - 1]):
        return None
    joints = [n - 1]
    for c in range(count, 0, -1):
        joints.append(int(back[c, joints[-1]]))
    return joints[::-1], float(best[count, n - 1])


def choose_joints(
    points: np.ndarray,
    min_piece_mm: float = MIN_PIECE_MM,
    max_pieces: int = MAX_PIECES,
    tolerance_mm: float = COURSE_TOLERANCE_MM,
    count: int | None = None,
):
    """The joints for ``count`` pieces, or else the fewest pieces within the tolerance.

    Returns ``(joints, deviation)``, or None if not even one piece fits.
    """
    if count is not None:
        return fit_joints(points, int(count), min_piece_mm)
    fits = []
    for c in range(1, int(max_pieces) + 1):
        fit = fit_joints(points, c, min_piece_mm)
        if fit is None:
            break
        fits.append(fit)
        if fit[1] <= tolerance_mm:
            return fit
    return min(fits, key=lambda f: f[1]) if fits else None


# -- onto the fibula -----------------------------------------------------------------


def _rotate(vector: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = _unit(axis)
    a = np.radians(angle_deg)
    return (
        vector * np.cos(a)
        + np.cross(axis, vector) * np.sin(a)
        + axis * float(axis @ vector) * (1.0 - np.cos(a))
    )


def _transport(vector: np.ndarray, before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """``vector`` turned by the least rotation that takes ``before`` to ``after``."""
    axis = np.cross(before, after)
    sine = float(np.linalg.norm(axis))
    cosine = float(before @ after)
    if sine < 1e-12:
        return vector.copy()
    return _rotate(vector, axis / sine, float(np.degrees(np.arctan2(sine, cosine))))


def _faces_out(course: Course, joints: list[int], directions: list[np.ndarray], roll_deg: float):
    """Which way the fibula's lateral side faces in each piece, and how far from out.

    Carried from piece to piece by the least rotation at each joint, so no piece
    is twisted on the next (a rotation-minimising frame): every joint is a plain
    wedge and the vessels stay along one side. The first piece's direction is
    the one that faces out of the jaw best over all the pieces together, then
    turned by ``roll_deg``.
    """
    n = len(directions)
    # The transport from the first piece to each, as a matrix.
    carry = [np.eye(3)]
    for k in range(1, n):
        step = np.column_stack([_transport(e, directions[k - 1], directions[k]) for e in np.eye(3)])
        carry.append(step @ carry[-1])
    wanted = []
    for k in range(n):
        span = slice(joints[k], joints[k + 1] + 1)
        out = course.outward[span].mean(axis=0)
        wanted.append(_unit(out - float(out @ directions[k]) * directions[k]))
    # The first piece's out that best matches every piece's own, carried.
    pull = sum(carry[k].T @ wanted[k] for k in range(n))
    first = _unit(pull - float(pull @ directions[0]) * directions[0])
    first = _rotate(first, directions[0], roll_deg)
    outs = [_unit(carry[k] @ first) for k in range(n)]
    facing = [
        float(np.degrees(np.arccos(np.clip(float(outs[k] @ wanted[k]), -1.0, 1.0)))) for k in range(n)
    ]
    return outs, facing


def _frame(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """A right-handed frame, columns: ``first``, ``second`` made square to it, their cross."""
    f1 = _unit(first)
    f2 = _unit(second - float(second @ f1) * f1)
    return np.column_stack([f1, f2, np.cross(f1, f2)])


def _face(point, normal, axis) -> Face:
    normal = _unit(normal)
    lean = float(np.degrees(np.arccos(np.clip(abs(float(normal @ _unit(axis))), 0.0, 1.0))))
    return Face(np.asarray(point, dtype=float), normal, lean)


def _gap_needed(axis, face_p: Face, face_q: Face, rim: np.ndarray, clearance: float) -> float:
    """The axial gap between two cuts' crossings of the line that keeps them
    ``clearance`` apart everywhere across the bone.

    ``face_p`` ends the distal piece (its normal points up the fibula), ``face_q``
    starts the proximal one (its normal points down). ``rim`` are offsets square
    to the axis covering the bone's section.
    """
    e = _unit(axis)
    tp = -(rim @ face_p.normal) / float(e @ face_p.normal)
    tq = -(rim @ face_q.normal) / float(e @ face_q.normal)
    return float(clearance + np.max(tp - tq))


def plan_fibula(
    course: Course,
    fibula,
    joints: list[int],
    deviation_mm: float,
    pedicle_end: str = "second",
    roll_deg: float = 0.0,
    start_mm: float | None = None,
    distal_keep_mm: float = DISTAL_KEEP_MM,
    proximal_keep_mm: float = PROXIMAL_KEEP_MM,
    clearance_mm: float = CUT_CLEARANCE_MM,
    rim_at=None,
    min_piece_mm: float = MIN_PIECE_MM,
) -> FibulaPlan:
    """Lay the chain's pieces out on ``fibula`` and carry them into the jaw.

    ``pedicle_end`` is the end of the defect the vessels leave from, "first" or
    "second" cut: the most proximal piece goes there. ``roll_deg`` turns the
    fibula about its line: at 0 its lateral side (``fibula.lateral_at``) faces
    out of the jaw. ``start_mm`` is where the graft's lowest cut reaches, mm up
    from the fibula's distal end; by default as low as ``distal_keep_mm`` allows.
    ``rim_at(s, frame)`` gives offsets square to the fibula's line covering its
    section near ``s``, mm; by default a circle a third wider than its section's
    equivalent radius there.
    """
    joint_points = course.points[joints]
    n = len(joints) - 1
    chords = [joint_points[k + 1] - joint_points[k] for k in range(n)]
    lengths = [float(np.linalg.norm(c)) for c in chords]
    directions = [_unit(c) for c in chords]
    if rim_at is None:
        rim_at = lambda s, frame: _circle_rim(fibula, s, frame)  # noqa: E731

    def jaw_face(k: int, at_start: bool) -> tuple[np.ndarray, np.ndarray]:
        """The plane ending course piece ``k`` at its start or its end in the jaw:
        a point, and the normal pointing out of the piece."""
        if at_start:
            if k == 0:
                # A cut's normal points into the defect, into the piece.
                return joint_points[0], -course.first_plane.normal
            return joint_points[k], -_unit(directions[k - 1] + directions[k])
        if k == n - 1:
            return joint_points[n], -course.second_plane.normal
        return joint_points[k + 1], _unit(directions[k] + directions[k + 1])

    outs, facing = _faces_out(course, joints, directions, roll_deg)

    def lay(k: int, s0: float) -> tuple[Piece, np.ndarray]:
        """Course piece ``k`` taken from the fibula from ``s0`` up."""
        s1 = _chord_end(fibula, s0, lengths[k])
        a, b = fibula.point_at(s0), fibula.point_at(s1)
        axis = _unit(b - a)
        leg = _frame(axis, fibula.lateral_at(0.5 * (s0 + s1)))
        # In the jaw the fibula runs distal to proximal towards the vessels' end,
        # its lateral side the way the transported frame says.
        towards = directions[k] if pedicle_end == "second" else -directions[k]
        rotation = _frame(towards, outs[k]) @ leg.T
        distal_in_jaw = joint_points[k] if pedicle_end == "second" else joint_points[k + 1]
        to_jaw = RigidTransform(rotation, distal_in_jaw - rotation @ a)
        back = to_jaw.inverse()
        faces = []
        for at_start in (True, False):
            point, normal = jaw_face(k, at_start)
            faces.append(_face(back.apply(point), back.rotation @ normal, axis))
        faces.sort(key=lambda f: float((f.point - a) @ axis))
        piece = Piece(
            number=k + 1,
            start=joint_points[k].copy(),
            end=joint_points[k + 1].copy(),
            s_distal=s0,
            s_proximal=s1,
            distal_face=faces[0],
            proximal_face=faces[1],
            to_jaw=to_jaw,
        )
        return piece, leg

    def layout(start: float) -> tuple[list[Piece], list[float], float, float]:
        """Every piece up the leg, the lowest line from ``start``, each just clear
        of the one below. Also the lowest and highest any cut reaches, mm."""
        order = list(range(n)) if pedicle_end == "second" else list(range(n))[::-1]
        laid: dict[int, tuple[Piece, np.ndarray]] = {}
        gaps: list[float] = []
        below = None
        s0 = start
        for k in order:
            piece, leg = lay(k, s0)
            if below is not None:
                previous, previous_leg = below
                for _ in range(2):  # the frame moves a little with the piece
                    rim = np.vstack([rim_at(previous.s_proximal, previous_leg), rim_at(piece.s_distal, leg)])
                    need = _gap_needed(leg[:, 0], previous.proximal_face, piece.distal_face, rim, clearance_mm)
                    s0 = previous.s_proximal + max(need, float(clearance_mm))
                    piece, leg = lay(k, s0)
                gaps.append(piece.s_distal - previous.s_proximal)
            laid[k] = (piece, leg)
            below = (piece, leg)
            s0 = piece.s_proximal
        lowest = min(
            p.s_distal + _face_reach(p.distal_face, leg[:, 0], rim_at(p.s_distal, leg), down=True)
            for p, leg in laid.values()
        )
        highest = max(
            p.s_proximal + _face_reach(p.proximal_face, leg[:, 0], rim_at(p.s_proximal, leg), down=False)
            for p, leg in laid.values()
        )
        return [laid[k][0] for k in range(n)], gaps, lowest, highest

    s_low = float(distal_keep_mm)
    s_high = float(fibula.length_mm) - float(proximal_keep_mm)
    wanted = s_low if start_mm is None else float(start_mm)
    start = wanted
    pieces, gaps, lowest, highest = layout(start)
    # Move the graft until its lowest cut, wherever it leans, is at the start.
    for _ in range(4):
        if abs(lowest - wanted) < 0.05:
            break
        start += wanted - lowest
        pieces, gaps, lowest, highest = layout(start)
    warnings: list[str] = []
    if lowest < s_low - 0.05:
        warnings.append(
            f"The graft reaches {s_low - lowest:.0f} mm into the {distal_keep_mm:.0f} mm "
            "of fibula to be left at the ankle."
        )
    if highest > s_high + 0.05:
        warnings.append(
            f"The graft needs {highest - lowest:.0f} mm of fibula and only "
            f"{s_high - s_low:.0f} mm can be taken between the {distal_keep_mm:.0f} mm "
            f"left at the ankle and the {proximal_keep_mm:.0f} mm left at the knee."
        )
    first_end = pieces[0].distal_face if pedicle_end == "second" else pieces[0].proximal_face
    last_end = pieces[-1].proximal_face if pedicle_end == "second" else pieces[-1].distal_face
    for name, face in (("first", first_end), ("second", last_end)):
        if face.lean_deg > MAX_END_LEAN_DEG:
            warnings.append(
                f"The graft's end at the {name} cut leans {face.lean_deg:.0f}° from square to the "
                "fibula: the jaw's cut there is far from square to the course. Turn the cut, or "
                "expect a long oblique end to saw and to seat."
            )
    short = [p.number for p in pieces if p.length_mm < min_piece_mm - 0.5]
    if short:
        warnings.append(
            "Piece " + ", ".join(str(i) for i in short) + f" is shorter than {min_piece_mm:.0f} mm."
        )
    angles = [
        float(np.degrees(np.arccos(np.clip(directions[k] @ directions[k + 1], -1.0, 1.0))))
        for k in range(n - 1)
    ]
    return FibulaPlan(
        pieces=pieces,
        joints=joint_points.copy(),
        joint_angles_deg=angles,
        deviation_mm=float(deviation_mm),
        gaps_mm=gaps,
        harvest_mm=(lowest, highest),
        above_graft_mm=max(s_high - highest, 0.0),
        pedicle_end=pedicle_end,
        facing_deg=facing,
        warnings=warnings,
    )


def _face_reach(face: Face, axis: np.ndarray, rim: np.ndarray, down: bool) -> float:
    """How far along the line, past its crossing of it, a cut reaches across the
    bone: the lowest (``down``) or the highest offset, mm. ``rim`` are offsets
    square to ``axis`` covering the bone's section."""
    along = float(_unit(axis) @ face.normal)
    if abs(along) < 1e-9:
        return 0.0
    t = -(rim @ face.normal) / along
    return float(t.min() if down else t.max())


def _circle_rim(fibula, s: float, frame: np.ndarray, share: float = 4.0 / 3.0) -> np.ndarray:
    """Offsets round the fibula's line at ``s``, a circle a third wider than its section."""
    radius = share * float(np.interp(s, fibula.s, fibula.radius_mm))
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    return radius * (np.cos(angles)[:, None] * frame[:, 1] + np.sin(angles)[:, None] * frame[:, 2])


def _chord_end(fibula, s0: float, length: float) -> float:
    """The position up the fibula whose point is ``length`` from the point at ``s0``."""
    s1 = s0 + length
    a = fibula.point_at(s0)
    for _ in range(4):
        d = float(np.linalg.norm(fibula.point_at(s1) - a))
        if abs(d - length) < 1e-4:
            break
        s1 += length - d
    return s1


def section_rim(points: np.ndarray, fibula, s: float, frame: np.ndarray, half_mm: float = 2.0) -> np.ndarray:
    """The fibula's own section near ``s``: offsets, square to its line, of its
    surface ``points`` within ``half_mm`` of the plane square to it there.

    Falls back to a circle a third wider than its equivalent radius where the
    surface has too few points there.
    """
    axis = frame[:, 0]
    rel = np.asarray(points, dtype=float) - fibula.point_at(s)
    along = rel @ axis
    near = np.abs(along) < half_mm
    if near.sum() < 12:
        return _circle_rim(fibula, s, frame)
    return rel[near] - along[near, None] * axis


def with_guide(
    rim: np.ndarray, out: np.ndarray, axis: np.ndarray, reach_mm: float, beyond_mm: float = 0.0
) -> np.ndarray:
    """``rim`` with the room a cutting guide takes round it, on its side facing ``out``.

    The guide's slots run on ``beyond_mm`` past the bone at each side, and its
    towers stand ``reach_mm`` out from the bone, down to the bone's middle: a box,
    in the plane square to ``axis``. Two cuts of a joint must stay apart through
    all of it as through the bone. How far apart two planes are changes evenly
    across a plane, so the box's corners and the rim, grown by ``beyond_mm``,
    are all that need be kept apart.
    """
    out = _unit(out)
    axis = _unit(axis)
    rim = np.asarray(rim, dtype=float)
    if beyond_mm > 0.0:
        size = np.linalg.norm(rim, axis=1, keepdims=True)
        rim = rim * (1.0 + float(beyond_mm) / np.maximum(size, 1e-9))
    across = _unit(np.cross(axis, out))
    ys = rim @ across
    top = float((rim @ out).max()) + float(reach_mm)
    corners = [height * out + side * across for height in (0.0, top) for side in (float(ys.min()), float(ys.max()))]
    return np.vstack([rim, np.array(corners)])


def vessels_in_jaw(plan: FibulaPlan, fibula, course: Course) -> tuple[float, float]:
    """Where the peroneal vessels end up round the fibula in the jaw.

    They run down the back of the fibula, on its inner side (posterior and
    medial in the leg). Returns how much that side faces up (+1) or down (-1)
    and out of the jaw (+1) or in (-1), in the piece the vessels leave from.
    """
    pedicle = max(plan.pieces, key=lambda p: p.s_proximal)
    s = 0.5 * (pedicle.s_distal + pedicle.s_proximal)
    axis = fibula.axis_at(s)
    posterior = np.array([0.0, 1.0, 0.0])  # patient LPS: +y is posterior
    posterior = _unit(posterior - float(posterior @ axis) * axis)
    medial = -fibula.lateral_at(s)
    back = _unit(posterior + medial)
    in_jaw = pedicle.to_jaw.rotation @ back
    middle = int(np.argmin(np.linalg.norm(course.points - 0.5 * (pedicle.start + pedicle.end), axis=1)))
    return float(in_jaw @ course.up[middle]), float(in_jaw @ course.outward[middle])
