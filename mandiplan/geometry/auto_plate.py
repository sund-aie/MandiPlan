"""Putting a plate on the jaw without being told where, and saying if it will do.

Given only the jaw, the arch curve, any planned resection and a plate, this
works out where that plate belongs, which holes to put screws in, and whether
the plate is applicable at all.

**The route.** A reconstruction plate runs along the outer face of the jaw just
above its lower border, clear of the tooth roots and the nerve above it, and up
the posterior border of the ramus. At each station of the arch curve the bone's
section is read for its lower border and its outer face at that height.

**The position.** The plate's holes are walked along the route at the plate's
own spacing, sliding it along the jaw and trying it each way round. A position
is scored on how evenly it holds the bone either side of the resection, on how
little bending in the plane of the plate it asks for (which is what finds the
angle of the jaw for a preformed angle plate, and the right side for a left or
right one), and on keeping every hole on the jaw.

**The screws.** Next to a cut, a screw needs bone round it: holes too close to
the cut, over the gap, off the bone or facing it badly get none. Of the rest,
the nearest few on each side of the resection are the ones recommended.

**The verdict.** Good, caution or not applicable, with the reasons in words.

Everything is in millimetres on patient coordinates. No Qt or VTK here: the
surface is given as its vertices, and the projector as an object.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .plate_bend import _gaussian, _walk_chain

#: The plate's lower edge stands this far above the jaw's lower border, mm.
LOWER_BORDER_MARGIN_MM = 2.0
#: Stations of the route along the curve, and half the slice each reads, mm.
ROUTE_STEP_MM = 1.0
SECTION_HALF_MM = 1.2
#: Within this of the curve in the section's plane, the bone is this part of the
#: jaw and not the far side of the arch, mm.
SECTION_REACH_MM = (28.0, 50.0)
#: Smoothing of the route, mm: a plate does not follow a bump this short.
ROUTE_SMOOTH_MM = 5.0
#: A screw needs this much bone between it and a cut, along the jaw, mm.
MIN_CUT_MARGIN_MM = 5.0
#: Screws recommended on each side of a resection, and the fewest acceptable.
SCREWS_PER_SIDE = 4
MIN_SCREWS_PER_SIDE = 3
#: With no resection, screws recommended at each end.
SCREWS_AT_END = 3
#: Beyond this from the bone a hole holds nothing, mm.
OFF_BONE_MM = 3.0
#: A gap this large under a hole is worth saying, mm.
GAP_CAUTION_MM = 1.5
#: A screw this far off the bone's normal is poor, degrees.
SCREW_ANGLE_MAX_DEG = 35.0
#: A hole with less bone than this under it takes no screw, mm.
MIN_BONE_DEPTH_MM = 4.0
#: The in-plane bending a position asks for is forgiven up to this, degrees.
TURN_FORGIVEN_DEG = 4.0
#: How strongly a position is scored against sliding the plate off centre, per
#: mm squared, against one degree squared of in-plane bending per hole.
OFF_CENTRE_WEIGHT = 0.02
#: In-plane bending above this per hole on a preformed plate means it was made
#: for somewhere else, degrees.
PREFORM_MISMATCH_DEG = 25.0
#: The bone must be at least this share of the plate's width where it lies.
MIN_BONE_HEIGHT_SHARE = 0.9
#: With no resection, the length recommended is at least this, mm.
DEFAULT_LENGTH_MM = 60.0

GOOD, CAUTION, BAD = "good", "caution", "bad"
SCREW, SPARE, SKIP, NO_BONE = "screw", "spare", "skip", "no bone"


@dataclass
class JawRoute:
    """Where a plate of a given width lies along the jaw."""

    s: np.ndarray  # arc positions on the arch curve, mm
    points: np.ndarray  # on the outer face of the bone
    outward: np.ndarray  # away from the bone
    bone_height_mm: np.ndarray  # lower border to top of the section there
    valid: np.ndarray  # whether the section at this station was read


@dataclass
class Placement:
    """Where along the route a plate's holes land."""

    positions: np.ndarray  # route sample position of each hole
    arcs_mm: np.ndarray  # arc position on the curve of each hole
    points: np.ndarray  # route point at each hole
    outward: np.ndarray
    reversed: bool
    holes_before: int  # sound holes before the resection
    holes_after: int
    turn_needed_deg: float  # worst in-plane bend this position asks for
    off_route: int  # holes past the ends of the route
    score: float


@dataclass
class HoleAdvice:
    index: int  # 0-based, in the plate's own hole order
    state: str  # SCREW, SPARE, SKIP or NO_BONE
    why: str = ""
    bone_depth_mm: float = float("nan")  # along the screw's line, first bone to last
    gap_mm: float = float("nan")


@dataclass
class PlateVerdict:
    level: str
    headline: str
    reasons: list[str] = field(default_factory=list)
    holes: list[HoleAdvice] = field(default_factory=list)
    #: Suggested other length, in holes, when that would answer it.
    suggested_holes: int | None = None


def jaw_route(vertices: np.ndarray, frames, plate_width_mm: float) -> JawRoute | None:
    """The line along the jaw a plate of this width lies on; None if there is none."""
    vertices = np.asarray(vertices, dtype=float).reshape(-1, 3)
    if len(vertices) < 100:
        return None
    stations = np.arange(float(frames.s[0]), float(frames.s[-1]) + 1e-9, ROUTE_STEP_MM)
    centre = frames.points.mean(axis=0)
    points = np.zeros((len(stations), 3))
    outward = np.zeros((len(stations), 3))
    height = np.zeros(len(stations))
    valid = np.zeros(len(stations), dtype=bool)
    previous = None
    for i, s in enumerate(stations):
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
        u = rel @ up
        b = rel @ side
        near = (np.abs(b) < SECTION_REACH_MM[0]) & (np.abs(u) < SECTION_REACH_MM[1])
        if near.sum() < 12:
            continue
        u, b = u[near], b[near]
        low, high = float(np.percentile(u, 1.0)), float(np.percentile(u, 99.0))
        level = low + LOWER_BORDER_MARGIN_MM + 0.5 * float(plate_width_mm)
        level = min(level, 0.5 * (low + high) if high - low < float(plate_width_mm) else level)
        band = np.abs(u - level) < 1.5
        if band.sum() < 3:
            continue
        face = float(b[band].max())
        points[i] = at + level * up + face * side
        outward[i] = side
        height[i] = high - low
        valid[i] = True
    if valid.sum() < 10:
        return None
    # Where a section could not be read the neighbours carry the line across.
    known = np.flatnonzero(valid)
    keep = np.arange(known[0], known[-1] + 1)
    for c in range(3):
        points[:, c] = np.interp(np.arange(len(stations)), known, points[known, c])
        outward[:, c] = np.interp(np.arange(len(stations)), known, outward[known, c])
    height = np.interp(np.arange(len(stations)), known, height[known])
    sigma = ROUTE_SMOOTH_MM / ROUTE_STEP_MM
    smooth = _gaussian(points[keep], sigma)
    faces = _gaussian(outward[keep], sigma)
    faces = faces / np.maximum(np.linalg.norm(faces, axis=1, keepdims=True), 1e-9)
    return JawRoute(stations[keep], smooth, faces, height[keep], valid[keep])


def plate_turns(centres: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """A plate's own bend in its plane at each inner hole, degrees, signed."""
    centres = np.asarray(centres, dtype=float).reshape(-1, 3)
    axes = np.asarray(axes, dtype=float).reshape(-1, 3)
    return _signed_turns(centres, axes)


def _signed_turns(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return np.zeros(0)
    chords = np.diff(points, axis=0)
    chords = chords / np.maximum(np.linalg.norm(chords, axis=1, keepdims=True), 1e-9)
    a, b = chords[:-1], chords[1:]
    n = normals[1:-1]
    return np.degrees(np.arctan2(np.einsum("ij,ij->i", np.cross(a, b), n), np.einsum("ij,ij->i", a, b)))


def place_chain(
    route: JawRoute,
    spacings: np.ndarray,
    own_turns: np.ndarray,
    defect: tuple[float, float] | None = None,
    min_holes: int = MIN_SCREWS_PER_SIDE,
) -> Placement | None:
    """The best place on ``route`` for a plate with these holes.

    ``defect`` is the arc span of the resection on the same curve as
    ``route.s``. A plate's hole order is kept: walking backwards along the route
    is the plate turned end for end, with its face still outward.
    """
    spacings = np.asarray(spacings, dtype=float).ravel()
    length = float(spacings.sum())
    samples = len(route.s)
    middle = 0.5 * (samples - 1)
    if defect is not None:
        wanted = np.interp(0.5 * (defect[0] + defect[1]), route.s, np.arange(samples))
    else:
        wanted = middle
    best = None
    for reverse in (False, True):
        points = route.points[::-1] if reverse else route.points
        faces = route.outward[::-1] if reverse else route.outward
        arcs = route.s[::-1] if reverse else route.s
        centre_wanted = samples - 1 - wanted if reverse else wanted
        # Slide through everywhere the plate lies wholly on the route, or, when
        # it is longer than the jaw, centred.
        room = samples - 1 - length / ROUTE_STEP_MM
        starts = np.arange(0.0, room + 1e-9, 2.0) if room > 0 else np.array([0.5 * room])
        for start in starts:
            positions = _walk_chain(points, max(float(start), 0.0), spacings, ROUTE_STEP_MM)
            if room <= 0:
                positions = positions + min(float(start), 0.0)
            index = np.arange(samples)
            hole_points = np.column_stack([np.interp(positions, index, points[:, c]) for c in range(3)])
            hole_faces = np.column_stack([np.interp(positions, index, faces[:, c]) for c in range(3)])
            hole_faces = hole_faces / np.maximum(np.linalg.norm(hole_faces, axis=1, keepdims=True), 1e-9)
            hole_arcs = np.interp(positions, index, arcs)
            off = int(np.count_nonzero((positions < 0.0) | (positions > samples - 1)))
            turns = _signed_turns(hole_points, hole_faces)
            excess = np.abs(turns - own_turns[: len(turns)]) if len(turns) == len(own_turns) else np.abs(turns)
            worst = float(excess.max()) if len(excess) else 0.0
            cost = float((np.maximum(excess - TURN_FORGIVEN_DEG, 0.0) ** 2).sum())
            cost += 500.0 * off
            before, after = 0, 0
            if defect is not None:
                before = int(np.count_nonzero(hole_arcs < defect[0] - MIN_CUT_MARGIN_MM))
                after = int(np.count_nonzero(hole_arcs > defect[1] + MIN_CUT_MARGIN_MM))
                short = max(min_holes - before, 0) + max(min_holes - after, 0)
                cost += 2000.0 * short + 8.0 * float(before - after) ** 2
            centre = 0.5 * (positions[0] + positions[-1])
            cost += OFF_CENTRE_WEIGHT * float(centre - centre_wanted) ** 2
            if best is None or cost < best.score:
                best = Placement(
                    positions=positions,
                    arcs_mm=hole_arcs,
                    points=hole_points,
                    outward=hole_faces,
                    reversed=reverse,
                    holes_before=before,
                    holes_after=after,
                    turn_needed_deg=worst,
                    off_route=off,
                    score=cost,
                )
    return best


def recommend_length(route, plates, defect, min_holes: int = MIN_SCREWS_PER_SIDE):
    """The plate to use of ``plates``: ``(spacings, own_turns, key)`` triples, shortest first.

    The shortest with ``SCREWS_PER_SIDE`` sound holes on each side of a
    resection, else the shortest with the minimum, else the longest. With no
    resection, the shortest of at least ``DEFAULT_LENGTH_MM``. Returns its key.
    """
    if not plates:
        return None
    if defect is None:
        for spacings, _, key in plates:
            if float(np.sum(spacings)) >= DEFAULT_LENGTH_MM:
                return key
        return plates[-1][2]
    fallback = None
    for spacings, own, key in plates:
        placed = place_chain(route, spacings, own, defect, min_holes)
        if placed is None or placed.off_route:
            continue
        sides = min(placed.holes_before, placed.holes_after)
        if sides >= SCREWS_PER_SIDE:
            return key
        if sides >= min_holes and fallback is None:
            fallback = key
    return fallback if fallback is not None else plates[-1][2]


def advise_holes(
    arcs_mm,
    gaps_mm,
    screw_angles_deg,
    depths_mm,
    defect: tuple[float, float] | None,
) -> list[HoleAdvice]:
    """What each hole of a seated plate is for."""
    arcs = np.asarray(arcs_mm, dtype=float)
    holes: list[HoleAdvice] = []
    for i, arc in enumerate(arcs):
        gap = float(gaps_mm[i])
        angle = float(screw_angles_deg[i])
        depth = float(depths_mm[i])
        state, why = SPARE, ""
        if defect is not None and defect[0] <= arc <= defect[1]:
            state, why = NO_BONE, "over the resection"
            depth = float("nan")
        elif gap > OFF_BONE_MM or not np.isfinite(gap):
            state, why = NO_BONE, f"stands {gap:.1f} mm off the bone" if np.isfinite(gap) else "no bone under it"
            depth = float("nan")
        elif defect is not None and min(abs(arc - defect[0]), abs(arc - defect[1])) < MIN_CUT_MARGIN_MM:
            state, why = SKIP, f"under {MIN_CUT_MARGIN_MM:.0f} mm from the cut"
        elif not depth >= MIN_BONE_DEPTH_MM:
            state, why = SKIP, "too little bone under it" if np.isfinite(depth) else "no bone under it"
        elif angle > SCREW_ANGLE_MAX_DEG:
            state, why = SKIP, f"screw would run {angle:.0f}° off the bone"
        holes.append(HoleAdvice(i, state, why, depth, gap))

    def choose(indices, count):
        chosen = 0
        for i in indices:
            if holes[i].state == SPARE:
                holes[i].state = SCREW
                chosen += 1
                if chosen == count:
                    break

    n = len(holes)
    order = np.argsort(arcs)
    if defect is None:
        choose(order, SCREWS_AT_END)
        choose(order[::-1], SCREWS_AT_END)
    else:
        before = [int(i) for i in order if arcs[i] < defect[0]]
        after = [int(i) for i in order if arcs[i] > defect[1]]
        # Outward from each cut.
        choose(before[::-1], SCREWS_PER_SIDE)
        choose(after, SCREWS_PER_SIDE)
    assert len(holes) == n
    return holes


def _plural(count: int, word: str = "hole") -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def judge_plate(
    holes: list[HoleAdvice],
    arcs_mm,
    defect: tuple[float, float] | None,
    jaw_arc: tuple[float, float] | None,
    placement: Placement | None = None,
    route: JawRoute | None = None,
    plate_width_mm: float = 0.0,
    bend_problems=(),
    other_cautions=(),
    is_preformed: bool = False,
    hole_count: int = 0,
    longer_holes: int | None = None,
) -> PlateVerdict:
    """Whether this plate will do here, in plain words."""
    arcs = np.asarray(arcs_mm, dtype=float)
    fatal: list[str] = []
    caution: list[str] = []
    suggested = None
    screws = [h for h in holes if h.state == SCREW]
    if defect is not None:
        order = {h.index: arcs[h.index] for h in holes}
        before = [h for h in screws if order[h.index] < defect[0]]
        after = [h for h in screws if order[h.index] > defect[1]]
        if jaw_arc is not None and (defect[0] <= jaw_arc[0] + 1.0 or defect[1] >= jaw_arc[1] - 1.0):
            fatal.append(
                "The resection runs through to the condyle: there is no bone beyond it to "
                "fix this plate to. It needs a second cut, or a plate with a condylar head."
            )
        fewest = min(len(before), len(after))
        if not fatal and fewest < MIN_SCREWS_PER_SIDE:
            held = "No screw" if fewest == 0 else f"Only {_plural(fewest, 'screw')}"
            fatal.append(
                f"{held} can hold on the bone on one side of the resection; "
                f"at least {MIN_SCREWS_PER_SIDE} are needed on each side."
            )
            suggested = longer_holes
        elif not fatal and fewest < SCREWS_PER_SIDE:
            caution.append(
                f"{_plural(fewest, 'screw')} on the bone on one side of the resection; "
                f"{SCREWS_PER_SIDE} are better."
            )
            suggested = longer_holes
    elif len(screws) < 2 * MIN_SCREWS_PER_SIDE:
        fatal.append(
            f"Only {_plural(len(screws), 'screw')} can hold on the bone along this plate's route."
        )

    fatal.extend(bend_problems)

    if placement is not None:
        if placement.off_route:
            fatal.append(
                f"This plate is longer than the jaw: {_plural(placement.off_route)} would run past its ends."
            )
        elif is_preformed and placement.turn_needed_deg > PREFORM_MISMATCH_DEG:
            fatal.append(
                f"This plate's preformed shape does not suit this site: it would need "
                f"{placement.turn_needed_deg:.0f}° of bending in its own plane. Try the other side's "
                "plate, or a straight one."
            )
        if route is not None and plate_width_mm > 0.0:
            heights = np.interp(placement.arcs_mm, route.s, route.bone_height_mm)
            sound = [
                heights[h.index]
                for h in holes
                if h.state in (SCREW, SPARE) and h.index < len(heights)
            ]
            if sound and min(sound) < MIN_BONE_HEIGHT_SHARE * plate_width_mm:
                fatal.append(
                    f"The jaw is only {min(sound):.0f} mm tall where this {plate_width_mm:.0f} mm "
                    "wide plate lies; the plate would hang over it."
                )

    off_bone = [h for h in holes if h.state == NO_BONE and "off the bone" in h.why]
    if off_bone:
        caution.append(
            f"{_plural(len(off_bone))} stand clear of the bone: cut the plate shorter if they are at its end."
        )
    gaps = [h for h in holes if h.state in (SCREW, SPARE) and h.gap_mm > GAP_CAUTION_MM]
    if gaps:
        caution.append(
            f"{_plural(len(gaps))} sit over {GAP_CAUTION_MM:.1f} mm off the bone, "
            f"worst {max(h.gap_mm for h in gaps):.1f} mm."
        )
    caution.extend(other_cautions)

    if fatal:
        headline = "Not applicable"
        level = BAD
    elif caution:
        headline = "Usable, with care"
        level = CAUTION
    else:
        headline = "Applicable"
        level = GOOD
    return PlateVerdict(level, headline, fatal + caution, holes, suggested)
