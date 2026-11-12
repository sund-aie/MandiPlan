"""Finding the fibula in a CT of the lower legs.

A leg CT is taken along the legs, so each axial slice cuts both bones of the leg
across: the tibia, the large one, and the fibula, the small one beside it,
lateral and a little posterior. Slice by slice the bone above the threshold is
split into its pieces; in each leg the largest piece is the tibia and the piece
of fibula's size near it, on its lateral side, is the fibula. Followed from slice
to slice they give the fibula's run from its lowest slice to its highest, the
centres of its sections its line. The side the fibula lies on, away from the
tibia, says which leg it is.

Its lateral side is the patient's: towards patient right in a right leg, as
BoneReconstructionPlanner takes it too. The fibula lies behind the tibia as well
as beside it, so the way from the tibia to it is no lateral; and a leg turned
out in the scanner turns both, which the roll of the plan is there to correct.

Where the fibula meets other bone in a slice, at the head against the tibia or
the malleolus against the talus, it is lost there and the run ends a little
short of its true tip. That only ever moves the ends inwards, so harvest margins
measured from them are, if anything, longer than they need be.

Patient coordinates (LPS, mm): +z is towards the head, so the proximal end of a
fibula is its higher one; patient left is +x. No Qt or VTK here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: The bone threshold of a CT, Hounsfield units: cortex is far above it, the
#: muscle and fat round it below. User-adjustable, as every threshold here is.
DEFAULT_THRESHOLD_HU = 200.0
#: A fibula's section is this big, mm2: more than a vessel's calcification, less than
#: a tibia's. Its head and malleolus are wider than its shaft.
FIBULA_AREA_MM2 = (25.0, 900.0)
#: The tibia's section is at least this big along most of the leg, mm2.
TIBIA_MIN_AREA_MM2 = 150.0
#: The fibula lies beside the tibia, its centre this far from the tibia's, mm.
TIBIA_FIBULA_MM = (8.0, 60.0)
#: From one slice to the next a section's centre moves no more than this, mm.
TRACK_STEP_MM = 6.0
#: Slices without a fibula this long still belong to one run, mm.
GAP_MM = 8.0
#: The shortest run that is a fibula, mm.
MIN_LENGTH_MM = 120.0
#: Two legs are this far apart, centre to centre, mm.
LEG_SPLIT_MM = 70.0
#: The fibula's line is smoothed over this much, mm.
LINE_SMOOTH_MM = 6.0


@dataclass
class Fibula:
    """One leg's fibula, as found in its CT."""

    side: str  # "left" or "right": which leg
    points: np.ndarray  # (N, 3) its line, from the distal end to the proximal, mm
    s: np.ndarray  # distance along the line from the distal end, mm
    radius_mm: np.ndarray  # equivalent radius of each section
    lateral: np.ndarray  # the patient's lateral for this leg (LPS); see lateral_at
    #: The way from the tibia to the fibula, across the leg: lateral and back.
    away_from_tibia: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: The tibia beside it, slice by slice: its centres and equivalent radii.
    tibia_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    tibia_radius_mm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    notes: list[str] = field(default_factory=list)

    @property
    def length_mm(self) -> float:
        return float(self.s[-1])

    @property
    def distal_end(self) -> np.ndarray:
        return self.points[0].copy()

    @property
    def proximal_end(self) -> np.ndarray:
        return self.points[-1].copy()

    def point_at(self, s: float) -> np.ndarray:
        """The point of the line ``s`` mm from the distal end, carried on straight past either end."""
        s = float(s)
        if s <= 0.0:
            return self.points[0] + s * self.axis_at(0.0)
        if s >= self.s[-1]:
            return self.points[-1] + (s - float(self.s[-1])) * self.axis_at(float(self.s[-1]))
        return np.array([np.interp(s, self.s, self.points[:, c]) for c in range(3)])

    def axis_at(self, s: float, reach_mm: float = 10.0) -> np.ndarray:
        """The line's direction at ``s``, distal to proximal, over ``reach_mm`` either side."""
        a = float(np.clip(s - reach_mm, self.s[0], self.s[-1]))
        b = float(np.clip(s + reach_mm, self.s[0], self.s[-1]))
        d = self._at(b) - self._at(a)
        return d / max(float(np.linalg.norm(d)), 1e-9)

    def _at(self, s: float) -> np.ndarray:
        return np.array([np.interp(s, self.s, self.points[:, c]) for c in range(3)])

    def lateral_at(self, s: float) -> np.ndarray:
        """The leg's lateral side at ``s``, square to the line there."""
        axis = self.axis_at(s)
        out = self.lateral - (self.lateral @ axis) * axis
        return out / max(float(np.linalg.norm(out)), 1e-9)

    def anterior_at(self, s: float) -> np.ndarray:
        """The leg's front at ``s``, square to the line there (patient anterior is -y)."""
        axis = self.axis_at(s)
        front = np.array([0.0, -1.0, 0.0])
        front = front - (front @ axis) * axis
        return front / max(float(np.linalg.norm(front)), 1e-9)

    def side_at(self, s: float, face: str) -> np.ndarray:
        """The way out of the bone on its ``face``: "lateral" or "anterior"."""
        return self.anterior_at(s) if face == "anterior" else self.lateral_at(s)


def _pieces(mask_slice: np.ndarray):
    """The connected pieces of one slice: ``(area_px, centroid_rc)`` for each."""
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(mask_slice.astype(np.uint8))
    labels = sitk.ConnectedComponent(image)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(labels)
    out = []
    for label in stats.GetLabels():
        cx, cy = stats.GetCentroid(label)  # x is column, y is row, in pixels
        out.append((stats.GetNumberOfPixels(label), (cy, cx)))
    return out


def find_fibulae(volume, threshold: float = DEFAULT_THRESHOLD_HU) -> list[Fibula]:
    """Every fibula in ``volume``, one per leg in it, those found."""
    array = volume.array
    sx, sy, sz = (float(v) for v in volume.spacing)
    ox, oy, oz = (float(v) for v in volume.origin)
    pixel = sx * sy
    mask = array > float(threshold)
    # Each slice's pieces of bone, in mm.
    slices = []
    for k in range(array.shape[0]):
        pieces = []
        if mask[k].any():
            for count, (row, col) in _pieces(mask[k]):
                pieces.append((count * pixel, np.array([ox + col * sx, oy + row * sy])))
        slices.append((oz + k * sz, pieces))
    # Where the legs are: the tibiae, the large pieces, gathered by x.
    big = [(c[0], c[1]) for _, pieces in slices for area, c in pieces if area >= TIBIA_MIN_AREA_MM2]
    if not big:
        return []
    xs = np.array([b[0] for b in big])
    legs = _legs(xs)
    found = []
    for centre_x in legs:
        fibula = _track(slices, centre_x, legs, sz)
        if fibula is not None:
            found.append(fibula)
    return found


def _legs(xs: np.ndarray) -> list[float]:
    """The x of each leg's tibia: one leg, or two well apart."""
    lo, hi = float(np.percentile(xs, 5)), float(np.percentile(xs, 95))
    if hi - lo < LEG_SPLIT_MM:
        return [float(np.median(xs))]
    middle = 0.5 * (lo + hi)
    return [float(np.median(xs[xs < middle])), float(np.median(xs[xs >= middle]))]


def _track(slices, centre_x: float, legs: list[float], sz: float) -> Fibula | None:
    """The fibula of the leg whose tibia is at ``centre_x``, slice by slice."""
    others = [x for x in legs if x != centre_x]
    rows = []  # (z, tibia centre, fibula candidates, tibia area)
    for z, pieces in slices:
        mine = [p for p in pieces if not others or abs(p[1][0] - centre_x) < abs(p[1][0] - others[0])]
        if not mine:
            continue
        area, tibia = max(mine, key=lambda p: p[0])
        if area < TIBIA_MIN_AREA_MM2:
            continue
        candidates = [
            p for p in mine
            if p[1] is not tibia and FIBULA_AREA_MM2[0] <= p[0] <= FIBULA_AREA_MM2[1]
            and TIBIA_FIBULA_MM[0] <= np.linalg.norm(p[1] - tibia) <= TIBIA_FIBULA_MM[1]
            and p[0] < area
        ]
        rows.append((z, tibia, candidates or None, area))
    if not rows:
        return None
    # The run of slices with a fibula in them, following its centre from slice to slice.
    lateral_guess = np.median(
        np.array([min(c, key=lambda p: -p[0])[1] - t for _, t, c, _ in rows if c is not None]), axis=0
    ) if any(c is not None for _, _, c, _ in rows) else None
    if lateral_guess is None:
        return None
    runs, current, last = [], [], None
    for z, tibia, candidates, tibia_area in rows:
        if candidates is None:
            if current and z - current[-1][0] > GAP_MM:
                runs.append(current)
                current, last = [], None
            continue
        if last is None:
            pick = min(candidates, key=lambda p: np.linalg.norm((p[1] - tibia) - lateral_guess))
        else:
            pick = min(candidates, key=lambda p: np.linalg.norm(p[1] - last))
            if np.linalg.norm(pick[1] - last) > TRACK_STEP_MM + 2.0 * abs(sz):
                runs.append(current)
                current = []
                pick = min(candidates, key=lambda p: np.linalg.norm((p[1] - tibia) - lateral_guess))
        current.append((z, tibia, pick[1], pick[0], tibia_area))
        last = pick[1]
    if current:
        runs.append(current)
    run = max(runs, key=lambda r: r[-1][0] - r[0][0]) if runs else []
    if not run or run[-1][0] - run[0][0] < MIN_LENGTH_MM:
        return None
    z = np.array([r[0] for r in run])
    centres = np.array([[r[2][0], r[2][1], r[0]] for r in run])
    tibiae = np.array([[r[1][0], r[1][1], r[0]] for r in run])
    radius = np.sqrt(np.array([r[3] for r in run]) / np.pi)
    tibia_radius = np.sqrt(np.array([r[4] for r in run]) / np.pi)
    # Smooth the line along the leg; its order from distal (low z) to proximal.
    order = np.argsort(z)
    centres, tibiae, radius, tibia_radius = centres[order], tibiae[order], radius[order], tibia_radius[order]
    sigma = max(LINE_SMOOTH_MM / max(abs(sz), 1e-6), 0.5)
    centres = _smooth(centres, sigma)
    steps = np.linalg.norm(np.diff(centres, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(steps)])
    away = centres - tibiae
    away[:, 2] = 0.0
    away = np.mean(away / np.maximum(np.linalg.norm(away, axis=1, keepdims=True), 1e-9), axis=0)
    away = away / max(float(np.linalg.norm(away)), 1e-9)
    # Patient left is +x: a fibula lateral to its tibia towards +x is the left leg's.
    side = "left" if away[0] > 0.0 else "right"
    lateral = np.array([1.0, 0.0, 0.0]) if side == "left" else np.array([-1.0, 0.0, 0.0])
    return Fibula(
        side=side, points=centres, s=s, radius_mm=radius, lateral=lateral,
        away_from_tibia=away, tibia_points=tibiae, tibia_radius_mm=tibia_radius,
    )


def _smooth(points: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(np.ceil(3.0 * sigma))
    if radius < 1 or len(points) < 3:
        return points.copy()
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    padded = np.concatenate([np.repeat(points[:1], radius, axis=0), points, np.repeat(points[-1:], radius, axis=0)])
    out = np.zeros_like(points, dtype=float)
    for w, o in zip(weights, offsets):
        out += w * padded[radius + o : radius + o + len(points)]
    return out / weights.sum()


#: The tube round the fibula's line: this many times its widest section, plus a margin, mm.
TUBE_SHARE = 1.6
TUBE_MARGIN_MM = 3.0
#: Past its ends the fibula's tube runs on this far, mm.
END_MARGIN_MM = 4.0


def fibula_volume(volume, fibula, background: float | None = None):
    """``volume`` with everything but the fibula blanked out.

    The bone is kept only within a tube round the fibula's line, between its two
    ends, and on the fibula's side of the tibia: nearer the fibula's rim than the
    tibia's, each bone's rim taken as a circle of its own section's size. A split
    half-way between their centres would run through the tibia, which is twice
    the fibula's width. Where the two touch, at the head, it ends in a facet.
    """
    array = volume.array
    sx, sy, sz = (float(v) for v in volume.spacing)
    ox, oy, oz = (float(v) for v in volume.origin)
    nz, ny, nx = array.shape
    fill = float(array.min()) if background is None else float(background)
    out = np.full_like(array, fill)
    xs = ox + np.arange(nx) * sx
    ys = oy + np.arange(ny) * sy
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    reach = TUBE_SHARE * float(np.max(fibula.radius_mm)) + TUBE_MARGIN_MM
    zs = fibula.points[:, 2]
    low, high = float(zs.min()) - END_MARGIN_MM, float(zs.max()) + END_MARGIN_MM
    tibia = fibula.tibia_points
    tibia_radius = getattr(fibula, "tibia_radius_mm", np.zeros(0))
    for k in range(nz):
        z = oz + k * sz
        if z < low or z > high:
            continue
        cx = np.interp(z, zs, fibula.points[:, 0])
        cy = np.interp(z, zs, fibula.points[:, 1])
        near = np.hypot(xx - cx, yy - cy)
        keep = near < reach
        if len(tibia):
            tx = np.interp(z, tibia[:, 2], tibia[:, 0])
            ty = np.interp(z, tibia[:, 2], tibia[:, 1])
            rf = float(np.interp(z, zs, fibula.radius_mm))
            rt = float(np.interp(z, tibia[:, 2], tibia_radius)) if len(tibia_radius) else rf
            keep &= near - rf < np.hypot(xx - tx, yy - ty) - rt
        out[k][keep] = array[k][keep]
    from .volume import Volume

    return Volume(array=out, spacing=volume.spacing.copy(), origin=volume.origin.copy(),
                  resampled_from_mm=volume.resampled_from_mm)
