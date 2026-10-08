"""Separate the mandible from the rest of the skull.

A real CBCT thresholded at bone is one connected mass more often than not:
with the teeth in occlusion the lower crowns touch the upper ones, and at
the voxel sizes of a planning scan the coronoid process or the condyle can
touch the skull as well. Everything downstream of that is wrong: a cutting
plane is infinite, so a resection would take a slice of maxilla and skull
with it; the mirror would copy the maxilla into the defect; the exported
"jaw" would be the whole head.

The separation follows an arch curve through the mandibular body. It does
not wait for the operator to draw one: :func:`find_arch` finds the mandible
in the scan by itself — the lowest wide U of bone, open towards the back,
that holds its shape for a centimetre upwards, which is the mandibular body
and not the smaller, thinner hyoid below it — and lays the curve through it.
An arch the operator draws can be used instead. Then three steps, each only
when the one before has not finished the job:

1. **Already free?** The bone connected to the arch curve is checked for
   anything that can only be skull: bone far above the bite, or palate
   inside the arch. A mandible scanned with the mouth open, or a phantom,
   stops here unchanged.
2. **The bite.** The occlusal gap is found along the arch as the darkest
   smooth line with bright teeth both above and below it — dynamic
   programming over a panoramic band, the standard way upper and lower
   teeth are separated on panoramic images. Bone within 0.75 mm of that
   line, and only in the dental band, is cut. Stations with no teeth (no
   bright crowns on both sides) are not cut.
3. **The joints.** If the mandible is still attached — a condyle touching
   its fossa, a coronoid touching the zygoma — a marker-controlled
   watershed on the distance map splits it at the narrowest connection.
   Everything below the bite, rami included, seeds the mandible; bone
   well above the condyles and the upper dental arch seed the skull.

Only the voxel labels are decided here. Gray values are never changed, so
the bone surface of the mandible is the same isosurface it always was.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .plate_profile import point_in_polygon

#: Half-width of the dental band across the arch, mm.
BAND_MM = 10.0
#: How far the bite is searched above the arch curve, mm.
BITE_SEARCH_MM = (-5.0, 50.0)
#: Half-thickness of the cut made along the bite, mm.
BITE_CUT_MM = 0.75
#: A station has teeth when both crowns reach this fraction of the band's
#: brightest gray values, and the gap between them is at least this deep.
CROWN_LEVEL = 0.55
MIN_GAP_DEPTH = 0.06
#: Gray values are scaled to the crowns (1.0); metal is clipped at this.
METAL_CLIP = 1.25
#: The second bite search keeps within this of the occlusal plane fitted to
#: the first, mm: the curve of Spee and an overbite stray about this far.
BITE_PLANE_BAND_MM = 3.5
#: The plane is fitted only when at least this many stations show teeth.
MIN_PLANE_STATIONS = 10
#: The rami continue behind the last station of the arch curve by about
#: this much, mm.
RAMUS_EXTENSION_MM = 35.0
#: Bone this far above the bite plane is skull, never mandible, mm. The
#: condyles sit some 25-40 mm above the occlusal plane; measuring from the
#: plane, not from world height, keeps a head tipped chin-down from putting
#: its condyles above the line.
SKULL_ABOVE_BITE_MM = 55.0
#: Each ramus seeds the mandible up to this far above the bite plane, mm,
#: and its processes are followed up from there (see ``_processes``). Low
#: enough to stay below the condyles of a jaw without teeth, whose bite
#: plane is only an estimate and whose condyles may stand 20 mm above it.
RAMUS_SEED_MM = 16.0
#: ... where the ramus is measured, above the bite and below the notch, mm.
RAMUS_SAMPLE_MM = (2.0, 16.0)
#: Half-thickness of the ramus plate seeded, and of the corridor above it
#: kept free of skull markers, mm. The corridor is wider: a condyle is some
#: 20 mm across from pole to pole.
RAMUS_SEED_HALF_MM = 5.0
RAMUS_CORRIDOR_HALF_MM = 13.0
#: Bone this close to the brain is the braincase, mm: the floor of the
#: cranial fossae, which over each joint is all that separates the condyle's
#: socket from the brain.
BRAINCASE_LINING_MM = 2.5
#: A slice's enclosed soft tissue is brain when there is this much of it, cm².
MIN_BRAIN_SECTION_CM2 = 15.0
#: The corridor is kept free of skull markers up to this far above the bite
#: plane, mm. A mouth held open, or a jaw without teeth, puts the condyles
#: higher above the bite than ``SKULL_ABOVE_BITE_MM``.
CONDYLE_CLEAR_MM = 80.0
#: Along the tooth row, bone more than this above the bite is upper teeth and
#: maxilla, and bone more than this below it lower teeth and mandible, mm.
#: The few millimetres between are left to the watershed, so a bite height a
#: little off, as it is where metal hides the gap, costs nothing.
BITE_MARGIN_MM = 2.0
#: ... and the upper row's markers reach this much higher, mm: crowns and
#: the alveolar bone round their roots.
UPPER_ROW_MM = 20.0
#: At the front the upper incisors overlap the lower ones by a few
#: millimetres (the overbite), so mandible seeds start this far below the
#: bite there, mm, within this of the chin along the arch, mm.
OVERBITE_MM = 4.5
INCISOR_SPAN_MM = 22.0
#: The arch curve is found in the body, where the last molar can stand a
#: little past its end; the tooth row reaches at most this far past it, mm.
TOOTH_ROW_BEYOND_MM = 8.0
#: The upper teeth stand up to this far from the arch curve in plan, mm: the
#: curve runs through the middle of the mandibular body; the upper molars
#: stand over the lower ones, but the upper incisors stand in front of the
#: lower ones, tipped forward. Beside the molars, further out than this,
#: rise the coronoid and the front of the ramus.
UPPER_REACH_MM = (8.0, 13.0)


@dataclass
class Bite:
    """The occlusal line along the arch curve."""

    station_points: np.ndarray  # (N, 3) arch curve points the bite was found at
    heights_mm: np.ndarray  # (N,) world z of the gap at each station
    has_teeth: np.ndarray  # (N,) bool, teeth on both sides of the gap
    depth: np.ndarray  # (N,) gap depth, fraction of the band's brightest values

    @property
    def found(self) -> bool:
        return bool(self.has_teeth.any())


@dataclass
class MandibleIsolation:
    """Which bone voxels are mandible, and how they were separated."""

    mask: np.ndarray  # bool, volume shape: mandible voxels
    other_bone: np.ndarray  # bool, volume shape: bone that is not mandible
    bite: Bite | None
    steps: list[str] = field(default_factory=list)
    volume_mm3: float = 0.0
    #: How high each ramus reaches above the bite plane, mm: (right, left).
    ramus_heights_mm: tuple[float, float] = (float("nan"), float("nan"))
    warnings: list[str] = field(default_factory=list)

    @property
    def separated(self) -> bool:
        return bool(self.other_bone.any())

    def summary(self) -> str:
        if not self.steps:
            return (
                f"The mandible is {self.volume_mm3 / 1000:.1f} cm³ of bone and is not "
                "attached to any other bone in the scan."
            )
        text = (
            f"Mandible separated from the rest of the skull {' and '.join(self.steps)} "
            f"({self.volume_mm3 / 1000:.1f} cm³ of bone)."
        )
        return " ".join([text, *self.warnings])


def _stations(frames, step_mm: float = 0.5):
    """Arch curve stations about ``step_mm`` apart, with horizontal normals."""
    stride = max(int(round(step_mm / frames.step_mm)), 1)
    points = np.asarray(frames.points[::stride], dtype=float)
    tangents = np.asarray(frames.tangents[::stride], dtype=float)
    across = np.cross(tangents, [0.0, 0.0, 1.0])
    length = np.linalg.norm(across, axis=1, keepdims=True)
    across = np.where(length > 1e-6, across / np.maximum(length, 1e-12), [1.0, 0.0, 0.0])
    return points, tangents, across


def find_bite(volume, frames, band_mm: float = BAND_MM, step_mm: float = 0.5) -> Bite:
    """The occlusal gap along the arch curve (see the module docstring)."""
    points, _, across = _stations(frames, step_mm)
    return bite_along(volume, points, across, band_mm, step_mm)


def bite_along(volume, points, across, band_mm: float = BAND_MM, step_mm: float = 0.5) -> Bite:
    """The occlusal gap along a polyline with horizontal across-directions.

    Searched twice. The first search finds the darkest smooth line with
    crowns either side anywhere in the band; an occlusal plane is fitted to
    it robustly, so the stretches where it went astray carry no weight; the
    second search keeps within ``BITE_PLANE_BAND_MM`` of that plane. A metal
    filling or crown shades the slices it lies in with dark and bright
    streaks, and on a CT reconstructed in slices a few millimetres thick that
    streak band can be darker, with brighter metal beside it, than the bite
    itself: the first search follows it into the crowns, the second cannot.
    """
    offsets = np.arange(-band_mm, band_mm + 1e-9, step_mm)
    rise = np.arange(BITE_SEARCH_MM[0], BITE_SEARCH_MM[1] + 1e-9, step_mm)
    grid = (
        points[:, None, None, :]
        + offsets[None, :, None, None] * across[:, None, None, :]
        + rise[None, None, :, None] * np.array([0.0, 0.0, 1.0])
    )
    fill = float(np.min(volume.array))
    image = volume.sample(grid, fill=fill).mean(axis=1)  # (stations, rise)
    # Scale to the crowns, not to metal: fillings and brackets are a few
    # percent of the band at most and would make every crown look dim. Above
    # the crowns the scale stops, so a filling is no brighter a crown than
    # enamel and its glare beside a dark streak is not taken for a bite.
    low = float(np.percentile(image, 1.0))
    high = float(np.percentile(image, 97.0))
    image = np.minimum((image - low) / max(high - low, 1e-6), METAL_CLIP)

    # A gap is dark with bright crowns on both sides within a crown height.
    reach = int(round(6.0 / step_mm))
    padded = np.pad(image, ((0, 0), (reach, reach)), mode="edge")
    width = image.shape[1]
    below = np.max(
        np.stack([padded[:, reach - d : reach - d + width] for d in range(2, reach + 1)]),
        axis=0,
    )
    above = np.max(
        np.stack([padded[:, reach + d : reach + d + width] for d in range(2, reach + 1)]),
        axis=0,
    )
    crowns = np.minimum(below, above)
    depth = crowns - image

    rows = np.arange(len(points))
    path = _darkest_path(-depth)
    has_teeth = (crowns[rows, path] >= CROWN_LEVEL) & (depth[rows, path] >= MIN_GAP_DEPTH)
    if int(has_teeth.sum()) >= MIN_PLANE_STATIONS:
        plane = robust_plane(
            points[has_teeth], points[has_teeth, 2] + rise[path[has_teeth]], depth[rows, path][has_teeth]
        )
        if plane is not None:
            expected = plane(points[:, 0], points[:, 1]) - points[:, 2]
            allowed = np.abs(rise[None, :] - expected[:, None]) <= BITE_PLANE_BAND_MM
            # A station whose plane lies outside the searched band keeps its
            # nearest row, so every station still has a path through it.
            nearest = np.argmin(np.abs(rise[None, :] - expected[:, None]), axis=1)
            allowed[rows, nearest] = True
            path = _darkest_path(np.where(allowed, -depth, np.inf))
            has_teeth = (crowns[rows, path] >= CROWN_LEVEL) & (depth[rows, path] >= MIN_GAP_DEPTH)
    return Bite(
        station_points=points,
        heights_mm=points[:, 2] + rise[path],
        has_teeth=has_teeth,
        depth=depth[rows, path],
    )


def _darkest_path(cost: np.ndarray) -> np.ndarray:
    """The cheapest path across the stations (rows of ``cost``), moving at
    most one height step per station: dynamic programming."""
    n, m = cost.shape
    penalty = np.array([[0.02], [0.0], [0.02]])
    total = cost[0].copy()
    step = np.zeros((n, m), dtype=np.int8)
    for i in range(1, n):
        options = np.stack([np.r_[np.inf, total[:-1]], total, np.r_[total[1:], np.inf]])
        with np.errstate(invalid="ignore"):
            choice = np.argmin(options + penalty, axis=0)
        step[i] = choice - 1
        total = options[choice, np.arange(m)] + cost[i]
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmin(total))
    for i in range(n - 1, 0, -1):
        path[i - 1] = path[i] + step[i, path[i]]
    return path


def robust_plane(points: np.ndarray, z: np.ndarray, weights=None, max_slope: float = 0.6):
    """``z(x, y)`` of the plane through ``(points[:, :2], z)``, outliers aside.

    Iteratively reweighted least squares with Tukey's biweight: a stretch of
    stations where the bite search followed a streak instead of the bite
    gets no weight once it is far from the plane the rest agree on. None
    when there are too few points or the fit is implausibly steep (a head
    is never tipped so far that its occlusal plane rises 0.6 mm per mm).
    """
    points = np.asarray(points, dtype=float)
    z = np.asarray(z, dtype=float)
    if len(z) < 3:
        return None
    base = np.ones(len(z)) if weights is None else np.clip(np.asarray(weights, dtype=float), 1e-3, None)
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(len(z))])
    w = base.copy()
    coef = np.zeros(3)
    for _ in range(20):
        sw = np.sqrt(w)
        coef, *_ = np.linalg.lstsq(design * sw[:, None], z * sw, rcond=None)
        residual = z - design @ coef
        scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 0.75)
        u = residual / (4.685 * scale)
        w = base * np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
        if np.count_nonzero(w) < 3:
            return None
    a, b, c = (float(v) for v in coef)
    if np.hypot(a, b) >= max_slope:
        return None
    return lambda x, y: a * np.asarray(x, dtype=float) + b * np.asarray(y, dtype=float) + c


def _extended_polyline(points: np.ndarray, extension_mm: float, step_mm: float):
    """The arch polyline continued straight on past both ends (into the rami)."""
    n = int(round(extension_mm / step_mm))
    if len(points) < 2 or n == 0:
        return points, 0
    k = min(10, len(points) - 1)
    head_dir = points[0] - points[k]
    tail_dir = points[-1] - points[-1 - k]
    head_dir[2] = tail_dir[2] = 0.0
    head_dir /= max(np.linalg.norm(head_dir), 1e-9)
    tail_dir /= max(np.linalg.norm(tail_dir), 1e-9)
    steps = step_mm * np.arange(1, n + 1)[:, None]
    head = points[0] + head_dir * steps[::-1]
    tail = points[-1] + tail_dir * steps
    return np.vstack([head, points, tail]), n


def _column_map(xs: np.ndarray, ys: np.ndarray, stations_xy: np.ndarray):
    """Nearest station and horizontal distance to it, for every (y, x) column."""
    X, Y = np.meshgrid(xs, ys)
    columns = np.stack([X.ravel(), Y.ravel()], axis=1)
    nearest = np.empty(len(columns), dtype=np.int32)
    distance = np.empty(len(columns), dtype=np.float32)
    chunk = max(1, 4_000_000 // max(len(stations_xy), 1))
    for start in range(0, len(columns), chunk):
        part = columns[start : start + chunk]
        d2 = ((part[:, None, :] - stations_xy[None, :, :]) ** 2).sum(axis=2)
        nearest[start : start + chunk] = np.argmin(d2, axis=1)
        distance[start : start + chunk] = np.sqrt(d2.min(axis=1))
    return nearest.reshape(X.shape), distance.reshape(X.shape)


def _components(mask: np.ndarray) -> np.ndarray:
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    return sitk.GetArrayFromImage(sitk.ConnectedComponent(image, True))


def _label_at(labels: np.ndarray, indices: np.ndarray, radius: np.ndarray) -> int:
    """The most common non-zero label near the given voxel indices.

    The arch curve is drawn on the buccal cortex, where the plate sits, so it
    can lie a little outside the bone; ``radius`` (voxels per axis) is how far
    to look for it.
    """
    found = []
    shape = np.array(labels.shape)
    for index in indices:
        start = np.maximum(index - radius, 0)
        stop = np.minimum(index + radius + 1, shape)
        block = labels[start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]]
        values = block[block > 0]
        if values.size:
            found.append(np.bincount(values).argmax())
    if not found:
        return 0
    return int(np.bincount(found).argmax())


#: The mandible's cross-section is at least this wide, mm; the hyoid is not.
MIN_ARCH_WIDTH_MM = 50.0
#: ... and keeps its U shape for at least this far up from its lower border.
MIN_BODY_HEIGHT_MM = 10.0


#: A head may lie turned in the scanner; if no U is found straight ahead,
#: it is looked for at these angles, nearest first, deg.
ARCH_TURNS_DEG = (-12.0, 12.0, -24.0, 24.0, -36.0, 36.0, -48.0, 48.0)
#: A head turned by less than this is treated as straight, deg: the arch's
#: own asymmetry turns its axes by a few degrees on any jaw.
STRAIGHT_DEG = 15.0


def _arch_section(section: np.ndarray, pixel_mm: np.ndarray, min_width_mm: float,
                  max_width_mm: float = np.inf, turns=(0.0,)):
    """The widest U-shaped piece of bone in an axial section, open to the back.

    Returns ``(mask, turn_deg)`` for that piece, or None. Patient axes are
    LPS, so the back of the head is +y: a mandibular section has bone across
    the front at the midline and an arm on either side running back. A head
    turned in the scanner turns the U with it, so each piece is also tried
    turned back by each of ``turns``, deg.
    """
    import SimpleITK as sitk

    labels = sitk.GetArrayFromImage(
        sitk.ConnectedComponent(sitk.GetImageFromArray(section.astype(np.uint8)), True)
    )
    sizes = np.bincount(labels.ravel())
    best, best_width = None, 0.0
    for label in np.flatnonzero(sizes[1:] * float(np.prod(pixel_mm)) >= 60.0) + 1:
        rows, cols = np.nonzero(labels == label)
        x0, y0 = cols * pixel_mm[0], rows * pixel_mm[1]
        for turn in turns:
            angle = np.radians(turn)
            cx, cy = x0.mean(), y0.mean()
            x = cx + (x0 - cx) * np.cos(angle) - (y0 - cy) * np.sin(angle)
            y = cy + (x0 - cx) * np.sin(angle) + (y0 - cy) * np.cos(angle)
            if _is_arch(x, y, min_width_mm, max_width_mm):
                width = float(np.ptp(x))
                if width > best_width:
                    best, best_width = (labels == label, turn), width
                break
    return best


def _is_arch(x: np.ndarray, y: np.ndarray, min_width_mm: float, max_width_mm: float) -> bool:
    """Whether points (mm, +y to the back) make a U open to the back."""
    width, depth = float(np.ptp(x)), float(np.ptp(y))
    if width < min_width_mm or width > max_width_mm or depth < 15.0:
        return False
    centre = 0.5 * (x.min() + x.max())
    midline = np.abs(x - centre) < 4.0
    if not midline.any() or y[midline].max() - y.min() > 0.45 * depth:
        return False  # solid across the middle: not an arch
    back = y > y.min() + 0.6 * depth
    return bool((x[back] < centre - 10.0).any() and (x[back] > centre + 10.0).any())


def find_arch(volume, threshold: float, count: int = 11, max_width_mm: float = np.inf,
              search_fraction: float = 0.7, turns=None) -> np.ndarray | None:
    """Arch curve points through the mandibular body, found from the scan alone.

    Axial sections are searched from the bottom of the scan up for the
    mandible's U (see :func:`_arch_section`); the curve is laid through the
    middle of the bone ten millimetres above the lowest level where the U
    holds for a centimetre. Returns ``(count, 3)`` world points, or None when
    no mandible-shaped bone is found. Only the lower ``search_fraction`` of
    the scan is searched, and only Us at most ``max_width_mm`` wide count:
    in a CT that reaches the shoulders, the shoulder girdle is a U too.
    Straight ahead is tried first, then a head turned by ``ARCH_TURNS_DEG``.
    """
    if turns is None:
        found = find_arch(volume, threshold, count, max_width_mm, search_fraction, (0.0,))
        if found is not None:
            return found
        return find_arch(volume, threshold, count, max_width_mm, search_fraction,
                         (0.0, *ARCH_TURNS_DEG))
    spacing = volume.spacing
    stride = np.maximum(np.round(1.0 / spacing[:2]).astype(int), 1)
    coarse_pixel = spacing[:2] * stride
    z_step = max(int(round(2.0 / spacing[2])), 1)
    array = volume.array
    levels = [
        k
        for k in range(0, max(int(array.shape[0] * search_fraction), 1), z_step)
        if _arch_section(
            array[k, :: stride[1], :: stride[0]] >= threshold, coarse_pixel, MIN_ARCH_WIDTH_MM,
            max_width_mm, turns,
        )
        is not None
    ]
    if not levels:
        return None
    present = set(levels)
    span = max(int(round(MIN_BODY_HEIGHT_MM / (z_step * spacing[2]))), 1)
    start = next(
        (
            k
            for k in levels
            if sum((k + i * z_step) in present for i in range(span + 1)) >= 0.8 * (span + 1)
        ),
        None,
    )
    if start is None:
        return None
    level = min(levels, key=lambda k: abs(k - (start + MIN_BODY_HEIGHT_MM / spacing[2])))
    found = _arch_section(array[level] >= threshold, spacing[:2], MIN_ARCH_WIDTH_MM, max_width_mm, turns)
    if found is None:
        return None
    section, turn = found
    rows, cols = np.nonzero(section)
    # Bin the bone by angle about a point just behind the arch: the mean of
    # each bin is a point on the middle of the body. "Behind" is in the
    # frame the U was recognised in, for a head lying turned.
    t = np.radians(turn)
    x, y = cols * spacing[0], rows * spacing[1]
    cx, cy = x.mean(), y.mean()
    u = (x - cx) * np.cos(t) - (y - cy) * np.sin(t)
    v = (x - cx) * np.sin(t) + (y - cy) * np.cos(t)
    angle = np.arctan2(v - (v.max() + 5.0), u)
    edges = np.linspace(angle.min(), angle.max(), count + 1)
    points = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (angle >= low) & (angle < high)
        if inside.sum() > 5:
            points.append(volume.index_to_world([cols[inside].mean(), rows[inside].mean(), level]))
    return np.array(points) if len(points) >= 3 else None


#: Spacing of the points laid along the automatic arch curve. The curve is a
#: spline through them; closer points would follow every bump of the trace.
CENTRELINE_SPACING_MM = 12.0


def mandible_centreline(volume, isolation: "MandibleIsolation", body_points: np.ndarray,
                        step_mm: float = 4.0,
                        spacing_mm: float = CENTRELINE_SPACING_MM) -> np.ndarray:
    """Points along the middle of the mandible, from one condyle to the other.

    The body part is ``body_points`` (from :func:`find_arch`). Each ramus is
    traced upwards from a little above the bite, where nothing but ramus
    stands beside the teeth, as the centre of the mandible's section on that
    side every ``step_mm``; where the ramus forks into coronoid and condylar
    processes the trace follows the back one, which ends at the condyle. A
    spline through the whole set joins the body to each ramus round the angle.
    Returned right condyle first (patient right is -x), as points
    ``spacing_mm`` apart along the trace.
    """
    import SimpleITK as sitk

    spacing, origin, mask = volume.spacing, volume.origin, isolation.mask
    body = np.asarray(body_points, dtype=float)
    front, back, lateral = arch_axes(body)
    if float((body[0, :2] - body[-1, :2]) @ lateral) > 0:
        body = body[::-1]  # patient right first
    centre = float(np.mean(body[:, :2] @ lateral))
    bite = isolation.bite
    bite_level = (
        float(np.median(bite.heights_mm[bite.has_teeth]))
        if bite is not None
        else float(body[0, 2]) + 20.0
    )
    k_step = max(int(round(step_mm / spacing[2])), 1)
    k_start = max(int(np.ceil((bite_level + 6.0 - origin[2]) / spacing[2])), 0)
    xs = origin[0] + np.arange(mask.shape[2]) * spacing[0]
    ys = origin[1] + np.arange(mask.shape[1]) * spacing[1]
    across = xs[None, :] * lateral[0] + ys[:, None] * lateral[1]
    pixel_area = float(spacing[0] * spacing[1])
    rami = []
    for side in (-1.0, 1.0):
        columns = (across - centre) * side > 15.0
        trail, previous = [], None
        for k in range(k_start, mask.shape[0], k_step):
            section = mask[k] & columns
            if not section.any():
                if trail:
                    break
                continue
            labels = sitk.GetArrayFromImage(
                sitk.ConnectedComponent(sitk.GetImageFromArray(section.astype(np.uint8)), True)
            )
            found = []
            for label in range(1, int(labels.max()) + 1):
                rows, cols = np.nonzero(labels == label)
                if len(rows) * pixel_area < 4.0:
                    continue
                found.append(
                    np.array([origin[0] + cols.mean() * spacing[0], origin[1] + rows.mean() * spacing[1]])
                )
            if previous is not None:
                found = [c for c in found if np.linalg.norm(c - previous) < 15.0]
            if not found:
                if trail:
                    break
                continue
            best = max(found, key=lambda c: float(c @ back))  # the back process: the condyle
            previous = best
            trail.append([best[0], best[1], origin[2] + k * spacing[2]])
        trail = np.array(trail, dtype=float).reshape(-1, 3)
        if len(trail) >= 3:
            smooth = trail.copy()
            smooth[1:-1, :2] = (trail[:-2, :2] + trail[1:-1, :2] + trail[2:, :2]) / 3.0
            trail = smooth
        rami.append(trail)
    right, left = rami
    return even_points(np.vstack([right[::-1], body, left]), spacing_mm)


def even_points(points: np.ndarray, spacing_mm: float) -> np.ndarray:
    """Points evenly spaced along a polyline, both ends kept."""
    points = np.asarray(points, dtype=float)
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], step > 1e-9])
    points, s = points[keep], np.concatenate([[0.0], np.cumsum(step[step > 1e-9])])
    if len(points) < 2:
        return points
    count = max(int(round(s[-1] / spacing_mm)), 2)
    wanted = np.linspace(0.0, s[-1], count + 1)
    return np.column_stack([np.interp(wanted, s, points[:, k]) for k in range(3)])


def arch_axes(points: np.ndarray):
    """``(front, back, lateral)`` of an arch in plan, as 2-D vectors.

    ``front`` is the point of the arch farthest from the line through its
    ends (the chin, on any U); ``back`` is square to that line, pointing
    from the front towards it, and ``lateral`` to the patient's left. For a
    head lying straight these are +y and +x; for one lying turned in the
    scanner they turn with it.
    """
    xy = np.asarray(points, dtype=float)[:, :2]
    between = 0.5 * (xy[0] + xy[-1])
    chord = xy[-1] - xy[0]
    length = float(np.linalg.norm(chord))
    if length < 1e-6:
        return xy[0], np.array([0.0, 1.0]), np.array([1.0, 0.0])
    square = np.array([-chord[1], chord[0]]) / length
    depth = (xy - between) @ square
    front = xy[int(np.argmax(np.abs(depth)))]
    back = square if float((between - front) @ square) > 0 else -square
    if back[1] > np.cos(np.radians(STRAIGHT_DEG)):
        back = np.array([0.0, 1.0])
    lateral = np.array([back[1], -back[0]])
    return front, back, lateral


def frames_for(points: np.ndarray, step_mm: float = 0.25):
    """Arch frames through ``points``, for :func:`isolate_mandible`."""
    from . import cpr
    from .spline import ArchCurve

    return cpr.build_frames(ArchCurve(points), step_mm)


def isolate_mandible(volume, threshold: float, frames) -> MandibleIsolation:
    """Label the mandible's voxels (see the module docstring)."""
    spacing = volume.spacing
    origin = volume.origin
    step = float(max(np.min(spacing), 0.5))
    bone = volume.array >= threshold
    full_shape = bone.shape
    points, _, _ = _stations(frames, step)

    # The curve may stop short of the last molars, and the rami rise behind
    # it: carry it straight on past both ends and look for teeth there too.
    extended, n_head = _extended_polyline(points, RAMUS_EXTENSION_MM, step)
    across = np.cross(np.gradient(extended, axis=0), [0.0, 0.0, 1.0])
    across /= np.maximum(np.linalg.norm(across, axis=1, keepdims=True), 1e-9)
    bite = bite_along(volume, extended, across, step_mm=step)
    ref = _reference_heights(bite, extended).astype(np.float32)
    plane = _bite_plane(bite, extended, ref)

    # Work inside a box around the arch: the whole mandible, and just enough
    # skull above it to recognise.
    lo = points.min(axis=0) - np.array([55.0, 55.0, 0.0])
    hi = points.max(axis=0) + np.array([55.0, 55.0, 0.0])
    corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
    lo[2] = origin[2]
    hi[2] = float(np.max(plane(corners[:, 0], corners[:, 1]))) + CONDYLE_CLEAR_MM + 8.0
    i0 = np.maximum(np.floor((lo - origin) / spacing).astype(int), 0)
    i1 = np.minimum(np.ceil((hi - origin) / spacing).astype(int) + 1, volume.size_xyz)
    box = (slice(i0[2], i1[2]), slice(i0[1], i1[1]), slice(i0[0], i1[0]))
    crop = bone[box]
    xs = origin[0] + spacing[0] * np.arange(i0[0], i1[0])
    ys = origin[1] + spacing[1] * np.arange(i0[1], i1[1])
    zs = (origin[2] + spacing[2] * np.arange(i0[2], i1[2])).astype(np.float32)

    station_index = np.round((points - origin) / spacing).astype(int)[:, ::-1] - i0[::-1]
    inside = np.all((station_index >= 0) & (station_index < crop.shape), axis=1)
    station_index = station_index[inside]
    radius = np.ceil(3.0 / spacing[::-1]).astype(int)

    labels = _components(crop)
    target = _label_at(labels, station_index, radius)
    if target == 0:
        raise ValueError(
            "No bone at the arch curve at this threshold; lower the threshold or "
            "redraw the curve on the bone."
        )
    connected = labels == target
    del labels

    # Which station each column of voxels belongs to. Half a millimetre is
    # fine enough for a band 20 mm wide; on a finer scan the map is made at
    # that pitch and repeated.
    fine = max(int(round(0.5 / float(min(spacing[0], spacing[1])))), 1)
    nearest, distance = _column_map(xs[::fine], ys[::fine], extended[:, :2])
    if fine > 1:
        rows, cols = len(ys), len(xs)
        nearest = np.repeat(np.repeat(nearest, fine, axis=0), fine, axis=1)[:rows, :cols]
        distance = np.repeat(np.repeat(distance, fine, axis=0), fine, axis=1)[:rows, :cols]
    height_above = zs[:, None, None] - ref[nearest][None, :, :]
    X, Y = np.meshgrid(xs, ys)
    above_plane = zs[:, None, None] - plane(X, Y).astype(np.float32)[None, :, :]

    # The palate and the palatal side of the upper jaw: inside the arch's
    # outline, above the bite. Nothing of the mandible stands there: the
    # coronoids rise outside the line of the arch, the rami behind its ends,
    # and only the inner edge of a ramus comes within 5 mm of the line.
    outline = points[:: max(len(points) // 60, 1), :2]
    columns = np.column_stack([X.ravel(), Y.ravel()])
    inside_arch = point_in_polygon(columns, outline).reshape(X.shape)
    palatal = (inside_arch & (distance > 5.0))[None] & (above_plane > 5.0) & (above_plane < 40.0)

    # Upper teeth: crowns and fillings are the brightest things in the scan,
    # brighter than any cortex. Where one stands above the bite along the
    # arch it is an upper tooth, even where a filling's streaks hid the gap
    # from the bite search and it was not cut free. Only along the arch
    # itself: its straight continuation runs up the rami.
    upper_teeth = np.zeros_like(crop)
    filled_teeth = np.zeros(len(extended), dtype=bool)
    if bite.found:
        on_arch = np.zeros(len(extended), dtype=bool)
        on_arch[n_head : n_head + len(points)] = True
        band = ((distance < 8.0) & on_arch[nearest])[None]
        gray = volume.array[box]
        near_bite = band & (np.abs(above_plane) < 10.0) & crop
        if near_bite.any():
            enamel = float(np.percentile(gray[near_bite], 97.0))
            bright = gray >= enamel
            upper_teeth = band & (above_plane > 2.0) & (above_plane < 14.0) & bright
            # Stations with crowns at the bite but no gap found between them:
            # fillings, whose streaks fill the gap the bite search looks for.
            crowns = (band & (np.abs(above_plane) < 4.0) & bright).any(axis=0)
            filled_teeth[np.unique(nearest[crowns])] = True
            filled_teeth &= on_arch & ~bite.has_teeth

    braincase = _braincase(volume.array[box], crop, spacing) & (above_plane > RAMUS_SEED_MM)

    # The tooth row, from the last tooth on one side to the last on the
    # other, and its front, where the upper incisors overlap the lower ones.
    dental = _tooth_row(bite.has_teeth | filled_teeth, n_head, n_head + len(points) - 1, step)
    along = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(extended, axis=0), axis=1))])
    front, _, _ = arch_axes(points)
    chin = int(np.argmin(np.linalg.norm(extended[:, :2] - front, axis=1)))
    front_teeth = np.abs(along - along[chin]) < INCISOR_SPAN_MM
    below_bite = np.where(front_teeth, OVERBITE_MM, BITE_MARGIN_MM)
    upper_reach = np.where(front_teeth, UPPER_REACH_MM[1], UPPER_REACH_MM[0])[nearest]
    # Along the bite: the crowns and the bone round them. A crown's enamel and
    # dentin are brighter than any cortex, a ramus's included.
    gray = volume.array[box]
    near_bite = ((distance < STREAK_REACH_MM) & dental[nearest])[None] & (
        np.abs(height_above) < STREAK_BAND_MM
    )
    crown_level = (
        float(np.percentile(gray[near_bite & crop], 90.0)) if (near_bite & crop).sum() >= 100 else np.inf
    )

    def skull_markers(mask: np.ndarray, clear=None, ramus_columns=None, rami=None) -> np.ndarray:
        """The skull's markers; ``clear`` (columns) is kept free of the
        far-above-the-bite ones below the condyles' greatest height, and
        ``ramus_columns`` and ``rami`` (voxels) of the upper teeth's."""
        high = above_plane > SKULL_ABOVE_BITE_MM
        if clear is not None:
            high = high & ~(clear[None] & (above_plane < CONDYLE_CLEAR_MM))
        # Above the bite along the whole tooth row: the upper teeth and the
        # alveolar bone they stand in. The rami, rising behind and beside
        # the last molars, are kept clear.
        reach = (distance < upper_reach) | inside_arch
        upper_arch = (
            (reach & dental[nearest])[None]
            & (height_above > BITE_MARGIN_MM)
            & (height_above < BITE_MARGIN_MM + UPPER_ROW_MM)
        )
        if ramus_columns is not None:
            upper_arch &= ~ramus_columns[None]
        if rami is not None:
            upper_arch &= ~rami
        return mask & (high | ((upper_arch | palatal | upper_teeth) & bite.found))

    steps: list[str] = []
    mandible = connected
    if skull_markers(connected).any():
        cut = np.zeros_like(crop)
        if bite.found:
            teeth = bite.has_teeth
            gap = zs[:, None, None] - bite.heights_mm.astype(np.float32)[nearest][None, :, :]
            with np.errstate(invalid="ignore"):
                cut = (
                    crop
                    & (distance < BAND_MM + 2.0)[None]
                    & teeth[nearest][None]
                    & (np.abs(gap) < BITE_CUT_MM)
                )
            # Where fillings hid the gap, cut along the bite plane instead.
            cut |= (
                crop
                & (distance < BAND_MM + 2.0)[None]
                & filled_teeth[nearest][None]
                & (np.abs(above_plane) < BITE_CUT_MM)
            )
            steps.append("at the bite")
        freed = crop & ~cut
        labels = _components(freed)
        target = _label_at(labels, station_index, radius)
        mandible = labels == target if target else connected
        del labels
        # Each ramus is mandible up to below its notch, and the corridor it
        # rises in is no place for a skull marker below the condyles' height.
        # Without this the watershed is left a whole ramus to decide, and on a
        # thick-slice CT, where the joint space is blurred away, the thin
        # middle of the ramus is where it cuts.
        plate, corridor = _ramus_corridors(mandible, X, Y, above_plane, points, spacing)
        # Never inside the arch: that is upper teeth and palate above the bite.
        ramus = mandible & (plate & ~inside_arch)[None] & (above_plane > -2.0) & (
            above_plane < RAMUS_SEED_MM
        )
        processes, joint_roofs = _processes(
            mandible, ramus & (above_plane > RAMUS_SEED_MM - 4.0), spacing, above_plane,
            plate & ~inside_arch, inside_arch,
        )
        # The front of a ramus rises beside the last molar, ahead of where its
        # plate was fitted; in each axial slice it is still one small piece
        # of bone with the ramus, and no upper tooth's.
        rami = _pieces_touching(mandible, ramus | processes, spacing)
        skull = skull_markers(mandible, clear=corridor, ramus_columns=plate | corridor, rami=rami)
        # The braincase is skull even in the corridor: over each joint it is
        # the fossa roof, the skull's side of the joint.
        skull |= mandible & braincase
        # So is the spine, which joins the skull behind the jaws and would
        # otherwise be anyone's.
        _, back_dir, lateral = arch_axes(points)
        across = X * lateral[0] + Y * lateral[1]
        behind = X * back_dir[0] + Y * back_dir[1]
        spine = (np.abs(across - float(np.mean(points[:, :2] @ lateral))) < SPINE_HALF_WIDTH_MM) & (
            behind > float((points[:, :2] @ back_dir).max()) + SPINE_BEHIND_ARCH_MM
        )
        skull |= mandible & spine[None]
        skull |= joint_roofs
        if skull.any():
            seeds = mandible & (distance < 15.0)[None] & (height_above < -below_bite[nearest][None])
            seeds |= ramus | processes
            markers = seeds.astype(np.uint8) + 2 * (skull & ~seeds).astype(np.uint8)
            split = _split_at_narrowest(
                mandible, markers, volume.array[box], threshold, spacing
            )
            region = (split == 1) & mandible
            labels = _components(region)
            target = _label_at(labels, station_index, radius)
            if target:
                mandible = labels == target
            steps.append("at the jaw joints")
        if bite.found:
            # Metal in the teeth throws streaks across the slices it lies in,
            # and where they pass the threshold they stand off the crowns as
            # thin fins. They are opened away at their roots, round the
            # crowns along the bite near metal, and fall away with them; a
            # ramus beside the last molar, whose middle is as thin, is left.
            metal = _metal(gray, crop, near_bite)
            if metal.any():
                crowns = gray >= float(np.percentile(gray[near_bite & crop], CROWN_PERCENTILE))
                mandible = _without_streaks(mandible, metal, spacing, volume.resampled_from_mm, crowns)
            band = near_bite & _round_crowns(gray, crop, near_bite, metal, crown_level, spacing)
            mandible = _without_fins(mandible, band, spacing, volume.resampled_from_mm)
            labels = _components(mandible)
            target = _label_at(labels, station_index, radius)
            if target:
                mandible = labels == target
            # The bite cut runs on past the last molar into the front of each
            # ramus; where the mandible lies both above and below a cut
            # voxel it was never a bite, and the slot is closed again.
            mandible |= cut & _between(mandible, max(int(np.ceil(1.5 / spacing[2])), 1))

    mask = np.zeros(full_shape, dtype=bool)
    mask[box] = mandible
    other = bone & ~mask
    heights, warnings = _ramus_check(mandible, above_plane, X, Y, points)
    return MandibleIsolation(
        mask=mask,
        other_bone=other,
        bite=bite if bite.found else None,
        steps=steps,
        volume_mm3=float(mask.sum() * np.prod(spacing)),
        ramus_heights_mm=heights,
        warnings=warnings,
    )


def _braincase(gray: np.ndarray, bone: np.ndarray, spacing) -> np.ndarray:
    """Bone lining the cranial cavity, where the scan reaches it.

    In each axial slice through the cranium bone closes off a large region
    of soft tissue: the brain. The bone within ``BRAINCASE_LINING_MM`` of it
    is skull whatever its height, which matters over the joints, where the
    condyles may rise above any fixed height and the fossa roof between
    them and the brain is the skull's only bone there.
    """
    import SimpleITK as sitk

    values = gray[::2, ::4, ::4].ravel()
    lo, hi = (float(v) for v in np.percentile(values, [0.5, 99.5]))
    if hi <= lo:
        return np.zeros_like(bone)
    from .threshold import otsu_threshold

    counts, edges = np.histogram(np.clip(values, lo, hi), 128, (lo, hi))
    air = otsu_threshold(counts, edges)
    pixel_cm2 = float(spacing[0] * spacing[1]) / 100.0
    brain = np.zeros_like(bone)
    for k in range(bone.shape[0]):
        section = bone[k]
        if not section.any():
            continue
        # The box round the jaws stops short of the back of the skull: close
        # it there, or the cranium is a ring cut open at the back.
        closed = section.copy()
        closed[-1, :] = True
        filled = sitk.GetArrayFromImage(
            sitk.BinaryFillhole(sitk.GetImageFromArray(closed.astype(np.uint8)))
        ) > 0
        enclosed = filled & ~section & (gray[k] > air)
        if enclosed.sum() * pixel_cm2 < MIN_BRAIN_SECTION_CM2:
            continue
        labels = sitk.GetArrayFromImage(
            sitk.ConnectedComponent(sitk.GetImageFromArray(enclosed.astype(np.uint8)))
        )
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        if sizes.max() * pixel_cm2 >= MIN_BRAIN_SECTION_CM2:
            brain[k] = labels == int(sizes.argmax())
    if not brain.any():
        return brain
    image = sitk.GetImageFromArray(brain.astype(np.uint8))
    image.SetSpacing([float(v) for v in spacing])
    distance = sitk.GetArrayFromImage(
        sitk.SignedMaurerDistanceMap(image, insideIsPositive=False, squaredDistance=False,
                                     useImageSpacing=True)
    )
    return bone & (distance <= BRAINCASE_LINING_MM)


def _ramus_corridors(mandible, X, Y, above_plane, points, spacing):
    """Each ramus's plate, and the corridor above it, as column masks.

    A ramus is a thin, near-vertical plate rising behind the last molars.
    Just above the bite, below the notch, nothing else of the mandible
    stands there, so the plate is fitted (a line through its bone in plan)
    from the bone at that height on either side, behind the arch's end.
    Returns ``(plate, corridor)``: the columns within the plate's half
    thickness of that line, and within the wider corridor half-width.
    """
    _, back_dir, lateral = arch_axes(points)
    sideways = X * lateral[0] + Y * lateral[1]
    behind = X * back_dir[0] + Y * back_dir[1]
    on_arch = points[:, :2] @ lateral
    centre = float(np.mean(on_arch))
    half = 0.25 * float(np.ptp(on_arch))
    sample = (above_plane > RAMUS_SAMPLE_MM[0]) & (above_plane < RAMUS_SAMPLE_MM[1]) & mandible
    plate = np.zeros(X.shape, dtype=bool)
    corridor = np.zeros(X.shape, dtype=bool)
    ends = points[[int(np.argmin(on_arch)), int(np.argmax(on_arch))]]
    voxel_mm3 = float(np.prod(spacing))
    for side, end in zip((-1.0, 1.0), ends):
        columns = ((sideways - centre) * side > half) & (behind > float(end[:2] @ back_dir) - 5.0)
        # The ramus rises from the body just behind the last tooth: in the
        # bone up to the slab's top on this side, it is the piece nearest the
        # arch's end that reaches both down below the bite and up into the
        # slab. An upper molar or the tuberosity stands in the slab too, but
        # the bite cut parted it from everything below; the spine reaches
        # below the bite too, but far behind.
        labels = _components(mandible & columns[None] & (above_plane < RAMUS_SAMPLE_MM[1]))
        low = np.unique(labels[(above_plane < -3.0) & (labels > 0)])
        count = int(labels.max()) + 1
        labels = np.where(sample, labels, 0)
        in_slab = np.bincount(labels.ravel(), minlength=count)
        best, best_near = 0, 15.0
        for label in low:
            if in_slab[label] * voxel_mm3 < 200.0:
                continue
            _, r, c = np.nonzero(labels == label)
            near = float(np.min(np.hypot(X[r, c] - end[0], Y[r, c] - end[1])))
            if near < best_near:
                best, best_near = label, near
        if not best:
            continue  # no ramus to speak of at this height on this side
        _, rows, cols = np.nonzero(labels == best)
        xy = np.column_stack([X[rows, cols], Y[rows, cols]])
        middle = xy.mean(axis=0)
        _, _, axes = np.linalg.svd(xy - middle, full_matrices=False)
        along, across = axes[0], axes[1]
        offset = (np.column_stack([X.ravel(), Y.ravel()]) - middle)
        t = (offset @ along).reshape(X.shape)
        d = np.abs(offset @ across).reshape(X.shape)
        if float(along @ back_dir) < 0:  # point it backwards: the condyle is at the back
            along, t = -along, -t
        span = (xy - middle) @ along
        within = (t > span.min() - 10.0) & (t < span.max() + 10.0) & columns
        plate |= within & (d < RAMUS_SEED_HALF_MM)
        # Only the back of the ramus rises to a condyle; the coronoid in front
        # stops under the zygomatic arch, which stays the skull's.
        back = t > 0.5 * (span.min() + span.max())
        corridor |= within & back & (d < RAMUS_CORRIDOR_HALF_MM)
    return plate, corridor


#: An axial section of a condylar or coronoid process is at most a few cm²;
#: one of the skull base, which a process traced upwards runs into where
#: the joint space is lost, is many times that.
PROCESS_MAX_CM2 = 6.0
#: Mandible seeds stop this far below where a process meets the skull, and
#: skull markers start this far above it, mm; the watershed decides between.
PROCESS_MARGIN_MM = 3.0
#: Skull markers are put this far up above the meeting, mm.
PROCESS_ROOF_MM = 12.0
#: Below this height above the bite plane, mm, a big section a process runs
#: into is not the skull base: no joint is that low. It is the maxilla, met
#: where a coronoid rests against the tuberosity or a filling's streaks join
#: the two in the axial slices.
PROCESS_JOINT_MIN_MM = 20.0


def _processes(mandible, start, spacing, above_plane=None, columns=None, inside=None):
    """The condylar and coronoid processes, followed up from the ramus seeds.

    ``start`` is the top of the ramus seeds. Slice by slice upwards, the
    pieces of bone (in the axial section) touching what was traced below are
    the processes. A process that ends in a joint space simply stops being
    found. One that has lost its joint space to thick slices, or rests on
    its eminence with the mouth open, runs into a piece far too big to be a
    process: the skull base. It is taken to end there; the bone just above
    its tip is the skull's, the bone just below the mandible's, and the few
    millimetres between are left to the watershed. A big piece met lower
    than any joint (``PROCESS_JOINT_MIN_MM`` above the bite plane, from
    ``above_plane``) is the maxilla instead: the process is followed through
    it, keeping to its own footprint from the slice below, which may grow
    by a pixel per slice only within ``columns`` (the ramus's plate), until
    it stands free again. Below that height an upper molar can touch a
    ramus, and is a small piece with it; the processes rise outside the
    arch, so there nothing ``inside`` it (columns) is followed.
    Returns ``(seeds, roofs)``.
    """
    import SimpleITK as sitk

    seeds = np.zeros_like(mandible)
    roofs = np.zeros_like(mandible)
    levels = np.flatnonzero(start.any(axis=(1, 2)))
    if not len(levels):
        return seeds, roofs
    pixel_cm2 = float(spacing[0] * spacing[1]) / 100.0
    margin = max(int(round(PROCESS_MARGIN_MM / spacing[2])), 1)
    roof = max(int(round(PROCESS_ROOF_MM / spacing[2])), 1)
    reach = max(int(round(3.0 / min(spacing[0], spacing[1]))), 1)
    # Each ramus from the top of its own start: the two end at different
    # heights when the bite plane is tilted, and a trace begun lower, where
    # a ramus can touch the upper molars, would wander along the upper teeth.
    pieces = _components(start)
    counts = np.bincount(pieces.ravel())
    counts[0] = 0
    traced = []
    meetings = []
    tips = []
    for part in np.flatnonzero(counts >= 0.05 * counts.max()):
        own = pieces == part
        k = int(np.flatnonzero(own.any(axis=(1, 2))).max())
        current = own[k]
        while k + 1 < mandible.shape[0] and current.any():
            k += 1
            labels = sitk.GetArrayFromImage(
                sitk.ConnectedComponent(sitk.GetImageFromArray(mandible[k].astype(np.uint8)), True)
            )
            touching = np.unique(labels[current & (labels > 0)])
            sizes = np.bincount(labels.ravel(), minlength=int(labels.max()) + 1)
            following = np.zeros_like(current)
            for label in touching:
                piece = labels == label
                if sizes[label] * pixel_cm2 <= PROCESS_MAX_CM2:
                    following |= piece
                    continue
                met = current & _grow2d(piece, reach)
                if above_plane is not None and float(np.mean(above_plane[k][met])) < PROCESS_JOINT_MIN_MM:
                    footprint = current if columns is None else current | (_grow2d(current, 1) & columns)
                    following |= piece & footprint
                else:
                    meetings.append((k, met))
            low = above_plane is not None and float(np.mean(above_plane[k][current])) < PROCESS_JOINT_MIN_MM
            if low and inside is not None:
                following &= ~inside
            # A process with no bone above it has ended in its joint space.
            ended = current & ~_grow2d(mandible[k], 1)
            if ended.any():
                tips.append((k - 1, ended))
            if following.any():
                traced.append((k, following))
            current = following
    for level, section in traced:
        seeds[level] |= section
    for level, tip in meetings:
        tip = _grow2d(tip, reach)
        for below in range(max(level - margin, 0), level):
            seeds[below] &= ~tip
        for above in range(level + margin, min(level + roof, mandible.shape[0])):
            roofs[above] |= mandible[above] & tip
    # A process that simply stops keeps clear of its joint too.
    for level, tip in tips:
        tip = _grow2d(tip, reach)
        for below in range(max(level - margin + 1, 0), level + 1):
            seeds[below] &= ~tip
    return seeds, roofs


#: Sheets of "bone" thinner than twice this, mm, standing off the crowns
#: along the bite are metal streaks, and are opened away; within this of the
#: bite, mm, and this of the arch curve in plan, mm.
FIN_MM = 1.0
STREAK_BAND_MM = 5.0
STREAK_REACH_MM = 30.0
#: ... within this of metal, mm, and this of a crown in plan, mm.
METAL_REACH_MM = 12.0
CROWN_REACH_MM = 3.0
#: Bone carries on at least this far above or below the slices a piece of
#: metal lies in, mm; its streaks do not.
STREAK_CONTINUES_MM = 2.0
#: Crowns are bone near the bite at least this bright, as a percentile of
#: the bone there; a streak, just over the threshold, never is.
CROWN_PERCENTILE = 75.0


def _metal(gray: np.ndarray, bone: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Metal in ``band``: brighter than any tooth, above the band's 99.5th
    percentile of bone and above 1.3 times its 90th, or at the scan's
    ceiling where it is clipped there (a CT stores at most 3071 HU).
    Relative, because CBCT gray values are not Hounsfield units; on a scan
    with no metal it finds only the brightest enamel, which is thick and
    loses nothing to what is done near metal."""
    values = gray[band & bone]
    if values.size < 100:
        return np.zeros_like(band)
    level = min(
        max(float(np.percentile(values, 99.5)), 1.3 * float(np.percentile(values, 90.0))),
        float(values.max()),
    )
    return band & (gray >= level)


def _without_streaks(
    mask: np.ndarray, metal: np.ndarray, spacing, slice_mm: float = 0.0, crowns=None
) -> np.ndarray:
    """``mask`` without the streaks round each piece of metal.

    A filling's streaks lie in the slices the metal lies in. Bone does not
    stop there: a ramus, a tooth or the body beside the metal carries on
    above or below that slab. So, within ``STREAK_REACH_MM`` of metal in
    plan, what of ``mask`` lies in the slab and has nothing of ``mask`` just
    above or just below it in its column is streak. Where several pieces of
    metal reach a column, at different heights where the bite is tilted,
    the slab there runs from the lowest to the highest, for their streaks
    cross and stack. A crown that leans can leave its column within the
    slab; ``crowns`` (enamel and dentin, far brighter than any streak) and
    the voxel round them are kept whatever their column.
    """
    import SimpleITK as sitk

    if not metal.any():
        return mask
    nz = mask.shape[0]
    pixel = float(min(spacing[0], spacing[1]))
    margin = max(int(round(0.5 * max(slice_mm, 1.0) / spacing[2])), 1)
    beyond = max(int(round(STREAK_CONTINUES_MM / spacing[2])), 1)
    low = np.full(mask.shape[1:], nz, dtype=np.int64)
    high = np.full(mask.shape[1:], -1, dtype=np.int64)
    pieces = _components(metal)
    sizes = np.bincount(pieces.ravel())
    for piece in np.flatnonzero(sizes[1:] >= 8) + 1:
        own = pieces == piece
        levels = np.flatnonzero(own.any(axis=(1, 2)))
        footprint = sitk.GetImageFromArray((~own.any(axis=0)).astype(np.uint8))
        footprint.SetSpacing([pixel, pixel])
        near = sitk.GetArrayFromImage(
            sitk.SignedMaurerDistanceMap(footprint, insideIsPositive=True, squaredDistance=False,
                                         useImageSpacing=True)
        ) <= STREAK_REACH_MM
        low[near] = np.minimum(low[near], int(levels.min()) - margin)
        high[near] = np.maximum(high[near], int(levels.max()) + margin)
    tested = (high >= 0) & (low - beyond >= 0) & (high + beyond < nz)
    if not tested.any():
        return mask
    # Whether the column holds anything of the mask just below its slab or
    # just above it, by cumulative counts down each column.
    count = np.concatenate([np.zeros((1,) + mask.shape[1:], dtype=np.int32), np.cumsum(mask, axis=0, dtype=np.int32)])
    lo = np.clip(low, beyond, nz)
    hi = np.clip(high, -1, nz - beyond - 1)
    rows, cols = np.indices(mask.shape[1:])
    below = count[lo, rows, cols] - count[lo - beyond, rows, cols]
    above = count[hi + 1 + beyond, rows, cols] - count[hi + 1, rows, cols]
    streak_columns = tested & ~((below > 0) | (above > 0))
    levels = np.arange(nz)[:, None, None]
    streak = mask & streak_columns[None] & (levels >= low[None]) & (levels <= high[None])
    if crowns is not None and (crowns & streak).any():
        keep = sitk.GetArrayFromImage(
            sitk.BinaryDilate(sitk.GetImageFromArray((crowns & mask).astype(np.uint8)), [1, 1, 1])
        ) > 0
        streak &= ~keep
    return mask & ~streak


def _round_crowns(
    gray: np.ndarray, bone: np.ndarray, band: np.ndarray, metal: np.ndarray, crown_level: float, spacing
) -> np.ndarray:
    """The part of ``band`` within ``CROWN_REACH_MM`` (in plan) of a crown
    (bone at ``crown_level`` or brighter) and within ``METAL_REACH_MM`` of
    metal."""
    import SimpleITK as sitk

    if not metal.any() or not np.isfinite(crown_level):
        return np.zeros_like(band)
    image = sitk.GetImageFromArray(metal.astype(np.uint8))
    image.SetSpacing([float(v) for v in spacing])
    distance = sitk.GetArrayFromImage(
        sitk.SignedMaurerDistanceMap(image, insideIsPositive=False, squaredDistance=False,
                                     useImageSpacing=True)
    )
    crowns = (band & bone & (gray >= crown_level)).any(axis=0)
    reach = max(int(round(CROWN_REACH_MM / float(min(spacing[0], spacing[1])))), 1)
    return (distance <= METAL_REACH_MM) & _grow2d(crowns, reach)[None]


def _without_fins(mask: np.ndarray, band: np.ndarray, spacing, slice_mm: float = 0.0) -> np.ndarray:
    """``mask`` with thin sheets opened away inside ``band``.

    On a volume interpolated from thick slices (``slice_mm``) a streak is a
    sheet as thick as the slices it lies in, and is opened that much in z.
    """
    import SimpleITK as sitk

    if not band.any():
        return mask
    kk, jj, ii = np.nonzero(band)
    pad = 4
    box = tuple(
        slice(max(int(lo) - pad, 0), int(hi) + pad + 1)
        for lo, hi in ((kk.min(), kk.max()), (jj.min(), jj.max()), (ii.min(), ii.max()))
    )
    radius = max(int(round(FIN_MM / float(min(spacing[0], spacing[1])))), 1)
    radius_z = max(int(round(max(FIN_MM, 0.75 * slice_mm) / float(spacing[2]))), 1)
    opened = sitk.GetArrayFromImage(
        sitk.BinaryMorphologicalOpening(
            sitk.GetImageFromArray(mask[box].astype(np.uint8)), [radius, radius, radius_z],
            sitk.sitkBall,
        )
    ) > 0
    out = mask.copy()
    out[box] = np.where(band[box], opened, mask[box])
    return out


def _between(mask: np.ndarray, reach: int) -> np.ndarray:
    """Voxels with ``mask`` within ``reach`` voxels both above and below."""
    above = np.zeros_like(mask)
    below = np.zeros_like(mask)
    for d in range(1, reach + 1):
        above[:-d] |= mask[d:]
        below[d:] |= mask[:-d]
    return above & below


def _pieces_touching(bone: np.ndarray, seeds: np.ndarray, spacing) -> np.ndarray:
    """In each axial slice, the pieces of ``bone`` that hold ``seeds`` and
    are small enough to be a ramus or a process (``PROCESS_MAX_CM2``), not
    one merged with the maxilla or the skull base."""
    import SimpleITK as sitk

    pixel_cm2 = float(spacing[0] * spacing[1]) / 100.0
    out = np.zeros_like(bone)
    for k in np.flatnonzero(seeds.any(axis=(1, 2))):
        labels = sitk.GetArrayFromImage(
            sitk.ConnectedComponent(sitk.GetImageFromArray(bone[k].astype(np.uint8)), True)
        )
        sizes = np.bincount(labels.ravel())
        for label in np.unique(labels[seeds[k] & (labels > 0)]):
            if sizes[label] * pixel_cm2 <= PROCESS_MAX_CM2:
                out[k] |= labels == label
    return out


def _grow2d(mask: np.ndarray, radius: int) -> np.ndarray:
    """A 2-D mask grown by ``radius`` pixels (a square)."""
    out = mask.copy()
    for axis in (0, 1):
        grown = out.copy()
        for shift in range(1, radius + 1):
            grown |= np.roll(out, shift, axis=axis) | np.roll(out, -shift, axis=axis)
        out = grown
    return out


#: The cervical spine stands within this of the midline, mm, behind the
#: arch; nothing of the mandible does.
SPINE_HALF_WIDTH_MM = 25.0
SPINE_BEHIND_ARCH_MM = 15.0


#: The two rami reach to within this of the same height on any jaw that is
#: whole; a bigger difference means one was cut short.
RAMUS_MISMATCH_MM = 15.0


def _ramus_check(mandible: np.ndarray, above_plane: np.ndarray, X: np.ndarray, Y: np.ndarray,
                 points: np.ndarray):
    """How high each ramus reaches above the bite plane, and a warning if one
    falls well short of the other: the mandible should run condyle to condyle."""
    _, _, lateral = arch_axes(points)
    across = X * lateral[0] + np.broadcast_to(Y, X.shape) * lateral[1]
    on_arch = points[:, :2] @ lateral
    centre = float(np.mean(on_arch))
    half = 0.25 * float(np.ptp(on_arch))
    heights = []
    for side in (across < centre - half, across > centre + half):  # patient right, left
        region = mandible & side[None]
        heights.append(float(above_plane[region].max()) if region.any() else float("nan"))
    right, left = heights
    warnings = []
    if np.isfinite(right) and np.isfinite(left) and abs(right - left) > RAMUS_MISMATCH_MM:
        short = "right" if right < left else "left"
        warnings.append(
            f"The patient's {short} ramus reaches {abs(right - left):.0f} mm lower than the "
            "other and may have been cut short: check it, or draw the arch curve along the "
            "whole jaw and use Separate again along my arch curve."
        )
    return (right, left), warnings


def _pool(mask: np.ndarray, factor: int) -> np.ndarray:
    """Any-pooling over ``factor``³ blocks: a thin bridge stays a bridge."""
    pad = [(0, (-n) % factor) for n in mask.shape]
    padded = np.pad(mask, pad)
    nz, ny, nx = (n // factor for n in padded.shape)
    return padded.reshape(nz, factor, ny, factor, nx, factor).any(axis=(1, 3, 5))


#: Pitch of the watershed's grid, mm. Fine enough to follow the contact
#: between an upper and a lower incisor; on a coarser grid the labels come
#: back to the scan's voxels in visible blocks.
WATERSHED_MM = 0.6


def _split_at_narrowest(
    region: np.ndarray, markers: np.ndarray, gray: np.ndarray, threshold: float, spacing
) -> np.ndarray:
    """Marker watershed of ``region``: the split follows thin, dark bone.

    Two cues mark where the mandible meets the skull. The connection is
    narrow (a condyle in its fossa, a coronoid tip), and it is dark: the joint
    space is only bridged by voxels that straddle it and average down to
    near the threshold, while the condylar neck is solid cortex. At 0.5 mm
    the narrowness alone finds the joint; at the 1 mm of a medical CT the
    joint space is often bridged across its whole width and only the gray
    value still tells it from the neck. So the flooding height is the sum
    of the two, each scaled to 0-1.

    The flooding runs on a grid of about ``WATERSHED_MM``, inside the box
    round ``region``, and the labels are carried back to the full grid.
    Neighbours are face neighbours only: a contact one voxel across, corner
    to corner, is no path for either side to flood through.
    """
    if not region.any():
        return np.zeros(region.shape, dtype=np.uint8)
    kk, jj, ii = (np.flatnonzero(region.any(axis=axes)) for axes in ((1, 2), (0, 2), (0, 1)))
    box = tuple(
        slice(max(int(found[0]) - 1, 0), int(found[-1]) + 2) for found in (kk, jj, ii)
    )
    labels = np.zeros(region.shape, dtype=np.uint8)
    labels[box] = _flood(region[box], markers[box], gray[box], threshold, spacing)
    return labels


def _flood(region: np.ndarray, markers: np.ndarray, gray: np.ndarray, threshold: float, spacing):
    """The watershed of :func:`_split_at_narrowest`, on ``region``'s box."""
    import SimpleITK as sitk

    factor = max(int(round(WATERSHED_MM / float(np.min(spacing)))), 1)
    if factor > 1:
        coarse = _pool(region, factor)
        seeds = _pool(markers == 1, factor)
        skull = _pool(markers == 2, factor) & ~seeds
        coarse_markers = seeds.astype(np.uint8) + 2 * skull.astype(np.uint8)
        coarse_gray = _block_mean(gray, factor)
        coarse_spacing = [float(v) * factor for v in spacing]
    else:
        coarse, coarse_markers, coarse_gray = region, markers, gray
        coarse_spacing = [float(v) for v in spacing]

    image = sitk.GetImageFromArray(coarse.astype(np.uint8))
    image.SetSpacing(coarse_spacing)
    distance = sitk.GetArrayFromImage(
        sitk.SignedMaurerDistanceMap(
            image, insideIsPositive=True, squaredDistance=False, useImageSpacing=True
        )
    )
    thin = np.clip(distance / 3.0, 0.0, 1.0)
    inside = coarse_gray[coarse] if coarse.any() else coarse_gray.ravel()
    bright = float(np.percentile(inside, 90.0)) if inside.size else threshold + 1.0
    dense = np.clip((coarse_gray - threshold) / max(bright - threshold, 1e-6), 0.0, 1.0)
    height = sitk.GetImageFromArray((-(thin + dense)).astype(np.float32))
    height.SetSpacing(coarse_spacing)
    marker_image = sitk.GetImageFromArray(coarse_markers)
    marker_image.CopyInformation(height)
    labels = sitk.GetArrayFromImage(
        sitk.MorphologicalWatershedFromMarkers(
            height, marker_image, markWatershedLine=False, fullyConnected=False
        )
    )
    if factor > 1:
        for axis in range(3):
            labels = np.repeat(labels, factor, axis=axis)
        labels = labels[: region.shape[0], : region.shape[1], : region.shape[2]]
    return labels


def _block_mean(array: np.ndarray, factor: int) -> np.ndarray:
    pad = [(0, (-n) % factor) for n in array.shape]
    padded = np.pad(array.astype(np.float32), pad, mode="edge")
    nz, ny, nx = (n // factor for n in padded.shape)
    return padded.reshape(nz, factor, ny, factor, nx, factor).mean(axis=(1, 3, 5))


def _reference_heights(bite: Bite, points: np.ndarray) -> np.ndarray:
    """Bite height at every station, carried across stations without teeth."""
    if not bite.found:
        # No teeth: the alveolar crest is some 25 mm above a body-level curve.
        return points[:, 2] + 25.0
    stations = np.arange(len(points))
    known = bite.has_teeth
    return np.interp(stations, stations[known], bite.heights_mm[known])


def _tooth_row(teeth: np.ndarray, first: int, last: int, step_mm: float) -> np.ndarray:
    """The tooth row: the stations from the first with teeth to the last,
    gaps between teeth included, reaching at most ``TOOTH_ROW_BEYOND_MM``
    past the ends of the arch curve (stations ``first`` to ``last``).

    The arch curve is carried on straight past its ends, which takes it up
    the rami; there the bone of a ramus below and the maxilla above can look
    like a bite to the search, and an upper-teeth marker on a ramus would
    hand it to the skull.
    """
    beyond = int(round(TOOTH_ROW_BEYOND_MM / step_mm))
    allowed = np.zeros(len(teeth), dtype=bool)
    allowed[max(first - beyond, 0) : last + beyond + 1] = True
    row = np.zeros(len(teeth), dtype=bool)
    found = np.flatnonzero(teeth & allowed)
    if len(found):
        row[found[0] : found[-1] + 1] = True
    return row


def _bite_plane(bite: Bite, stations: np.ndarray, heights: np.ndarray):
    """The occlusal plane as a height function ``z(x, y)``.

    Fitted to the bite where there are teeth; with too few, the mean height
    of the reference line. A head scanned tilted tilts this plane with it.
    """
    if bite.found and int(bite.has_teeth.sum()) >= MIN_PLANE_STATIONS:
        # Robustly, and a plausible tilt only: a fit pulled steep by a few
        # stray stations would put the skull markers on one side's condyle.
        plane = robust_plane(
            bite.station_points[bite.has_teeth],
            bite.heights_mm[bite.has_teeth],
            bite.depth[bite.has_teeth],
        )
        if plane is not None:
            return plane
    level = float(np.mean(heights))
    return lambda x, y: np.full(np.broadcast(np.asarray(x), np.asarray(y)).shape, level)


def masked_bone_volume(volume, isolation: MandibleIsolation):
    """The scan with every bone voxel that is not mandible pushed below bone.

    Soft tissue next to the mandible keeps its gray value, so the mandible's
    isosurface is exactly where it was; only the other bone disappears. Where
    the mandible touched other bone (the cut contacts) the surface closes
    half a voxel from the cut.
    """
    from .volume import Volume

    array = volume.array.astype(np.float32, copy=True)
    background = float(np.min(array))
    array[isolation.other_bone] = background
    return Volume(
        array=array,
        spacing=volume.spacing.copy(),
        origin=volume.origin.copy(),
        resampled_from_mm=volume.resampled_from_mm,
    )
