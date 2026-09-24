"""Separate the mandible from the rest of the skull.

A real CBCT thresholded at bone is one connected mass more often than not:
with the teeth in occlusion the lower crowns touch the upper ones, and at
the voxel sizes of a planning scan the coronoid process or the condyle can
touch the skull as well. Everything downstream of that is wrong: a cutting
plane is infinite, so a resection would take a slice of maxilla and skull
with it; the mirror would copy the maxilla into the defect; the exported
"jaw" would be the whole head.

The separation uses the arch curve the operator has already drawn, and
three steps, each only when the one before has not finished the job:

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
#: The rami continue behind the last station of the arch curve by about
#: this much, mm.
RAMUS_EXTENSION_MM = 35.0
#: Bone this far above the highest bite is skull, never mandible, mm.
SKULL_ABOVE_BITE_MM = 45.0


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

    @property
    def separated(self) -> bool:
        return bool(self.other_bone.any())

    def summary(self) -> str:
        if not self.steps:
            return (
                f"The mandible is {self.volume_mm3 / 1000:.1f} cm³ of bone and is not "
                "attached to any other bone in the scan."
            )
        return (
            f"Mandible separated from the rest of the skull {' and '.join(self.steps)} "
            f"({self.volume_mm3 / 1000:.1f} cm³ of bone)."
        )


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
    """The occlusal gap along a polyline with horizontal across-directions."""
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
    # percent of the band at most and would make every crown look dim.
    low = float(np.percentile(image, 1.0))
    high = float(np.percentile(image, 97.0))
    image = (image - low) / max(high - low, 1e-6)

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

    # Smooth, darkest path across the stations: at most one row per station.
    cost = -depth
    n, m = cost.shape
    penalty = np.array([[0.02], [0.0], [0.02]])
    total = cost[0].copy()
    step = np.zeros((n, m), dtype=np.int8)
    for i in range(1, n):
        options = np.stack([np.r_[np.inf, total[:-1]], total, np.r_[total[1:], np.inf]])
        choice = np.argmin(options + penalty, axis=0)
        step[i] = choice - 1
        total = options[choice, np.arange(m)] + cost[i]
    path = np.empty(n, dtype=int)
    path[-1] = int(np.argmin(total))
    for i in range(n - 1, 0, -1):
        path[i - 1] = path[i] + step[i, path[i]]

    rows = np.arange(n)
    gap_depth = depth[rows, path]
    has_teeth = (crowns[rows, path] >= CROWN_LEVEL) & (gap_depth >= MIN_GAP_DEPTH)
    return Bite(
        station_points=points,
        heights_mm=points[:, 2] + rise[path],
        has_teeth=has_teeth,
        depth=gap_depth,
    )


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
    skull_level = float(np.max(ref[n_head : n_head + len(points)])) + SKULL_ABOVE_BITE_MM

    # Work inside a box around the arch: the whole mandible, and just enough
    # skull above it to recognise.
    lo = points.min(axis=0) - np.array([55.0, 55.0, 0.0])
    hi = points.max(axis=0) + np.array([55.0, 55.0, 0.0])
    lo[2] = origin[2]
    hi[2] = skull_level + 8.0
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
    arch_interior = _interior_stations(len(extended), n_head, len(points), step)

    def skull_markers(mask: np.ndarray) -> np.ndarray:
        high = zs[:, None, None] > skull_level
        upper_arch = (
            (distance < 8.0)[None]
            & arch_interior[nearest][None]
            & (height_above > 3.0)
            & (height_above < 12.0)
        )
        return mask & (high | (upper_arch & bite.found))

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
            steps.append("at the bite")
        freed = crop & ~cut
        labels = _components(freed)
        target = _label_at(labels, station_index, radius)
        mandible = labels == target if target else connected
        del labels
        skull = skull_markers(mandible)
        if skull.any():
            seeds = mandible & (distance < 15.0)[None] & (height_above < -2.0)
            markers = seeds.astype(np.uint8) + 2 * (skull & ~seeds).astype(np.uint8)
            split = _split_at_narrowest(mandible, markers, spacing)
            region = (split == 1) & mandible
            labels = _components(region)
            target = _label_at(labels, station_index, radius)
            if target:
                mandible = labels == target
            steps.append("at the jaw joints")

    mask = np.zeros(full_shape, dtype=bool)
    mask[box] = mandible
    other = bone & ~mask
    return MandibleIsolation(
        mask=mask,
        other_bone=other,
        bite=bite if bite.found else None,
        steps=steps,
        volume_mm3=float(mask.sum() * np.prod(spacing)),
    )


def _pool(mask: np.ndarray, factor: int) -> np.ndarray:
    """Any-pooling over ``factor``³ blocks: a thin bridge stays a bridge."""
    pad = [(0, (-n) % factor) for n in mask.shape]
    padded = np.pad(mask, pad)
    nz, ny, nx = (n // factor for n in padded.shape)
    return padded.reshape(nz, factor, ny, factor, nx, factor).any(axis=(1, 3, 5))


def _split_at_narrowest(region: np.ndarray, markers: np.ndarray, spacing) -> np.ndarray:
    """Marker watershed of ``region`` on its inverted distance map.

    Where two marked parts meet through bone, the boundary settles on the
    narrowest connection. The flooding is done on a grid of about 1 mm —
    the split only has to find a joint space, and at 0.5 mm it costs ten
    times as long — and the labels are carried back to the full grid.
    """
    import SimpleITK as sitk

    factor = max(int(round(1.0 / float(np.min(spacing)))), 1)
    if factor > 1:
        coarse = _pool(region, factor)
        seeds = _pool(markers == 1, factor)
        skull = _pool(markers == 2, factor) & ~seeds
        coarse_markers = seeds.astype(np.uint8) + 2 * skull.astype(np.uint8)
        coarse_spacing = [float(v) * factor for v in spacing]
    else:
        coarse, coarse_markers = region, markers
        coarse_spacing = [float(v) for v in spacing]

    image = sitk.GetImageFromArray(coarse.astype(np.uint8))
    image.SetSpacing(coarse_spacing)
    distance_map = sitk.SignedMaurerDistanceMap(
        image, insideIsPositive=True, squaredDistance=False, useImageSpacing=True
    )
    marker_image = sitk.GetImageFromArray(coarse_markers)
    marker_image.CopyInformation(image)
    labels = sitk.GetArrayFromImage(
        sitk.MorphologicalWatershedFromMarkers(
            sitk.InvertIntensity(distance_map, maximum=0.0),
            marker_image,
            markWatershedLine=False,
            fullyConnected=True,
        )
    )
    if factor > 1:
        for axis in range(3):
            labels = np.repeat(labels, factor, axis=axis)
        labels = labels[: region.shape[0], : region.shape[1], : region.shape[2]]
    return labels


def _reference_heights(bite: Bite, points: np.ndarray) -> np.ndarray:
    """Bite height at every station, carried across stations without teeth."""
    if not bite.found:
        # No teeth: the alveolar crest is some 25 mm above a body-level curve.
        return points[:, 2] + 25.0
    stations = np.arange(len(points))
    known = bite.has_teeth
    return np.interp(stations, stations[known], bite.heights_mm[known])


def _interior_stations(total: int, n_head: int, n_arch: int, step_mm: float) -> np.ndarray:
    """Arch stations away from both ends, where no ramus rises beside the teeth."""
    margin = int(round(10.0 / step_mm))
    interior = np.zeros(total, dtype=bool)
    interior[n_head + margin : n_head + n_arch - margin] = True
    return interior


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
    return Volume(array=array, spacing=volume.spacing.copy(), origin=volume.origin.copy())
