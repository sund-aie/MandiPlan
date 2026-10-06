"""Curved planar reformation (CPR).

The arch curve is a planar curve in the axial (x, y) plane.  Sampling it at
uniform arc length gives, at every station ``i``:

    t_i  unit tangent (direction of travel along the arch)
    u_i  patient superior, parallel-transported along the curve
    n_i  = t_i x u_i                -- the buccolingual direction

For a curve lying in the axial plane ``u_i`` stays exactly ``z_hat`` and
``n_i`` is the ``t_i x z_hat`` this module used before the frame was
transported, so the reformat is unchanged. Transport matters where the curve
leaves the axial plane, which is where a fixed global up-axis degenerates.

The panoramic reformat is what an OPG shows: a vertical sheet standing on the
curve's footprint in the axial plane, unrolled flat. Its x-axis is arc length
along that footprint (mm) and its y-axis world ``z`` (mm), so *both* axes are
millimetres and measurement in that view is 1:1. For a curve drawn on one
axial slice the footprint is the curve itself and x is its arc length ``s``.
A curve that climbs the rami to the condyles stands almost still in the
axial plane while it climbs, so each ramus rises in the panoramic as it does
on an OPG instead of being smeared sideways; :func:`sheet_x` and
:func:`s_on_sheet` convert between ``s`` and the sheet.

The cross-section at ``s`` is the plane through the curve perpendicular to
it, spanned by the buccolingual axis and the transported up-axis; on the
body that is the vertical buccolingual section, on a ramus a horizontal one.
Nothing in this module ever returns a pixel count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spline import ArchCurve

SUPERIOR = np.array([0.0, 0.0, 1.0])

#: Aggregation modes for the panoramic slab.
AGGREGATION_MODES = ("max", "mean")


@dataclass
class ArchFrames:
    """Arc-length-uniform stations along the arch curve, with local frames.

    The local mandibular frame at each station is a right-handed triad:

    ``tangents`` (T)
        Unit direction of travel along the arch, toward increasing arc length.
    ``ups`` (U)
        The anatomical superior reference, parallel-transported along the
        curve from patient superior at ``s = 0`` and kept perpendicular to T.
        Transport is what stops the frame from spinning or flipping where the
        curve leaves the axial plane and climbs the ramus.
    ``normals`` (B)
        Buccolingual, ``T x U``. Named ``normals`` for the panoramic reformat,
        which has used it as the slab direction since before the frame was a
        full triad.

    For a curve lying in an axial plane every rotation between consecutive
    tangents is about the superior axis, so transporting patient-superior
    returns patient-superior exactly and ``normals`` is identical to the
    ``T x z`` this class used to compute. Nothing in the reformat changes.
    """

    s: np.ndarray  # (M,) cumulative arc length, mm
    points: np.ndarray  # (M, 3) world mm
    tangents: np.ndarray  # (M, 3) unit, direction of travel
    normals: np.ndarray  # (M, 3) unit, buccolingual (T x U)
    step_mm: float
    ups: np.ndarray | None = None  # (M, 3) unit, transported superior

    @property
    def length_mm(self) -> float:
        return float(self.s[-1])

    def index_of(self, s_mm: float) -> int:
        return int(np.clip(round(float(s_mm) / self.step_mm), 0, len(self.s) - 1))

    def clamp(self, s_mm: float) -> float:
        """``s_mm`` brought inside the curve's valid range."""
        return float(np.clip(float(s_mm), 0.0, self.length_mm))

    def frame_at(self, s_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The local mandibular triad ``(T, U, B)`` at an arc position."""
        i = self.index_of(s_mm)
        tangent = self.tangents[i]
        up = SUPERIOR if self.ups is None else self.ups[i]
        # Re-orthogonalise defensively: callers may have built frames by hand.
        up = up - float(np.dot(up, tangent)) * tangent
        norm = np.linalg.norm(up)
        if norm < 1e-9:
            raise ValueError(f"degenerate arch frame at s = {s_mm:.2f} mm")
        up = up / norm
        return tangent, up, np.cross(tangent, up)

    def point_at(self, s_mm: float) -> np.ndarray:
        """The point on the arch curve at an arc position, in world mm."""
        return self.points[self.index_of(s_mm)].copy()


@dataclass
class Reformat:
    """A resampled 2-D image whose both axes are millimetres.

    ``image[row, col]``; column ``c`` is at ``x0 + c * pixel_mm`` and row ``r``
    is at ``y0 + r * pixel_mm`` in the coordinates named by the labels.
    ``row_axis_up`` says the row coordinate increases superiorly and should be
    drawn bottom-up.
    """

    image: np.ndarray
    pixel_mm: float
    x0: float
    y0: float
    x_label: str
    y_label: str
    row_axis_up: bool = True

    @property
    def width_mm(self) -> float:
        return (self.image.shape[1] - 1) * self.pixel_mm

    @property
    def height_mm(self) -> float:
        return (self.image.shape[0] - 1) * self.pixel_mm

    def col_to_mm(self, col: float) -> float:
        return self.x0 + col * self.pixel_mm

    def row_to_mm(self, row: float) -> float:
        return self.y0 + row * self.pixel_mm

    def mm_to_col(self, x_mm: float) -> float:
        return (x_mm - self.x0) / self.pixel_mm

    def mm_to_row(self, y_mm: float) -> float:
        return (y_mm - self.y0) / self.pixel_mm


def parallel_transport(tangents: np.ndarray, up=SUPERIOR, start: int = 0) -> np.ndarray:
    """Carry an up-vector along a curve without letting it spin about it.

    At each step the frame is rotated by exactly the rotation that takes the
    previous tangent onto the current one — the minimal rotation, so no twist
    is added. Building the up-axis from a fixed global reference instead
    (``t x z`` and friends) degenerates wherever the curve runs parallel to
    that reference, which on a mandible is the ramus.

    ``up`` is taken as the up-vector at station ``start`` and carried both
    ways from there. Starting where the curve is level (the body, for a curve
    from condyle to condyle) makes the frame there exactly superior; started
    at a condyle, the turn round the angle would leave it tilted.
    """
    tangents = np.asarray(tangents, dtype=float).reshape(-1, 3)
    if start:
        ahead = parallel_transport(tangents[start:], up)
        behind = parallel_transport(tangents[start::-1], up)
        return np.vstack([behind[:0:-1], ahead])
    up = np.asarray(up, dtype=float).reshape(3)

    seed = up - float(np.dot(up, tangents[0])) * tangents[0]
    if np.linalg.norm(seed) < 1e-9:
        # The curve starts along the reference direction; any perpendicular
        # will do as a seed, and transport keeps it consistent from there.
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(helper, tangents[0]))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        seed = helper - float(np.dot(helper, tangents[0])) * tangents[0]
    ups = np.empty_like(tangents)
    ups[0] = seed / np.linalg.norm(seed)

    for i in range(1, len(tangents)):
        previous, current = tangents[i - 1], tangents[i]
        axis = np.cross(previous, current)
        sine = float(np.linalg.norm(axis))
        carried = ups[i - 1]
        if sine > 1e-12:
            axis = axis / sine
            angle = float(np.arctan2(sine, float(np.dot(previous, current))))
            cos, sin = np.cos(angle), np.sin(angle)
            carried = (
                carried * cos
                + np.cross(axis, carried) * sin
                + axis * float(np.dot(axis, carried)) * (1.0 - cos)
            )
        # Rounding accumulates over thousands of stations; re-project.
        carried = carried - float(np.dot(carried, current)) * current
        ups[i] = carried / np.linalg.norm(carried)
    return ups


def build_frames(curve: ArchCurve, step_mm: float, up=SUPERIOR) -> ArchFrames:
    """Resample ``curve`` at uniform arc length and build local frames.

    The frame is transported rather than derived from a global axis, so a
    curve that climbs out of the axial plane into the angle and ramus keeps a
    continuous, non-flipping frame instead of being refused.
    """
    samples = curve.resample(step_mm)
    ups = parallel_transport(samples.tangents, up, start=_level_station(samples.tangents))
    normals = np.cross(samples.tangents, ups)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(norm < 1e-9):
        raise ValueError("arch curve has a degenerate tangent; check the seed points")
    return ArchFrames(
        s=samples.s,
        points=samples.points,
        tangents=samples.tangents,
        normals=normals / norm,
        step_mm=samples.step_mm,
        ups=ups,
    )


#: How far above its lowest point the curve may rise and still be on the
#: mandibular body rather than climbing a ramus, mm.
BODY_RISE_MM = 10.0


def body_span(frames: ArchFrames, rise_mm: float = BODY_RISE_MM) -> tuple[float, float]:
    """Arc positions where the curve runs along the body: ``(start, end)``.

    A curve drawn on one axial slice is body all along; one that climbs the
    rami to the condyles is body only between the angles.
    """
    z = frames.points[:, 2]
    low = np.flatnonzero(z <= z.min() + rise_mm)
    return float(frames.s[low[0]]), float(frames.s[low[-1]])


def _level_station(tangents: np.ndarray) -> int:
    """The station the frame is anchored at: the middle, if the curve is level
    there, else the most level one."""
    middle = len(tangents) // 2
    if abs(float(tangents[middle, 2])) < 0.5:
        return middle
    return int(np.argmin(np.abs(tangents[:, 2])))


def image_pixel_mm(volume, frames: ArchFrames) -> float:
    """Pixel size of the reformatted images, mm.

    The arch is stepped finely for the geometry, but an image sampled far
    finer than the scan's own voxels shows nothing more and costs the square
    of the difference: a 0.2 mm panoramic of a 0.5 mm scan is six times the
    work, several seconds a click on a real scan. So the images are sampled at
    the voxel size, or the arch step if that is coarser.
    """
    return float(max(frames.step_mm, float(np.min(volume.spacing))))


#: Window over which the footprint's direction is taken, mm: the footprint
#: of a climbing ramus moves a few millimetres while the curve rises tens.
FOOTPRINT_WINDOW_MM = 3.0


def sheet_x(frames: ArchFrames) -> np.ndarray:
    """Panoramic x of every station: arc length along the curve's footprint."""
    step = np.linalg.norm(np.diff(frames.points[:, :2], axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def sheet_point(frames: ArchFrames, s_mm: float, z_mm: float | None = None):
    """Panoramic ``(x, y)`` of arc position ``s_mm`` (at height ``z_mm``)."""
    i = frames.index_of(s_mm)
    z = frames.points[i, 2] if z_mm is None else z_mm
    return np.array([sheet_x(frames)[i], float(z)])


def s_on_sheet(frames: ArchFrames, x_mm: float, y_mm: float) -> float:
    """Arc position of the curve point nearest a point picked on the sheet."""
    d = np.hypot(sheet_x(frames) - float(x_mm), frames.points[:, 2] - float(y_mm))
    return float(frames.s[int(np.argmin(d))])


def sheet_line(frames: ArchFrames, s_mm: float, origin, normal, half_mm: float):
    """Where a plane through the curve at ``s_mm`` crosses the panoramic sheet.

    Returned as the two ends of a segment ``half_mm`` either side of
    ``origin``, in panoramic ``(x, y)``. A cut across the body shows as a
    near-vertical line tilted as the cut is; one across a ramus as a near-
    horizontal one.
    """
    x = sheet_x(frames)
    i = frames.index_of(s_mm)
    along = _footprint_direction(frames, x, x[i])
    normal = np.asarray(normal, dtype=float)
    direction = np.array([float(normal[2]), -float(np.dot(normal[:2], along))])
    length = np.linalg.norm(direction)
    direction = np.array([0.0, 1.0]) if length < 1e-9 else direction / length
    centre = np.array([x[i], float(np.asarray(origin, dtype=float)[2])])
    return np.vstack([centre - half_mm * direction, centre + half_mm * direction])


def _footprint_direction(frames: ArchFrames, x: np.ndarray, at) -> np.ndarray:
    """Unit direction of the footprint at sheet position(s) ``at``."""
    at = np.atleast_1d(np.asarray(at, dtype=float))
    h = FOOTPRINT_WINDOW_MM / 2.0
    ahead = np.column_stack([np.interp(at + h, x, frames.points[:, k]) for k in range(2)])
    behind = np.column_stack([np.interp(at - h, x, frames.points[:, k]) for k in range(2)])
    d = ahead - behind
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return d[0] if d.shape[0] == 1 else d


def _z_grid(volume, z_min, z_max, pixel_mm):
    lo, hi = volume.bounds_mm
    z_min = float(lo[2]) if z_min is None else float(z_min)
    z_max = float(hi[2]) if z_max is None else float(z_max)
    rows = max(int(round((z_max - z_min) / pixel_mm)) + 1, 2)
    return z_min, z_min + np.arange(rows) * pixel_mm


def build_panoramic(
    volume,
    frames: ArchFrames,
    slab_mm: float = 10.0,
    mode: str = "max",
    z_min: float | None = None,
    z_max: float | None = None,
) -> Reformat:
    """Flatten the volume along the arch curve into a panoramic image.

    Each output pixel aggregates the volume over a slab of ``slab_mm``
    centred on the sheet standing on the curve's footprint and running along
    +/- its horizontal normal. ``mode`` is ``"max"`` (maximum intensity,
    crisper cortical outline) or ``"mean"`` (ray-sum, looks like an OPG).
    """
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"aggregation mode must be one of {AGGREGATION_MODES}")
    pixel_mm = image_pixel_mm(volume, frames)
    z0, z = _z_grid(volume, z_min, z_max, pixel_mm)
    x = sheet_x(frames)
    s = np.arange(0.0, x[-1] + 1e-9, pixel_mm)
    points = np.column_stack([np.interp(s, x, frames.points[:, a]) for a in range(2)])
    along = _footprint_direction(frames, x, s).reshape(-1, 2)
    normals = np.column_stack([along[:, 1], -along[:, 0]])

    n_off = max(int(round(slab_mm / pixel_mm)) + 1, 2)
    offsets = np.linspace(-slab_mm / 2.0, slab_mm / 2.0, n_off)

    rows, cols = len(z), len(s)
    fill = float(volume.array.min())
    acc = np.full((rows, cols), -np.inf if mode == "max" else 0.0)

    pts = np.empty((rows, cols, 3))
    pts[:, :, 2] = z[:, None]
    for d in offsets:
        xy = points[:, :2] + d * normals[:, :2]
        pts[:, :, 0] = xy[None, :, 0]
        pts[:, :, 1] = xy[None, :, 1]
        values = volume.sample(pts, fill=fill)
        if mode == "max":
            np.maximum(acc, values, out=acc)
        else:
            acc += values
    if mode == "mean":
        acc /= len(offsets)

    return Reformat(
        image=acc,
        pixel_mm=pixel_mm,
        x0=0.0,
        y0=z0,
        x_label="arc length along the arch (mm)",
        y_label="superior-inferior (mm)",
    )


#: The cross-section always reaches this far either side of the curve along
#: its up-axis, mm, even where the curve is near the top of the scan.
CROSS_HALF_HEIGHT_MM = 40.0


def _cross_axes(frames: ArchFrames, s_mm: float):
    i = frames.index_of(s_mm)
    up = SUPERIOR if frames.ups is None else frames.ups[i]
    return frames.points[i], frames.normals[i], up


def build_cross_section(
    volume,
    frames: ArchFrames,
    s_mm: float,
    width_mm: float = 40.0,
    z_min: float | None = None,
    z_max: float | None = None,
) -> Reformat:
    """Cross-section of the volume perpendicular to the curve at ``s_mm``.

    The plane is spanned by the buccolingual normal ``n_i`` and the
    transported up-axis ``u_i``. The x-axis of the result is the signed
    offset along ``+n_i``; the y-axis is the curve point's height plus the
    offset along ``u_i``, which on the body, where ``u_i`` is superior, is
    world ``z``.
    """
    pixel_mm = image_pixel_mm(volume, frames)
    p, n, up = _cross_axes(frames, s_mm)

    lo, hi = volume.bounds_mm
    if z_min is None:
        z_min = min(float(lo[2]), float(p[2]) - CROSS_HALF_HEIGHT_MM)
    if z_max is None:
        z_max = max(float(hi[2]), float(p[2]) + CROSS_HALF_HEIGHT_MM)
    z0, z = _z_grid(volume, z_min, z_max, pixel_mm)
    cols = max(int(round(width_mm / pixel_mm)) + 1, 2)
    u = (np.arange(cols) - (cols - 1) / 2.0) * pixel_mm
    v = z - float(p[2])

    pts = p + u[None, :, None] * n + v[:, None, None] * up
    image = volume.sample(pts, fill=float(volume.array.min()))

    return Reformat(
        image=image,
        pixel_mm=pixel_mm,
        x0=float(u[0]),
        y0=z0,
        x_label="buccolingual offset (mm)",
        # Up the ramus the section is near-horizontal: its second axis runs
        # front to back, not up and down.
        y_label="superior-inferior (mm)" if abs(float(up[2])) > 0.9 else "across the section (mm)",
    )


def cross_section_world_point(frames: ArchFrames, s_mm: float, u_mm: float, z_mm: float):
    """World point of a location picked in a cross-section image."""
    p, n, up = _cross_axes(frames, s_mm)
    return p + u_mm * n + (z_mm - float(p[2])) * up
