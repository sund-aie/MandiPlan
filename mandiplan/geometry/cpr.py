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

The panoramic reformat has arc length ``s`` (mm) on its x-axis and world
``z`` (mm) on its y-axis, so *both* axes are millimetres and measurement in
that view is 1:1.  Nothing in this module ever returns a pixel count.
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


def parallel_transport(tangents: np.ndarray, up=SUPERIOR) -> np.ndarray:
    """Carry an up-vector along a curve without letting it spin about it.

    At each step the frame is rotated by exactly the rotation that takes the
    previous tangent onto the current one — the minimal rotation, so no twist
    is added. Building the up-axis from a fixed global reference instead
    (``t x z`` and friends) degenerates wherever the curve runs parallel to
    that reference, which on a mandible is the ramus.
    """
    tangents = np.asarray(tangents, dtype=float).reshape(-1, 3)
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
    ups = parallel_transport(samples.tangents, up)
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
    centred on the curve and running along +/- the buccolingual normal.
    ``mode`` is ``"max"`` (maximum intensity, crisper cortical outline) or
    ``"mean"`` (ray-sum, looks like an OPG).
    """
    if mode not in AGGREGATION_MODES:
        raise ValueError(f"aggregation mode must be one of {AGGREGATION_MODES}")
    pixel_mm = frames.step_mm
    z0, z = _z_grid(volume, z_min, z_max, pixel_mm)

    n_off = max(int(round(slab_mm / pixel_mm)) + 1, 2)
    offsets = np.linspace(-slab_mm / 2.0, slab_mm / 2.0, n_off)

    rows, cols = len(z), len(frames.s)
    fill = float(volume.array.min())
    acc = np.full((rows, cols), -np.inf if mode == "max" else 0.0)

    pts = np.empty((rows, cols, 3))
    pts[:, :, 2] = z[:, None]
    for d in offsets:
        xy = frames.points[:, :2] + d * frames.normals[:, :2]
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
        x_label="arc length along arch curve (mm)",
        y_label="superior-inferior (mm)",
    )


def build_cross_section(
    volume,
    frames: ArchFrames,
    s_mm: float,
    width_mm: float = 40.0,
    z_min: float | None = None,
    z_max: float | None = None,
) -> Reformat:
    """Buccolingual cross-section of the volume at arc position ``s_mm``.

    The plane is spanned by the buccolingual normal ``n_i`` and the superior
    axis.  The x-axis of the result is the signed offset along ``+n_i``.
    """
    pixel_mm = frames.step_mm
    i = frames.index_of(s_mm)
    p = frames.points[i]
    n = frames.normals[i]

    z0, z = _z_grid(volume, z_min, z_max, pixel_mm)
    cols = max(int(round(width_mm / pixel_mm)) + 1, 2)
    u = (np.arange(cols) - (cols - 1) / 2.0) * pixel_mm

    pts = np.empty((len(z), cols, 3))
    pts[:, :, 0] = p[0] + u[None, :] * n[0]
    pts[:, :, 1] = p[1] + u[None, :] * n[1]
    pts[:, :, 2] = z[:, None]
    image = volume.sample(pts, fill=float(volume.array.min()))

    return Reformat(
        image=image,
        pixel_mm=pixel_mm,
        x0=float(u[0]),
        y0=z0,
        x_label="buccolingual offset (mm)",
        y_label="superior-inferior (mm)",
    )


def cross_section_world_point(frames: ArchFrames, s_mm: float, u_mm: float, z_mm: float):
    """World point of a location picked in a cross-section image."""
    i = frames.index_of(s_mm)
    p = frames.points[i]
    n = frames.normals[i]
    return np.array([p[0] + u_mm * n[0], p[1] + u_mm * n[1], z_mm])
