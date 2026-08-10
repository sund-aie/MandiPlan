"""Curved planar reformation (CPR).

The arch curve is a planar curve in the axial (x, y) plane.  Sampling it at
uniform arc length gives, at every station ``i``:

    t_i  unit tangent (direction of travel along the arch)
    n_i  = normalize(t_i x z_hat)   -- the buccolingual direction
    z_hat = (0, 0, 1)               -- the superior axis

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
    """Arc-length-uniform stations along the arch curve, with local frames."""

    s: np.ndarray  # (M,) cumulative arc length, mm
    points: np.ndarray  # (M, 3) world mm
    tangents: np.ndarray  # (M, 3) unit
    normals: np.ndarray  # (M, 3) unit, buccolingual
    step_mm: float

    @property
    def length_mm(self) -> float:
        return float(self.s[-1])

    def index_of(self, s_mm: float) -> int:
        return int(np.clip(round(float(s_mm) / self.step_mm), 0, len(self.s) - 1))


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


def build_frames(curve: ArchCurve, step_mm: float, up=SUPERIOR) -> ArchFrames:
    """Resample ``curve`` at uniform arc length and build buccolingual frames."""
    samples = curve.resample(step_mm)
    up = np.asarray(up, dtype=float)
    normals = np.cross(samples.tangents, up)
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(norm < 1e-9):
        raise ValueError(
            "arch curve is locally parallel to the superior axis; "
            "seed points must lie in an axial plane"
        )
    normals = normals / norm
    return ArchFrames(
        s=samples.s,
        points=samples.points,
        tangents=samples.tangents,
        normals=normals,
        step_mm=samples.step_mm,
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


def panoramic_world_point(frames: ArchFrames, s_mm: float, z_mm: float):
    """World point on the arch curve for a location picked in the panoramic view.

    The buccolingual coordinate is collapsed by the reformat, so the point
    returned lies on the curve itself.
    """
    i = frames.index_of(s_mm)
    p = frames.points[i]
    return np.array([p[0], p[1], z_mm])
