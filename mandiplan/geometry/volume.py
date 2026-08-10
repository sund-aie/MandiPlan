"""Axis-aligned scalar volume with real-world millimetre geometry.

Coordinate conventions used throughout MandiPlan
------------------------------------------------
World space is the DICOM patient coordinate system (LPS), in millimetres:

    +x -> patient Left, +y -> patient Posterior, +z -> patient Superior

A :class:`Volume` is always axis-aligned to that frame (the DICOM reader
re-orients the series and refuses anything it cannot align, see
``mandiplan.dicom_io``).  ``array`` is indexed ``array[k, j, i]`` where the
index axes ``(i, j, k)`` map to world ``(x, y, z)``.  The world position of
voxel ``(i, j, k)`` is::

    world = origin + (i, j, k) * spacing

``spacing`` is per-axis and is never assumed isotropic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Volume:
    """A scalar volume plus the geometry needed to measure it in millimetres."""

    array: np.ndarray  # (nz, ny, nx)
    spacing: np.ndarray  # (sx, sy, sz) in mm
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.array = np.asarray(self.array)
        if self.array.ndim != 3:
            raise ValueError("volume array must be 3-D (nz, ny, nx)")
        self.spacing = np.asarray(self.spacing, dtype=float).reshape(3)
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)
        if np.any(self.spacing <= 0):
            raise ValueError(f"voxel spacing must be positive, got {self.spacing}")

    # -- shape / extent ---------------------------------------------------

    @property
    def size_xyz(self) -> np.ndarray:
        """Number of voxels along world x, y, z."""
        nz, ny, nx = self.array.shape
        return np.array([nx, ny, nz], dtype=int)

    @property
    def extent_mm(self) -> np.ndarray:
        """Physical size along world x, y, z, measured centre-to-centre."""
        return (self.size_xyz - 1) * self.spacing

    @property
    def bounds_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """(min_xyz, max_xyz) of the voxel-centre bounding box, in mm."""
        return self.origin.copy(), self.origin + self.extent_mm

    @property
    def value_range(self) -> tuple[float, float]:
        return float(self.array.min()), float(self.array.max())

    # -- index <-> world --------------------------------------------------

    def index_to_world(self, index_ijk) -> np.ndarray:
        idx = np.asarray(index_ijk, dtype=float)
        return self.origin + idx * self.spacing

    def world_to_index(self, points_xyz) -> np.ndarray:
        """Continuous (fractional) voxel index (i, j, k) of world points."""
        pts = np.asarray(points_xyz, dtype=float)
        return (pts - self.origin) / self.spacing

    # -- sampling ---------------------------------------------------------

    def sample(self, points_xyz, fill: float | None = None) -> np.ndarray:
        """Trilinearly interpolate the volume at arbitrary world points.

        ``points_xyz`` has shape ``(..., 3)``; the result has shape ``(...)``.
        Points outside the volume take ``fill`` (default: the volume minimum).
        """
        pts = np.asarray(points_xyz, dtype=float)
        out_shape = pts.shape[:-1]
        idx = self.world_to_index(pts.reshape(-1, 3))
        nx, ny, nz = (int(v) for v in self.size_xyz)
        n = np.array([nx, ny, nz])

        inside = np.all((idx >= 0) & (idx <= n - 1), axis=1)
        base = np.floor(idx).astype(np.int64)
        frac = idx - base
        # Clamp so the +1 neighbour always exists; out-of-range samples are
        # masked out afterwards anyway.
        base = np.clip(base, 0, np.maximum(n - 2, 0))
        frac = np.clip(idx - base, 0.0, 1.0)

        i0, j0, k0 = base[:, 0], base[:, 1], base[:, 2]
        i1 = np.minimum(i0 + 1, nx - 1)
        j1 = np.minimum(j0 + 1, ny - 1)
        k1 = np.minimum(k0 + 1, nz - 1)
        fi, fj, fk = frac[:, 0], frac[:, 1], frac[:, 2]

        a = self.array
        c000 = a[k0, j0, i0]
        c100 = a[k0, j0, i1]
        c010 = a[k0, j1, i0]
        c110 = a[k0, j1, i1]
        c001 = a[k1, j0, i0]
        c101 = a[k1, j0, i1]
        c011 = a[k1, j1, i0]
        c111 = a[k1, j1, i1]

        c00 = c000 * (1 - fi) + c100 * fi
        c10 = c010 * (1 - fi) + c110 * fi
        c01 = c001 * (1 - fi) + c101 * fi
        c11 = c011 * (1 - fi) + c111 * fi
        c0 = c00 * (1 - fj) + c10 * fj
        c1 = c01 * (1 - fj) + c11 * fj
        values = c0 * (1 - fk) + c1 * fk

        if fill is None:
            fill = float(self.array.min())
        values = np.where(inside, values, fill)
        return values.reshape(out_shape)

    # -- orthogonal slices -------------------------------------------------

    def orthogonal_slice(self, axis: int, index: int) -> np.ndarray:
        """2-D slice perpendicular to world ``axis``.

        Rows and columns follow the two remaining world axes in ascending
        order::

            axis=0 (sagittal): rows = z, cols = y
            axis=1 (coronal):  rows = z, cols = x
            axis=2 (axial):    rows = y, cols = x
        """
        index = int(np.clip(index, 0, self.size_xyz[axis] - 1))
        if axis == 2:
            return self.array[index, :, :]
        if axis == 1:
            return self.array[:, index, :]
        return self.array[:, :, index]

    # -- statistics --------------------------------------------------------

    def histogram(self, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.value_range
        if hi <= lo:
            hi = lo + 1.0
        counts, edges = np.histogram(self.array, bins=bins, range=(lo, hi))
        return counts, edges
