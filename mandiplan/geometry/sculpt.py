"""Small hand edits to the reconstructed jaw, on top of what was computed.

The reconstruction is computed, and is flush by construction; this is for
the places where the operator still wants a rounder junction, a dent filled
or a ridge taken down. Three brushes act on the vertices within a radius of
the point under the cursor, with a smooth falloff to nothing at the edge:

``smooth``
    Taubin smoothing (a shrinking Laplacian step, then an inflating one), so
    the region evens out without the whole patch sinking, which is what
    plain Laplacian smoothing does to a convex bone surface.
``fill``
    Pushes the patch out along its mean normal: adds bone.
``carve``
    Pushes it in: takes bone away.

Only vertex positions change, never the triangles, so a closed surface stays
closed. Every stroke can be undone, and the computed surface can always be
restored. ``smooth_near_planes`` is the same smoothing applied by the
computer in a band around each cut, for the junctions.

Numpy only.
"""

from __future__ import annotations

import numpy as np

BRUSHES = ("smooth", "fill", "carve")
#: Taubin's pass-band parameters: shrink by lambda, inflate by mu.
TAUBIN_LAMBDA = 0.5
TAUBIN_MU = -0.53
#: How far one fill or carve stroke moves the surface at full strength, mm.
PUSH_PER_STROKE_MM = 0.25
#: Taubin passes in one dab of the smooth brush.
SMOOTH_PASSES = 3
#: Strokes kept for undo.
UNDO_DEPTH = 30


class SurfaceSculptor:
    """Brush edits on a closed triangle mesh, with undo."""

    def __init__(self, points: np.ndarray, triangles: np.ndarray):
        self.points = np.array(points, dtype=float, copy=True)
        self.triangles = np.asarray(triangles, dtype=np.int64)
        self.original = self.points.copy()
        self._undo: list[np.ndarray] = []
        self._neighbours, self._offsets = _adjacency(len(self.points), self.triangles)

    # -- state -------------------------------------------------------------

    @property
    def edited(self) -> bool:
        return not np.array_equal(self.points, self.original)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    def checkpoint(self) -> None:
        """Remember the surface before a stroke, for undo."""
        self._undo.append(self.points.copy())
        del self._undo[:-UNDO_DEPTH]

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.points = self._undo.pop()
        return True

    def reset(self) -> None:
        """Back to the surface as computed."""
        if self.edited:
            self.checkpoint()
        self.points = self.original.copy()

    def max_change_mm(self) -> float:
        return float(np.max(np.linalg.norm(self.points - self.original, axis=1))) if len(self.points) else 0.0

    # -- brushes -----------------------------------------------------------

    def weights(self, centre, radius_mm: float) -> tuple[np.ndarray, np.ndarray]:
        """Vertices inside the brush and their falloff weights, 1 at the centre."""
        d = np.linalg.norm(self.points - np.asarray(centre, dtype=float), axis=1)
        inside = np.flatnonzero(d < radius_mm)
        # Full strength over the inner half, then a smoothstep down to nothing
        # at the rim, so a dab has a clear effect and no visible edge.
        t = np.clip(2.0 * d[inside] / max(radius_mm, 1e-9) - 1.0, 0.0, 1.0)
        return inside, 1.0 - t * t * (3.0 - 2.0 * t)

    def stroke(self, centre, radius_mm: float, strength: float, brush: str = "smooth") -> int:
        """Apply one dab of ``brush``; returns how many vertices it moved."""
        if brush not in BRUSHES:
            raise ValueError(f"brush must be one of {BRUSHES}")
        strength = float(np.clip(strength, 0.0, 1.0))
        inside, w = self.weights(centre, radius_mm)
        if inside.size == 0 or strength == 0.0:
            return 0
        w = w * strength
        if brush == "smooth":
            for _ in range(SMOOTH_PASSES):
                self._taubin(inside, w)
        else:
            normal = self._mean_normal(inside, w)
            sign = 1.0 if brush == "fill" else -1.0
            self.points[inside] += sign * PUSH_PER_STROKE_MM * w[:, None] * normal
            # A push leaves a lip at the brush edge; one smoothing pass
            # blends it into the surrounding surface.
            self._taubin(inside, 0.5 * w)
        return int(inside.size)

    def smooth_near_planes(self, planes, band_mm: float = 3.0, passes: int = 6) -> int:
        """Smooth the surface within ``band_mm`` of each cut plane.

        ``planes`` are objects with ``origin`` and ``normal``. The weight
        falls from 1 on the plane to 0 at the edge of the band, so the
        smoothing fades into the untouched surface on both sides.
        """
        moved = np.zeros(len(self.points), dtype=bool)
        for plane in planes:
            origin = np.asarray(plane.origin, dtype=float)
            normal = np.asarray(plane.normal, dtype=float)
            normal = normal / max(np.linalg.norm(normal), 1e-12)
            distance = np.abs((self.points - origin) @ normal)
            # Only near the bone the plane cuts, not wherever the infinite
            # plane happens to pass through the rest of the jaw.
            near = np.linalg.norm(self.points - origin, axis=1) < 40.0
            inside = np.flatnonzero((distance < band_mm) & near)
            if inside.size == 0:
                continue
            t = distance[inside] / band_mm
            w = (1.0 - t * t) ** 2
            for _ in range(passes):
                self._taubin(inside, w)
            moved[inside] = True
        return int(moved.sum())

    # -- internals -----------------------------------------------------------

    def _laplacian(self, vertices: np.ndarray) -> np.ndarray:
        """Mean of each vertex's neighbours, minus the vertex."""
        starts = self._offsets[vertices]
        counts = self._offsets[vertices + 1] - starts
        first = np.concatenate([[0], np.cumsum(counts)[:-1]])
        flat = self._neighbours[np.repeat(starts - first, counts) + np.arange(counts.sum())]
        owner = np.repeat(np.arange(len(vertices)), counts)
        sums = np.zeros((len(vertices), 3))
        np.add.at(sums, owner, self.points[flat])
        means = sums / np.maximum(counts, 1)[:, None]
        step = means - self.points[vertices]
        # A point no triangle uses has no neighbours; it must stay put rather
        # than be pulled to the origin.
        step[counts == 0] = 0.0
        return step

    def _taubin(self, vertices: np.ndarray, weights: np.ndarray) -> None:
        self.points[vertices] += TAUBIN_LAMBDA * weights[:, None] * self._laplacian(vertices)
        self.points[vertices] += TAUBIN_MU * weights[:, None] * self._laplacian(vertices)

    def _mean_normal(self, vertices: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Area-weighted outward normal of the faces touching the brush."""
        touched = np.zeros(len(self.points), dtype=bool)
        touched[vertices] = True
        faces = self.triangles[touched[self.triangles].any(axis=1)]
        a, b, c = (self.points[faces[:, k]] for k in range(3))
        n = np.cross(b - a, c - a).sum(axis=0)
        length = np.linalg.norm(n)
        return n / length if length > 0 else np.array([0.0, 0.0, 1.0])


def _adjacency(count: int, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertex neighbours as a CSR list: ``neighbours[offsets[i]:offsets[i+1]]``."""
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    edges = np.unique(edges, axis=0)
    order = np.argsort(edges[:, 0], kind="stable")
    edges = edges[order]
    offsets = np.zeros(count + 1, dtype=np.int64)
    np.add.at(offsets, edges[:, 0] + 1, 1)
    return edges[:, 1].copy(), np.cumsum(offsets)


def vertex_normals(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals, for re-shading an edited surface."""
    a, b, c = (points[triangles[:, k]] for k in range(3))
    face = np.cross(b - a, c - a)
    normals = np.zeros_like(points)
    for k in range(3):
        np.add.at(normals, triangles[:, k], face)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(length > 0, length, 1.0)
