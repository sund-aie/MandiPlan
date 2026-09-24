"""Flush mirror reconstruction of a resected mandibular segment, in the volume.

Why the old approach could not be flush
---------------------------------------
It reflected the retained surface in the estimated mid-sagittal plane,
clipped out the part inside the cuts, and set that beside the real stumps as a
second mesh. Three things guaranteed a step at every junction:

* two meshes meeting at a seam are not one surface;
* the symmetry plane is an estimate, and a fraction of a degree of error at
  the midline is millimetres at a cut several centimetres away;
* no mandible is perfectly symmetric, so even a perfect plane leaves the
  mirrored stump offset from the real one.

What this does instead
----------------------
It works on the scan's own gray values, so the answer is one field and one
surface:

1. **Register the mirror to each stump.** Bone-surface points in a band just
   outside each cut are matched by iterative closest point: the reflected
   opposite side against the real stump. One rigid correction per cut.
2. **Warp between them.** Across the defect the correction is interpolated
   from the first cut's to the second's, so the donor meets both stumps, not
   just one.
3. **Blend inside the retained bone.** The transition band lies on the
   retained side of each cut and ends at the cut: everything inside the
   resection is donor, and the hand-over happens where the two fields already
   describe the same, registered bone.
4. **One surface.** The combined field is contoured once at the bone
   threshold, giving a single watertight mandible.

Where the defect crosses the midline the mirror has no donor — its source
would itself be resected tissue. Those voxels are marked, reported, and filled
from a fallback field instead (the reference library when one is installed,
otherwise the patient's own pre-operative contour, clearly flagged).

Everything here is numpy; contouring is done by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .plate_fit import kabsch


# -- rigid transforms -------------------------------------------------------


@dataclass
class RigidTransform:
    """``x -> R x + t``."""

    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(np.eye(3), np.zeros(3))

    def apply(self, points) -> np.ndarray:
        return np.asarray(points, dtype=float) @ self.rotation.T + self.translation

    def inverse(self) -> "RigidTransform":
        r_t = self.rotation.T
        return RigidTransform(r_t, -r_t @ self.translation)

    def compose(self, first: "RigidTransform") -> "RigidTransform":
        """``self`` applied after ``first``."""
        return RigidTransform(
            self.rotation @ first.rotation,
            self.rotation @ first.translation + self.translation,
        )

    @property
    def angle_deg(self) -> float:
        cos = (np.trace(self.rotation) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    @property
    def shift_mm(self) -> float:
        return float(np.linalg.norm(self.translation))


def _to_quaternion(rotation: np.ndarray) -> np.ndarray:
    m = rotation
    trace = np.trace(m)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def _rotations_from_quaternions(q: np.ndarray) -> np.ndarray:
    """(N, 4) unit quaternions (w, x, y, z) to (N, 3, 3) matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty((len(q), 3, 3))
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - z * w)
    out[:, 0, 2] = 2 * (x * z + y * w)
    out[:, 1, 0] = 2 * (x * y + z * w)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - x * w)
    out[:, 2, 0] = 2 * (x * z - y * w)
    out[:, 2, 1] = 2 * (y * z + x * w)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def blend_transforms(a: RigidTransform, b: RigidTransform, weights) -> tuple[np.ndarray, np.ndarray]:
    """Per-point rigid transforms, slerped from ``a`` (w=0) to ``b`` (w=1).

    Returns ``(rotations (N, 3, 3), translations (N, 3))``.
    """
    w = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, 1.0)
    qa, qb = _to_quaternion(a.rotation), _to_quaternion(b.rotation)
    if np.dot(qa, qb) < 0:
        qb = -qb
    dot = float(np.clip(np.dot(qa, qb), -1.0, 1.0))
    theta = np.arccos(dot)
    if theta < 1e-8:
        q = qa[None, :] * (1 - w)[:, None] + qb[None, :] * w[:, None]
    else:
        s = np.sin(theta)
        q = (np.sin((1 - w) * theta) / s)[:, None] * qa + (np.sin(w * theta) / s)[:, None] * qb
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    translations = a.translation[None, :] * (1 - w)[:, None] + b.translation[None, :] * w[:, None]
    return _rotations_from_quaternions(q), translations


# -- registration ------------------------------------------------------------


def nearest(source: np.ndarray, target: np.ndarray, chunk: int = 1500):
    """For each source point, index of and distance to the closest target point."""
    source = np.asarray(source, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    index = np.empty(len(source), dtype=np.int64)
    distance = np.empty(len(source))
    t2 = np.einsum("ij,ij->i", target, target)
    for start in range(0, len(source), chunk):
        block = source[start : start + chunk]
        d2 = np.einsum("ij,ij->i", block, block)[:, None] - 2.0 * block @ target.T + t2[None, :]
        k = np.argmin(d2, axis=1)
        index[start : start + chunk] = k
        distance[start : start + chunk] = np.sqrt(np.maximum(d2[np.arange(len(block)), k], 0.0))
    return index, distance


@dataclass
class Registration:
    """How a donor band was brought onto a stump."""

    transform: RigidTransform
    rms_before_mm: float
    rms_after_mm: float
    points_used: int


def icp(
    source: np.ndarray,
    target: np.ndarray,
    iterations: int = 30,
    keep_fraction: float = 0.8,
    max_points: int = 1200,
    seed: int = 0,
) -> Registration:
    """Rigid iterative closest point, trimmed: ``source`` onto ``target``.

    The worst-matching fifth of pairs are ignored at each step, so a band that
    overlaps the target only partly still registers on the part it shares.
    """
    source = np.asarray(source, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    if len(source) < 10 or len(target) < 10:
        return Registration(RigidTransform.identity(), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    if len(source) > max_points:
        source = source[rng.choice(len(source), max_points, replace=False)]
    if len(target) > max_points * 5:
        target = target[rng.choice(len(target), max_points * 5, replace=False)]

    def trimmed_rms(points):
        _, d = nearest(points, target)
        keep = np.sort(d)[: max(3, int(len(d) * keep_fraction))]
        return float(np.sqrt(np.mean(keep**2)))

    before = trimmed_rms(source)
    total = RigidTransform.identity()
    moved = source.copy()
    previous = np.inf
    for _ in range(iterations):
        idx, d = nearest(moved, target)
        keep = d <= np.quantile(d, keep_fraction)
        if keep.sum() < 3:
            break
        rotation, translation = kabsch(moved[keep], target[idx[keep]])
        step = RigidTransform(rotation, translation)
        moved = step.apply(moved)
        total = step.compose(total)
        error = float(np.sqrt(np.mean(d[keep] ** 2)))
        if abs(previous - error) < 1e-4:
            break
        previous = error
    return Registration(total, before, trimmed_rms(moved), len(source))


# -- the resected region and the blend ---------------------------------------


def plane_depths(planes, points) -> np.ndarray:
    """Signed distance of each point into the resected side of each plane.

    Shape (len(points), len(planes)); positive means on the side the plane
    removes.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.column_stack([plane.signed_distance(pts) for plane in planes])


def depth_into_resection(planes, points) -> np.ndarray:
    """How far inside the resected region a point is; negative outside it."""
    return plane_depths(planes, points).min(axis=1)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class ReconstructionReport:
    """What the reconstruction did and how flush the result is."""

    registrations: list[Registration] = field(default_factory=list)
    donorless_voxels: int = 0
    donorless_volume_mm3: float = 0.0
    fallback: str = ""
    warnings: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        lines = []
        for i, reg in enumerate(self.registrations, start=1):
            if reg.points_used == 0:
                lines.append(f"Junction {i}: too little bone near the cut to register.")
                continue
            lines.append(
                f"Junction {i}: mirror-to-stump mismatch {reg.rms_before_mm:.2f} mm "
                f"before registration, {reg.rms_after_mm:.2f} mm after "
                f"({reg.transform.shift_mm:.2f} mm, {reg.transform.angle_deg:.1f}° correction)."
            )
        if self.donorless_voxels:
            lines.append(
                f"{self.donorless_volume_mm3:.0f} mm³ of the defect has no mirror donor "
                f"(the defect crosses the midline); filled from {self.fallback}."
            )
        lines.extend(self.warnings)
        return lines


def compose_field(
    volume,
    planes,
    symmetry,
    registrations: list[tuple[int, RigidTransform]],
    box_min,
    box_max,
    band_mm: float = 3.0,
    threshold: float = float("-inf"),
    fallback_field=None,
    fallback_name: str = "the pre-operative contour",
):
    """The reconstructed gray-value field inside ``box``.

    ``registrations`` pairs each plane index with the rigid correction that
    brings the reflected donor onto that plane's stump. Returns
    ``(field, (i0, j0, k0), donorless_mask, report)`` where ``field`` covers
    the index box starting at ``(i0, j0, k0)``.
    """
    spacing, origin = volume.spacing, volume.origin
    lo = np.maximum(np.floor((np.asarray(box_min) - origin) / spacing).astype(int), 0)
    hi = np.minimum(
        np.ceil((np.asarray(box_max) - origin) / spacing).astype(int),
        volume.size_xyz - 1,
    )
    ii = np.arange(lo[0], hi[0] + 1)
    jj = np.arange(lo[1], hi[1] + 1)
    kk = np.arange(lo[2], hi[2] + 1)
    K, J, I = np.meshgrid(kk, jj, ii, indexing="ij")
    world = origin + np.stack([I, J, K], axis=-1).reshape(-1, 3) * spacing
    block = volume.array[lo[2] : hi[2] + 1, lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
    shape = block.shape
    original = block.astype(np.float32).reshape(-1)

    depths = plane_depths(planes, world)
    inside = depths.min(axis=1)
    report = ReconstructionReport(fallback=fallback_name)

    # Donor weight: 1 inside the resection, falling to 0 across a band that
    # lies in the retained bone and ends exactly at the cut.
    alpha = _smoothstep((inside + band_mm) / band_mm)
    active = alpha > 0.0
    result = original.copy()
    donorless = np.zeros(len(world), dtype=bool)
    if not np.any(active):
        return result.reshape(shape), tuple(lo), donorless.reshape(shape), report

    by_plane = dict(registrations)
    if len(planes) >= 2 and 0 in by_plane and 1 in by_plane:
        # Across the defect, weight toward whichever cut is nearer.
        d0 = np.maximum(depths[active, 0], 0.0)
        d1 = np.maximum(depths[active, 1], 0.0)
        w = d0 / np.maximum(d0 + d1, 1e-9)
        # In a stump band, use that stump's own correction outright.
        w = np.where(depths[active, 0] < 0, 0.0, w)
        w = np.where(depths[active, 1] < 0, 1.0, w)
        rotations, translations = blend_transforms(by_plane[0], by_plane[1], w)
    else:
        only = next(iter(by_plane.values()), RigidTransform.identity())
        n = int(active.sum())
        rotations = np.repeat(only.rotation[None], n, axis=0)
        translations = np.repeat(only.translation[None], n, axis=0)

    # Sample the donor: undo the correction, then reflect to the source side.
    x = world[active]
    undone = np.einsum("nji,nj->ni", rotations, x - translations)
    source = symmetry.reflect(undone)
    donor = volume.sample(source, fill=float(volume.array.min())).astype(np.float32)

    # A source inside the resection is diseased tissue, not a donor.
    source_inside = depth_into_resection(planes, source) >= 0.0
    if np.any(source_inside):
        fallback = (
            fallback_field(x[source_inside])
            if fallback_field is not None
            else original[active][source_inside]
        )
        donor[source_inside] = fallback
        mask = np.zeros(len(world), dtype=bool)
        mask[np.flatnonzero(active)[source_inside]] = True
        donorless = mask & (inside >= 0)
        report.donorless_voxels = int(donorless.sum())
        report.donorless_volume_mm3 = float(report.donorless_voxels * np.prod(spacing))

    a = alpha[active].astype(np.float32)
    result[active] = (1.0 - a) * original[active] + a * donor
    if report.donorless_voxels:
        # Report the bone that had to be filled, not the air around it.
        donorless &= result >= threshold
        report.donorless_voxels = int(donorless.sum())
        report.donorless_volume_mm3 = float(report.donorless_voxels * np.prod(spacing))
    return result.reshape(shape), tuple(lo), donorless.reshape(shape), report


def stump_bands(planes, points, near_mm: float = 1.0, far_mm: float = 12.0, radius_mm: float = 25.0):
    """Which points sit in each cut's stump band.

    Returns a list of boolean masks, one per plane: points on the retained
    side of that plane, between ``near_mm`` and ``far_mm`` from it, and within
    ``radius_mm`` of the cut's centre (so the opposite side of the jaw, which
    may also lie behind the plane, is not mistaken for this stump).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    depths = plane_depths(planes, pts)
    masks = []
    for i, plane in enumerate(planes):
        behind = -depths[:, i]
        near_cut = np.linalg.norm(pts - plane.origin, axis=1) < radius_mm
        others = [j for j in range(len(planes)) if j != i]
        # Not inside the resection by way of another plane's region.
        clear = np.ones(len(pts), dtype=bool)
        for j in others:
            clear &= depths[:, j] > -far_mm - radius_mm
        masks.append((behind >= near_mm) & (behind <= far_mm) & near_cut & clear)
    return masks


def cross_section_areas(volume_like, threshold: float, plane, offsets_mm, half_size_mm: float = 25.0, step_mm: float = 0.25):
    """Bone area on slices parallel to a cut, at signed offsets from it.

    ``volume_like`` needs a ``sample(points)`` method. A flush junction has an
    area profile with no jump across offset 0.
    """
    n = np.asarray(plane.normal, dtype=float)
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    grid = np.arange(-half_size_mm, half_size_mm + 1e-9, step_mm)
    A, B = np.meshgrid(grid, grid, indexing="ij")
    base = plane.origin + A.reshape(-1, 1) * u + B.reshape(-1, 1) * v
    areas = []
    for offset in offsets_mm:
        values = volume_like.sample(base + n * float(offset))
        areas.append(float(np.count_nonzero(values >= threshold)) * step_mm * step_mm)
    return np.asarray(areas)


def junction_step_mm(
    volume_like,
    threshold: float,
    plane,
    delta_mm: float = 0.3,
    half_size_mm: float = 25.0,
    step_mm: float = 0.1,
) -> float:
    """Mean step in the bone surface across a junction, in mm.

    Bone sections on two thin slices ``delta_mm`` either side of the cut are
    compared: the area that is bone on one slice and not the other, divided by
    the section's perimeter, is the mean distance the outline moves across
    the junction. The same measure on a pair of slices well inside the stump
    is subtracted, so the natural taper of the jaw is not counted as a step.
    A flush junction scores close to zero.
    """
    n = np.asarray(plane.normal, dtype=float)
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    grid = np.arange(-half_size_mm, half_size_mm + 1e-9, step_mm)
    A, B = np.meshgrid(grid, grid, indexing="ij")
    base = plane.origin + A.reshape(-1, 1) * u + B.reshape(-1, 1) * v

    def mask(offset):
        return (volume_like.sample(base + n * offset) >= threshold).reshape(A.shape)

    def moved(a, b):
        ma, mb = mask(a), mask(b)
        changed = np.count_nonzero(ma ^ mb) * step_mm * step_mm
        edge = ma ^ np.roll(ma, 1, 0) | ma ^ np.roll(ma, 1, 1)
        perimeter = max(np.count_nonzero(edge) * step_mm, 1e-6)
        return changed / perimeter

    across = moved(-delta_mm, +delta_mm)
    control = moved(-6.0 - delta_mm, -6.0 + delta_mm)
    return float(max(across - control, 0.0))
