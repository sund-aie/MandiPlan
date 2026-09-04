"""Placing a real plate mesh on a planned path.

Stage A: rigid placement. The asset is moved as a solid body — rotated and
translated, never scaled, sheared or stretched — so its thickness, hole
diameters and hole-to-hole spacing are preserved exactly rather than
approximately. Whatever the rigid fit cannot reach is reported as residual,
which is the honest measure of how much bending the plate would actually
need.

Local plate space, as the catalogue declares it: ``+x`` along the plate,
``+y`` across its width, ``+z`` out of the face that looks at the surgeon —
so ``-z`` is the face that meets bone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Lever arm for the orientation constraints added to the point match, in mm.
#: A straight plate's hole centres are collinear, so matching them alone
#: leaves the roll about the plate's long axis undetermined; pairing each hole
#: with points offset along the plate's own axes pins it down.
_FRAME_ARM_MM = 5.0


@dataclass
class FittedPlate:
    """A plate asset placed on a path, and how well it actually sits."""

    points: np.ndarray  # (N, 3) transformed mesh vertices, world mm
    triangles: np.ndarray  # (M, 3)
    hole_centres: np.ndarray  # (H, 3) world mm, from the asset's own holes
    hole_axes: np.ndarray  # (H, 3) unit, out of the outer face
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    residual_mm: np.ndarray  # (H,) distance from each hole to its target
    holes_used: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def max_residual_mm(self) -> float:
        return float(np.max(self.residual_mm)) if len(self.residual_mm) else 0.0

    @property
    def rms_residual_mm(self) -> float:
        if not len(self.residual_mm):
            return 0.0
        return float(np.sqrt(np.mean(self.residual_mm**2)))

    def screw_trajectories(self, length_mm: float = 14.0) -> np.ndarray:
        """``(H, 2, 3)`` segments from each fitted hole centre into bone.

        The trajectory starts at the real hole centre and runs along the real
        hole axis, both carried through the placement transform — not at a
        decorative marker dropped on the path.
        """
        into_bone = -self.hole_axes
        return np.stack(
            [self.hole_centres, self.hole_centres + into_bone * float(length_mm)],
            axis=1,
        )


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best rigid transform taking ``source`` onto ``target``.

    Returns ``(rotation, translation)`` with ``x -> R x + t``. Reflections are
    excluded: a mirrored plate is a different part, not a fit.
    """
    source = np.asarray(source, dtype=float).reshape(-1, 3)
    target = np.asarray(target, dtype=float).reshape(-1, 3)
    if len(source) != len(target):
        raise ValueError("point sets must be the same length")
    if len(source) < 3:
        raise ValueError("a rigid fit needs at least three point pairs")

    source_centre = source.mean(axis=0)
    target_centre = target.mean(axis=0)
    covariance = (source - source_centre).T @ (target - target_centre)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(vt.T @ u.T)))])
    rotation = vt.T @ correction @ u.T
    return rotation, target_centre - rotation @ source_centre


def plate_targets(
    nodes: np.ndarray,
    normals: np.ndarray,
    binormals: np.ndarray,
    clearance_mm: float,
    thickness_mm: float,
) -> np.ndarray:
    """Where the plate's hole centres should end up.

    The path runs on the bone surface. The plate's inner face has to stand
    ``clearance_mm`` off it, and its holes sit on the midplane, so the target
    for a hole centre is the bone point pushed out along the surface normal by
    the clearance plus half the plate's thickness.
    """
    nodes = np.asarray(nodes, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    if clearance_mm < 0:
        raise ValueError("clearance must not be negative")
    offset = float(clearance_mm) + float(thickness_mm) / 2.0
    return nodes + normals * offset


def _centred_slice(count: int, wanted: int) -> np.ndarray:
    """``wanted`` consecutive indices taken from the middle of ``count``."""
    wanted = min(wanted, count)
    start = (count - wanted) // 2
    return np.arange(start, start + wanted)


def rigid_fit(
    asset_points: np.ndarray,
    asset_triangles: np.ndarray,
    hole_centres: np.ndarray,
    hole_axes: np.ndarray,
    targets: np.ndarray,
    target_normals: np.ndarray,
    target_binormals: np.ndarray,
) -> FittedPlate:
    """Place a plate mesh on a path as a rigid body.

    The correspondence is not hole-centre to target alone: a straight plate's
    holes are collinear, which leaves the roll about the plate's long axis
    free and lets the plate arrive lying on its edge. Each hole is therefore
    matched three times — at its centre, at a point along the plate's own hole
    axis, and at a point across the plate's width — against the target point,
    the bone's outward normal and the path's binormal. That pins all six
    degrees of freedom with no scaling anywhere.
    """
    hole_centres = np.asarray(hole_centres, dtype=float).reshape(-1, 3)
    hole_axes = np.asarray(hole_axes, dtype=float).reshape(-1, 3)
    targets = np.asarray(targets, dtype=float).reshape(-1, 3)
    target_normals = np.asarray(target_normals, dtype=float).reshape(-1, 3)
    target_binormals = np.asarray(target_binormals, dtype=float).reshape(-1, 3)

    warnings: list[str] = []
    holes, stations = len(hole_centres), len(targets)
    if holes < 2 or stations < 2:
        raise ValueError("a plate fit needs at least two holes and two stations")

    used = min(holes, stations)
    hole_index = _centred_slice(holes, used)
    target_index = _centred_slice(stations, used)
    if holes < stations:
        warnings.append(
            f"The plate has {holes} holes but the path has {stations} stations; "
            f"{stations - holes} station(s) at the ends carry no screw hole."
        )
    elif holes > stations:
        warnings.append(
            f"The plate has {holes} holes but the path only reaches "
            f"{stations} of them; the plate overhangs the planned path."
        )

    # Along-plate direction, taken from the asset's own hole line.
    plate_axis = hole_centres[-1] - hole_centres[0]
    plate_axis = plate_axis / np.linalg.norm(plate_axis)

    source, target = [], []
    for h, t in zip(hole_index, target_index):
        axis = hole_axes[h] / np.linalg.norm(hole_axes[h])
        # Handedness matters. The path's binormal is tangent x normal, so the
        # plate's matching axis is along x out-of-face, in that order. Taking
        # the cross product the other way round asks Kabsch to match a
        # left-handed frame to a right-handed one, and since reflections are
        # excluded it answers with a visibly wrong rotation instead.
        across = np.cross(plate_axis, axis)
        across_norm = np.linalg.norm(across)
        across = across / across_norm if across_norm > 1e-9 else np.zeros(3)

        source.extend(
            [
                hole_centres[h],
                hole_centres[h] + axis * _FRAME_ARM_MM,
                hole_centres[h] + across * _FRAME_ARM_MM,
            ]
        )
        normal = target_normals[t] / np.linalg.norm(target_normals[t])
        binormal = target_binormals[t] / np.linalg.norm(target_binormals[t])
        target.extend(
            [
                targets[t],
                targets[t] + normal * _FRAME_ARM_MM,
                targets[t] + binormal * _FRAME_ARM_MM,
            ]
        )

    rotation, translation = kabsch(np.array(source), np.array(target))

    placed_holes = hole_centres @ rotation.T + translation
    placed_axes = hole_axes @ rotation.T
    residual = np.full(len(hole_centres), np.nan)
    residual[hole_index] = np.linalg.norm(
        placed_holes[hole_index] - targets[target_index], axis=1
    )

    return FittedPlate(
        points=np.asarray(asset_points, dtype=float) @ rotation.T + translation,
        triangles=np.asarray(asset_triangles, dtype=np.int64),
        hole_centres=placed_holes,
        hole_axes=placed_axes,
        rotation=rotation,
        translation=translation,
        residual_mm=residual[hole_index],
        holes_used=int(used),
        warnings=warnings,
    )


def hole_spacings_mm(hole_centres: np.ndarray) -> np.ndarray:
    """Centre-to-centre distance between consecutive holes."""
    centres = np.asarray(hole_centres, dtype=float).reshape(-1, 3)
    return np.linalg.norm(np.diff(centres, axis=0), axis=1)
