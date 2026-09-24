"""Bending a plate onto a path, and checking whether the result is usable.

Stage B of fitting. Rigid placement (:mod:`plate_fit`) puts the plate where it
belongs; this bends it to follow a path a rigid body cannot reach.

What may bend and what may not
------------------------------
A screw hole is a machined feature with a diameter a screw has to pass
through. Bending it out of round is not a small error, it is a hole that no
longer takes a screw. So the plate is divided along its own length:

* **Protected zones** — the neighbourhood of every screw hole, out to
  ``protected_radius_mm``. These move by a single rigid transform, so the
  hole stays exactly circular, exactly its diameter, and exactly its
  thickness.
* **Bridges** — the spans between holes. These are re-swept along the target
  path, which is where all the bending actually happens, as it does on a
  bench with irons.

A vertex is described by where it sits along the plate's own centreline and
how far it stands off it. Re-embedding those coordinates on the target path
preserves arc length along the plate, and the lateral offsets are carried
through untouched, so width and thickness survive. Nothing is scaled anywhere.

The frames along the path are parallel-transported for the same reason the
resection frames are: a frame built from a fixed global axis spins and flips
where the path turns, and a plate whose faces spin is a plate with a twist in
it that nobody asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BentPlate:
    """A plate bent onto a path, with the evidence it is still a plate."""

    points: np.ndarray  # (N, 3) world mm
    triangles: np.ndarray  # (M, 3)
    hole_centres: np.ndarray  # (H, 3) world mm
    hole_axes: np.ndarray  # (H, 3) unit
    bend_angles_deg: np.ndarray  # (H,) turn at each hole station
    #: Arc position of each hole along the plate, mm. Bends are located
    #: against these to work out which holes a bend actually lands on.
    hole_s_mm: np.ndarray | None = None
    #: Tangent along the plate at each hole, for orienting hole ovalisation.
    hole_tangents: np.ndarray | None = None
    distortion: object | None = None
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return not self.problems

    def screw_trajectories(self, length_mm: float = 14.0) -> np.ndarray:
        return np.stack(
            [self.hole_centres, self.hole_centres - self.hole_axes * float(length_mm)],
            axis=1,
        )


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1.0, n)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def transported_frames(
    points: np.ndarray, seed_up: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Arc length and a non-spinning frame field along a polyline.

    Returns ``(s, tangents, ups, binormals)``. The up-axis is carried by the
    minimal rotation between consecutive tangents, so the plate's faces do not
    twist as the path turns.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(steps)])

    tangents = np.gradient(points, axis=0)
    tangents = _unit(tangents)

    seed = np.asarray(seed_up, dtype=float).reshape(3)
    seed = seed - float(np.dot(seed, tangents[0])) * tangents[0]
    if np.linalg.norm(seed) < 1e-9:
        helper = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(helper, tangents[0]))) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
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
        carried = carried - float(np.dot(carried, current)) * current
        ups[i] = carried / np.linalg.norm(carried)
    return s, tangents, ups, np.cross(tangents, ups)


@dataclass
class _FrameField:
    """A smooth curve with a frame at every station, sampled by arc length.

    ``U`` is the direction out of the plate face — the plate's own hole axes on
    the source side, the bone's own outward normals on the target side — and
    ``B = T x U`` runs across the plate's width. Taking both from the real
    geometry, rather than from a global axis, is what keeps width and
    thickness from being confused once the plate has been rotated onto the
    jaw.
    """

    s: np.ndarray
    P: np.ndarray
    T: np.ndarray
    U: np.ndarray
    B: np.ndarray
    station_s: np.ndarray

    def at(self, s_query):
        """Frame at arbitrary arc positions; straight extension past the ends."""
        s_query = np.atleast_1d(np.asarray(s_query, dtype=float))
        clipped = np.clip(s_query, self.s[0], self.s[-1])

        def interp(field):
            return np.column_stack(
                [np.interp(clipped, self.s, field[:, c]) for c in range(3)]
            )

        T = _unit(interp(self.T))
        U = interp(self.U)
        U = _unit(U - np.einsum("ij,ij->i", U, T)[:, None] * T)
        B = np.cross(T, U)
        P = interp(self.P) + (s_query - clipped)[:, None] * T
        return P, T, U, B

    def rotation_at(self, s_value: float) -> np.ndarray:
        """Columns T, B, U at one arc position, as a rotation matrix."""
        _, T, U, B = self.at([s_value])
        return np.column_stack([T[0], B[0], U[0]])


def _frame_field(points, normals, step_mm: float = 0.25) -> _FrameField:
    """Build a frame field on a smooth curve through ``points``.

    The curve is the same centripetal Catmull-Rom used for the arch, so the
    span between two screw holes follows the curve rather than the straight
    chord across it; on convex bone a chord cuts inside the surface by the
    sagitta. The out-of-face direction is interpolated from the given normals
    and re-orthogonalised against the tangent, so the frame twists exactly as
    much as the surface underneath it does.
    """
    from .spline import ArchCurve

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = _unit(np.asarray(normals, dtype=float).reshape(-1, 3))
    curve = ArchCurve(points)
    samples = curve.resample(min(step_mm, curve.length_mm / 4.0))
    station_s = np.array([curve.arc_position_of(p) for p in points])
    station_s[0], station_s[-1] = 0.0, samples.s[-1]

    # Keep neighbouring normals on the same side before interpolating them.
    oriented = normals.copy()
    for i in range(1, len(oriented)):
        if np.dot(oriented[i], oriented[i - 1]) < 0.0:
            oriented[i] = -oriented[i]
    U = np.column_stack(
        [np.interp(samples.s, station_s, oriented[:, c]) for c in range(3)]
    )
    T = _unit(samples.tangents)
    U = _unit(U - np.einsum("ij,ij->i", U, T)[:, None] * T)
    return _FrameField(
        s=samples.s,
        P=samples.points,
        T=T,
        U=U,
        B=np.cross(T, U),
        station_s=station_s,
    )


def _nearest_station(points: np.ndarray, field: _FrameField, chunk: int = 2048):
    """Index of the closest field station to each point, in memory-safe chunks."""
    out = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        diff = block[:, None, :] - field.P[None, :, :]
        out[start : start + chunk] = np.argmin(
            np.einsum("ijk,ijk->ij", diff, diff), axis=1
        )
    return out


def bend_to_path(
    points: np.ndarray,
    triangles: np.ndarray,
    hole_centres: np.ndarray,
    hole_axes: np.ndarray,
    path_points: np.ndarray,
    path_normals: np.ndarray,
    protected_radius_mm: float = 3.5,
    min_bend_radius_mm: float = 15.0,
    max_bend_deg_per_node: float = 15.0,
) -> BentPlate:
    """Bend an already-placed plate so it follows ``path_points``.

    ``points`` and ``hole_centres`` are the plate as rigidly placed;
    ``path_points`` are the positions its hole centres should reach, with
    ``path_normals`` the direction out of the bone at each.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    hole_centres = np.asarray(hole_centres, dtype=float).reshape(-1, 3)
    hole_axes = _unit(np.asarray(hole_axes, dtype=float).reshape(-1, 3))
    path_points = np.asarray(path_points, dtype=float).reshape(-1, 3)
    path_normals = _unit(np.asarray(path_normals, dtype=float).reshape(-1, 3))

    if len(path_points) < 2 or len(hole_centres) < 2:
        raise ValueError("bending needs at least two holes and two path points")
    if len(path_points) != len(hole_centres):
        raise ValueError(
            f"{len(hole_centres)} holes but {len(path_points)} path points; "
            "the caller must pair them before bending"
        )

    warnings: list[str] = []
    problems: list[str] = []

    # Source: the plate's own centreline, with its own hole axes as the
    # out-of-face direction. Target: the planned path, with the bone's
    # outward normals, so the plate's bone face lands on the bone.
    source = _frame_field(hole_centres, hole_axes)
    target = _frame_field(path_points, path_normals)
    s_src = source.station_s

    if abs(target.s[-1] - source.s[-1]) > 0.5:
        warnings.append(
            f"The path is {target.s[-1]:.1f} mm between end holes but the plate is "
            f"{source.s[-1]:.1f} mm. Bending cannot change a plate's length; the "
            "screw holes will not land on the planned stations."
        )

    # Each vertex in the source frame nearest to it: arc position along the
    # plate, offset across its width, offset out of its face.
    nearest = _nearest_station(points, source)
    offset = points - source.P[nearest]
    vertex_s = source.s[nearest] + np.einsum("ij,ij->i", offset, source.T[nearest])
    across = np.einsum("ij,ij->i", offset, source.B[nearest])
    out = np.einsum("ij,ij->i", offset, source.U[nearest])

    # Swept: re-embed those coordinates on the target frame. This is where
    # the bridges bend.
    sP, _, sU, sB = target.at(vertex_s)
    swept = sP + across[:, None] * sB + out[:, None] * sU

    # Rigid: each protected zone moves by one exact rigid transform taking
    # its hole's source frame onto that hole's target frame.
    home = np.argmin(np.abs(vertex_s[:, None] - s_src[None, :]), axis=1)
    rigid = np.empty_like(points)
    rotations = []
    hole_origin = np.empty_like(hole_centres)
    hole_up = np.empty_like(hole_axes)
    hole_tangents = np.empty_like(hole_centres)
    for h, s_h in enumerate(s_src):
        src_p, _, _, _ = source.at([s_h])
        dst_p, dst_t, _, _ = target.at([s_h])
        rotation = target.rotation_at(s_h) @ source.rotation_at(s_h).T
        rotations.append(rotation)
        hole_origin[h] = dst_p[0] + rotation @ (hole_centres[h] - src_p[0])
        hole_up[h] = rotation @ hole_axes[h]
        hole_tangents[h] = dst_t[0]
        members = home == h
        rigid[members] = dst_p[0] + (points[members] - src_p[0]) @ rotation.T

    # Weight 0 inside a protected zone, ramping to 1 across the bridge.
    span = float(np.median(np.diff(s_src))) if len(s_src) > 1 else 1.0
    free = max(span / 2.0 - protected_radius_mm, 1e-6)
    distance_from_hole = np.abs(vertex_s - s_src[home])
    weight = _smoothstep((distance_from_hole - protected_radius_mm) / free)
    bent = rigid + weight[:, None] * (swept - rigid)

    bend_angles = _turn_angles(hole_origin)
    steep = bend_angles > max_bend_deg_per_node
    if np.any(steep):
        worst = float(np.nanmax(bend_angles))
        problems.append(
            f"The path turns {worst:.1f}° at a screw hole, above the "
            f"{max_bend_deg_per_node:.0f}° per-node working limit for this "
            "plate. Redraw the path or choose a plate rated to bend further."
        )
    radius = _bend_radii(hole_origin)
    tight = np.isfinite(radius) & (radius < min_bend_radius_mm)
    if np.any(tight):
        problems.append(
            f"The path bends to a {float(np.nanmin(radius[tight])):.1f} mm "
            f"radius, tighter than this plate's {min_bend_radius_mm:.0f} mm "
            "minimum. Bending it this far risks cracking it."
        )

    fold = _self_intersection_warning(hole_origin, protected_radius_mm)
    if fold:
        problems.append(fold)

    return BentPlate(
        points=bent,
        triangles=np.asarray(triangles, dtype=np.int64),
        hole_centres=hole_origin,
        hole_axes=hole_up,
        bend_angles_deg=bend_angles,
        hole_s_mm=s_src,
        hole_tangents=hole_tangents,
        warnings=warnings,
        problems=problems,
    )


def _turn_angles(points: np.ndarray) -> np.ndarray:
    """Degrees of turn at each interior station; NaN at the two ends."""
    angles = np.full(len(points), np.nan)
    v = np.diff(points, axis=0)
    v = _unit(v)
    for i in range(1, len(points) - 1):
        angles[i] = np.degrees(
            np.arccos(float(np.clip(np.dot(v[i - 1], v[i]), -1.0, 1.0)))
        )
    return angles


def _bend_radii(points: np.ndarray) -> np.ndarray:
    """Radius of the circle through each consecutive triple of stations."""
    radii = np.full(len(points), np.inf)
    for i in range(1, len(points) - 1):
        a, b, c = points[i - 1], points[i], points[i + 1]
        ab, bc, ca = (
            np.linalg.norm(b - a),
            np.linalg.norm(c - b),
            np.linalg.norm(a - c),
        )
        area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
        radii[i] = np.inf if area < 1e-9 else (ab * bc * ca) / (4.0 * area)
    return radii


def _self_intersection_warning(
    hole_centres: np.ndarray, protected_radius_mm: float
) -> str:
    """Detect the plate folding back onto itself.

    Non-neighbouring screw holes closer together than two protected zones means
    the bar has been bent past a hairpin and is now overlapping its own body.
    """
    n = len(hole_centres)
    limit = 2.0 * protected_radius_mm
    for i in range(n):
        for j in range(i + 2, n):
            if float(np.linalg.norm(hole_centres[i] - hole_centres[j])) < limit:
                return (
                    f"The bent plate folds onto itself: holes {i + 1} and "
                    f"{j + 1} end up {limit:.1f} mm or less apart. This path "
                    "cannot be made from one plate."
                )
    return ""


# -- contact and clearance ------------------------------------------------


@dataclass
class ClearanceReport:
    """How the fitted plate actually sits against the bone."""

    clearance_mm: np.ndarray  # (H,) inner face to bone, signed; negative = into bone
    hole_axis_error_deg: np.ndarray  # (H,) hole axis against the bone normal
    collisions: np.ndarray  # (H,) bool
    excessive_gaps: np.ndarray  # (H,) bool
    poor_screw_angles: np.ndarray  # (H,) bool
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def clearance_report(
    hole_centres: np.ndarray,
    hole_axes: np.ndarray,
    bone_points: np.ndarray,
    bone_normals: np.ndarray,
    thickness_mm: float,
    target_clearance_mm: float = 0.5,
    max_gap_mm: float = 2.0,
    max_screw_angle_deg: float = 35.0,
) -> ClearanceReport:
    """Measure the fitted plate against the bone it is supposed to sit on.

    Clearance is signed along the bone's own outward normal: the inner face of
    the plate minus the bone surface. Negative means the plate is inside the
    bone, which is a collision and not a fit.
    """
    hole_centres = np.asarray(hole_centres, dtype=float).reshape(-1, 3)
    hole_axes = _unit(np.asarray(hole_axes, dtype=float).reshape(-1, 3))
    bone_points = np.asarray(bone_points, dtype=float).reshape(-1, 3)
    bone_normals = _unit(np.asarray(bone_normals, dtype=float).reshape(-1, 3))

    diff = hole_centres[:, None, :] - bone_points[None, :, :]
    nearest = np.argmin(np.einsum("ijk,ijk->ij", diff, diff), axis=1)
    normals = bone_normals[nearest]
    offset = hole_centres - bone_points[nearest]

    # Distance from the bone surface to the plate's inner face.
    clearance = np.einsum("ij,ij->i", offset, normals) - thickness_mm / 2.0
    angle = np.degrees(
        np.arccos(np.clip(np.einsum("ij,ij->i", hole_axes, normals), -1.0, 1.0))
    )

    collisions = clearance < 0.0
    gaps = clearance > max_gap_mm
    poor = angle > max_screw_angle_deg

    problems: list[str] = []
    warnings: list[str] = []
    if np.any(collisions):
        problems.append(
            f"The plate cuts into bone at {int(collisions.sum())} screw "
            f"position(s), worst {float(clearance.min()):.1f} mm. Increase the "
            "standoff or redraw the path."
        )
    if np.any(gaps):
        warnings.append(
            f"The plate stands off more than {max_gap_mm:.1f} mm at "
            f"{int(gaps.sum())} screw position(s), worst "
            f"{float(clearance.max()):.1f} mm. It will not be seated there."
        )
    if np.any(poor):
        warnings.append(
            f"{int(poor.sum())} screw hole(s) point more than "
            f"{max_screw_angle_deg:.0f}° off the bone surface, worst "
            f"{float(angle.max()):.0f}°. Those screws would skive."
        )
    return ClearanceReport(
        clearance_mm=clearance,
        hole_axis_error_deg=angle,
        collisions=collisions,
        excessive_gaps=gaps,
        poor_screw_angles=poor,
        problems=problems,
        warnings=warnings,
    )
