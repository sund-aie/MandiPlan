"""Reconstruction-plate path: bend instructions and bending template.

Input is an ordered list of points on the bone surface together with the
outward surface normal at each point.  The path is resampled at the screw-hole
pitch, and at every interior node the turn from the incoming to the outgoing
segment is decomposed into three signed angles in the plate's own frame.

Local frame at interior node ``i``
----------------------------------
    t_i = normalize(v_{i-1} + v_i)          averaged direction of travel
    n_i = outward surface normal at P_i, orthogonalised against t_i
    b_i = normalize(t_i x n_i)              binormal, in the plate's plane

Sign conventions (identical in the table, the CSV and the 3-D overlay)
----------------------------------------------------------------------
All three angles are signed rotations following the right-hand rule about the
named axis, reported in degrees.

``in_plane_deg``
    Rotation from v_{i-1} to v_i about **+n_i**, after projecting both
    segments onto the plane perpendicular to n_i (the plate's own plane).
    This is the contour bend made with bending irons.  Positive is
    counter-clockwise when looking at the plate's outer face, which turns the
    path toward **-b_i**.

``out_of_plane_deg``
    Rotation from v_{i-1} to v_i about **+b_i**, after projecting both
    segments onto the plane perpendicular to b_i.  Positive turns the path
    toward **+n_i**, i.e. the plate lifts away from the bone; negative bends
    it into the bone.

``twist_deg``
    Rotation of the plate's outer face about the direction of travel,
    measured as the signed angle from the segment normal of v_{i-1} to that
    of v_i about **+t_i**.  Positive is a right-handed twist about the
    direction of travel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spline import interpolate_along_polyline, resample_polyline


@dataclass
class BendNode:
    """One screw-hole station of the plate."""

    index: int
    cumulative_mm: float
    in_plane_deg: float
    out_of_plane_deg: float
    twist_deg: float
    segment_length_mm: float  # length of the segment leaving this node (NaN at the end)


@dataclass
class PlatePlan:
    nodes: np.ndarray  # (N, 3) world mm
    normals: np.ndarray  # (N, 3) unit outward normals at the nodes
    tangents: np.ndarray  # (N, 3) unit averaged tangents
    binormals: np.ndarray  # (N, 3) unit binormals
    bends: list[BendNode]
    pitch_mm: float

    @property
    def total_length_mm(self) -> float:
        return float(np.linalg.norm(np.diff(self.nodes, axis=0), axis=1).sum())

    @property
    def segment_lengths_mm(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.nodes, axis=0), axis=1)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _project_out(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Component of ``v`` perpendicular to unit vector ``axis``."""
    return v - np.dot(v, axis) * axis


def _signed_angle(u: np.ndarray, w: np.ndarray, axis: np.ndarray) -> float:
    """Signed angle (rad) from ``u`` to ``w`` about unit ``axis``, right-handed."""
    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
    if nu < 1e-12 or nw < 1e-12:
        return float("nan")
    u, w = u / nu, w / nw
    return float(np.arctan2(np.dot(np.cross(u, w), axis), np.dot(u, w)))


def _segment_normals(nodes: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Outward normal attached to each segment, perpendicular to that segment."""
    v = np.diff(nodes, axis=0)
    avg = 0.5 * (normals[:-1] + normals[1:])
    out = np.empty_like(v)
    for i in range(len(v)):
        t = _unit(v[i])
        out[i] = _unit(_project_out(avg[i], t))
    return out


def compute_plate_plan(path_points, path_normals, pitch_mm: float) -> PlatePlan:
    """Resample a surface path at ``pitch_mm`` and compute its bend instructions."""
    path_points = np.asarray(path_points, dtype=float).reshape(-1, 3)
    path_normals = np.asarray(path_normals, dtype=float).reshape(-1, 3)
    if len(path_points) != len(path_normals):
        raise ValueError("each path point needs a surface normal")
    if pitch_mm <= 0:
        raise ValueError("screw-hole pitch must be positive")

    nodes = resample_polyline(path_points, pitch_mm)
    normals = interpolate_along_polyline(path_points, path_normals, pitch_mm)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    if len(nodes) < 2:
        raise ValueError("path is shorter than one screw-hole pitch")

    v = np.diff(nodes, axis=0)
    lengths = np.linalg.norm(v, axis=1)
    seg_normals = _segment_normals(nodes, normals)

    n_nodes = len(nodes)
    tangents = np.empty((n_nodes, 3))
    frame_normals = np.empty((n_nodes, 3))
    binormals = np.empty((n_nodes, 3))
    bends: list[BendNode] = []
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])

    for i in range(n_nodes):
        if i == 0:
            t = _unit(v[0])
        elif i == n_nodes - 1:
            t = _unit(v[-1])
        else:
            t = _unit(_unit(v[i - 1]) + _unit(v[i]))
        n_perp = _project_out(normals[i], t)
        if np.linalg.norm(n_perp) < 1e-9:
            # Surface normal parallel to the path: fall back to any normal in
            # the plane perpendicular to t so the frame stays defined.
            helper = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(helper, t)) > 0.9:
                helper = np.array([1.0, 0.0, 0.0])
            n_perp = _project_out(helper, t)
        n_i = _unit(n_perp)
        b_i = _unit(np.cross(t, n_i))
        tangents[i], frame_normals[i], binormals[i] = t, n_i, b_i

        if 0 < i < n_nodes - 1:
            in_plane = _signed_angle(
                _project_out(v[i - 1], n_i), _project_out(v[i], n_i), n_i
            )
            out_plane = _signed_angle(
                _project_out(v[i - 1], b_i), _project_out(v[i], b_i), b_i
            )
            twist = _signed_angle(
                _project_out(seg_normals[i - 1], t), _project_out(seg_normals[i], t), t
            )
        else:
            in_plane = out_plane = twist = float("nan")

        bends.append(
            BendNode(
                index=i,
                cumulative_mm=float(cumulative[i]),
                in_plane_deg=float(np.degrees(in_plane)),
                out_of_plane_deg=float(np.degrees(out_plane)),
                twist_deg=float(np.degrees(twist)),
                segment_length_mm=(
                    float(lengths[i]) if i < n_nodes - 1 else float("nan")
                ),
            )
        )

    return PlatePlan(
        nodes=nodes,
        normals=frame_normals,
        tangents=tangents,
        binormals=binormals,
        bends=bends,
        pitch_mm=float(pitch_mm),
    )


def ribbon_mesh(
    plan: PlatePlan, width_mm: float, thickness_mm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep a rectangular ribbon along the plate path.

    The inner face of the ribbon lies on the path (i.e. on the bone surface)
    and the ribbon extends ``thickness_mm`` outward along the surface normal;
    its width runs along the binormal.  Returns ``(points, triangles)`` for a
    closed surface, wound so the triangle normals point outward.
    """
    if width_mm <= 0 or thickness_mm <= 0:
        raise ValueError("ribbon width and thickness must be positive")
    p = plan.nodes
    n = plan.normals
    b = plan.binormals
    half = width_mm / 2.0

    corners = np.stack(
        [
            p - half * b,
            p + half * b,
            p + half * b + thickness_mm * n,
            p - half * b + thickness_mm * n,
        ],
        axis=1,
    )  # (N, 4, 3)
    points = corners.reshape(-1, 3)

    tris: list[tuple[int, int, int]] = []
    n_st = len(p)
    for i in range(n_st - 1):
        a0 = 4 * i
        a1 = 4 * (i + 1)
        for e in range(4):
            e2 = (e + 1) % 4
            # Wound so the side faces point away from the ribbon interior.
            tris.append((a0 + e, a1 + e, a1 + e2))
            tris.append((a0 + e, a1 + e2, a0 + e2))
    # End caps.
    tris.append((0, 1, 2))
    tris.append((0, 2, 3))
    last = 4 * (n_st - 1)
    tris.append((last, last + 2, last + 1))
    tris.append((last, last + 3, last + 2))

    triangles = np.array(tris, dtype=np.int64)
    if _signed_volume(points, triangles) < 0:
        triangles = triangles[:, ::-1].copy()
    return points, triangles


def _signed_volume(points: np.ndarray, triangles: np.ndarray) -> float:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def bend_table_rows(plan: PlatePlan) -> list[list[str]]:
    """Bend table as display strings, units included."""
    rows = []
    for node in plan.bends:
        rows.append(
            [
                str(node.index),
                f"{node.cumulative_mm:.1f}",
                _fmt_angle(node.in_plane_deg),
                _fmt_angle(node.out_of_plane_deg),
                _fmt_angle(node.twist_deg),
                (
                    f"{node.segment_length_mm:.2f}"
                    if np.isfinite(node.segment_length_mm)
                    else "—"
                ),
            ]
        )
    return rows


def _fmt_angle(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    direction = "→" if value >= 0 else "←"
    return f"{value:+.1f} {direction}"
