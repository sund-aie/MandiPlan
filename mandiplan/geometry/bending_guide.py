"""A printable guide that clips onto the plate and stops each bend at its angle.

The guide is a chain of saddles, one over each screw hole, printed in one
piece from a stiff flexible filament (TPU 95A, or PETG). Each saddle clips
over the plate's edges; a square window over the hole leaves it free for the
bending irons; a thin strap over the plate's centreline links each saddle to
the next.

Where two saddles meet, over the bridge between two holes, their facing ends
are the stop. They are cut on the two halves of a mitre: the plane the ends
would share if the plate were already bent to the target, taken back to the
straight plate. On the side the bend closes, the ends then stand apart by a
wedge that closes exactly when the bridge has turned through the stop angle,
and the ends meet face to face. On the side the bend opens they are simply
square, and move apart. So the bench procedure is: bend each bridge the way
its gap closes until the two saddles touch, and let go.

The stop angle is the target angle overbent for springback (see
``materials.Alloy.overbend_deg``), so the plate relaxes onto the target
rather than short of it. Twist about the plate's axis closes no gap and has
no stop; it is labelled, and listed in the guide table with the rest.

The guide is designed on the plate as supplied, straight or preformed: each
saddle sits on its hole in the supplied shape, and the bend at each bridge is
the change from the supplied shape to the planned one.

Numpy only. The solid is an implicit function (each part the intersection of
half-spaces, parts combined by min/max); ``render/bending_guide_mesh.py``
contours it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Fit between saddle and plate, per side, mm.
FIT_CLEARANCE_MM = 0.15
#: Saddle wall and top thickness, mm.
WALL_MM = 1.2
#: How far the walls reach past the plate's faces, mm. The stop is on the
#: walls, and the further from the bend axis it is, the more a printing
#: error of 0.1 mm is spread: about 1.3° at this reach.
WALL_REACH_MM = 3.0
#: Clearance between neighbouring saddles where the bend opens, mm.
JOINT_GAP_MM = 0.4
#: A bend smaller than this gets no stop: the wedge would be thinner than a
#: printer can hold apart.
MIN_STOP_DEG = 1.5
#: The snap lips tuck this far under the plate's edges, mm.
LIP_MM = 0.6
#: Strap linking the saddles, mm.
STRAP_WIDTH_MM = 2.4
STRAP_THICKNESS_MM = 0.6


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1.0, n)


def _frame(tangent, up) -> np.ndarray:
    """Columns ``(along, across, out)``: right-handed, ``out`` off the outer face."""
    t = _unit(tangent)
    u = _unit(np.asarray(up, dtype=float) - np.dot(up, t) * t)
    b = np.cross(u, t)
    return np.column_stack([t, b, u])


def _rotation(axis, angle_rad: float) -> np.ndarray:
    a = _unit(axis)
    x, y, z = a
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    angle = float(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    if angle < 1e-9:
        return np.array([0.0, 1.0, 0.0]), 0.0
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if np.linalg.norm(axis) < 1e-9:  # a half turn
        w, v = np.linalg.eigh(rotation + rotation.T)
        axis = v[:, int(np.argmax(w))]
    return _unit(axis), angle


def hole_frames(centres, axes) -> np.ndarray:
    """``(H, 3, 3)`` frames at the holes of a plate, along its hole order."""
    centres = np.asarray(centres, dtype=float)
    axes = np.asarray(axes, dtype=float)
    # Second-order differences: exact along a preform's circular arc inside,
    # and at the two end holes off by the square of the step, not half of it.
    if len(centres) >= 3:
        tangents = np.gradient(centres, axis=0, edge_order=2)
    else:
        tangents = np.repeat((centres[-1] - centres[0])[None], len(centres), axis=0)
    return np.stack([_frame(t, u) for t, u in zip(tangents, axes)])


@dataclass
class GuideJoint:
    """The bridge between hole ``index`` and hole ``index + 1``."""

    index: int
    origin: np.ndarray  # joint point, plate (supplied) frame, mm
    frame: np.ndarray  # (3, 3) columns along / across / out, supplied frame
    rotation: np.ndarray  # target bend of the next segment, joint frame
    stop_rotation: np.ndarray  # the bend at which the saddles meet
    angle_deg: float
    stop_angle_deg: float
    in_plane_deg: float
    out_of_plane_deg: float
    twist_deg: float

    @property
    def has_stop(self) -> bool:
        return self.stop_angle_deg >= MIN_STOP_DEG

    def instruction(self) -> str:
        if not self.has_stop and abs(self.twist_deg) < MIN_STOP_DEG:
            return f"Holes {self.index + 1}-{self.index + 2}: leave straight."
        parts = []
        if abs(self.out_of_plane_deg) >= 0.5:
            way = "away from the bone" if self.out_of_plane_deg > 0 else "into the bone"
            parts.append(f"curl {abs(self.out_of_plane_deg):.1f}° {way}")
        if abs(self.in_plane_deg) >= 0.5:
            way = "counter-clockwise" if self.in_plane_deg > 0 else "clockwise"
            parts.append(f"turn {abs(self.in_plane_deg):.1f}° {way} seen from the outer face")
        if abs(self.twist_deg) >= 0.5:
            parts.append(f"twist {self.twist_deg:+.1f}° (no stop: check by eye)")
        text = ", ".join(parts)
        if self.has_stop:
            text += f"; bend until the saddles meet ({self.stop_angle_deg:.1f}° with springback)"
        return f"Holes {self.index + 1}-{self.index + 2}: {text}."


@dataclass
class GuideSaddle:
    """The saddle over one hole, in that hole's supplied frame."""

    hole: int
    centre: np.ndarray  # supplied frame, mm
    frame: np.ndarray  # (3, 3)
    start_x: float  # local extent along the plate, mm
    end_x: float
    start_joint: GuideJoint | None
    end_joint: GuideJoint | None


@dataclass
class BendingGuide:
    width_mm: float
    thickness_mm: float
    window_mm: float
    saddles: list[GuideSaddle]
    joints: list[GuideJoint]
    alloy_name: str = ""
    notes: list[str] = field(default_factory=list)

    def instructions(self) -> list[str]:
        return [joint.instruction() for joint in self.joints]


def _decompose(rotation: np.ndarray) -> tuple[float, float, float]:
    """In-plane, out-of-plane and twist of a bend, degrees, as the bend table.

    Conventions follow ``geometry/plate.py``: in-plane positive is
    counter-clockwise seen from the outer face, out-of-plane positive lifts
    the plate away from the bone, twist is right-handed about the direction
    of travel.
    """
    v = rotation[:, 0]
    w = rotation[:, 2]
    in_plane = np.degrees(np.arctan2(v[1], v[0]))
    out_of_plane = np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))
    twist = np.degrees(np.arctan2(-w[1], w[2]))
    return float(in_plane), float(out_of_plane), float(twist)


def plan_bending_guide(
    supplied_centres,
    supplied_axes,
    bent_centres,
    bent_axes,
    bent_tangents,
    width_mm: float,
    thickness_mm: float,
    seat_diameter_mm: float,
    overbend=None,
    alloy_name: str = "",
) -> BendingGuide:
    """Lay out the guide for bending a supplied plate into a planned shape.

    ``supplied_*`` are the plate's holes as it comes out of the packet (the
    asset's own frame); ``bent_*`` the same holes as planned. ``overbend`` maps
    a target angle in degrees to the angle to bend to before springback, or
    is ``None`` for none.
    """
    sc = np.asarray(supplied_centres, dtype=float)
    supplied = hole_frames(sc, supplied_axes)
    bent_axes = np.asarray(bent_axes, dtype=float)
    bent_tangents = np.asarray(bent_tangents, dtype=float)
    bent = np.stack([_frame(t, u) for t, u in zip(bent_tangents, bent_axes)])
    if len(sc) != len(bent):
        raise ValueError("the supplied and planned plates have different hole counts")
    if len(sc) < 2:
        raise ValueError("a bending guide needs at least two holes")

    joints: list[GuideJoint] = []
    for i in range(len(sc) - 1):
        origin = 0.5 * (sc[i] + sc[i + 1])
        frame = _frame(sc[i + 1] - sc[i], supplied[i][:, 2] + supplied[i + 1][:, 2])
        # The change in the relative orientation of the two segments, taken
        # into the joint's own frame.
        before = supplied[i].T @ supplied[i + 1]
        after = bent[i].T @ bent[i + 1]
        change = after @ before.T  # in segment i's frame
        to_joint = frame.T @ supplied[i]
        rotation = to_joint @ change @ to_joint.T
        axis, angle = _axis_angle(rotation)
        angle_deg = float(np.degrees(angle))
        stop_deg = float(overbend(angle_deg)) if overbend is not None else angle_deg
        stop = _rotation(axis, np.radians(stop_deg))
        in_plane, out_plane, twist = _decompose(rotation)
        joints.append(
            GuideJoint(
                index=i,
                origin=origin,
                frame=frame,
                rotation=rotation,
                stop_rotation=stop,
                angle_deg=angle_deg,
                stop_angle_deg=stop_deg,
                in_plane_deg=in_plane,
                out_of_plane_deg=out_plane,
                twist_deg=twist,
            )
        )

    saddles: list[GuideSaddle] = []
    for i in range(len(sc)):
        frame = supplied[i]
        before = joints[i - 1] if i > 0 else None
        after = joints[i] if i < len(joints) else None
        pitch_back = np.linalg.norm(sc[i] - sc[i - 1]) if i > 0 else np.linalg.norm(sc[1] - sc[0])
        pitch_on = (
            np.linalg.norm(sc[i + 1] - sc[i]) if i < len(sc) - 1 else np.linalg.norm(sc[-1] - sc[-2])
        )
        saddles.append(
            GuideSaddle(
                hole=i,
                centre=sc[i],
                frame=frame,
                start_x=-0.5 * float(pitch_back),
                end_x=0.5 * float(pitch_on),
                start_joint=before,
                end_joint=after,
            )
        )
    return BendingGuide(
        width_mm=float(width_mm),
        thickness_mm=float(thickness_mm),
        window_mm=float(seat_diameter_mm) + 1.0,
        saddles=saddles,
        joints=joints,
        alloy_name=alloy_name,
    )


# -- the solid ----------------------------------------------------------------
#
# Every part of a saddle is a convex block: the intersection of half-spaces
# ``A p <= b`` in the saddle's own frame (origin at the hole centre, +x along
# the plate toward the next hole, +z out of the outer face). The same list
# drives the printed mesh (each block clipped out of a box) and the contact
# check (a point is inside if it is inside any block), so the two cannot
# disagree.


@dataclass
class Block:
    normals: np.ndarray  # (K, 3)
    offsets: np.ndarray  # (K,)
    name: str = ""

    def value(self, points: np.ndarray) -> np.ndarray:
        """Negative inside."""
        return np.max(points @ self.normals.T - self.offsets, axis=-1)

    def with_limits(self, normals, offsets) -> "Block":
        return Block(
            np.vstack([self.normals, np.asarray(normals, dtype=float).reshape(-1, 3)]),
            np.concatenate([self.offsets, np.asarray(offsets, dtype=float).ravel()]),
            self.name,
        )


def _box_block(lo, hi, name: str = "") -> Block:
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    eye = np.eye(3)
    return Block(np.vstack([-eye, eye]), np.concatenate([-lo, hi]), name)


def saddle_dimensions(guide: BendingGuide) -> dict[str, float]:
    half_w = guide.width_mm / 2.0 + FIT_CLEARANCE_MM
    half_t = guide.thickness_mm / 2.0 + FIT_CLEARANCE_MM
    return {
        "inner_y": half_w,
        "outer_y": half_w + WALL_MM,
        "plate_top": half_t,
        "top": half_t + WALL_MM,
        "wall_top": half_t + WALL_REACH_MM,
        "wall_bottom": -(half_t + WALL_REACH_MM),
    }


def saddle_blocks(guide: BendingGuide, saddle: GuideSaddle, include_strap: bool = True) -> list[Block]:
    """The convex parts of one saddle, ends already cut at its joints."""
    d = saddle_dimensions(guide)
    x0, x1 = saddle.start_x - 0.5, saddle.end_x + 0.5
    hw = guide.window_mm / 2.0
    blocks = [
        # The two walls, reaching past both faces of the plate.
        _box_block([x0, d["inner_y"], d["wall_bottom"]], [x1, d["outer_y"], d["wall_top"]], "wall"),
        _box_block([x0, -d["outer_y"], d["wall_bottom"]], [x1, -d["inner_y"], d["wall_top"]], "wall"),
        # The top between the walls, around a window over the hole that
        # leaves it free for the bending irons.
        _box_block([x0, -d["inner_y"], d["plate_top"]], [-hw, d["inner_y"], d["top"]], "top"),
        _box_block([hw, -d["inner_y"], d["plate_top"]], [x1, d["inner_y"], d["top"]], "top"),
        _box_block([-hw, hw, d["plate_top"]], [hw, d["inner_y"], d["top"]], "top"),
        _box_block([-hw, -d["inner_y"], d["plate_top"]], [hw, -hw, d["top"]], "top"),
    ]
    # Snap lips under the plate's edges, in the middle of the saddle only, so
    # the ends are free to meet.
    span = 0.25 * (saddle.end_x - saddle.start_x)
    lip_top = -(guide.thickness_mm / 2.0 + FIT_CLEARANCE_MM)
    lip_in = guide.width_mm / 2.0 - LIP_MM
    blocks += [
        _box_block([-span, lip_in, lip_top - WALL_MM], [span, d["inner_y"], lip_top], "lip"),
        _box_block([-span, -d["inner_y"], lip_top - WALL_MM], [span, -lip_in, lip_top], "lip"),
    ]

    # The ends: square where the bend opens, on the stop mitre where it closes.
    for joint, sign in ((saddle.end_joint, 1.0), (saddle.start_joint, -1.0)):
        if joint is None:
            continue
        normals, offsets = _end_limits(saddle, joint, sign)
        blocks = [block.with_limits(normals, offsets) for block in blocks]

    # Half of each strap to a neighbour, reaching just past the joint so the
    # halves overlap and print as one. It crosses the gap, so it takes no
    # end cut; it is thin enough to buckle rather than stop the bend.
    if include_strap:
        for joint, sign in ((saddle.end_joint, 1.0), (saddle.start_joint, -1.0)):
            if joint is None:
                continue
            along = float((saddle.frame.T @ (joint.origin - saddle.centre))[0])
            lo_x, hi_x = (along - 2.0, along + 0.1) if sign > 0 else (along - 0.1, along + 2.0)
            blocks.append(
                _box_block(
                    [lo_x, -STRAP_WIDTH_MM / 2.0, d["top"] - STRAP_THICKNESS_MM],
                    [hi_x, STRAP_WIDTH_MM / 2.0, d["top"]],
                    "strap",
                )
            )
    return blocks


def _end_limits(saddle: GuideSaddle, joint: GuideJoint, sign: float):
    """Half-spaces bounding a saddle at one joint, in the saddle's frame.

    ``sign`` is +1 at the saddle's far end (the joint's first segment) and -1
    at its near end (the second). In the joint frame the first segment turns
    by minus half the stop rotation and the second by plus half, and the
    shared mitre is then the plane x = 0; taken back to the straight plate,
    the first segment's end is the plane with normal ``H x`` and the
    second's ``H^-1 x``.
    """
    to_joint = joint.frame.T @ saddle.frame  # saddle-local -> joint frame
    origin = saddle.frame.T @ (joint.origin - saddle.centre)
    x = np.array([1.0, 0.0, 0.0])
    normals = [sign * x]
    offsets = [-JOINT_GAP_MM / 2.0]
    if joint.has_stop:
        axis, angle = _axis_angle(joint.stop_rotation)
        half = _rotation(axis, angle / 2.0)
        normals.append(sign * (half @ x if sign > 0 else half.T @ x))
        offsets.append(0.0)
    # n . (to_joint (p - origin)) <= c  ->  (to_joint^T n) . p <= c + (to_joint^T n) . origin
    local = [to_joint.T @ n for n in normals]
    return np.array(local), np.array([c + n @ origin for n, c in zip(local, offsets)])


def saddle_field(
    guide: BendingGuide, saddle: GuideSaddle, local: np.ndarray, include_strap: bool = True
) -> np.ndarray:
    """Implicit value of one saddle at points in its own frame (<= 0 inside)."""
    points = np.asarray(local, dtype=float)
    return np.min(
        np.stack([block.value(points) for block in saddle_blocks(guide, saddle, include_strap)]),
        axis=0,
    )


def joint_overlap(
    guide: BendingGuide, joint: GuideJoint, angle_scale: float, step_mm: float = 0.1
) -> float:
    """Volume (mm³) where two neighbouring saddles overlap when the bridge is
    bent through ``angle_scale`` times the stop angle: zero up to the stop,
    growing past it. The flexible strap is left out; it buckles rather than
    stops."""
    first = guide.saddles[joint.index]
    second = guide.saddles[joint.index + 1]
    d = saddle_dimensions(guide)
    # The saddles can only meet near the joint: a slab either side of it,
    # as wide as the widest wedge the stop can open.
    r = float(np.hypot(d["outer_y"], d["wall_top"])) + 0.5
    reach = min(r * np.tan(np.radians(max(joint.stop_angle_deg, 1.0)) * angle_scale / 2.0 + 0.05) + 1.0, r)
    across = np.arange(-r, r + 1e-9, step_mm)
    along = np.arange(-reach, reach + 1e-9, step_mm)
    X, Y, Z = np.meshgrid(along, across, across, indexing="ij")
    q = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)  # joint frame, bent
    axis, angle = _axis_angle(joint.stop_rotation)
    half = _rotation(axis, angle_scale * angle / 2.0)

    def inside(saddle, turned):
        # The segment was turned by ``turned``; take the points back.
        straight = q @ turned  # == (turned.T @ q.T).T
        world = joint.origin + straight @ joint.frame.T
        local = (world - saddle.centre) @ saddle.frame
        return saddle_field(guide, saddle, local, include_strap=False) <= 0.0

    overlap = inside(first, half.T) & inside(second, half)
    return float(overlap.sum() * step_mm**3)
