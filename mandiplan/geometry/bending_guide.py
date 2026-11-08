"""The bending guide: a thin comb the plate wraps round, made for one plate.

It stands on the bench (and on the print bed) on a flat base strip that runs
inside the curve of the plate. One thin tooth rises from it at each screw hole
of the finished plate. Each tooth ends in a pad exactly where the bone face of
the plate lies at that hole, turned exactly as the plate is there, and a pin
exactly where the hole's centre is, as thick as this plate's bore allows.

So the guide is used with the bender in hand and nothing clipped on. Hang
hole 1 on the flagged tooth. Bend the next bridge until the next hole drops
over the next pin and lies flat on its pad, let go, and move on. The plate is
right when every hole is on its pin. A pin is an answer no angle reading can
give: a hole on its pin has the right place and the right turn.

Everything is the plate's own, and the kit's:

* hole count and spacing come from the plate (after any trimming), pin
  diameter from its bore, tooth height from its width, pad position from its
  thickness;
* between the teeth the bridges are free on all sides. The base stands back
  from the plate by the depth the chosen bender needs behind the bridge, and
  the guide reports where that bender's jaws would meet a tooth at this
  plate's pitch, so the plate is lifted off only then.

Numpy only. The solid is a set of simple convex parts given as distance-like
fields; ``render/bending_guide_mesh.py`` contours their union.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Base strip under the teeth: thickness and width, mm.
BASE_THICKNESS_MM = 1.6
BASE_WIDTH_MM = 5.0
#: Plate's lowest edge above the base, mm: room for the tool's jaw.
EDGE_ABOVE_BASE_MM = 2.0
#: Pin as a fraction of the plate's thickness: it locates without standing
#: out of the outer face.
PIN_HEIGHT_FRACTION = 0.75
#: How far past the plate's edge the first tooth's flag stands, mm.
FLAG_MM = 3.5
#: Below this a bend is left alone, degrees.
NEGLIGIBLE_DEG = 0.5
#: How close a tool's jaw may come to a tooth and still clear it, mm.
JAW_MARGIN_MM = 1.0


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
    angle_deg: float
    stop_angle_deg: float  # loaded angle after the alloy's springback
    in_plane_deg: float
    out_of_plane_deg: float
    twist_deg: float


@dataclass
class GuidePart:
    """One convex piece of the guide, in print coordinates."""

    centre: np.ndarray
    frame: np.ndarray  # (3, 3) columns: the part's local axes
    half: np.ndarray  # half sizes along them; for a pin (radius, radius, half height)
    name: str

    def field(self, local):
        """Negative inside, about the distance to the surface outside."""
        x, y, z = local
        if self.name == "pin":
            return np.maximum(np.hypot(x, y) - self.half[0], np.abs(z) - self.half[2])
        return np.maximum.reduce(
            (np.abs(x) - self.half[0], np.abs(y) - self.half[1], np.abs(z) - self.half[2])
        )


@dataclass
class BendingGuide:
    centres: np.ndarray  # the finished plate's hole centres, patient space
    frames: np.ndarray  # (H, 3, 3) columns along / across / out, patient space
    stock_distances: np.ndarray  # along the supplied plate from hole 1, mm
    joints: list[GuideJoint]
    parts: list[GuidePart]
    kit: object
    width_mm: float
    thickness_mm: float
    tooth_mm: float
    tooth_width_mm: float
    play_mm: float
    pin_diameter_mm: float
    pin_height_mm: float
    setback_mm: float
    jaws_meet_teeth: bool
    basis: np.ndarray  # (3, 3) columns: print x, y, z in patient space
    origin: np.ndarray  # the print origin in patient space
    alloy_name: str = ""

    def to_print(self, points) -> np.ndarray:
        return (np.asarray(points, dtype=float) - self.origin) @ self.basis

    def to_patient(self, points) -> np.ndarray:
        return np.asarray(points, dtype=float) @ self.basis.T + self.origin

    def instructions(self) -> list[str]:
        return [self.instruction(joint) for joint in self.joints]

    def instruction(self, joint: GuideJoint) -> str:
        prefix = f"Holes {joint.index + 1}-{joint.index + 2}: "
        wanted = [
            ("in_plane", joint.in_plane_deg, "turn in the plane of the plate",
             "counter-clockwise from the outer face", "clockwise from the outer face"),
            ("out_of_plane", joint.out_of_plane_deg, "curl", "away from the bone", "towards the bone"),
            ("twist", joint.twist_deg, "twist", "right-handed along the plate", "left-handed along the plate"),
        ]
        pieces, usable = [], []
        for kind, angle, verb, positive, negative in wanted:
            if abs(angle) < NEGLIGIBLE_DEG:
                continue
            text = f"{verb} {abs(angle):.1f} deg {positive if angle > 0 else negative}"
            if self.kit.can_make(kind):
                usable.append(abs(angle))
            else:
                text += " (this needs another tool)"
            pieces.append(text)
        if not pieces:
            return prefix + "leave as supplied; the next hole already sits on its pin."
        text = prefix + ", ".join(pieces) + f"; until hole {joint.index + 2} drops onto its pin."
        if usable and self.kit.max_angle_per_pass_deg > 0:
            passes = int(np.ceil(max(usable) / self.kit.max_angle_per_pass_deg))
            if passes > 1:
                text += f" {self.kit.name}: in {passes} passes, checking on the pins between them."
        if joint.stop_angle_deg > joint.angle_deg + 0.05:
            text += f" Expect to load about {joint.stop_angle_deg:.1f} deg for springback."
        return text


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
    hole_diameter_mm: float,
    kit,
    overbend=None,
    alloy_name: str = "",
    tooth_mm: float = 1.2,
    play_mm: float = 0.25,
) -> BendingGuide:
    """Lay out the guide that bends a supplied plate into a planned shape.

    ``supplied_*`` are the plate's holes as it comes out of the packet;
    ``bent_*`` the same holes as planned. ``overbend`` maps a target angle in
    degrees to the angle to load it to before springback. ``tooth_mm`` is how
    thin the teeth are and ``play_mm`` the fit of a pin in its hole per side.
    """
    sc = np.asarray(supplied_centres, dtype=float)
    centres = np.asarray(bent_centres, dtype=float)
    supplied = hole_frames(sc, supplied_axes)
    bent_axes = np.asarray(bent_axes, dtype=float)
    bent_tangents = np.asarray(bent_tangents, dtype=float)
    if len(sc) != len(bent_axes) or len(sc) != len(centres):
        raise ValueError("the supplied and planned plates have different hole counts")
    if len(sc) < 2:
        raise ValueError("a bending guide needs at least two holes")
    if not (min(width_mm, thickness_mm, hole_diameter_mm) > 0 and np.isfinite(centres).all()):
        raise ValueError("Invalid plate section or planned hole positions.")
    if not (0.8 <= tooth_mm <= 2.4 and 0.05 <= play_mm <= 0.8):
        raise ValueError("Teeth must be 0.8-2.4 mm thick and pin play 0.05-0.8 mm.")
    pin_radius = 0.5 * float(hole_diameter_mm) - float(play_mm)
    if pin_radius < 0.4:
        raise ValueError("That much play leaves no pin in a hole this small.")
    bent = np.stack([_frame(t, u) for t, u in zip(bent_tangents, bent_axes)])
    if np.min(np.linalg.norm(np.diff(centres, axis=0), axis=1)) < 1.0:
        raise ValueError("Coincident hole positions cannot form a bending guide.")

    joints: list[GuideJoint] = []
    for i in range(len(sc) - 1):
        frame = _frame(sc[i + 1] - sc[i], supplied[i][:, 2] + supplied[i + 1][:, 2])
        # The change in the relative orientation of the two segments, taken
        # into the joint's own frame.
        change = (bent[i].T @ bent[i + 1]) @ (supplied[i].T @ supplied[i + 1]).T
        to_joint = frame.T @ supplied[i]
        rotation = to_joint @ change @ to_joint.T
        _, angle = _axis_angle(rotation)
        angle_deg = float(np.degrees(angle))
        stop_deg = float(overbend(angle_deg)) if overbend is not None else angle_deg
        in_plane, out_plane, twist = _decompose(rotation)
        joints.append(GuideJoint(i, angle_deg, stop_deg, in_plane, out_plane, twist))

    pitch = float(np.median(np.linalg.norm(np.diff(sc, axis=0), axis=1)))
    tooth_width = float(min(max(2.4, 2.0 * pin_radius + 0.8), max(pitch - 3.0, 2.0 * pin_radius + 0.4)))
    grip = float(kit.grip_span_mm)
    setback = float(np.clip(0.35 * grip, 3.0, 10.0))
    jaws_meet = False
    if grip > 0:
        # Jaws either side of a bridge, at holes half a pitch off its middle.
        phase = (0.5 * grip / pitch - 0.5) % 1.0
        jaws_meet = min(phase, 1.0 - phase) * pitch < 0.5 * tooth_width + JAW_MARGIN_MM

    # Print space: the plate stands on edge, its width upright, its holes
    # facing outward, wrapped round the teeth.
    up = _unit(bent[:, :, 1].mean(axis=0))
    middle = centres.mean(axis=0)
    flat = (centres - middle) - np.outer((centres - middle) @ up, up)
    _, singular, axes = np.linalg.svd(flat, full_matrices=False)
    x = axes[0] if singular[0] > 1e-6 else bent[0, :, 0]
    x = _unit(x - (x @ up) * up)
    basis = np.column_stack([x, np.cross(up, x), up])
    placed = (centres - middle) @ basis
    frames = np.einsum("ji,hjk->hik", basis, bent)
    half_t = 0.5 * float(thickness_mm)
    half_w = 0.5 * float(width_mm)
    reach = (
        np.abs(frames[:, 2, 1]) * half_w
        + np.abs(frames[:, 2, 0]) * 0.5 * tooth_width
        + np.abs(frames[:, 2, 2]) * (half_t + tooth_mm)
    )
    lift = BASE_THICKNESS_MM + EDGE_ABOVE_BASE_MM - float(np.min(placed[:, 2] - reach))
    placed[:, 2] += lift
    origin = middle - lift * up

    pads = placed - frames[:, :, 2] * (half_t + 0.5 * tooth_mm)
    inward = -frames[:, :, 2].copy()
    inward[:, 2] = 0.0
    if np.min(np.linalg.norm(inward, axis=1)) < 0.2:
        raise ValueError("The plate turns too far out of one plane for a guide to stand on a bench.")
    inward = _unit(inward)
    # The base keeps the setback unless the plate's curve is too tight for it,
    # when it would fold back on itself: stand it nearer rather than fail.
    tangents = np.gradient(pads[:, :2], axis=0)
    while True:
        feet = pads[:, :2] + setback * inward[:, :2]
        if np.all(np.einsum("ij,ij->i", np.diff(feet, axis=0), 0.5 * (tangents[:-1] + tangents[1:])) > 0.5):
            break
        if setback <= 2.0:
            raise ValueError("The plate curves too tightly for a guide behind it; check the plate path.")
        setback = max(setback - 0.5, 2.0)

    parts: list[GuidePart] = []
    pin_height = PIN_HEIGHT_FRACTION * float(thickness_mm)
    z_axis = np.array([0.0, 0.0, 1.0])
    for i, (hole, frame, pad, into, foot) in enumerate(zip(placed, frames, pads, inward, feet)):
        parts.append(GuidePart(pad, frame, np.array([0.5 * tooth_width, half_w, 0.5 * tooth_mm]), "pad"))
        pin_low = hole - frame[:, 2] * (half_t + 0.25 * tooth_mm)
        pin_high = hole + frame[:, 2] * (pin_height - half_t)
        parts.append(
            GuidePart(0.5 * (pin_low + pin_high), frame,
                      np.array([pin_radius, pin_radius, 0.5 * np.linalg.norm(pin_high - pin_low)]), "pin")
        )
        across = np.cross(z_axis, into)
        upright = np.column_stack([across, into, z_axis])
        top = float(pad[2]) + (half_w + FLAG_MM if i == 0 else 0.0)
        parts.append(
            GuidePart(np.array([pad[0], pad[1], 0.5 * top]), upright,
                      np.array([0.5 * tooth_width, 0.5 * tooth_mm, 0.5 * top]), "tooth")
        )
        # From the tooth back to the base; nothing stands out under the plate.
        run = setback + 0.5 * tooth_mm
        foot_centre = pad + into * (0.5 * run - 0.5 * tooth_mm)
        parts.append(
            GuidePart(np.array([foot_centre[0], foot_centre[1], 0.5 * BASE_THICKNESS_MM]), upright,
                      np.array([0.5 * tooth_width, 0.5 * run, 0.5 * BASE_THICKNESS_MM]), "foot")
        )
    for a, b in zip(feet[:-1], feet[1:]):
        along = np.array([b[0] - a[0], b[1] - a[1], 0.0])
        length = float(np.linalg.norm(along))
        along = along / length
        frame = np.column_stack([along, np.cross(z_axis, along), z_axis])
        parts.append(
            GuidePart(np.array([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * BASE_THICKNESS_MM]), frame,
                      np.array([0.5 * length + 0.5 * BASE_WIDTH_MM, 0.5 * BASE_WIDTH_MM, 0.5 * BASE_THICKNESS_MM]),
                      "base")
        )

    distances = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(sc, axis=0), axis=1))])
    return BendingGuide(
        centres=centres.copy(),
        frames=bent,
        stock_distances=distances,
        joints=joints,
        parts=parts,
        kit=kit,
        width_mm=float(width_mm),
        thickness_mm=float(thickness_mm),
        tooth_mm=float(tooth_mm),
        tooth_width_mm=tooth_width,
        play_mm=float(play_mm),
        pin_diameter_mm=2.0 * pin_radius,
        pin_height_mm=pin_height,
        setback_mm=setback,
        jaws_meet_teeth=jaws_meet,
        basis=basis,
        origin=origin,
        alloy_name=alloy_name,
    )
