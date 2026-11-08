"""The bending guide: a thin comb the plate wraps round, made for one plate.

It stands on the bench (and on the print bed) on a flat base strip that runs
inside the curve of the plate. One thin tooth rises from it at each screw hole
of the finished plate. Each tooth ends in a pad exactly where the bone face of
the plate lies at that hole, turned exactly as the plate is there, and a pin
exactly where the hole's centre is, as thick as this plate's bore allows.

So the guide is used with the bender in hand and nothing clipped on. Take the
plate as it comes out of the packet, hang hole 1 on the flagged tooth with its
lower edge on the ledge, and bend the next bridge until the next hole drops
over the next pin and lies flat on its pad. Let go, and move on: the plate
wraps round the comb from one end. It is right when every hole is on its pin.
A pin is an answer no angle reading can give: a hole on its pin has the right
place and the right turn. Each tooth's hole number is on the base beside it.

Everything is the plate's own, and the kit's:

* hole count and spacing come from the plate (after any trimming), each pin
  takes the shape of its bore, oval or round, tooth height comes from its width,
  pad position from its thickness;
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
#: The ledge under the plate's lower edge at each tooth: gap to the plate, how
#: far past the plate's outer face it stands, its thinnest, mm. Its underside
#: slopes at 45 degrees, so it prints without support.
LEDGE_GAP_MM = 0.1
LEDGE_PAST_MM = 0.6
LEDGE_THIN_MM = 0.6
#: Chamfer on a pin's tip, which lets a hole find it, mm.
PIN_LEAD_IN_MM = 0.35
#: Hole numbers on the base: height, stroke and how far they stand up, mm.
DIGIT_HEIGHT_MM = 3.4
DIGIT_STROKE_MM = 0.55
DIGIT_RAISE_MM = 0.5
#: Play per side of the pins on the printed test piece, mm.
PIN_TEST_PLAYS_MM = (0.15, 0.25, 0.35)
#: Pin as a fraction of the plate's thickness: it locates without standing
#: out of the outer face.
PIN_HEIGHT_FRACTION = 0.75
#: A tooth is at least its height over this thick, mm: a thin tall one bends.
TOOTH_SLENDERNESS = 20.0
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
    #: For a pin: how much longer than wide along the plate, as an oval hole is.
    stretch: float = 0.0

    def field(self, local):
        """Negative inside, about the distance to the surface outside."""
        x, y, z = local
        if self.name == "pin":
            along = np.maximum(np.abs(x) - 0.5 * self.stretch, 0.0)
            radius = self.half[0] - np.maximum(z - (self.half[2] - PIN_LEAD_IN_MM), 0.0)
            return np.maximum(np.hypot(along, y) - radius, np.abs(z) - self.half[2])
        if self.name == "ledge":
            # Flat on top (local y = -half), sloping up to the tooth at z = -half.
            top = y + self.half[1]
            out = z + self.half[2]
            underside = top - (LEDGE_THIN_MM + (2.0 * self.half[2] - out))
            return np.maximum.reduce(
                (np.abs(x) - self.half[0], -top, underside, np.abs(z) - self.half[2])
            )
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
    hole_stretch_mm: float
    hole_diameter_mm: float
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

    def steps(self) -> list[tuple[int, str]]:
        """The bench sequence: ``(bridges bent so far, what to do)``."""
        first = (
            "Fresh plate: hang hole 1 on the flagged tooth, bone face to the teeth, "
            "its lower edge on the ledge."
        )
        out = [(0, first)]
        for joint in self.joints:
            if max(abs(joint.in_plane_deg), abs(joint.out_of_plane_deg), abs(joint.twist_deg)) >= NEGLIGIBLE_DEG:
                out.append((joint.index + 1, self.instruction(joint)))
        out.append((len(self.joints), "Done: every hole on its pin and flat on its pad, after letting go."))
        return out

    def pin_test_parts(self) -> list[GuidePart]:
        """A small tab with pins of each play, to try in a real plate before
        printing the guide. Pins stand straight up; a row of dots counts them."""
        radius = 0.5 * self.hole_diameter_mm
        frame = np.eye(3)
        gap = max(2.0 * radius + self.hole_stretch_mm + 4.0, 7.0)
        count = len(PIN_TEST_PLAYS_MM)
        length = gap * count
        parts = [
            GuidePart(np.array([0.5 * length, 0.0, 0.5 * BASE_THICKNESS_MM]), frame,
                      np.array([0.5 * length, 4.5 + radius, 0.5 * BASE_THICKNESS_MM]), "base")
        ]
        tall = max(self.pin_height_mm, 1.0)
        for i, play in enumerate(PIN_TEST_PLAYS_MM):
            x = gap * (i + 0.5)
            parts.append(
                GuidePart(np.array([x, 0.0, BASE_THICKNESS_MM + 0.5 * tall - 0.2]), frame,
                          np.array([radius - play, radius - play, 0.5 * tall + 0.2]), "pin",
                          stretch=self.hole_stretch_mm)
            )
            parts.extend(_digit_parts(str(i + 1), np.array([x, -(radius + 2.4)]), np.array([1.0, 0.0]), BASE_THICKNESS_MM))
        return parts

    def chain_after(self, supplied_centres, supplied_axes, bridges: int):
        """The plate's holes with its first ``bridges`` bridges bent: ``(centres, axes)``.

        Those holes lie on their pins; the rest of the plate runs on from the
        last of them still as it came, rigid.
        """
        sc = np.asarray(supplied_centres, dtype=float)
        supplied = hole_frames(sc, supplied_axes)
        last = int(np.clip(bridges, 0, len(sc) - 1))
        turn = self.frames[last] @ supplied[last].T
        centres = self.centres.copy()
        axes = self.frames[:, :, 2].copy()
        for j in range(last + 1, len(sc)):
            centres[j] = self.centres[last] + turn @ (sc[j] - sc[last])
            axes[j] = turn @ supplied[j][:, 2]
        return centres, axes

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


#: Which of a seven-segment digit's strokes each numeral lights.
_SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcfgd",
}


def _digit_parts(text: str, centre_xy, ahead_xy, base_top: float) -> list[GuidePart]:
    """Raised seven-segment numerals on a flat surface at height ``base_top``."""
    ahead = np.array([ahead_xy[0], ahead_xy[1], 0.0], dtype=float)
    # Read from the inside of the curve, so "up" the numeral is ahead turned left
    # as seen from above, and writing runs along ``ahead``.
    up = np.array([-ahead[1], ahead[0], 0.0])
    z = np.array([0.0, 0.0, 1.0])
    frame = np.column_stack([ahead, up, z])
    height = DIGIT_HEIGHT_MM
    width = 0.6 * height
    stroke = DIGIT_STROKE_MM
    spans = {
        "a": (0.0, 0.5, True), "g": (0.0, 0.0, True), "d": (0.0, -0.5, True),
        "f": (-0.5, 0.25, False), "b": (0.5, 0.25, False),
        "e": (-0.5, -0.25, False), "c": (0.5, -0.25, False),
    }
    pitch = width + 0.7
    parts = []
    for n, char in enumerate(text):
        offset = (n - 0.5 * (len(text) - 1)) * pitch
        for segment in _SEGMENTS[char]:
            sx, sy, flat = spans[segment]
            local = ahead * (offset + sx * width) + up * (sy * height)
            half = (
                np.array([0.5 * (width + stroke), 0.5 * stroke, 0.5 * DIGIT_RAISE_MM + 0.1])
                if flat
                else np.array([0.5 * stroke, 0.25 * height + 0.5 * stroke, 0.5 * DIGIT_RAISE_MM + 0.1])
            )
            centre = np.array([centre_xy[0], centre_xy[1], base_top + 0.5 * DIGIT_RAISE_MM - 0.1]) + local
            parts.append(GuidePart(centre, frame, half, "digit"))
    return parts


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
    hole_length_mm: float | None = None,
) -> BendingGuide:
    """Lay out the guide that bends a supplied plate into a planned shape.

    ``supplied_*`` are the plate's holes as it comes out of the packet;
    ``bent_*`` the same holes as planned. ``overbend`` maps a target angle in
    degrees to the angle to load it to before springback. ``tooth_mm`` is how
    thin the teeth are and ``play_mm`` the fit of a pin in its hole per side.
    ``hole_length_mm`` is the bore's length along the plate, if oval.
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
    stretch = max(float(hole_length_mm or hole_diameter_mm) - float(hole_diameter_mm), 0.0)
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
    pin_length = 2.0 * pin_radius + stretch
    tooth_width = float(min(max(2.4, pin_length + 0.8), max(pitch - 3.0, pin_length + 0.4)))
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
    ledge_run = 2.0 * half_t + LEDGE_PAST_MM + 0.5 * tooth_mm
    ledge_depth = LEDGE_THIN_MM + ledge_run
    # The ledge's underside, which starts at the plate's lower edge, stays off the base.
    lift = BASE_THICKNESS_MM + max(EDGE_ABOVE_BASE_MM, ledge_depth + LEDGE_GAP_MM + 0.4) - float(np.min(placed[:, 2] - reach))
    placed[:, 2] += lift
    origin = middle - lift * up

    pads = placed - frames[:, :, 2] * (half_t + 0.5 * tooth_mm)
    inward = -frames[:, :, 2].copy()
    inward[:, 2] = 0.0
    if np.min(np.linalg.norm(inward, axis=1)) < 0.2:
        raise ValueError("The plate turns too far out of one plane for a guide to stand on a bench.")
    inward = _unit(inward)
    # Which way along its width each pad's end points down. A tooth comes up to
    # that end, never past it, so a plate leaning towards it does not hit it.
    lows = frames[:, :, 1] * np.where(frames[:, 2, 1] < 0.0, 1.0, -1.0)[:, None]
    posts = pads + lows * half_w
    # The base keeps the setback unless the plate's curve is too tight for it,
    # when it would fold back on itself: stand it nearer rather than fail.
    tangents = np.gradient(posts[:, :2], axis=0)
    while True:
        feet = posts[:, :2] + setback * inward[:, :2]
        if np.all(np.einsum("ij,ij->i", np.diff(feet, axis=0), 0.5 * (tangents[:-1] + tangents[1:])) > 0.5):
            break
        if setback <= 2.0:
            raise ValueError("The plate curves too tightly for a guide behind it; check the plate path.")
        setback = max(setback - 0.5, 2.0)

    parts: list[GuidePart] = []
    pin_height = PIN_HEIGHT_FRACTION * float(thickness_mm)
    z_axis = np.array([0.0, 0.0, 1.0])
    for i, (hole, frame, pad, post, low, into) in enumerate(zip(placed, frames, pads, posts, lows, inward)):
        parts.append(GuidePart(pad, frame, np.array([0.5 * tooth_width, half_w, 0.5 * tooth_mm]), "pad"))
        pin_low = hole - frame[:, 2] * (half_t + 0.25 * tooth_mm)
        pin_high = hole + frame[:, 2] * (pin_height - half_t)
        parts.append(
            GuidePart(0.5 * (pin_low + pin_high), frame,
                      np.array([pin_radius, pin_radius, 0.5 * np.linalg.norm(pin_high - pin_low)]), "pin",
                      stretch=stretch)
        )
        # The ledge the plate's lower edge rests on.
        ledge_frame = np.column_stack([frame[:, 0], low, frame[:, 2]])
        ledge_start = pad + low * (half_w + LEDGE_GAP_MM)
        parts.append(
            GuidePart(
                ledge_start + low * 0.5 * ledge_depth + frame[:, 2] * 0.5 * ledge_run,
                ledge_frame,
                np.array([0.5 * tooth_width, 0.5 * ledge_depth, 0.5 * ledge_run]),
                "ledge",
            )
        )
        # The tooth: thin, but stiffer the taller it stands, keeping its outer
        # face where the pad's is.
        across = np.cross(z_axis, into)
        upright = np.column_stack([across, into, z_axis])
        top = float(post[2]) + 0.3
        thick = max(tooth_mm, top / TOOTH_SLENDERNESS)
        parts.append(
            GuidePart(np.array([post[0], post[1], 0.5 * top]) + into * 0.5 * (thick - tooth_mm), upright,
                      np.array([0.5 * tooth_width, 0.5 * thick, 0.5 * top]), "tooth")
        )
        if i == 0:
            parts.append(
                GuidePart(pad - low * (half_w + 0.5 * FLAG_MM), frame,
                          np.array([0.5 * tooth_width, 0.5 * FLAG_MM + 0.2, 0.5 * tooth_mm]), "flag")
            )
        # From the tooth back to the base; nothing stands out under the plate.
        run = setback + 0.5 * tooth_mm
        foot_centre = post + into * (0.5 * run - 0.5 * tooth_mm)
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

    # Each tooth's hole number, on the base beside it, read from inside the curve.
    for i, (foot, into) in enumerate(zip(feet, inward)):
        ahead = feet[min(i + 1, len(feet) - 1)] - feet[max(i - 1, 0)]
        ahead = ahead / max(float(np.linalg.norm(ahead)), 1e-9)
        parts.extend(_digit_parts(str(i + 1), foot, ahead, BASE_THICKNESS_MM))

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
        hole_stretch_mm=stretch,
        hole_diameter_mm=float(hole_diameter_mm),
        setback_mm=setback,
        jaws_meet_teeth=jaws_meet,
        basis=basis,
        origin=origin,
        alloy_name=alloy_name,
    )
