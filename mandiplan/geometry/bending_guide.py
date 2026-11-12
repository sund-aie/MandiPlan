"""Bending keys: one small printed form for each bend of a plate, and the checks they pass.

The plate comes out of the packet straight. Each bridge between two holes has to
turn by some angle, and the bender has no way of knowing how much. So each bend
gets its own key, made for this plate as cut and this bender.

**Press keys, for bends out of the plate's plane.** With a three-point bender (a post
on one side of the plate, two lobes and an arch between them on the other) the key
is a short block that sits up in the arch, on the plate's convex side. Its underside
is a shallow V, a die: two faces that are the plate's own surface over the two lobes
either side of the bridge, in the finished shape with the alloy's springback allowed
for, a round channel along the fold between them. The post is the punch. Squeezing
the handles pushes the plate up into the V, bending it at the bridge, until both lobes
lie flat on the faces. Then each lobe is held between the post and its face, and the
bridge takes no more bending however hard the handles are pressed: the angle comes
from the key, not from the hand. Its two pins sit in the two holes; the second one is
flattened along the plate, so the holes can come together or apart as the bridge
between them bends without either pin binding.

The key bears on the arch, not on the lobes: under the lobes there is room for only a
thin key, which a hard squeeze would crack. The lobes stand clear of the plate the whole
way. Past its two faces the key's underside rises steeply, so whatever the plate does
beyond them, bent already or not yet, never touches it.

**Check keys, for bends in the plate's own plane.** A lobe of a plate is round about its
hole: pressed against a flat face it can turn about the hole and still touch it, so no
face can stop an in-plane bend. These keys are flat trays with pins for up to four
holes, two either side of the bridge, where the planned turn puts them. The bend is
made with an in-plane bender, a little at a time, until the plate lies flat on the
tray with every pin in its hole.

The plan follows the plate maker's order: in the plate's own plane first, then out of
it. A twist goes in the same key as that bridge's out-of-plane bend.

**What is checked.** For every press key the press is followed step by step, from the
plate as it reaches that key to the stop: the rest of the plate has to stay clear of
the key and of the bender's head, and each pin clear of its hole's wall. The key's
bending stress is worked out for a hard squeeze. ``BendingGuide.check`` gives it all.

Numpy only. The solids are given as distance-like fields, and
``render/bending_guide_mesh.py`` contours their union.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: A bridge turning less than this is left alone: it would barely yield, degrees.
MIN_BEND_DEG = 1.0
#: A key's face covers its lobe of the plate this far from the hole, as a share of the
#: hole spacing, or half the plate's width if less: the lobe, not the next bridge.
LOBE_SHARE_OF_PITCH = 0.42
#: Past its faces a key's underside rises this steeply (tan 60 degrees).
BEVEL_SLOPE = float(np.tan(np.radians(60.0)))
#: How far a key runs on past its faces, mm.
KEY_END_MM = 1.5
#: How close to the bender's head a key's top is made, mm.
TOP_CLEARANCE_MM = 0.15
#: The head's lobes stand at least this far off the plate throughout the press, mm.
LOBE_GAP_MM = 1.0
#: The thinnest a key may be at its ends, mm.
MIN_KEY_END_MM = 3.0
#: A round channel along the fold of the key, mm: the plate's bridge curves there
#: instead of folding, and the key has no sharp notch to crack from.
RIDGE_RELIEF_MM = 0.8
#: How much wider than the plate a key is, as the head allows, mm.
KEY_EXTRA_WIDTH_MM = 3.0
#: Pins taper to this share of their base, so that they find a hole that is still
#: turning and end up snug; how long they are, as a share of the plate's thickness;
#: and how far they run into the key, mm.
PIN_TIP_SHARE = 0.55
PIN_LENGTH_SHARE = 0.85
PIN_EMBED_MM = 0.5
#: All but the first pin of a key are flattened along the plate by this much a side (a
#: diamond pin, as fixtures for two holes use), mm.
DIAMOND_RELIEF_MM = 0.35
#: How thick the block of a lever kit's key is under the plate, mm.
LEVER_BASE_MM = 7.0
#: Check keys: the tray's thickness, how far it runs past the lobes, and the strip that
#: carries the name, mm.
TRAY_MM = 2.0
TRAY_MARGIN_MM = 1.5
TRAY_LABEL_MM = 5.0
#: The numerals on each key: height, stroke and how far they stand up, mm.
DIGIT_HEIGHT_MM = 3.2
DIGIT_STROKE_MM = 0.55
DIGIT_RAISE_MM = 0.4
#: Twist a press can be asked to give in one key, degrees.
MAX_PRESSED_TWIST_DEG = 8.0
#: Turns the plate's plan asks for are gathered into as few bends, and so as few keys,
#: as keep every hole within these of where all of them would put it, mm. In the plate's
#: own plane a hole only moves along the bone. Out of it, it moves off the bone.
IN_PLANE_TOLERANCE_MM = 0.8
OUT_OF_PLANE_TOLERANCE_MM = 0.25
#: No gathered turn is to be more than this, unless the plan itself asks for it, degrees.
MAX_GATHERED_DEG = 15.0
#: Twist left over from bridges with no key goes into the next key; beyond this it
#: gets a key of its own, degrees.
MAX_CARRIED_TWIST_DEG = 3.0
#: A plate this far from its plan after all the keys is worth saying, mm.
SHAPE_CAUTION_MM = 1.5
#: Play per side of the pins tried on the fit test, mm.
PIN_TEST_PLAYS_MM = (0.05, 0.1, 0.15)
#: The bend the fit test's key and gauge are made for, degrees.
FIT_TEST_ANGLE_DEG = 15.0
#: The squeeze a key is checked against, as force on the post, N: a hard one.
DESIGN_FORCE_N = 2000.0
#: The bending stress a key may see then, MPa: about half what a rigid printing resin
#: takes in bending.
KEY_STRESS_LIMIT_MPA = 50.0
#: How many steps the press is followed in, from the plate reaching the key to the stop.
PRESS_STEPS = 12
#: Contact this close counts as touching, mm.
TOUCH_MM = 0.02


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


def _half(rotation: np.ndarray) -> np.ndarray:
    axis, angle = _axis_angle(rotation)
    return _rotation(axis, 0.5 * angle)


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


def _decompose(rotation: np.ndarray) -> tuple[float, float, float]:
    """In-plane, out-of-plane and twist of a bend, degrees, as the bend table.

    ``rotation = Rz(in) Ry(-out) Rx(twist)`` in the bridge's frame: in the plate's
    own plane first, then out of it, then the twist. In-plane positive is
    counter-clockwise seen from the outer face, out-of-plane positive lifts the
    plate away from the bone, twist is right-handed about the direction of travel.
    """
    v = rotation[:, 0]
    in_plane = np.arctan2(v[1], v[0])
    out_plane = np.arctan2(v[2], np.hypot(v[0], v[1]))
    undone = _rotation([0, 1, 0], out_plane) @ _rotation([0, 0, 1], -in_plane) @ rotation
    twist = np.arctan2(undone[2, 1], undone[1, 1])
    return float(np.degrees(in_plane)), float(np.degrees(out_plane)), float(np.degrees(twist))


def _compose(in_plane_deg: float, out_plane_deg: float, twist_deg: float) -> np.ndarray:
    return (
        _rotation([0, 0, 1], np.radians(in_plane_deg))
        @ _rotation([0, 1, 0], -np.radians(out_plane_deg))
        @ _rotation([1, 0, 0], np.radians(twist_deg))
    )


def die_height(head, x):
    """How high the underside of a three-point head stands above its lobes' lowest
    points, at distance ``x`` from the middle: the arch, then each lobe."""
    x = np.abs(np.asarray(x, dtype=float))
    a = 0.5 * head.lobe_spacing_mm
    rho = head.lobe_radius_mm

    def lobe(at):
        off = np.minimum(np.abs(at - a), rho)
        return rho - np.sqrt(np.maximum(rho * rho - off * off, 0.0))

    if head.arch_rise_mm <= 0.0:
        return lobe(x)  # rollers on a flat-faced press
    radius = head.arch_radius_mm

    def arch(at):
        return head.arch_rise_mm - radius + np.sqrt(np.maximum(radius * radius - at * at, 0.0))

    # The arch runs out into each lobe where the two curves come closest.
    start = max(a - rho, 0.0)
    probe = np.linspace(start, max(min(a, radius), start + 1e-3), 48)
    meet = float(probe[int(np.argmin(np.abs(arch(probe) - lobe(probe))))])
    return np.where(x < meet, arch(x), lobe(x))


@dataclass
class GuidePart:
    """One convex piece of a solid, its own axes given as the columns of ``frame``."""

    centre: np.ndarray
    frame: np.ndarray
    half: np.ndarray  # half sizes along those axes
    name: str
    #: For a pin: how much longer than wide along its local x, as an oval hole is.
    stretch: float = 0.0
    #: For a pin: the radius at its tip, its base radius being ``half[0]``.
    tip_radius: float = 0.0
    #: For a pin: how much it is flattened along its local x, a side.
    relief: float = 0.0

    def field(self, local):
        """Negative inside, about the distance to the surface outside."""
        x, y, z = local
        if self.name == "pin":
            along = np.maximum(np.abs(x) - 0.5 * self.stretch, 0.0)
            # Local +z is towards the tip.
            share = (z + self.half[2]) / (2.0 * self.half[2])
            radius = self.half[0] + (self.tip_radius - self.half[0]) * share
            inside = np.maximum(np.hypot(along, y) - radius, np.abs(z) - self.half[2])
            if self.relief > 0.0:
                inside = np.maximum(inside, np.abs(x) - (self.half[0] + 0.5 * self.stretch - self.relief))
            return inside
        if self.name == "post":
            return np.maximum(np.hypot(x, z) - self.half[0], np.abs(y) - self.half[1])
        return np.maximum.reduce(
            (np.abs(x) - self.half[0], np.abs(y) - self.half[1], np.abs(z) - self.half[2])
        )


@dataclass
class KeyBody(GuidePart):
    """The block of a key, in the key's own frame: x along the plate, y along the
    bender's post, z from the plate towards the head. Origin at the bridge."""

    planes: tuple = ()  # ((point, normal), (point, normal)): the plate's surface
    head: object = None
    lobes_z: float = 0.0
    face_end: float = 0.0
    y_range: tuple[float, float] = (0.0, 0.0)
    x_reach: float = 0.0
    ridge: tuple = ()  # (point, direction): the channel along the fold
    channel_radius: float = RIDGE_RELIEF_MM
    lever_floor: float | None = None
    lever_top: float = 0.0

    def face_field(self, local):
        """The faces alone: negative on the key's side of them, the bevels included.

        Each face is its own lobe's, on its own side of the bridge. Where a bend twists
        or turns in plane too, the two faces cross on a slant, and either one carried on
        past the bridge would reach over the other lobe.
        """
        x, y, z = local
        (qa, na), (qb, nb) = self.planes
        above_a = (x - qa[0]) * na[0] + (y - qa[1]) * na[1] + (z - qa[2]) * na[2]
        above_b = (x - qb[0]) * nb[0] + (y - qb[1]) * nb[1] + (z - qb[2]) * nb[2]
        # Past the plate's two lobes the faces fall away steeply.
        lift = BEVEL_SLOPE * np.maximum(np.abs(x) - self.face_end, 0.0)
        return lift - np.where(x < 0.0, above_a, above_b)

    def field(self, local):
        x, y, z = local
        walls = np.maximum.reduce(
            (np.abs(x) - self.x_reach, self.y_range[0] - y, y - self.y_range[1])
        )
        faces = self.face_field(local)
        if self.lever_floor is not None:
            return np.maximum.reduce((faces, self.lever_floor - z, z - self.lever_top, walls))
        solid = np.maximum.reduce((faces, z - self.top(x), walls))
        # The round channel along the fold.
        point, direction = self.ridge
        px, py, pz = x - point[0], y - point[1], z - point[2]
        along = px * direction[0] + py * direction[1] + pz * direction[2]
        off = np.sqrt(
            (px - along * direction[0]) ** 2
            + (py - along * direction[1]) ** 2
            + (pz - along * direction[2]) ** 2
        )
        return np.maximum(solid, self.channel_radius - off)

    def face_height(self, x, y=0.0):
        """The height of a face over ``(x, y)``, the bevels included: its own lobe's."""
        x = np.asarray(x, dtype=float)
        lift = BEVEL_SLOPE * np.maximum(np.abs(x) - self.face_end, 0.0)
        heights = [q[2] + (lift - (x - q[0]) * n[0] - (y - q[1]) * n[1]) / n[2] for q, n in self.planes]
        return np.where(x < 0.0, heights[0], heights[1])

    def underside(self, x, y=0.0):
        """The height of a press key's underside over ``(x, y)``, the fold's channel aside."""
        return self.face_height(x, y)

    def top(self, x):
        """The height of a press key's top: just under the bender's head."""
        return self.lobes_z + die_height(self.head, x) - TOP_CLEARANCE_MM


@dataclass
class TrayBody(GuidePart):
    """The tray of a check key, in its own frame: the plate lies on it bone face down,
    its top at ``top``. Round each held hole's lobe and the run between them, with a
    strip for the name."""

    holes: np.ndarray = None  # (N, 2) x, y of the held holes
    radius: float = 0.0
    label: tuple = ()  # (x0, x1, y0, y1): the strip carrying the name
    top: float = 0.0

    def field(self, local):
        x, y, z = local
        slab = np.maximum(z - self.top, self.top - TRAY_MM - z)
        best = None
        for (ax, ay), (bx, by) in zip(self.holes[:-1], self.holes[1:]):
            dx, dy = bx - ax, by - ay
            length2 = max(dx * dx + dy * dy, 1e-12)
            t = np.clip(((x - ax) * dx + (y - ay) * dy) / length2, 0.0, 1.0)
            d = np.hypot(x - (ax + t * dx), y - (ay + t * dy)) - self.radius
            best = d if best is None else np.minimum(best, d)
        x0, x1, y0, y1 = self.label
        strip = np.maximum.reduce((x0 - x, x - x1, y0 - y, y - y1))
        return np.maximum(slab, np.minimum(best, strip))


@dataclass
class DieBody(GuidePart):
    """The bender's head, for the rehearsal and the checks: everything above its underside."""

    head: object = None
    lobes_z: float = 0.0

    def field(self, local):
        x, y, z = local
        z = z + self.centre[2]
        wide = 0.5 * self.head.lobe_spacing_mm + self.head.lobe_radius_mm
        top = self.lobes_z + self.head.arch_rise_mm + 5.0
        return np.maximum.reduce(
            (self.lobes_z + die_height(self.head, x) - z, z - top, np.abs(x) - wide, np.abs(y) - self.half[1])
        )


#: Which of a seven-segment character's strokes it lights.
_SEGMENTS = {
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcfgd",
    "A": "abcefg", "B": "cdefg", "P": "abefg",
}


def digit_parts(text: str, centre, ahead, up, normal) -> list[GuidePart]:
    """Raised seven-segment characters on a flat face whose outward normal is
    ``normal``, reading along ``ahead`` with ``up`` as their top."""
    frame = np.column_stack([ahead, up, normal])
    height = DIGIT_HEIGHT_MM
    width = 0.6 * height
    stroke = DIGIT_STROKE_MM
    spans = {
        "a": (0.0, 0.5, True), "g": (0.0, 0.0, True), "d": (0.0, -0.5, True),
        "f": (-0.5, 0.25, False), "b": (0.5, 0.25, False),
        "e": (-0.5, -0.25, False), "c": (0.5, -0.25, False),
    }
    pitch = width + 0.8
    parts = []
    for n, char in enumerate(text):
        offset = (n - 0.5 * (len(text) - 1)) * pitch
        for segment in _SEGMENTS[char]:
            sx, sy, flat = spans[segment]
            half = (
                np.array([0.5 * (width + stroke), 0.5 * stroke, 0.5 * DIGIT_RAISE_MM + 0.15])
                if flat
                else np.array([0.5 * stroke, 0.25 * height + 0.5 * stroke, 0.5 * DIGIT_RAISE_MM + 0.15])
            )
            place = (
                np.asarray(centre, dtype=float)
                + np.asarray(ahead) * (offset + sx * width)
                + np.asarray(up) * (sy * height)
                + np.asarray(normal) * (0.5 * DIGIT_RAISE_MM - 0.15)
            )
            parts.append(GuidePart(place, frame, half, "digit"))
    return parts


@dataclass
class GuideJoint:
    """The bridge between hole ``index`` and hole ``index + 1``."""

    index: int
    angle_deg: float
    stop_angle_deg: float  # how far it is loaded, the alloy springing back from there
    in_plane_deg: float
    out_of_plane_deg: float
    twist_deg: float
    #: What the keys make there, the plan's turns being gathered into fewer bends.
    in_plane_made_deg: float = 0.0
    out_of_plane_made_deg: float = 0.0
    twist_made_deg: float = 0.0


@dataclass
class KeyCheck:
    """What following the press of one key found."""

    plate_clear_of_key_mm: float  # the rest of the plate to the key, at its closest
    plate_clear_of_head_mm: float  # the plate to the bender's head, at its closest
    pin_clear_mm: float  # each pin to its hole's wall, at its closest
    stress_mpa: float  # in the key at DESIGN_FORCE_N on the post
    opening_mm: float  # how far the head must open to take the plate and key
    lobes_flat: bool  # the two lobes lie on the faces at the stop

    @property
    def ok(self) -> bool:
        return (
            self.plate_clear_of_key_mm >= -TOUCH_MM
            and self.plate_clear_of_head_mm >= -TOUCH_MM
            and self.pin_clear_mm >= -TOUCH_MM
            and self.stress_mpa <= KEY_STRESS_LIMIT_MPA
            and self.lobes_flat
        )


@dataclass
class BendKey:
    """The printed form for one bend."""

    number: int  # its place in the sequence, from 1
    bridge: int  # hole ``bridge`` and hole ``bridge + 1``
    kind: str  # "in_plane" or "out_of_plane"
    name: str  # what is printed on it
    angle_deg: float  # the released angle it makes
    stop_angle_deg: float  # the loaded angle its faces are at
    parts: list[GuidePart] = field(default_factory=list)
    #: The two lobes of the plate where the key holds them, in the key's frame.
    frame_a: np.ndarray = None
    frame_b: np.ndarray = None
    hole_a: np.ndarray = None
    hole_b: np.ndarray = None
    faces: tuple = ()  # the two planes the plate comes to lie on (press keys)
    from_bone_side: bool = False  # its pins enter from the plate's bone face
    #: A check key, a tray for an in-plane bend, rather than a press key.
    check: bool = False
    #: The holes it has pins in, by index.
    pinned: tuple[int, ...] = ()
    #: How far the head must open to load the plate and key, mm.
    loading_depth_mm: float = 0.0
    #: The head's lobes' lowest points, and the post's top at the stop, z.
    lobes_z: float = 0.0
    post_top_z: float = 0.0
    #: Each bridge's turn, in its own frame, made before this key.
    applied_before: list = None
    #: This bridge's turn before the press, at the stop, and as released.
    turn_start: np.ndarray = None
    turn_stop: np.ndarray = None
    turn_made: np.ndarray = None
    #: From the bridge's frame to the key's, and lobe A in the bridge's frame.
    to_key: np.ndarray = None
    lobe_a: tuple = ()  # (frame, hole)


@dataclass
class GuideStep:
    key: BendKey | None
    text: str


@dataclass
class BendingGuide:
    """The set of keys for one plate as cut, and one bender."""

    centres: np.ndarray  # the finished plate's hole centres, patient space
    frames: np.ndarray  # (H, 3, 3) the finished plate's hole frames, patient space
    stock_centres: np.ndarray
    stock_frames: np.ndarray
    stock_distances: np.ndarray  # along the supplied plate from hole 1, mm
    joints: list[GuideJoint]
    keys: list[BendKey]
    kit: object
    head: object
    width_mm: float
    thickness_mm: float
    hole_diameter_mm: float
    hole_stretch_mm: float
    play_mm: float
    pin_diameter_mm: float
    alloy_name: str = ""
    warnings: list[str] = field(default_factory=list)
    #: How far the plate the keys make lies from the planned plate, at its worst hole, mm.
    deviation_mm: float = 0.0
    #: Maps a released angle to the angle to load it to, degrees.
    overbend: object = None
    _checks: dict = field(default_factory=dict)

    # -- what to do -------------------------------------------------------

    def instruction(self, key: BendKey) -> str:
        a, b = key.bridge + 1, key.bridge + 2
        if key.check:
            held = ", ".join(str(i + 1) for i in key.pinned)
            return (
                f"Key {key.name} checks the turn in the plate's own plane between holes {a} and {b}. Bend it "
                "with your in-plane bender a little at a time, and lay the plate on the key, bone face down, after "
                f"each go: it is right when the plate lies flat on the key with its pins in holes {held}."
            )
        face = "bone" if key.from_bone_side else "outer"
        seat = (
            f"Key {key.name}: holes {a} and {b}. Seat its round pin in hole {a} and its flattened pin in hole {b}, "
            f"from the plate's {face} face."
        )
        if self.head.kind == "lever":
            return f"{seat} Lever the plate down onto the key until it lies flat on it."
        return (
            f"{seat} Put the key up into the bender's arch, the post under the bridge between the two holes. "
            f"Squeeze until the plate lies flat on the key: it bends {key.angle_deg:.1f} deg and stops there, "
            "however hard you squeeze. Let go."
        )

    def instructions(self) -> list[str]:
        return [self.instruction(key) for key in self.keys]

    def steps(self) -> list[GuideStep]:
        """The bench sequence, with what to do at each step."""
        start = (
            "A fresh plate, cut to length. Lay out the keys in their order. Count the plate's holes from the end "
            "that lies first in the plan: that is hole 1."
        )
        out = [GuideStep(None, start)]
        out.extend(GuideStep(key, self.instruction(key)) for key in self.keys)
        out.append(GuideStep(None, "All the bends are made. Try the plate on the printed jaw."))
        return out

    # -- where the plate is ------------------------------------------------

    def _chain(self, applied):
        """Hole places and frames of the plate with these turns at its bridges, hole 1
        at the origin as supplied."""
        count = len(self.stock_centres)
        world = [np.eye(3)]
        places = [np.zeros(3)]
        for j in range(count - 1):
            before = self.stock_frames[j].T @ self.stock_frames[j + 1]
            chord = self.stock_frames[j].T @ (self.stock_centres[j + 1] - self.stock_centres[j])
            to_joint = _joint_frame(self.stock_centres, self.stock_frames, j).T @ self.stock_frames[j]
            turn = to_joint.T @ applied[j] @ to_joint
            hinge = 0.5 * chord
            places.append(places[j] + world[j] @ (hinge + turn @ (chord - hinge)))
            world.append(world[j] @ turn @ before)
        return np.array(places), np.array(world)

    def _made(self, upto: int):
        """Each bridge's turn, in its own frame, once the keys before number ``upto``
        are made and released."""
        applied = []
        for joint in self.joints:
            in_key = next((k for k in self.keys if k.bridge == joint.index and k.kind == "in_plane"), None)
            out_key = next((k for k in self.keys if k.bridge == joint.index and k.kind == "out_of_plane"), None)
            phi = joint.in_plane_made_deg if in_key is not None and in_key.number < upto else 0.0
            done = out_key is not None and out_key.number < upto
            theta = joint.out_of_plane_made_deg if done else 0.0
            tau = joint.twist_made_deg if done else 0.0
            applied.append(_compose(phi, theta, tau))
        return applied

    def pose(self, key: BendKey, fraction: float = 1.0):
        """The plate's holes, ``(centres, frames)``, in ``key``'s frame.

        At ``fraction`` 1 the plate is at a press key's stop, or a check key's plate as it
        should come out. Below 1 the press is that far along: this bridge bent that share
        of the way, the earlier keys' bends made, the plate up against the key.
        """
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if key.check:
            turn = key.turn_made
        else:
            axis, angle = _axis_angle(key.turn_stop @ key.turn_start.T)
            turn = _rotation(axis, fraction * angle) @ key.turn_start
        applied = list(key.applied_before)
        applied[key.bridge] = turn
        places, world = self._chain(applied)
        half = _half(turn)
        frame_a = key.to_key @ half.T @ key.lobe_a[0]
        hole_a = key.to_key @ half.T @ key.lobe_a[1]
        to = frame_a @ world[key.bridge].T
        places = places @ to.T + (hole_a - to @ places[key.bridge])
        world = np.einsum("ij,hjk->hik", to, world)
        if not key.check and fraction < 1.0:
            places = places + np.array([0.0, 0.0, self._contact_shift(key, places, world)])
        return places, world

    def _contact_shift(self, key: BendKey, places, frames) -> float:
        """How far the plate must move along the key's z to just touch its faces."""
        body = key.parts[0]
        held = [key.bridge, key.bridge + 1]
        points = _lobe_points(places[held], frames[held], self._lobe_half(), self.thickness_mm)
        towards = 1.0 if float(body.planes[0][1][2]) > 0.0 else -1.0
        shift = 0.0
        for _ in range(8):
            gap = float(np.min(body.face_field(tuple((points + np.array([0.0, 0.0, shift])).T))))
            if abs(gap) < 1e-5:
                break
            shift += towards * gap
        return shift

    def stage(self, key: BendKey | None = None, done: bool = False):
        """The plate's holes, ``(centres, axes)``, for the rehearsal: at ``key``'s stop
        in that key's frame, or with no key fresh, or if ``done`` with every bend made."""
        if key is not None:
            places, world = self.pose(key, 1.0)
            return places, world[:, :, 2]
        if done:
            places, world = self._chain(self._made(len(self.keys) + 1))
        else:
            places, world = self._chain([np.eye(3)] * (len(self.stock_centres) - 1))
        return places, world[:, :, 2]

    # -- checks -------------------------------------------------------------

    def _lobe_half(self) -> float:
        pitch = float(np.median(np.linalg.norm(np.diff(self.stock_centres, axis=0), axis=1)))
        return min(0.5 * self.width_mm, LOBE_SHARE_OF_PITCH * pitch)

    def check(self, key: BendKey) -> KeyCheck:
        """Follow the press of ``key`` and report what was found."""
        if key.number not in self._checks:
            self._checks[key.number] = _check_key(self, key)
        return self._checks[key.number]

    def checks(self) -> list[KeyCheck]:
        return [self.check(key) for key in self.keys]

    def check_warnings(self) -> list[str]:
        """What following each key's press found wrong, in words."""
        out = []
        for key in self.keys:
            result = self.check(key)
            if result.ok:
                continue
            if result.stress_mpa > KEY_STRESS_LIMIT_MPA:
                out.append(
                    f"Key {key.name} is thin across its fold for a bend with this much twist: about "
                    f"{result.stress_mpa:.0f} MPa in it at a hard squeeze. Give the twist with twisting irons first, "
                    "then squeeze it on the key gently."
                )
            if min(result.plate_clear_of_key_mm, result.plate_clear_of_head_mm) < -TOUCH_MM:
                out.append(f"Key {key.name}: the plate touches the key or the head outside its two lobes during the press.")
            if result.pin_clear_mm < -TOUCH_MM:
                out.append(f"Key {key.name}: a pin binds in its hole during the press.")
            if not result.lobes_flat:
                out.append(f"Key {key.name}: the plate does not come to lie flat on it.")
        return out


def _lobe_points(centres, frames, radius: float, thickness: float, count: int = 16):
    """Points round the rim of each lobe and at its centre, on both of its faces."""
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    out = []
    for c, f in zip(centres, frames):
        along, across, normal = f[:, 0], f[:, 1], f[:, 2]
        ring = [c] + [c + radius * (np.cos(a) * along + np.sin(a) * across) for a in angles]
        for side in (-0.5, 0.5):
            out.extend(p + side * thickness * normal for p in ring)
    return np.array(out)


def head_clearance(head, lobes_z: float, points: np.ndarray) -> float:
    """How far the plate's ``points``, in a key's frame, stay from the bender's head: the
    least distance to its underside, the two lobes and the arch between them, mm. The
    head is as deep as its pocket; past its lobes it is open."""
    if head is None or head.kind != "three_point" or not len(points):
        return float("inf")
    a = 0.5 * head.lobe_spacing_mm
    reach = a + head.lobe_radius_mm
    near = (np.abs(points[:, 1]) < 0.5 * head.pocket_depth_mm) & (np.abs(points[:, 0]) < reach + 3.0)
    if not near.any():
        return float("inf")
    p = points[near]
    xs = np.linspace(-reach, reach, 2 * int(reach / 0.1) + 1)
    zs = lobes_z + die_height(head, xs)
    distance = np.min(np.hypot(p[:, 0, None] - xs[None, :], p[:, 2, None] - zs[None, :]), axis=1)
    inside = (np.abs(p[:, 0]) <= reach) & (p[:, 2] > lobes_z + die_height(head, p[:, 0]))
    return float(np.min(np.where(inside, -distance, distance)))


def _check_key(guide: BendingGuide, key: BendKey) -> KeyCheck:
    lobe_half = guide._lobe_half()
    width_half = 0.5 * guide.width_mm
    t = guide.thickness_mm
    pins = [p for p in key.parts if p.name == "pin"]
    if key.check:
        places, frames = guide.pose(key, 1.0)
        pin_clear = min(_pin_clearance(pin, places[i], frames[i], guide) for pin, i in zip(pins, key.pinned))
        points = _lobe_points(places, frames, width_half, t)
        clear_tray = float(np.min(key.parts[0].field(tuple(points.T))))
        return KeyCheck(clear_tray, float("inf"), pin_clear, 0.0, 0.0, True)
    body = key.parts[0]
    held = (key.bridge, key.bridge + 1)
    clear_key = clear_head = pin_clear = float("inf")
    for fraction in np.linspace(0.0, 1.0, PRESS_STEPS + 1):
        places, frames = guide.pose(key, fraction)
        rest = [i for i in range(len(places)) if i not in held]
        if rest:
            points = _lobe_points(places[rest], frames[rest], width_half, t)
            clear_key = min(clear_key, float(np.min(body.field(tuple(points.T)))))
        if body.lever_floor is None:
            everything = _lobe_points(places, frames, width_half, t)
            clear_head = min(clear_head, head_clearance(guide.head, key.lobes_z, everything))
        for pin, i in zip(pins, held):
            pin_clear = min(pin_clear, _pin_clearance(pin, places[i], frames[i], guide))
    # At the stop the face of each lobe towards the key lies on it.
    places, frames = guide.pose(key, 1.0)
    rims = _lobe_points(places[list(held)], frames[list(held)], 0.8 * lobe_half, t, 8)
    gaps = body.face_field(tuple(rims.T))
    flat = bool(np.sum(np.abs(gaps) < 0.05) >= len(rims) // 2 and np.min(gaps) > -0.05)
    return KeyCheck(clear_key, clear_head, pin_clear, key_stress_mpa(key, DESIGN_FORCE_N, lobe_half),
                    key.loading_depth_mm, flat)


def _pin_clearance(pin: GuidePart, centre, frame, guide: BendingGuide) -> float:
    """The least room between a pin and its hole's wall, within the plate, mm."""
    radius = 0.5 * guide.hole_diameter_mm
    stretch = guide.hole_stretch_mm
    t = guide.thickness_mm
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    best = float("inf")
    limit = pin.half[0] + 0.5 * pin.stretch - pin.relief if pin.relief > 0.0 else np.inf
    for h in np.linspace(0.0, 1.0, 11):
        z = -pin.half[2] + 2.0 * pin.half[2] * h
        r = pin.half[0] + (pin.tip_radius - pin.half[0]) * h
        x = np.clip(np.cos(angles) * r + np.sign(np.cos(angles)) * 0.5 * pin.stretch, -limit, limit)
        y = np.sin(angles) * r
        points = pin.centre + np.column_stack([x, y, np.full(len(x), z)]) @ pin.frame.T
        rel = (points - centre) @ frame
        within = np.abs(rel[:, 2]) <= 0.5 * t
        if not within.any():
            continue
        off = np.hypot(np.maximum(np.abs(rel[within, 0]) - 0.5 * stretch, 0.0), rel[within, 1])
        best = min(best, float(np.min(radius - off)))
    return best


def key_stress_mpa(key: BendKey, force_n: float, lobe_half: float) -> float:
    """The nominal bending stress in a press key with ``force_n`` on the post, MPa.

    The key is a short beam: the plate pushes up on its two faces, the head down on its
    top. Where on its top is not known for a head measured by eye, so the worst of three
    is taken: spread along it, all at its middle, all at its two ends.
    """
    body = key.parts[0]
    if key.check or body.lever_floor is not None:
        return 0.0
    width = body.y_range[1] - body.y_range[0]
    reach = body.x_reach
    xs = np.linspace(0.0, reach - 1e-3, 200)
    heights = np.maximum(body.top(xs) - body.underside(xs), 0.1)
    channel = body.channel_radius
    heights = np.where(xs < channel, np.maximum(heights - channel * np.sqrt(np.maximum(1.0 - (xs / channel) ** 2, 0.0)), 0.1), heights)
    inner = max(abs(float(key.hole_a[0])) - lobe_half, channel)
    outer = body.face_end
    half = 0.5 * force_n
    q = half / max(outer - inner, 1e-6)
    s0 = np.maximum(xs, inner)
    plate = np.where(outer > s0, 0.5 * q * ((outer - xs) ** 2 - (s0 - xs) ** 2), 0.0)
    worst = 0.0
    for head in (0.5 * half / reach * (reach - xs) ** 2, np.zeros_like(xs), half * (reach - xs)):
        stress = 6.0 * np.abs(plate - head) / (width * heights**2)
        worst = max(worst, float(stress.max()))
    return worst


def _laid_flat(turns_deg: np.ndarray, pitches: np.ndarray) -> np.ndarray:
    """A plate turning this much at each bridge, in a plane: where it starts, each
    bridge where it turns, and where it ends, the plate running straight between."""
    places = [np.zeros(2)]
    heading = 0.0
    for turn, pitch in zip(turns_deg, pitches):
        places.append(places[-1] + 0.5 * pitch * np.array([np.cos(heading), np.sin(heading)]))
        heading += np.radians(turn)
        last = places[-1] + 0.5 * pitch * np.array([np.cos(heading), np.sin(heading)])
        places.append(last)
    # Hole, bridge, hole, bridge, hole: only the bridges can turn. A hole on a straight
    # run through it comes out on that run, so the holes are dropped.
    return np.array([places[0], *places[1::2], places[-1]])


def _simplified(places: np.ndarray, tolerance: float) -> list[int]:
    """Which places a straight run between them may turn at, for all the rest to stay
    within ``tolerance`` of that run (Ramer, Douglas and Peucker)."""
    keep = {0, len(places) - 1}
    work = [(0, len(places) - 1)]
    while work:
        a, b = work.pop()
        if b - a < 2:
            continue
        chord = places[b] - places[a]
        length = float(np.linalg.norm(chord))
        off = places[a + 1 : b] - places[a]
        if length < 1e-9:
            distance = np.linalg.norm(off, axis=1)
        else:
            distance = np.abs(off[:, 0] * chord[1] - off[:, 1] * chord[0]) / length
        k = int(np.argmax(distance))
        if distance[k] > tolerance:
            keep.add(a + 1 + k)
            work.extend(((a, a + 1 + k), (a + 1 + k, b)))
    return sorted(keep)


def _gathered(turns_deg, pitches, tolerance: float) -> np.ndarray:
    """The plan's turns at its bridges, gathered into as few bends as stay within
    ``tolerance`` of it. A gathered bend much sharper than the plan's own sharpest is
    split again where it is, and only there."""
    turns = np.asarray(turns_deg, dtype=float)
    places = _laid_flat(turns, np.asarray(pitches, dtype=float))
    limit = max(MAX_GATHERED_DEG, float(np.abs(turns).max(initial=0.0))) + 1e-6
    kept = _simplified(places, tolerance)

    def changes(kept):
        runs = np.diff(places[kept], axis=0)
        headings = np.unwrap(np.arctan2(runs[:, 1], runs[:, 0]))
        return np.degrees(np.diff(headings))

    for _ in range(len(places)):
        change = changes(kept)
        sharp = np.flatnonzero(np.abs(change) > limit)
        if not len(sharp):
            break
        added = set()
        for i in sharp:
            # The bend at kept[i + 1]: put back the middle place of the run either side.
            for lo, hi in ((kept[i], kept[i + 1]), (kept[i + 1], kept[i + 2])):
                if hi - lo >= 2:
                    added.add((lo + hi) // 2)
        if not added:
            break
        kept = sorted(set(kept) | added)
    out = np.zeros(len(turns))
    for place, change in zip(kept[1:-1], changes(kept)):
        out[place - 1] = change  # place 1 is the first bridge
    return out


def _shape_error(made: np.ndarray, planned: np.ndarray) -> float:
    """The worst distance between two sets of points when laid over one another."""
    a = made - made.mean(axis=0)
    b = planned - planned.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    if np.linalg.det(u @ vt) < 0.0:
        u[:, -1] = -u[:, -1]
    return float(np.linalg.norm(a @ (u @ vt) - b, axis=1).max())


def _joint_frame(centres, frames, j: int) -> np.ndarray:
    return _frame(centres[j + 1] - centres[j], frames[j][:, 2] + frames[j + 1][:, 2])


def _pin(hole, normal, along, offset, thickness, radius, stretch, play, relief=0.0) -> GuidePart:
    """A pin standing from ``offset`` off a hole's centre along ``normal`` back into the hole."""
    normal = np.asarray(normal, dtype=float)
    tip_way = -normal
    along = _unit(np.asarray(along, dtype=float) - (np.asarray(along) @ tip_way) * tip_way)
    base = hole + normal * (offset + PIN_EMBED_MM)
    tip = hole + normal * (offset - PIN_LENGTH_SHARE * thickness)
    frame = np.column_stack([along, np.cross(tip_way, along), tip_way])
    r = radius - play
    return GuidePart(0.5 * (base + tip), frame, np.array([r, r, 0.5 * np.linalg.norm(tip - base)]),
                     "pin", stretch=stretch, tip_radius=PIN_TIP_SHARE * r, relief=relief)


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
    play_mm: float = 0.1,
    hole_length_mm: float | None = None,
    head=None,
) -> BendingGuide:
    """The keys that take a supplied plate into a planned shape.

    ``supplied_*`` are the plate's holes as it comes out of the packet, ``bent_*`` as
    planned. ``overbend`` maps a released angle in degrees to the angle to load it to.
    ``play_mm`` is the fit of a pin's base in its hole per side, ``hole_length_mm`` the
    bore's length along the plate if oval, and ``head`` the bender's head, if it was
    measured, in place of the kit's.
    """
    head = head if head is not None else kit.head
    sc = np.asarray(supplied_centres, dtype=float)
    centres = np.asarray(bent_centres, dtype=float)
    bent_axes = np.asarray(bent_axes, dtype=float)
    bent_tangents = np.asarray(bent_tangents, dtype=float)
    if len(sc) != len(bent_axes) or len(sc) != len(centres):
        raise ValueError("the supplied and planned plates have different hole counts")
    if len(sc) < 2:
        raise ValueError("a bending guide needs at least two holes")
    if not (min(width_mm, thickness_mm, hole_diameter_mm) > 0 and np.isfinite(centres).all()):
        raise ValueError("Invalid plate section or planned hole positions.")
    if head.kind == "none":
        raise ValueError(f"{kit.name} is shaped by hand: there is nothing for a key to stop. Choose a bender.")
    if not 0.0 <= play_mm <= 0.5:
        raise ValueError("Pin play must be 0 to 0.5 mm.")
    hole_radius = 0.5 * float(hole_diameter_mm)
    if hole_radius - float(play_mm) < 0.4:
        raise ValueError("That much play leaves no pin in a hole this small.")
    stretch = max(float(hole_length_mm or hole_diameter_mm) - float(hole_diameter_mm), 0.0)
    supplied = hole_frames(sc, supplied_axes)
    bent = np.stack([_frame(t, u) for t, u in zip(bent_tangents, bent_axes)])
    if np.min(np.linalg.norm(np.diff(centres, axis=0), axis=1)) < 1.0:
        raise ValueError("Coincident hole positions cannot be bent to.")

    def stop_of(angle: float) -> float:
        return float(overbend(angle)) if overbend is not None else angle

    joints: list[GuideJoint] = []
    data = []
    for i in range(len(sc) - 1):
        joint = _joint_frame(sc, supplied, i)
        change = (bent[i].T @ bent[i + 1]) @ (supplied[i].T @ supplied[i + 1]).T
        to_joint = joint.T @ supplied[i]
        rotation = to_joint @ change @ to_joint.T
        in_plane, out_plane, twist = _decompose(rotation)
        _, angle = _axis_angle(rotation)
        angle_deg = float(np.degrees(angle))
        joints.append(GuideJoint(i, angle_deg, stop_of(angle_deg), in_plane, out_plane, twist, in_plane, out_plane, twist))
        hinge = 0.5 * (sc[i] + sc[i + 1])
        data.append(
            (joint.T @ supplied[i], joint.T @ supplied[i + 1], joint.T @ (sc[i] - hinge), joint.T @ (sc[i + 1] - hinge))
        )

    # Fewer bends for the same plate: each one is a key to print and a press to make.
    pitches = np.linalg.norm(np.diff(sc, axis=0), axis=1)
    in_plane_made = _gathered([j.in_plane_deg for j in joints], pitches, IN_PLANE_TOLERANCE_MM)
    out_plane_made = _gathered([j.out_of_plane_deg for j in joints], pitches, OUT_OF_PLANE_TOLERANCE_MM)
    carried = 0.0
    for joint, in_made, out_made in zip(joints, in_plane_made, out_plane_made):
        joint.in_plane_made_deg = float(in_made) if abs(in_made) >= MIN_BEND_DEG else 0.0
        joint.out_of_plane_made_deg = float(out_made) if abs(out_made) >= MIN_BEND_DEG else 0.0
        carried += joint.twist_deg
        if abs(joint.out_of_plane_made_deg) > 0.0 or abs(carried) >= MAX_CARRIED_TWIST_DEG:
            joint.twist_made_deg, carried = carried, 0.0
        else:
            joint.twist_made_deg = 0.0

    guide = BendingGuide(
        centres=centres.copy(), frames=bent, stock_centres=sc.copy(), stock_frames=supplied,
        stock_distances=np.concatenate([[0.0], np.cumsum(pitches)]), joints=joints, keys=[], kit=kit,
        head=head, width_mm=float(width_mm), thickness_mm=float(thickness_mm),
        hole_diameter_mm=float(hole_diameter_mm), hole_stretch_mm=stretch, play_mm=float(play_mm),
        pin_diameter_mm=2.0 * (hole_radius - float(play_mm)), alloy_name=alloy_name, overbend=overbend,
    )
    warnings = guide.warnings

    # The order: in the plate's own plane first, then out of it, as the plate's maker says.
    schedule = []
    for joint in joints:
        if abs(joint.in_plane_made_deg) >= MIN_BEND_DEG:
            if kit.can_make("in_plane"):
                schedule.append(("in_plane", joint))
            else:
                warnings.append(
                    f"Holes {joint.index + 1}-{joint.index + 2} turn {abs(joint.in_plane_made_deg):.0f} deg in the "
                    f"plate's own plane, which {kit.name} cannot do."
                )
    in_plane_made_at = {joint.index for kind, joint in schedule}
    for joint in joints:
        if max(abs(joint.out_of_plane_made_deg), abs(joint.twist_made_deg)) >= MIN_BEND_DEG:
            if not kit.can_make("out_of_plane") and abs(joint.out_of_plane_made_deg) >= MIN_BEND_DEG:
                warnings.append(
                    f"Holes {joint.index + 1}-{joint.index + 2} curve {abs(joint.out_of_plane_made_deg):.0f} deg, "
                    f"which {kit.name} cannot do."
                )
                continue
            schedule.append(("out_of_plane", joint))
            if abs(joint.twist_made_deg) > MAX_PRESSED_TWIST_DEG and not kit.can_twist:
                warnings.append(
                    f"Holes {joint.index + 1}-{joint.index + 2} twist {abs(joint.twist_made_deg):.0f} deg: more than a "
                    "press gives reliably. Start it with twisting irons before the key."
                )

    for number, (kind, joint) in enumerate(schedule, 1):
        a0, b0, pa0, pb0 = data[joint.index]
        key = BendKey(number=number, bridge=joint.index, kind=kind,
                      name=f"{'A' if kind == 'in_plane' else 'B'}{joint.index + 1}", angle_deg=0.0, stop_angle_deg=0.0)
        guide.keys.append(key)
        key.applied_before = guide._made(number)
        key.lobe_a = (a0, pa0)
        if kind == "in_plane":
            size = abs(joint.in_plane_made_deg)
            key.angle_deg = key.stop_angle_deg = size
            key.turn_made = _compose(joint.in_plane_made_deg, 0.0, 0.0)
            key.turn_start = key.turn_stop = key.turn_made
            _build_check_key(guide, key)
        else:
            phi = joint.in_plane_made_deg if joint.index in in_plane_made_at else 0.0
            size = float(np.hypot(joint.out_of_plane_made_deg, joint.twist_made_deg))
            stop = stop_of(size)
            scale = stop / max(size, 1e-9)
            key.angle_deg, key.stop_angle_deg = size, stop
            key.turn_start = _compose(phi, 0.0, 0.0)
            key.turn_made = _compose(phi, joint.out_of_plane_made_deg, joint.twist_made_deg)
            key.turn_stop = _compose(phi, joint.out_of_plane_made_deg * scale, joint.twist_made_deg * scale)
            _build_press_key(guide, key, b0, pb0)

    if head.kind == "three_point":
        worst = max((k.loading_depth_mm for k in guide.keys if not k.check), default=0.0)
        if worst > head.opening_mm:
            warnings.append(
                f"The plate and key need the head {worst:.1f} mm open to go in, and this bender is set to open "
                f"{head.opening_mm:.1f} mm. Measure its widest opening, and check on the fit test."
            )
    guide.deviation_mm = _shape_error(guide.stage(done=True)[0], centres)
    if guide.deviation_mm > SHAPE_CAUTION_MM:
        warnings.append(
            f"The plate these keys make lies up to {guide.deviation_mm:.1f} mm from the planned plate. "
            "Check it on the printed jaw."
        )
    return guide


def _build_press_key(guide: BendingGuide, key: BendKey, b0, pb0) -> None:
    """A press key, in its own frame, for a bend out of the plate's plane."""
    head = guide.head
    lever = head.kind == "lever"
    t, width = guide.thickness_mm, guide.width_mm
    half = _half(key.turn_stop)
    a0, pa0 = key.lobe_a
    _, out_of_plane, _ = _decompose(key.turn_stop)
    # Bending towards the bone leaves the outer face convex, and the other way round.
    # A press key goes on the convex face, its z always pointing that way; a lever
    # kit's on the concave face.
    sign = -np.sign(out_of_plane) or 1.0
    key.to_key = np.diag([1.0, sign, sign])
    a_key, b_key = key.to_key @ half.T @ a0, key.to_key @ half @ b0
    hole_a, hole_b = key.to_key @ half.T @ pa0, key.to_key @ half @ pb0
    towards_key = -1.0 if lever else 1.0
    normal_a = towards_key * sign * a_key[:, 2]
    normal_b = towards_key * sign * b_key[:, 2]
    offset = 0.5 * t
    plane_a = (hole_a + offset * normal_a, normal_a)
    plane_b = (hole_b + offset * normal_b, normal_b)
    key.frame_a, key.frame_b, key.hole_a, key.hole_b = a_key, b_key, hole_a, hole_b
    key.faces = (plane_a, plane_b)
    key.from_bone_side = bool((sign < 0.0) != lever)
    key.pinned = (key.bridge, key.bridge + 1)

    pitch = float(np.linalg.norm(hole_b - hole_a))
    lobe_half = min(0.5 * width, LOBE_SHARE_OF_PITCH * pitch)
    face_end = max(abs(float(hole_a[0])), abs(float(hole_b[0]))) + lobe_half
    x_reach = face_end + KEY_END_MM

    def height_at(plane, x, y):
        """z of a face over ``(x, y)``."""
        q, n = plane
        return float(q[2] - (n[0] * (x - q[0]) + n[1] * (y - q[1])) / n[2])

    def height(plane, x):
        """z of a face over ``x``, along the key's middle."""
        return height_at(plane, x, 0.0)

    if lever:
        y_range = (-0.5 * width - 1.0, 0.5 * width + 1.0)
    else:
        y_half = 0.5 * max(width, min(width + KEY_EXTRA_WIDTH_MM, head.pocket_depth_mm - 0.5))
        y_range = (-y_half, y_half)
    y_bound = max(abs(y_range[0]), abs(y_range[1])) + 1.0
    direction = np.cross(normal_a, normal_b)
    length = float(np.linalg.norm(direction))
    direction = direction / length if length > 1e-6 else np.array([0.0, 1.0, 0.0])
    if direction[1] < 0.0:
        direction = -direction

    if lever:
        ridge_z = min(height(plane_a, 0.0), height(plane_b, 0.0))
        drop = abs(height(plane_a, -face_end) - height(plane_a, 0.0))
        floor = ridge_z - LEVER_BASE_MM - drop
        body = KeyBody(
            centre=np.zeros(3), frame=np.eye(3),
            half=np.array([x_reach + 1.0, y_bound, max(abs(floor), abs(ridge_z)) + 2.0]),
            name="key", planes=(plane_a, plane_b), head=head, face_end=face_end, y_range=y_range,
            x_reach=x_reach, ridge=(np.array([0.0, 0.0, ridge_z]), direction),
            lever_floor=floor, lever_top=ridge_z - 0.25,
        )
        key.parts = [body]
        key.lobes_z = ridge_z
        key.post_top_z = ridge_z + t
        middle_z = ridge_z - 0.5 * LEVER_BASE_MM
    else:
        # Along the fold the two faces meet; twisted or turned in plane, they meet with a
        # step that grows across the key. The channel there is deep enough to take it.
        ys = np.linspace(-0.5 * width - 0.3, 0.5 * width + 0.3, 23)  # where the plate's bridge is
        fold = np.array([[height_at(plane_a, 0.0, y), height_at(plane_b, 0.0, y)] for y in ys])
        ridge_z = float(fold.max())
        channel = max(RIDGE_RELIEF_MM, ridge_z - float(fold.min()) + 0.75 * RIDGE_RELIEF_MM)
        body = KeyBody(
            centre=np.zeros(3), frame=np.eye(3), half=np.zeros(3), name="key", planes=(plane_a, plane_b),
            head=head, lobes_z=0.0, face_end=face_end, y_range=y_range, x_reach=x_reach,
            ridge=(np.array([0.0, 0.0, ridge_z + 0.25 * RIDGE_RELIEF_MM]), np.array([0.0, 1.0, 0.0])),
            channel_radius=channel,
        )
        key.parts = [body]
        # Where the head goes: as low as leaves its lobes clear of the plate the whole
        # way, from the plate reaching the key to the stop, the key thick enough at its
        # ends. The key's top then follows the head's underside.
        lobes_z = -np.inf
        a = 0.5 * head.lobe_spacing_mm
        reach = a + head.lobe_radius_mm
        depth = 0.5 * head.pocket_depth_mm
        beyond = []
        for fraction in np.linspace(0.0, 1.0, PRESS_STEPS + 1):
            places, frames = guide.pose(key, fraction)
            points = _lobe_points(places, frames, 0.5 * width, t)
            outside = (np.abs(points[:, 0]) > x_reach) & (np.abs(points[:, 1]) < depth)
            beyond.append(points[outside])
            under = outside & (np.abs(points[:, 0]) < reach)
            if under.any():
                need = points[under, 2] + LOBE_GAP_MM - die_height(head, points[under, 0])
                lobes_z = max(lobes_z, float(need.max()))
        end = float(body.underside(x_reach - 1e-3))
        lobes_z = max(lobes_z, end + MIN_KEY_END_MM + TOP_CLEARANCE_MM - float(die_height(head, x_reach)))
        # Straight up is not the nearest way to a lobe's round side: raise the head until
        # the plate is clear of it by the gap every way.
        beyond = np.vstack(beyond) if beyond else np.zeros((0, 3))
        for _ in range(6):
            nearest = head_clearance(head, lobes_z, beyond)
            if nearest >= LOBE_GAP_MM - 1e-3:
                break
            lobes_z += LOBE_GAP_MM - nearest + 0.01
        body.lobes_z = lobes_z
        top = float(body.top(0.0))
        body.half = np.array([x_reach + 1.0, y_bound, max(abs(top), abs(end), abs(ridge_z)) + 4.0])
        key.lobes_z = lobes_z
        middle_z = 0.5 * (ridge_z + top)
        # The post's top at the stop: the plate's concave face under the bridge. Before
        # the press it is as low as the plate there, which is how far the head must open.
        places, _ = guide.pose(key, 1.0)
        key.post_top_z = float(0.5 * (places[key.bridge][2] + places[key.bridge + 1][2]) - 0.5 * t)
        places, _ = guide.pose(key, 0.0)
        key.loading_depth_mm = float(lobes_z - (0.5 * (places[key.bridge][2] + places[key.bridge + 1][2]) - 0.5 * t))

    radius = 0.5 * guide.hole_diameter_mm
    for hole, lobe, normal, relief in ((hole_a, a_key, normal_a, 0.0), (hole_b, b_key, normal_b, DIAMOND_RELIEF_MM)):
        key.parts.append(_pin(hole, normal, lobe[:, 0], offset, t, radius, guide.hole_stretch_mm, guide.play_mm, relief))
    # Its name, on the side that is up when it prints.
    key.parts.extend(
        digit_parts(key.name, [0.0, y_range[1], middle_z], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    )


def _build_check_key(guide: BendingGuide, key: BendKey) -> None:
    """A tray that the plate drops onto when its in-plane bend is right."""
    count = len(guide.stock_centres)
    key.to_key = np.eye(3)
    key.check = True
    key.pinned = tuple(i for i in (key.bridge - 1, key.bridge, key.bridge + 1, key.bridge + 2) if 0 <= i < count)
    places, frames = guide.pose(key, 1.0)
    t = guide.thickness_mm
    # In the bridge's frame the plate's mid-plane is at z = 0 and its outer face up. It
    # lies on the tray bone face down: the tray's top is at its bone face.
    top = -0.5 * t
    held = places[list(key.pinned)]
    width_half = 0.5 * guide.width_mm
    rim = width_half + TRAY_MARGIN_MM
    label_y = float(held[:, 1].min()) - rim - TRAY_LABEL_MM + 1.0
    label = (float(held[:, 0].min()), float(held[:, 0].max()), label_y, float(held[:, 1].min()))
    corners = np.vstack([held[:, :2] - rim, held[:, :2] + rim, [[label[0], label[2]]]])
    reach = np.abs(corners).max(axis=0) + 1.0
    tray = TrayBody(np.zeros(3), np.eye(3), np.array([reach[0], reach[1], abs(top) + TRAY_MM + 1.0]), "tray",
                    holes=held[:, :2].copy(), radius=rim, label=label, top=top)
    key.parts = [tray]
    radius = 0.5 * guide.hole_diameter_mm
    for n, i in enumerate(key.pinned):
        down = -frames[i][:, 2]  # through the bone face, into the tray
        key.parts.append(_pin(places[i], down, frames[i][:, 0], 0.5 * t, t, radius, guide.hole_stretch_mm,
                              guide.play_mm, 0.0 if n == 0 else DIAMOND_RELIEF_MM))
    key.parts.extend(
        digit_parts(key.name, [0.5 * (label[0] + label[1]), label[2] + 0.5 * TRAY_LABEL_MM, top],
                    [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    )
    key.frame_a, key.frame_b = frames[key.bridge], frames[key.bridge + 1]
    key.hole_a, key.hole_b = places[key.bridge], places[key.bridge + 1]
    key.from_bone_side = True


def tool_parts(guide: BendingGuide, key: BendKey) -> list[GuidePart]:
    """The bender's head as it stands at this key's stop, for the rehearsal."""
    head = guide.head
    if head.kind != "three_point" or key.check:
        return []
    depth = 0.5 * head.pocket_depth_mm
    radius = 0.5 * head.post_diameter_mm
    die = DieBody(
        centre=np.array([0.0, 0.0, key.lobes_z + 0.5 * head.arch_rise_mm + 2.5]),
        frame=np.eye(3),
        half=np.array([0.5 * head.lobe_spacing_mm + head.lobe_radius_mm + 1.0, depth, 0.5 * head.arch_rise_mm + 6.0]),
        name="die", head=head, lobes_z=key.lobes_z,
    )
    post = GuidePart(
        np.array([0.0, 0.0, key.post_top_z - radius]), np.eye(3),
        np.array([radius, depth, radius]), "post",
    )
    return [die, post]


def fit_test_key(guide: BendingGuide) -> BendKey:
    """A press key for one known bend on this plate and bender, springback allowed for."""
    pitch = float(np.median(np.linalg.norm(np.diff(guide.stock_centres, axis=0), axis=1)))
    stock = np.array([[(i - 1.5) * pitch, 0.0, 0.0] for i in range(4)])
    axes = np.tile([0.0, 0.0, 1.0], (4, 1))
    turn = _rotation([0.0, 1.0, 0.0], np.radians(FIT_TEST_ANGLE_DEG))  # towards the bone
    frames = [np.eye(3), np.eye(3), turn, turn]
    places = [stock[0], stock[1]]
    places.append(places[1] + np.array([0.5 * pitch, 0.0, 0.0]) + turn @ np.array([0.5 * pitch, 0.0, 0.0]))
    places.append(places[2] + turn @ np.array([pitch, 0.0, 0.0]))
    test = plan_bending_guide(
        stock, axes, np.array(places), np.array([f[:, 2] for f in frames]), np.array([f[:, 0] for f in frames]),
        guide.width_mm, guide.thickness_mm, guide.hole_diameter_mm, guide.kit, overbend=guide.overbend,
        play_mm=guide.play_mm, hole_length_mm=guide.hole_diameter_mm + guide.hole_stretch_mm, head=guide.head,
    )
    key = next(k for k in test.keys if not k.check)
    key.name = str(int(FIT_TEST_ANGLE_DEG))
    # Its name on it again, now that it has the test's.
    key.parts = [p for p in key.parts if p.name != "digit"]
    body = key.parts[0]
    middle = body.lever_floor if body.lever_floor is not None else 0.5 * (float(body.ridge[0][2]) + float(body.top(0.0)))
    key.parts.extend(
        digit_parts(key.name, [0.0, body.y_range[1], middle], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
    )
    return key


@dataclass
class _GaugeBody(GuidePart):
    """A flat tab with a shallow V in one edge, of a given included angle."""

    angle_deg: float = FIT_TEST_ANGLE_DEG

    def field(self, local):
        x, y, z = local
        walls = np.maximum.reduce((np.abs(x) - self.half[0], np.abs(z) - self.half[2], -self.half[1] - y))
        slope = np.tan(np.radians(0.5 * self.angle_deg))
        return np.maximum(walls, y + np.abs(x) * slope - self.half[1])


def gauge_parts() -> list[GuidePart]:
    """A gauge to lay on the fit test's released plate: flush means the springback
    allowance is right. Its edge has the released angle of the test bend."""
    gauge = _GaugeBody(np.zeros(3), np.eye(3), np.array([18.0, 7.0, 1.5]), "gauge", angle_deg=FIT_TEST_ANGLE_DEG)
    parts: list[GuidePart] = [gauge]
    parts.extend(
        digit_parts(str(int(FIT_TEST_ANGLE_DEG)), [0.0, -2.5, 1.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    )
    return parts


def pin_tab_parts(guide: BendingGuide) -> list[GuidePart]:
    """A small tab with a pin of each play, to try in a real hole of the plate."""
    radius = 0.5 * guide.hole_diameter_mm
    gap = max(2.0 * radius + guide.hole_stretch_mm + 4.0, 7.0)
    length = gap * len(PIN_TEST_PLAYS_MM)
    frame = np.eye(3)
    base = 2.0
    parts: list[GuidePart] = [
        GuidePart(np.array([0.5 * length, 0.0, 0.5 * base]), frame, np.array([0.5 * length, 4.5 + radius, 0.5 * base]), "tab")
    ]
    tall = PIN_LENGTH_SHARE * guide.thickness_mm
    for i, play in enumerate(PIN_TEST_PLAYS_MM):
        x = gap * (i + 0.5)
        pin_radius = radius - play
        parts.append(
            GuidePart(np.array([x, 0.0, base + 0.5 * tall - 0.25]), frame,
                      np.array([pin_radius, pin_radius, 0.5 * tall + 0.25]), "pin",
                      stretch=guide.hole_stretch_mm, tip_radius=PIN_TIP_SHARE * pin_radius)
        )
        parts.extend(
            digit_parts(str(i + 1), [x, -(radius + 2.4), base], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
        )
    return parts


def key_print_rotation(key: BendKey | None = None) -> np.ndarray:
    """How a key lies to print; its rows are the print x, y and z axes.

    A press key lies on its side, its name up: then the faces the plate comes to lie
    on stand upright, with no stair-stepping, at the printer's finest accuracy. A check
    key lies as it is used, its tray on the bed and its pins standing up."""
    if key is not None and key.check:
        return np.eye(3)
    return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
