"""Bending keys: one small printed form for each bend of a plate.

The plate comes out of the packet straight. Each bridge between two holes has to
turn by some angle, and the bender has no way of knowing how much. So each bend
gets its own key: a small block whose face is the plate's own surface at that
bridge, in the finished shape, with two pins that drop into the holes either side.

With a three-point bender (a post on one side of the plate, two lobes and an arch
on the other) the key sits on the plate's convex side, nested in the arch with its
shoulders under the lobes. The post pushes the plate into the key until the plate
lies flat on it. Then the plate is carried by the key from end to end and bends
no further, however hard the handles are pressed: the angle comes from the
key's shape and the springback allowed for, not from a reading or the hand's feel.

The stop does not depend on the bender's exact size. What the key takes from the
bender is only how to nest in it and how far apart its lobes are, so it stays
in place and clear of the rest of the head. With irons or forceps, which hold the
plate either side of a bridge and lever it, the key goes on the concave side and the
plate is levered down onto it.

The plan follows the plate maker's order: bends in the plate's own plane first,
then out of its plane. A twist goes in the same key as that bridge's out-of-plane
bend, its two faces being turned to one another, and the press gives it too.

A plan on a real jaw turns a little at every bridge, and a key for each of those would
be a lot of keys for no gain. So the turns are gathered into as few bends as keep every
hole within a fraction of a millimetre of the plan, tighter for what would lift the
plate off the bone than for what only moves it along the bone.

Everything is the plate's and the bender's own: faces from the plate's thickness or
width and the finished shape, pins from its bores (oval ones too, read from
its mesh), and the key's back from the bender's head, its stop angle from the alloy.

Numpy only. The solids are given as distance-like fields, and
``render/bending_guide_mesh.py`` contours their union.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: A bridge turning less than this is left alone: it would barely yield, degrees.
MIN_BEND_DEG = 1.0
#: The key's thickness under each lobe, and the least it may be, mm.
SHOULDER_MM = 2.6
MIN_SHOULDER_MM = 1.2
#: Clearance between the key's back and the bender's arch, mm. It rests on the
#: lobes, not the arch.
DIE_CLEARANCE_MM = 0.4
#: How far the key's face lies past the far edge of each lobe of the plate, mm.
FACE_PAST_LOBE_MM = 1.0
#: Beyond that the face lifts away, this many mm per mm, so only those two lobes
#: of the plate touch it, whatever has been bent or not bent beyond them.
RELIEF_SLOPE = float(np.tan(np.radians(8.0)))
#: A round channel along the ridge of the key, mm: the plate's bridge curves
#: there instead of folding, and the key has no sharp notch to crack from.
RIDGE_RELIEF_MM = 0.8
#: How far the key runs on past each lobe of the bender, mm.
END_PAST_LOBE_MM = 1.5
#: How much wider than the plate the key is, for stiffness, as the head allows, mm.
KEY_EXTRA_WIDTH_MM = 3.0
#: Pins taper to this share of their base, so that they stay in a hole that is
#: still turning, and end up snug. How long they are, as a share of the plate.
PIN_TIP_SHARE = 0.55
PIN_LENGTH_SHARE = 0.85
PIN_EMBED_MM = 0.5
#: How thick the block of a lever kit's key is under the plate, mm.
LEVER_BASE_MM = 7.0
#: The skin carrying the pins of an in-plane key over the plate's face, mm.
SKIN_MM = 1.2
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

    def field(self, local):
        """Negative inside, about the distance to the surface outside."""
        x, y, z = local
        if self.name == "pin":
            along = np.maximum(np.abs(x) - 0.5 * self.stretch, 0.0)
            # Local +z is towards the tip.
            share = (z + self.half[2]) / (2.0 * self.half[2])
            radius = self.half[0] + (self.tip_radius - self.half[0]) * share
            return np.maximum(np.hypot(along, y) - radius, np.abs(z) - self.half[2])
        if self.name == "post":
            return np.maximum(np.hypot(x, z) - self.half[0], np.abs(y) - self.half[1])
        return np.maximum.reduce(
            (np.abs(x) - self.half[0], np.abs(y) - self.half[1], np.abs(z) - self.half[2])
        )


@dataclass
class KeyBody(GuidePart):
    """The block of a bend key, in the key's own frame: x along the plate, y along
    the bender's post, z from the plate towards the head. Origin at the bridge."""

    planes: tuple = ()  # ((point, normal), (point, normal)): the plate's surface
    head: object = None
    lobes_z: float = 0.0
    face_reach: float = 0.0
    y_range: tuple[float, float] = (0.0, 0.0)
    x_reach: float = 0.0
    ridge: tuple = ()  # (point, direction)
    lever_floor: float | None = None
    lever_top: float = 0.0

    def field(self, local):
        x, y, z = local
        (qa, na), (qb, nb) = self.planes
        above_a = (x - qa[0]) * na[0] + (y - qa[1]) * na[1] + (z - qa[2]) * na[2]
        above_b = (x - qb[0]) * nb[0] + (y - qb[1]) * nb[1] + (z - qb[2]) * nb[2]
        walls = np.maximum.reduce(
            (np.abs(x) - self.x_reach, self.y_range[0] - y, y - self.y_range[1])
        )
        if self.lever_floor is not None:
            # Under the plate, on the side both normals point to; its ridge taken off.
            shape = np.maximum.reduce((-above_a, -above_b, self.lever_floor - z, z - self.lever_top))
            return np.maximum(shape, walls)
        # Past the plate's two lobes the face lifts away.
        lift = RELIEF_SLOPE * np.maximum(np.abs(x) - self.face_reach, 0.0)
        under = -np.maximum(above_a - lift, above_b - lift)
        back = self.lobes_z + np.maximum(die_height(self.head, x) - DIE_CLEARANCE_MM, 0.0)
        solid = np.maximum.reduce((under, z - back, walls))
        # The round channel along the ridge.
        point, direction = self.ridge
        px, py, pz = x - point[0], y - point[1], z - point[2]
        along = px * direction[0] + py * direction[1] + pz * direction[2]
        off = np.sqrt(
            (px - along * direction[0]) ** 2
            + (py - along * direction[1]) ** 2
            + (pz - along * direction[2]) ** 2
        )
        return np.maximum(solid, RIDGE_RELIEF_MM - off)


@dataclass
class DieBody(GuidePart):
    """The bender's head for the rehearsal: everything above its underside."""

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
    "A": "abcefg", "B": "cdefg",
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
class BendKey:
    """The printed form for one bend."""

    number: int  # its place in the sequence, from 1
    bridge: int  # hole ``bridge`` and hole ``bridge + 1``
    kind: str  # "in_plane" or "out_of_plane"
    name: str  # what is printed on it
    angle_deg: float  # the released angle it makes
    stop_angle_deg: float  # the loaded angle its faces are at
    parts: list[GuidePart]
    #: The two lobes of the plate at the stop, in the key's frame.
    frame_a: np.ndarray = None
    frame_b: np.ndarray = None
    hole_a: np.ndarray = None
    hole_b: np.ndarray = None
    faces: tuple = ()  # the two planes the plate comes to lie on
    from_bone_side: bool = False  # its pins enter from the plate's bone face
    shoulder_mm: float = SHOULDER_MM
    #: How far below the lobes the post's top must be to load a fresh plate, mm.
    loading_depth_mm: float = 0.0
    #: For the rehearsal: the lobes' lowest points and the post's top, z.
    lobes_z: float = 0.0
    post_top_z: float = 0.0


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

    def instruction(self, key: BendKey) -> str:
        a, b = key.bridge + 1, key.bridge + 2
        face = "bone" if key.from_bone_side else "outer"
        seat = f"Key {key.name}: holes {a} and {b}. Seat its pins in those holes from the plate's {face} face."
        if self.head.kind == "lever":
            return f"{seat} Lever the plate down onto the key until it lies flat on it."
        if key.kind == "in_plane":
            way = (
                f"This one turns the plate {key.angle_deg:.1f} deg in its own plane: lay the plate flat "
                "in the head, the key to the arch, the post against the plate's other edge at the bridge."
            )
        else:
            way = (
                f"This one curves the plate {key.angle_deg:.1f} deg: nest the key in the arch, "
                "the post under the bridge between the two holes."
            )
        return f"{seat} {way} Press until the plate lies flat on the key and the handles stop. Let go."

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

    def stage(self, key: BendKey | None = None, done: bool = False):
        """The plate's holes, ``(centres, axes)``, for the rehearsal.

        At ``key``'s stop, in that key's frame, with every earlier key's bend
        released and none of the later ones. With no key: fresh, or if ``done``
        with everything made; those two start at hole 1 with its frame as it comes.
        """
        count = len(self.stock_centres)
        applied = []
        upto = key.number if key is not None else (len(self.keys) + 1 if done else 0)
        for joint in self.joints:
            in_key = next((k for k in self.keys if k.bridge == joint.index and k.kind == "in_plane"), None)
            out_key = next((k for k in self.keys if k.bridge == joint.index and k.kind == "out_of_plane"), None)
            phi = joint.in_plane_made_deg if in_key is not None and in_key.number < upto else 0.0
            theta = joint.out_of_plane_made_deg if out_key is not None and out_key.number < upto else 0.0
            tau = joint.twist_made_deg if out_key is not None and out_key.number < upto else 0.0
            if key is not None and key.bridge == joint.index:
                if key.kind == "in_plane":
                    phi = np.sign(joint.in_plane_made_deg) * key.stop_angle_deg
                else:
                    scale = key.stop_angle_deg / max(key.angle_deg, 1e-9)
                    theta, tau = joint.out_of_plane_made_deg * scale, joint.twist_made_deg * scale
            applied.append(_compose(phi, theta, tau))
        world = [np.eye(3)]
        places = [np.zeros(3)]
        for j in range(count - 1):
            before = self.stock_frames[j].T @ self.stock_frames[j + 1]
            chord = self.stock_frames[j].T @ (self.stock_centres[j + 1] - self.stock_centres[j])
            joint_frame = _joint_frame(self.stock_centres, self.stock_frames, j)
            to_joint = joint_frame.T @ self.stock_frames[j]
            turn = to_joint.T @ applied[j] @ to_joint
            hinge = 0.5 * chord
            places.append(places[j] + world[j] @ (hinge + turn @ (chord - hinge)))
            world.append(world[j] @ turn @ before)
        places = np.array(places)
        world = np.array(world)
        if key is not None:
            # Lobe A of this key's bridge, where the key has it.
            to_key = key.frame_a @ world[key.bridge].T
            shift = key.hole_a - to_key @ places[key.bridge]
            places = places @ to_key.T + shift
            world = np.einsum("ij,hjk->hik", to_key, world)
        return places, world[:, :, 2]


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
    ``tolerance`` of it, none of them much sharper than the plan's own sharpest."""
    turns = np.asarray(turns_deg, dtype=float)
    places = _laid_flat(turns, np.asarray(pitches, dtype=float))
    limit = max(MAX_GATHERED_DEG, float(np.abs(turns).max(initial=0.0)))
    while True:
        kept = _simplified(places, tolerance)
        runs = np.diff(places[kept], axis=0)
        headings = np.unwrap(np.arctan2(runs[:, 1], runs[:, 0]))
        changes = np.degrees(np.diff(headings))
        if tolerance < 0.02 or not len(changes) or float(np.abs(changes).max()) <= limit:
            break
        tolerance *= 0.5
    out = np.zeros(len(turns))
    for place, change in zip(kept[1:-1], changes):
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


def _build_key(name, number, bridge, kind, angle, stop, a0, b0, pa0, pb0, turn, head, width, thickness,
               hole_radius, stretch, play) -> BendKey:
    """A key in its own frame, from the bridge's data given in the bridge's frame.

    ``a0``/``b0`` are the plate's two lobe frames and ``pa0``/``pb0`` its two holes as
    supplied, relative to the hinge; ``turn`` is the bridge's rotation at the stop.
    """
    axis, angle_rad = _axis_angle(turn)
    half = _rotation(axis, 0.5 * angle_rad)
    a_stop, b_stop = half.T @ a0, half @ b0
    hole_a, hole_b = half.T @ pa0, half @ pb0

    in_plane_turn, out_of_plane_turn, _ = _decompose(turn)
    if kind == "in_plane":
        # Towards the plate's convex edge, with its thickness along the key's y.
        sign = -np.sign(in_plane_turn) or 1.0
        to_key = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -sign], [0.0, sign, 0.0]])
        offset = 0.5 * width
        normal_column = 1
        # The skin lies on the key's +y side, which is the bridge's -sign z.
        from_bone = sign > 0.0
    else:
        # Bending towards the bone leaves the outer face convex, and the other way
        # round. The key goes on the convex face, its z always pointing that way.
        sign = -np.sign(out_of_plane_turn) or 1.0
        to_key = np.diag([1.0, sign, sign])
        offset = 0.5 * thickness
        normal_column = 2
        from_bone = sign < 0.0
    a_key, b_key = to_key @ a_stop, to_key @ b_stop
    hole_a, hole_b = to_key @ hole_a, to_key @ hole_b
    lever = head.kind == "lever"
    if lever:
        # A lever kit's key goes on the concave side, and its pins from there.
        from_bone = not from_bone if kind != "in_plane" else from_bone
    towards_key = -1.0 if lever else 1.0
    normal_a = towards_key * sign * a_key[:, normal_column]
    normal_b = towards_key * sign * b_key[:, normal_column]
    plane_a = (hole_a + offset * normal_a, normal_a)
    plane_b = (hole_b + offset * normal_b, normal_b)

    spacing = max(float(np.linalg.norm(hole_a)), float(np.linalg.norm(hole_b)))
    face_reach = 2.0 * spacing + FACE_PAST_LOBE_MM

    def height(plane, x):
        """z of a face over ``x``, along the key's middle."""
        q, n = plane
        return float(q[2] - (n[0] * (x - q[0]) + n[1] * (0.0 - q[1])) / n[2])

    if kind == "in_plane":
        y_range = (-0.5 * thickness - 0.2, 0.5 * thickness + 0.2 + SKIN_MM)
    elif lever:
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

    section = width if kind == "in_plane" else thickness
    shoulder = SHOULDER_MM
    depth = 0.0
    if lever:
        ridge_z = min(height(plane_a, 0.0), height(plane_b, 0.0))
        drop = abs(height(plane_a, -face_reach) - height(plane_a, 0.0))
        floor = ridge_z - LEVER_BASE_MM - drop
        body = KeyBody(
            centre=np.zeros(3), frame=np.eye(3),
            half=np.array([face_reach + 1.0, y_bound, max(abs(floor), abs(ridge_z)) + 2.0]),
            name="key", planes=(plane_a, plane_b), head=head, face_reach=face_reach, y_range=y_range,
            x_reach=face_reach, ridge=(np.array([0.0, 0.0, ridge_z]), direction),
            lever_floor=floor, lever_top=ridge_z - 0.25,
        )
        lobes_z = ridge_z
        middle_z = ridge_z - 0.5 * LEVER_BASE_MM
        post_top = ridge_z + section
    else:
        ridge_z = max(height(plane_a, 0.0), height(plane_b, 0.0))
        a = 0.5 * head.lobe_spacing_mm
        drop = max(height(plane_a, -a), height(plane_b, a))
        reach = max(a - face_reach, 0.0)
        relief = RELIEF_SLOPE * reach
        # A fresh plate lies across the key's two faces at their far ends, so the post
        # has to fall this far below the lobes to let it in.
        slope = abs(height(plane_b, face_reach) - height(plane_b, a)) / max(reach, 1e-6) if reach else 0.0
        extra = reach * (RELIEF_SLOPE - slope)
        shoulder = float(np.clip(head.opening_mm - section - extra - 0.3, MIN_SHOULDER_MM, SHOULDER_MM))
        depth = shoulder + extra + section
        lobes_z = drop + relief + shoulder
        x_reach = a + END_PAST_LOBE_MM
        crown = float(np.max(die_height(head, np.linspace(0.0, a, 64))))
        top = lobes_z + max(crown - DIE_CLEARANCE_MM, 0.0)
        body = KeyBody(
            centre=np.zeros(3), frame=np.eye(3),
            half=np.array([x_reach + 1.0, y_bound, max(abs(drop) + 4.0, abs(top) + 2.0)]),
            name="key", planes=(plane_a, plane_b), head=head, lobes_z=lobes_z, face_reach=face_reach,
            y_range=y_range, x_reach=x_reach, ridge=(np.array([0.0, 0.0, ridge_z + 0.25 * RIDGE_RELIEF_MM]), direction),
        )
        middle_z = 0.5 * (ridge_z + float(die_height(head, 0.0)) + lobes_z)
        post_top = ridge_z - section

    parts: list[GuidePart] = [body]

    radius = hole_radius - play
    if kind == "in_plane":
        # A skin over the plate's face from the key's body, with a pin into each hole.
        through = np.array([0.0, 1.0, 0.0])
        for hole, lobe in ((hole_a, a_key), (hole_b, b_key)):
            along = lobe[:, 0]
            towards = np.cross(along, through)
            if (towards[2] > 0.0) == lever:
                towards = -towards
            # Far enough to be well into the body, wherever a preformed plate's turn puts it.
            start = hole + through * (0.5 * thickness + 0.2 + 0.5 * SKIN_MM)
            reach_body = 0.5 * width + 1.0
            while reach_body < 3.0 * width and body.field(tuple(start + towards * (reach_body - 0.5))) > -0.3:
                reach_body += 0.5
            behind = radius + 1.0
            place = hole + through * (0.5 * thickness + 0.2 + 0.5 * SKIN_MM) + towards * 0.5 * (reach_body - behind)
            parts.append(
                GuidePart(place, np.column_stack([along, through, towards]),
                          np.array([radius + 1.2, 0.5 * SKIN_MM, 0.5 * (reach_body + behind)]), "skin")
            )
            tip_way = -through
            base = hole + through * (0.5 * thickness + 0.2 + PIN_EMBED_MM)
            tip = hole + through * (0.5 * thickness + 0.2 - PIN_LENGTH_SHARE * thickness)
            frame = np.column_stack([along, np.cross(tip_way, along), tip_way])
            parts.append(
                GuidePart(0.5 * (base + tip), frame, np.array([radius, radius, 0.5 * np.linalg.norm(tip - base)]),
                          "pin", stretch=stretch, tip_radius=PIN_TIP_SHARE * radius)
            )
    else:
        for hole, lobe, normal in ((hole_a, a_key, normal_a), (hole_b, b_key, normal_b)):
            along = lobe[:, 0]
            tip_way = -normal
            # Into the body by this much at least, wherever a preformed plate's turn puts it.
            embed = PIN_EMBED_MM
            while embed < 6.0 and body.field(tuple(hole + normal * (offset + embed))) > -0.2:
                embed += 0.25
            base = hole + normal * (offset + embed)
            tip = hole + normal * (offset - PIN_LENGTH_SHARE * thickness)
            frame = np.column_stack([along, np.cross(tip_way, along), tip_way])
            parts.append(
                GuidePart(0.5 * (base + tip), frame, np.array([radius, radius, 0.5 * np.linalg.norm(tip - base)]),
                          "pin", stretch=stretch, tip_radius=PIN_TIP_SHARE * radius)
            )

    # Its name, on the side that is up when it prints: the skin side of an in-plane
    # key goes down, so that its pins stand up from it.
    if kind == "in_plane":
        parts.extend(
            digit_parts(name, [0.0, y_range[0], middle_z], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0])
        )
    else:
        parts.extend(
            digit_parts(name, [0.0, y_range[1], middle_z], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
        )
    return BendKey(
        number=number, bridge=bridge, kind=kind, name=name, angle_deg=angle, stop_angle_deg=stop, parts=parts,
        frame_a=a_key, frame_b=b_key, hole_a=hole_a, hole_b=hole_b, faces=(plane_a, plane_b),
        from_bone_side=bool(from_bone), shoulder_mm=shoulder, loading_depth_mm=depth,
        lobes_z=lobes_z, post_top_z=post_top,
    )


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

    warnings: list[str] = []
    keys: list[BendKey] = []

    def add_key(kind, joint):
        a0, b0, pa0, pb0 = data[joint.index]
        if kind == "in_plane":
            size = abs(joint.in_plane_made_deg)
            stop = stop_of(size)
            turn = _compose(np.sign(joint.in_plane_made_deg) * stop, 0.0, 0.0)
            prefix = "A"
        else:
            size = float(np.hypot(joint.out_of_plane_made_deg, joint.twist_made_deg))
            stop = stop_of(size)
            scale = stop / max(size, 1e-9)
            turn = _compose(joint.in_plane_made_deg, joint.out_of_plane_made_deg * scale, joint.twist_made_deg * scale)
            prefix = "B"
        keys.append(
            _build_key(
                f"{prefix}{joint.index + 1}", len(keys) + 1, joint.index, kind, size, stop, a0, b0, pa0, pb0, turn,
                head, float(width_mm), float(thickness_mm), hole_radius, stretch, float(play_mm),
            )
        )

    # In the plate's own plane first, then out of it, as the plate's maker says.
    for joint in joints:
        if abs(joint.in_plane_made_deg) >= MIN_BEND_DEG:
            if kit.can_make("in_plane"):
                add_key("in_plane", joint)
            else:
                warnings.append(
                    f"Holes {joint.index + 1}-{joint.index + 2} turn {abs(joint.in_plane_made_deg):.0f} deg in the "
                    f"plate's own plane, which {kit.name} cannot do."
                )
    for joint in joints:
        if max(abs(joint.out_of_plane_made_deg), abs(joint.twist_made_deg)) >= MIN_BEND_DEG:
            if not kit.can_make("out_of_plane") and abs(joint.out_of_plane_made_deg) >= MIN_BEND_DEG:
                warnings.append(
                    f"Holes {joint.index + 1}-{joint.index + 2} curve {abs(joint.out_of_plane_made_deg):.0f} deg, "
                    f"which {kit.name} cannot do."
                )
                continue
            add_key("out_of_plane", joint)
            if abs(joint.twist_made_deg) > MAX_PRESSED_TWIST_DEG and not kit.can_twist:
                warnings.append(
                    f"Holes {joint.index + 1}-{joint.index + 2} twist {abs(joint.twist_made_deg):.0f} deg: more than a "
                    "press gives reliably. Start it with twisting irons before the key."
                )
    if head.kind != "lever":
        worst = max((k.loading_depth_mm for k in keys if k.kind != "in_plane"), default=0.0)
        if worst > head.opening_mm:
            warnings.append(
                f"The plate and key need the post {worst:.1f} mm below the lobes to load, and this bender is set "
                f"to open {head.opening_mm:.1f} mm. Check it on the fit test, and correct its measurements."
            )
        worst = max((k.loading_depth_mm for k in keys if k.kind == "in_plane"), default=0.0)
        if worst > head.opening_mm:
            warnings.append(
                f"An in-plane key (the A keys) goes in with the plate on its edge: {worst:.1f} mm in the head, and "
                f"this bender is set to open {head.opening_mm:.1f} mm. Measure its widest opening. If it really "
                "opens less, make those bends with its nose or with irons, and check each on its key."
            )
    distances = np.concatenate([[0.0], np.cumsum(pitches)])
    guide = BendingGuide(
        centres=centres.copy(),
        frames=bent,
        stock_centres=sc.copy(),
        stock_frames=supplied,
        stock_distances=distances,
        joints=joints,
        keys=keys,
        kit=kit,
        head=head,
        width_mm=float(width_mm),
        thickness_mm=float(thickness_mm),
        hole_diameter_mm=float(hole_diameter_mm),
        hole_stretch_mm=stretch,
        play_mm=float(play_mm),
        pin_diameter_mm=2.0 * (hole_radius - float(play_mm)),
        alloy_name=alloy_name,
        warnings=warnings,
    )
    guide.deviation_mm = _shape_error(guide.stage(done=True)[0], centres)
    if guide.deviation_mm > SHAPE_CAUTION_MM:
        warnings.append(
            f"The plate these keys make lies up to {guide.deviation_mm:.1f} mm from the planned plate. "
            "Check it on the printed jaw."
        )
    return guide


def tool_parts(guide: BendingGuide, key: BendKey) -> list[GuidePart]:
    """The bender's head as it stands at this key's stop, for the rehearsal."""
    head = guide.head
    if head.kind != "three_point":
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
    """A key for one known bend on this plate and bender."""
    pitch = float(np.linalg.norm(guide.stock_centres[1] - guide.stock_centres[0]))
    turn = _compose(0.0, -FIT_TEST_ANGLE_DEG, 0.0)
    name = str(int(FIT_TEST_ANGLE_DEG))
    return _build_key(
        name, 1, 0, "out_of_plane", FIT_TEST_ANGLE_DEG, FIT_TEST_ANGLE_DEG, np.eye(3), np.eye(3),
        np.array([-0.5 * pitch, 0.0, 0.0]), np.array([0.5 * pitch, 0.0, 0.0]), turn, guide.head,
        guide.width_mm, guide.thickness_mm, 0.5 * guide.hole_diameter_mm, guide.hole_stretch_mm, guide.play_mm,
    )


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
    """A key lies on its side to print, its name up: then the faces the plate comes to
    lie on stand upright, with no stair-stepping, at the printer's finest accuracy.
    Its rows are the print x, y and z axes. An in-plane key lies skin down, so that the
    pins stand up from the skin and do not hang from it."""
    if key is not None and key.kind == "in_plane":
        return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
