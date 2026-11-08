"""Support rods and a raft, for a part already standing in its print pose.

A resin printer builds a part hanging in the air, each layer needing something
already cured to hang from. Every surface turned towards the build plate at a
shallow angle needs a rod, and every lowest point (an island, where a layer would
start from nothing) needs one first of all or it fails. So the part is read along
vertical rays: each place a ray enters the solid from below is a surface turned to
the plate. The lowest of those on each island are taken first, then the rest, no
two rods closer than a spacing that keeps layers from sagging between them.

A rod stands on the raft if nothing is below it, and otherwise on the part's own
surface below, unless that surface must be kept clean. A caller can forbid
surfaces outright: nothing touches them. Its tip narrows to a point that snaps off.
Tall rods are tied together so they do not bow during peeling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

#: Sample spacing of the rays, mm, and the closest two rods may be, mm.
RAY_STEP_MM = 1.5
ROD_SPACING_MM = 5.0
#: A surface flatter than this to the horizontal needs holding up, degrees. Steeper
#: ones build on their own layer below.
SELF_SUPPORTING_DEG = 45.0
#: Raft thickness, how far it runs past the outer rods, mm.
RAFT_MM = 1.0
RAFT_MARGIN_MM = 1.8
#: The rod: radius of its body, foot and tip; how long the tip tapers; how far the tip
#: goes into the part so that the two are one solid when printed, mm.
ROD_RADIUS_MM = 0.6
FOOT_RADIUS_MM = 1.2
TIP_RADIUS_MM = 0.25
TIP_LENGTH_MM = 1.6
TIP_INTO_PART_MM = 0.2
#: How high above the raft's top the part's lowest point stands, mm.
LIFT_ABOVE_RAFT_MM = 3.0
#: Rods further apart than this are not tied; ties are this far apart up a rod, mm.
TIE_REACH_MM = 12.0
TIE_EVERY_MM = 10.0
TIE_RADIUS_MM = 0.4


def lift_mm() -> float:
    """How high a part has to stand above the bed to leave room for its supports."""
    return RAFT_MM + LIFT_ABOVE_RAFT_MM


@dataclass
class _Contact:
    x: float
    y: float
    z: float
    below: float | None  # z of the part's own surface under it, or None for the raft
    needs: bool  # a surface too flat to build by itself
    grid: tuple[int, int] = (0, 0)
    lowest: bool = False  # the bottom of an island


def _rod_mesh(x: float, y: float, z_top: float, z_base: float, foot: bool) -> tuple[np.ndarray, np.ndarray]:
    """A closed rod from ``z_base`` to its tip at ``z_top``: ``(points, triangles)``."""
    sides = 12
    angle = np.arange(sides) * 2.0 * np.pi / sides
    span = z_top - z_base
    tip = min(TIP_LENGTH_MM, 0.6 * span)
    if foot:
        rings = [
            (z_base, FOOT_RADIUS_MM),
            (z_base + min(0.6, 0.25 * span), ROD_RADIUS_MM),
            (z_top - tip, ROD_RADIUS_MM),
            (z_top, TIP_RADIUS_MM),
        ]
    else:
        # Standing on the part: a narrow tip there too.
        low = min(TIP_LENGTH_MM, 0.3 * span)
        rings = [
            (z_base, TIP_RADIUS_MM),
            (z_base + low, ROD_RADIUS_MM),
            (z_top - min(tip, 0.3 * span), ROD_RADIUS_MM),
            (z_top, TIP_RADIUS_MM),
        ]
    points = np.vstack(
        [np.column_stack((x + r * np.cos(angle), y + r * np.sin(angle), np.full(sides, z))) for z, r in rings]
    )
    faces = []
    for k in range(len(rings) - 1):
        for i in range(sides):
            j = (i + 1) % sides
            a, b, c, d = k * sides + i, k * sides + j, (k + 1) * sides + j, (k + 1) * sides + i
            faces.extend(((a, b, c), (a, c, d)))
    top = (len(rings) - 1) * sides
    for i in range(1, sides - 1):
        faces.append((0, i + 1, i))
        faces.append((top, top + i, top + i + 1))
    return points, np.array(faces, dtype=np.int64)


def _bar_mesh(p: np.ndarray, q: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """A closed round bar from ``p`` to ``q``."""
    axis = q - p
    length = float(np.linalg.norm(axis))
    axis = axis / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    sides = 8
    angle = np.arange(sides) * 2.0 * np.pi / sides
    ring = radius * (np.cos(angle)[:, None] * u + np.sin(angle)[:, None] * v)
    points = np.vstack([p + ring, q + ring])
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.extend(((i, j, sides + j), (i, sides + j, sides + i)))
    for i in range(1, sides - 1):
        faces.append((0, i, i + 1))
        faces.append((sides, sides + i + 1, sides + i))
    return points, np.array(faces, dtype=np.int64)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """The convex hull of 2-D points, counter-clockwise."""
    pts = sorted({(round(float(x), 6), round(float(y), 6)) for x, y in points})
    if len(pts) < 3:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def _raft_mesh(feet: np.ndarray, floor: float, top: float, limits) -> tuple[np.ndarray, np.ndarray]:
    """A plate under the rods: their outline grown a little, kept on the bed."""
    hull = _convex_hull(feet)
    if len(hull) < 3:
        lo, hi = feet.min(axis=0), feet.max(axis=0)
        hull = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]], dtype=float)
    # Grow it at each corner, along the corner's two neighbours' outward normals.
    grown = []
    count = len(hull)
    for i in range(count):
        before, here, after = hull[i - 1], hull[i], hull[(i + 1) % count]
        for start, end in ((before, here), (here, after)):
            edge = end - start
            length = float(np.linalg.norm(edge))
            if length < 1e-9:
                continue
            grown.append(here + RAFT_MARGIN_MM * np.array([edge[1], -edge[0]]) / length)
    outline = _convex_hull(np.array(grown))
    if limits is not None:
        outline = np.clip(outline, limits[0], limits[1])
    n = len(outline)
    points = np.vstack(
        [np.column_stack((outline, np.full(n, floor))), np.column_stack((outline, np.full(n, top)))]
    )
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.extend(((i, j, n + j), (i, n + j, n + i)))
    for i in range(1, n - 1):
        faces.append((0, i + 1, i))
        faces.append((n, n + i, n + i + 1))
    return points, np.array(faces, dtype=np.int64)


def _polydata(pieces) -> vtk.vtkPolyData:
    """One mesh of many closed shells from ``(points, triangles)`` pairs."""
    all_points, all_faces, offset = [], [], 0
    for points, faces in pieces:
        all_points.append(points)
        all_faces.append(faces + offset)
        offset += len(points)
    points = np.vstack(all_points)
    faces = np.vstack(all_faces)
    mesh = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    mesh.SetPoints(vtk_points)
    cells = vtk.vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(faces.reshape(-1).astype(np.int64), deep=True),
    )
    mesh.SetPolys(cells)
    return mesh


@dataclass
class Supports:
    """A part's raft and rods."""

    mesh: vtk.vtkPolyData
    rods: int
    #: Share of the surfaces that needed holding up with a rod within reach.
    covered: float
    #: Rods standing on the part itself, not the raft.
    on_part: int
    #: Where each rod touches the part, ``(N, 3)``.
    contacts: np.ndarray


def support_structure(part, keep_off=None, limits=None, spacing: float = ROD_SPACING_MM) -> Supports:
    """Rods and a raft for ``part``, in print pose, its lowest point at ``lift_mm()``.

    ``keep_off(points, normals)`` says which surface points no rod may touch, and
    ``limits`` is the bed as ``(low_xy, high_xy)``.
    """
    locator = vtk.vtkOBBTree()
    locator.SetDataSet(part)
    locator.BuildLocator()
    bounds = np.array(part.GetBounds()).reshape(3, 2)
    corners = vtk.vtkIdList()
    needs_cos = float(np.cos(np.radians(SELF_SUPPORTING_DEG)))
    allowed: list[_Contact] = []
    held_up: list[tuple[float, float, float]] = []
    forbidden_points = []
    xs = np.arange(bounds[0, 0] + 0.4, bounds[0, 1], RAY_STEP_MM)
    ys = np.arange(bounds[1, 0] + 0.4, bounds[1, 1], RAY_STEP_MM)
    columns: dict[tuple[int, int], list[_Contact]] = {}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            hits = vtk.vtkPoints()
            cells = vtk.vtkIdList()
            locator.IntersectWithLine((float(x), float(y), bounds[2, 0] - 1.0), (float(x), float(y), bounds[2, 1] + 1.0), hits, cells)
            count = hits.GetNumberOfPoints()
            if not count:
                continue
            order = sorted(range(count), key=lambda k: hits.GetPoint(k)[2])
            for rank, k in enumerate(order):
                part.GetCellPoints(cells.GetId(k), corners)
                a, b, c = (np.array(part.GetPoint(corners.GetId(m))) for m in range(3))
                normal = np.cross(b - a, c - a)
                size = float(np.linalg.norm(normal))
                if size < 1e-12 or normal[2] >= 0.0:
                    continue  # leaving the solid, going up: not a surface turned down
                normal = normal / size
                z = hits.GetPoint(k)[2]
                below = hits.GetPoint(order[rank - 1])[2] if rank > 0 else None
                contact = _Contact(float(x), float(y), float(z), below, bool(-normal[2] >= needs_cos), (i, j))
                columns.setdefault((i, j), []).append(contact)
                if contact.needs:
                    held_up.append((contact.x, contact.y, contact.z))
                bad = False
                if keep_off is not None:
                    bad = bool(keep_off(np.array([[x, y, z]], dtype=float), normal[None])[0])
                    if not bad and below is not None:
                        # Standing on the part: where it stands must be allowed too.
                        bad = bool(keep_off(np.array([[x, y, below]], dtype=float), np.array([[0.0, 0.0, 1.0]]))[0])
                if bad:
                    forbidden_points.append((x, y, z))
                    continue
                allowed.append(contact)
    # The bottom of each island: nothing lower beside it.
    for contact in allowed:
        i, j = contact.grid
        lower = False
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if not (di or dj):
                    continue
                for other in columns.get((i + di, j + dj), []):
                    if abs(other.z - contact.z) < 3.0 and other.z < contact.z - 1e-6:
                        lower = True
        contact.lowest = not lower
    first = sorted((c for c in allowed if c.lowest), key=lambda c: c.z)
    then = sorted((c for c in allowed if c.needs and not c.lowest), key=lambda c: c.z)
    picked: list[_Contact] = []
    for group, gap in ((first, 0.6 * spacing), (then, spacing)):
        for contact in group:
            place = np.array([contact.x, contact.y, contact.z])
            if all(np.linalg.norm(place - np.array([p.x, p.y, p.z])) >= gap for p in picked):
                picked.append(contact)
    if not picked:
        raise ValueError("Nothing under this part can take a support.")

    raft_top = RAFT_MM
    pieces = []
    grounded = []
    for contact in picked:
        if contact.below is None:
            grounded.append(contact)
            pieces.append(_rod_mesh(contact.x, contact.y, contact.z + TIP_INTO_PART_MM, raft_top - 0.1, True))
        else:
            pieces.append(
                _rod_mesh(contact.x, contact.y, contact.z + TIP_INTO_PART_MM, contact.below - TIP_INTO_PART_MM, False)
            )
    if not grounded:
        raise ValueError("No support under this part reaches the bed.")
    # Ties between neighbouring tall rods.
    tall = [c for c in grounded if c.z - raft_top > TIE_EVERY_MM + 4.0]
    tied = set()
    for m, one in enumerate(tall):
        for n, other in enumerate(tall):
            if n <= m or (m, n) in tied:
                continue
            apart = float(np.hypot(one.x - other.x, one.y - other.y))
            if apart > TIE_REACH_MM:
                continue
            tied.add((m, n))
            top = min(one.z, other.z) - TIP_LENGTH_MM - 1.0
            for z in np.arange(raft_top + TIE_EVERY_MM, top, TIE_EVERY_MM):
                pieces.append(
                    _bar_mesh(np.array([one.x, one.y, z]), np.array([other.x, other.y, z]), TIE_RADIUS_MM)
                )
    feet = np.array([[c.x, c.y] for c in grounded])
    pieces.append(_raft_mesh(feet, 0.0, raft_top, limits))

    covered = 1.0
    if held_up:
        held = np.array(held_up)
        chosen = np.array([[c.x, c.y, c.z] for c in picked])
        nearest = np.min(
            np.linalg.norm(held[:, None, :] - chosen[None, :, :], axis=2), axis=1
        )
        covered = float(np.mean(nearest <= 1.2 * spacing))
    contacts = np.array([[c.x, c.y, c.z] for c in picked])
    return Supports(_polydata(pieces), len(picked), covered, len(picked) - len(grounded), contacts)


def merged(part: vtk.vtkPolyData, supports: Supports) -> vtk.vtkPolyData:
    """The part and its supports as one mesh of closed shells, ready to print."""
    append = vtk.vtkAppendPolyData()
    append.AddInputData(part)
    append.AddInputData(supports.mesh)
    append.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(append.GetOutput())
    return out

