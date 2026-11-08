"""The mandible as a kit of printable pieces that lock together like a puzzle.

All inputs are patient millimetres. The jaw is split only if the chosen
printer needs it, along the arch and not along a straight axis, so each piece
is a run of jaw that lies well on a bed. The stretch to protect (the working
side and the resection on it) stays in one piece, and no seam comes near it.

An isosurface of real bone is hollow: thin cortex round a darker marrow, so
there is nothing for a joint to bite into. The kit is therefore built from the
jaw as a solid: the surface is filled on a fine grid and contoured again, which
keeps its form to a small fraction of a millimetre and closes its inside.

A seam is a jigsaw joint seen from the side of the jaw: a round head on a
narrower neck, on one piece, in a socket of the same shape on the next, with a
printing clearance between them. It only goes together one way, by sliding the
pieces together across the jaw, and then cannot be pulled apart along it.

Each piece is laid with the side to keep best facing up: on the resection's
piece, the outer face at the resection, which then never touches a support
and is finished as a top surface. Stems meet it from below, and keep clear of
the resection and the joints. Nothing is written unless the whole kit checks.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from .exporting import write_surface_stl

#: Build volumes, mm, as their makers publish them. "Custom" is always there
#: too; check your own machine's usable volume.
PRINTERS = {
    "Original Prusa MINI+": (180.0, 180.0, 180.0),
    "Original Prusa MK4S": (250.0, 210.0, 220.0),
    "Bambu Lab A1 mini": (180.0, 180.0, 180.0),
    "Bambu Lab P1S / X1 Carbon": (256.0, 256.0, 256.0),
    "Creality Ender-3 V3": (220.0, 220.0, 250.0),
    "Formlabs Form 3+": (145.0, 145.0, 185.0),
    "Formlabs Form 4": (200.0, 125.0, 210.0),
    "ELEGOO Mars 4 Ultra": (153.36, 77.76, 165.0),
    "ELEGOO Saturn 4 Ultra": (218.88, 122.88, 220.0),
}
#: Kept clear round the bed's edge, mm.
BED_MARGIN = 2.0
#: How far a seam keeps from the protected stretch, along the jaw, mm.
PROTECT_MM = 6.0
#: Height of the part above the bed on stems, mm.
SUPPORT_LIFT = 3.0
#: Grid the solid is built on, mm, and the smoothing of its surface, voxels.
GRID_MM = 0.3
SMOOTH_VOXELS = 1.1
#: Marrow is what closing by this fills, mm, where it is this thick from its
#: surface inwards, mm.
MARROW_CLOSE_MM = 2.1
MARROW_CORE_MM = 0.9
#: Along the jaw from a seam, within this the joint's shape decides which piece
#: bone goes to, mm; beyond it only its position along the jaw does.
SEAM_BAND_MM = 18.0
#: Stems keep this far from the resection, and from a joint, mm.
STEM_CLEAR_MM = 8.0
#: Pieces shorter than this along the jaw are not worth having, mm.
MIN_PIECE_MM = 14.0
MAX_PIECES = 10
#: Loose bits under this are dropped from a piece, mm3.
MIN_FRAGMENT_MM3 = 5.0
#: How far a joint reaches past the end of a piece, along the jaw, at most, mm.
JOINT_REACH_MM = 14.0
#: How much of a joint's head and neck must lie in bone.
MIN_JOINT_COVER = 0.7


@dataclass
class _Route:
    """The jaw's run: stations along it, and its arc position at each."""

    points: np.ndarray
    s: np.ndarray

    def along(self, points: np.ndarray) -> np.ndarray:
        out = np.empty(len(points), dtype=np.float32)
        chunk = max(1, 2_000_000 // max(len(self.points), 1))
        for i in range(0, len(points), chunk):
            part = points[i : i + chunk]
            d2 = ((part[:, None, :] - self.points[None, :, :]) ** 2).sum(axis=2)
            out[i : i + chunk] = self.s[np.argmin(d2, axis=1)]
        return out

    def tangent(self, s: float) -> np.ndarray:
        i = int(np.clip(np.searchsorted(self.s, s), 1, len(self.s) - 2))
        t = self.points[i + 1] - self.points[i - 1]
        return t / np.linalg.norm(t)

    def outward(self, s: float, centre: np.ndarray) -> np.ndarray:
        """Away from the middle of the arch, across the jaw."""
        i = int(np.clip(np.searchsorted(self.s, s), 0, len(self.s) - 1))
        t = self.tangent(s)
        out = self.points[i] - centre
        out = out - (out @ t) * t
        n = np.linalg.norm(out)
        return out / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])


@dataclass
class _Seam:
    """A jigsaw joint across the jaw at arc position ``s``."""

    s: float
    centre: np.ndarray
    along: np.ndarray  # along the jaw, towards higher s
    height: np.ndarray  # the long way of the jaw's section
    across: np.ndarray  # the way the pieces slide together
    head_radius: float
    neck_half: float
    head_offset: float

    def phi(self, points: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Negative in the piece before the seam (which owns the knob), positive after."""
        rel = points - self.centre
        n = rel @ self.along
        v = rel @ self.height
        neck = np.maximum.reduce((np.abs(v) - self.neck_half, n - self.head_offset, -n - 1.0))
        head = np.hypot(v, n - self.head_offset) - self.head_radius
        field = np.minimum.reduce((n, neck, head))
        far = np.abs(s - self.s) > SEAM_BAND_MM
        field[far] = np.where(s[far] > self.s, 6.0, -6.0)
        return field.astype(np.float32)

    def report(self) -> dict:
        return {
            "arc_position_mm": float(self.s),
            "centre_mm": self.centre.tolist(),
            "slide_together_along": self.across.tolist(),
            "head_diameter_mm": 2.0 * self.head_radius,
            "neck_width_mm": 2.0 * self.neck_half,
        }


def _route_of(frames, mesh_points: np.ndarray) -> _Route:
    if frames is not None:
        stride = max(int(round(1.0 / frames.step_mm)), 1)
        return _Route(np.asarray(frames.points[::stride], dtype=float), np.asarray(frames.s[::stride], dtype=float))
    # No arch curve: the run is the object's long axis.
    centre = mesh_points.mean(axis=0)
    _, _, axes = np.linalg.svd(mesh_points[:: max(len(mesh_points) // 5000, 1)] - centre, full_matrices=False)
    axis = axes[0] * (1.0 if axes[0][np.argmax(np.abs(axes[0]))] > 0 else -1.0)
    reach = (mesh_points - centre) @ axis
    s = np.arange(0.0, float(np.ptp(reach)) + 1.0, 1.0)
    return _Route(centre + (float(reach.min()) + s)[:, None] * axis, s)


def _solid_grid(surface, step: float):
    """The jaw filled solid on a grid: ``(mask, origin)``, mask indexed (z, y, x)."""
    import SimpleITK as sitk

    bounds = np.array(surface.GetBounds()).reshape(3, 2)
    # Room round the jaw for the closing below to work in.
    pad = int(np.ceil(1.5 * MARROW_CLOSE_MM / step)) + 2
    origin = bounds[:, 0] - pad * step
    size = np.ceil((bounds[:, 1] - bounds[:, 0]) / step).astype(int) + 2 * pad + 1
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(surface)
    stencil.SetOutputOrigin(*origin)
    stencil.SetOutputSpacing(step, step, step)
    stencil.SetOutputWholeExtent(0, int(size[0]) - 1, 0, int(size[1]) - 1, 0, int(size[2]) - 1)
    to_image = vtk.vtkImageStencilToImage()
    to_image.SetInputConnection(stencil.GetOutputPort())
    to_image.SetInsideValue(1)
    to_image.SetOutsideValue(0)
    to_image.SetOutputScalarTypeToUnsignedChar()
    to_image.Update()
    mask = vtk_to_numpy(to_image.GetOutput().GetPointData().GetScalars())
    mask = mask.reshape(int(size[2]), int(size[1]), int(size[0])).astype(np.uint8)
    solid = sitk.GetArrayFromImage(sitk.BinaryFillhole(sitk.GetImageFromArray(mask))) > 0

    def distance_to(array):
        image = sitk.GetImageFromArray(array.astype(np.uint8))
        image.SetSpacing([step, step, step])
        return sitk.GetArrayFromImage(
            sitk.SignedMaurerDistanceMap(image, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
        )

    # Marrow that reaches the surface through a gap in the cortex is no closed
    # hole. It is what a wide closing adds that is thick in itself, which leaves
    # the narrow spaces of the anatomy (between roots, foramina) as they are.
    dilated = distance_to(solid) <= MARROW_CLOSE_MM
    closed = distance_to(~dilated) >= MARROW_CLOSE_MM
    added = closed & ~solid
    core = distance_to(~added) >= MARROW_CORE_MM
    solid |= added & (distance_to(core) <= MARROW_CORE_MM + step)
    solid = sitk.GetArrayFromImage(sitk.BinaryFillhole(sitk.GetImageFromArray(solid.astype(np.uint8)))) > 0
    return solid, origin


def _coordinates(indices, origin, step) -> np.ndarray:
    return origin + np.column_stack([indices[2], indices[1], indices[0]]) * step


def _make_seam(route, solid_points, solid_s, s0: float, clearance: float) -> _Seam | None:
    """A joint at ``s0``, or None if the bone there cannot hold one."""
    near = np.abs(solid_s - s0) < 1.5
    if near.sum() < 30:
        return None
    tangent = route.tangent(s0)
    section = solid_points[near]
    centre = section.mean(axis=0)
    # Keep to the section the seam's own plane crosses, not a far part of it.
    section = section[np.abs((section - centre) @ tangent) < 2.0]
    if len(section) < 30:
        return None
    centre = section.mean(axis=0)
    flat = (section - centre) - np.outer((section - centre) @ tangent, tangent)
    _, _, axes = np.linalg.svd(flat[:: max(len(flat) // 4000, 1)], full_matrices=False)
    height = axes[0] - (axes[0] @ tangent) * tangent
    height = height / np.linalg.norm(height)
    across = np.cross(tangent, height)
    extent = float(np.ptp((section - centre) @ height))
    head = float(np.clip(0.24 * extent, 2.2, 6.0))
    if extent < 2.0 * head + 2.0 * clearance + 2.0:
        return None
    centre = centre + height * float(np.median((section - centre) @ height))
    seam = _Seam(s0, centre, tangent, height, across, head, max(0.62 * head, 1.2), 1.1 * head)
    # The head and neck must lie in bone, on the side they reach into.
    near = np.abs(solid_s - s0) < SEAM_BAND_MM
    rel = solid_points[near] - centre
    n, v, w = rel @ tangent, rel @ height, rel @ across
    close = np.abs(w) < 12.0
    n, v = n[close], v[close]
    grid_n, grid_v = np.meshgrid(np.arange(0.0, seam.head_offset + head, 0.5), np.arange(-head, head + 0.1, 0.5))
    wanted = (np.hypot(grid_v, grid_n - seam.head_offset) <= head) | (
        (np.abs(grid_v) <= seam.neck_half) & (grid_n <= seam.head_offset)
    )
    occupied = np.zeros(wanted.shape, dtype=bool)
    ii = np.round((v + head) / 0.5).astype(int)
    jj = np.round(n / 0.5).astype(int)
    keep = (ii >= 0) & (ii < wanted.shape[0]) & (jj >= 0) & (jj < wanted.shape[1])
    occupied[ii[keep], jj[keep]] = True
    if (occupied & wanted).sum() < MIN_JOINT_COVER * wanted.sum():
        return None
    return seam


def _pose_for(points: np.ndarray, bed: np.ndarray, ups, lifted: bool):
    """A rotation that fits ``points`` on ``bed``: first of ``ups`` that does.

    Rows of the rotation are the print x, y, z axes in patient space.
    """
    limit = bed - np.array([2.0 * BED_MARGIN, 2.0 * BED_MARGIN, SUPPORT_LIFT if lifted else 0.0])
    for up in ups:
        up = up / np.linalg.norm(up)
        helper = np.array([0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(helper, up)
        u = u / np.linalg.norm(u)
        v = np.cross(up, u)
        if np.ptp(points @ up) > limit[2]:
            continue
        flat_u, flat_v = points @ u, points @ v
        for yaw in np.radians(np.arange(0.0, 180.0, 10.0)):
            c, t = np.cos(yaw), np.sin(yaw)
            if np.ptp(c * flat_u + t * flat_v) <= limit[0] and np.ptp(-t * flat_u + c * flat_v) <= limit[1]:
                return np.array([c * u + t * v, -t * u + c * v, up])
    return None


def _ups(primary: np.ndarray) -> list[np.ndarray]:
    """Directions for a piece's top, best first: tilting from ``primary`` up
    towards the patient's superior, then round the other way."""
    superior = np.array([0.0, 0.0, 1.0])
    out = []
    for angle in (0.0, 20.0, 40.0, 60.0, 90.0, -20.0, -40.0, -60.0, -90.0):
        t = np.radians(angle)
        direction = np.cos(t) * primary + np.sin(t) * superior
        if np.linalg.norm(direction) > 1e-3:
            out.append(direction)
    return out


def _plan_seams(route, sample, sample_s, solid_points, solid_s, bed, guard, lifted, clearance, centre):
    """Where to split: arc positions, none in ``guard``, so every piece fits."""
    lo, hi = float(route.s[0]) - 1.0, float(route.s[-1]) + 1.0

    def fits(a, b):
        # A piece also holds the joint reaching past its end.
        chosen = sample[(sample_s >= a - JOINT_REACH_MM) & (sample_s <= b + JOINT_REACH_MM)]
        if not len(chosen):
            return True
        return _pose_for(chosen, bed, _ups(route.outward(0.5 * (a + b), centre)), lifted) is not None

    whole = _pose_for(sample, bed, _ups(route.outward(0.5 * (lo + hi), centre)), lifted)
    if whole is not None:
        return []
    if guard is None:
        raise ValueError("This jaw needs splitting for this printer. Choose the working side to protect first.")
    if not fits(*guard):
        raise ValueError("The protected working side does not fit this printer in one piece. Choose a larger printer.")

    def seam_at(s):
        return _make_seam(route, solid_points, solid_s, float(s), clearance)

    def out_to(here, end, sign, found):
        """Seams from ``here`` out to the end of the jaw, each piece fitting."""
        while (end - here) * sign > 0:
            if fits(min(here, end), max(here, end)):
                return found
            seam = None
            # The furthest place that still fits and leaves no sliver either side.
            for s in np.arange(end - sign * MIN_PIECE_MM, here + sign * (MIN_PIECE_MM - 1.0), -sign * 2.0):
                if fits(min(here, s), max(here, s)):
                    seam = seam_at(s)
                    if seam is not None:
                        break
            if seam is None:
                raise ValueError(
                    "No place on the jaw can take a joint for a piece this printer can hold. "
                    "Choose a larger printer."
                )
            found.append(seam)
            here = seam.s
            if len(found) >= MAX_PIECES:
                raise ValueError("This printer would need too many pieces. Choose a larger printer.")
        return found

    def first_seam(s, sign, limit):
        # From the far edge of the guarded piece, back towards the guard, until
        # the bone can hold a joint.
        while (limit - s) * sign >= 0:
            seam = seam_at(s)
            if seam is not None:
                return seam
            s += sign * 2.0
        raise ValueError("No joint fits between the protected side and the rest. Choose a larger printer.")

    # The guarded piece grows as far as it can each way, to leave fewer seams.
    a, b = guard
    while a - 2.0 > lo and fits(a - 2.0, b):
        a -= 2.0
    while b + 2.0 < hi and fits(a, b + 2.0):
        b += 2.0
    plan = []
    if a > lo + MIN_PIECE_MM:
        seam = first_seam(a, 1.0, guard[0])
        plan = out_to(seam.s, lo, -1.0, plan + [seam])
    if b < hi - MIN_PIECE_MM:
        seam = first_seam(b, -1.0, guard[1])
        plan = out_to(seam.s, hi, 1.0, plan + [seam])
    return sorted(plan, key=lambda item: item.s)


def _piece_mesh(smooth, along, origin, step, lower, upper, clearance, box):
    """One piece of the jaw, its surface in patient space, from its fields."""
    zz, yy, xx = np.meshgrid(*(np.arange(b.start, b.stop) for b in box), indexing="ij")
    points = origin + np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]) * step
    s = along[box].ravel()
    field = smooth[box].ravel() - 0.5
    if lower is not None:
        field = np.minimum(field, lower.phi(points, s) - 0.5 * clearance)
    if upper is not None:
        field = np.minimum(field, -upper.phi(points, s) - 0.5 * clearance)
    shape = zz.shape
    # Nothing may run out of the box: a contour there would be open.
    field = field.reshape(shape)
    field[[0, -1], :, :] = -1.0
    field[:, [0, -1], :] = -1.0
    field[:, :, [0, -1]] = -1.0
    field = field.ravel()
    image = vtk.vtkImageData()
    image.SetDimensions(shape[2], shape[1], shape[0])
    image.SetSpacing(step, step, step)
    image.SetOrigin(*(origin + np.array([box[2].start, box[1].start, box[0].start]) * step))
    image.GetPointData().SetScalars(numpy_to_vtk(field.astype(np.float32), deep=True))
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, 0.0)
    contour.ComputeNormalsOff()
    contour.Update()
    return contour.GetOutput()


def _tidy(mesh: vtk.vtkPolyData, name: str) -> vtk.vtkPolyData:
    """Drop specks, orient outward, and refuse anything that is not closed."""
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(mesh)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()
    labelled = connectivity.GetOutput()
    if not labelled.GetNumberOfPoints():
        raise ValueError(f"{name} came out empty. Check the working side and the printer.")
    regions = vtk_to_numpy(labelled.GetPointData().GetArray("RegionId"))
    points = vtk_to_numpy(labelled.GetPoints().GetData()).astype(float)
    faces = vtk_to_numpy(labelled.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    signed = np.einsum("ij,ij->i", a - points.mean(axis=0), np.cross(b - points.mean(axis=0), c - points.mean(axis=0))) / 6.0
    face_region = regions[faces[:, 0]]
    volumes = np.bincount(face_region, weights=signed)
    flip = volumes[int(np.argmax(np.abs(volumes)))] < 0
    faces = faces[(np.abs(volumes) >= MIN_FRAGMENT_MM3)[face_region]]
    if flip:
        faces = faces[:, ::-1]
    used, inverse = np.unique(faces, return_inverse=True)
    out = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points[used], deep=True))
    out.SetPoints(vtk_points)
    connectivity_array = numpy_to_vtk(inverse.reshape(-1).astype(np.int64), deep=True, array_type=vtk.VTK_ID_TYPE)
    offsets = numpy_to_vtk(np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True, array_type=vtk.VTK_ID_TYPE)
    cells = vtk.vtkCellArray()
    cells.SetData(offsets, connectivity_array)
    out.SetPolys(cells)
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(out)
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.Update()
    if edges.GetOutput().GetNumberOfCells():
        raise ValueError(f"{name} is not a closed solid. Try a different printer size or threshold.")
    return out


def _transform(mesh, rotation, shift):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = shift
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    moved = vtk.vtkPolyData()
    moved.ShallowCopy(mesh)
    new_points = vtk.vtkPoints()
    new_points.SetData(numpy_to_vtk(points @ rotation.T + shift, deep=True))
    moved.SetPoints(new_points)
    return moved, matrix


def _stem(cx, cy, height):
    """A stem: flared foot on the bed, thin neck, a narrow tip that snaps off."""
    n = 16
    angle = np.arange(n) * 2.0 * np.pi / n
    radii = [1.8, 1.8, 0.75, 0.75, 0.30]
    zs = [0.0, 0.35, 0.8, max(height - 0.8, 0.9), height + 0.10]
    rings = [np.column_stack((cx + r * np.cos(angle), cy + r * np.sin(angle), np.full(n, z))) for r, z in zip(radii, zs)]
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(4 * n, 5 * n))]
    for k in range(4):
        for i in range(n):
            j = (i + 1) % n
            faces.append((k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i))
    mesh = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.vstack(rings), deep=True))
    mesh.SetPoints(points)
    cells = vtk.vtkCellArray()
    for ids in faces:
        cells.InsertNextCell(len(ids), [int(i) for i in ids])
    mesh.SetPolys(cells)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(mesh)
    triangles.Update()
    return triangles.GetOutput()


def _stems(part, rotation, shift, avoid):
    """Stems under ``part`` (in print pose), clear of everything in ``avoid``."""
    locator = vtk.vtkOBBTree()
    locator.SetDataSet(part)
    locator.BuildLocator()
    bounds = np.array(part.GetBounds()).reshape(3, 2)
    lo, hi = bounds[:, 0], bounds[:, 1]
    candidates = []
    for x in np.arange(lo[0] + 2.0, hi[0] - 1.0, 3.0):
        for y in np.arange(lo[1] + 2.0, hi[1] - 1.0, 3.0):
            hits = vtk.vtkPoints()
            locator.IntersectWithLine((x, y, -1.0), (x, y, hi[2] + 1.0), hits, None)
            if not hits.GetNumberOfPoints():
                continue
            z = min(hits.GetPoint(i)[2] for i in range(hits.GetNumberOfPoints()))
            world = (np.array([x, y, z]) - shift) @ rotation
            if len(avoid) and np.min(np.linalg.norm(avoid - world, axis=1)) < STEM_CLEAR_MM:
                continue
            candidates.append((z, x, y))
    picked = []
    for z, x, y in sorted(candidates):
        if all(np.hypot(x - xx, y - yy) >= 12.0 for _, xx, yy in picked):
            picked.append((z, x, y))
    if len(picked) < 3 or np.linalg.matrix_rank(np.array([[x, y, 1.0] for _, x, y in picked])) < 3:
        raise ValueError("Too little safe underside for three stems on a piece. Switch stems off or use a larger printer.")
    out = vtk.vtkAppendPolyData()
    for z, x, y in picked:
        out.AddInputData(_stem(x, y, z))
    out.Update()
    return out.GetOutput(), len(picked)


def _guard_span(route, working_side, cut_arcs, midline_x):
    """The arc positions to keep whole, with the side actually chosen."""
    side = working_side
    x = route.points[:, 0]
    middle = len(x) // 2
    crossings = np.flatnonzero(np.sign(x[:-1] - midline_x) * np.sign(x[1:] - midline_x) <= 0)
    if len(crossings):
        s_mid = float(route.s[crossings[np.argmin(np.abs(crossings - middle))]])
    else:
        s_mid = float(route.s[middle])
    if side == "auto" and len(cut_arcs):
        at = np.array([np.argmin(np.abs(route.s - s)) for s in cut_arcs])
        off = float(np.mean(x[at]) - midline_x)
        side = "left" if off > 1.0 else "right" if off < -1.0 else "auto"
    spans = []
    if side in ("left", "right"):
        # Patient left is +x in LPS. Which end of the run that is, the run says.
        first_is_left = x[0] > x[-1]
        lower_half = (side == "left") == first_is_left
        spans.append((float(route.s[0]) - 1.0, s_mid) if lower_half else (s_mid, float(route.s[-1]) + 1.0))
    if len(cut_arcs):
        spans.append((float(np.min(cut_arcs)), float(np.max(cut_arcs))))
    if not spans:
        return side, None
    lo = min(span[0] for span in spans) - PROTECT_MM
    hi = max(span[1] for span in spans) + PROTECT_MM
    return side, (max(lo, float(route.s[0]) - 1.0), min(hi, float(route.s[-1]) + 1.0))


def _instructions(count: int, has_cut: bool, supports: bool) -> str:
    lines = ["Print at 100% scale, in millimetres, keeping each piece's exported position and orientation."]
    if has_cut:
        lines.append(
            "The piece holding the resection is laid with the outer face at the resection upward, so "
            "that face is finished as a top surface and no support touches it. Do not re-orient it."
        )
    if supports:
        lines.append(
            "Import each piece-NN-stems.stl with its piece as parts of one object, keeping their relative "
            "positions; do not drop either to the bed separately. The stems have flared feet and narrow "
            "tips that snap off. Add slicer supports only where it reports an overhang, kept to the underside."
        )
    if count > 1:
        lines.append(
            f"The {count} pieces are numbered along the jaw. Each seam is a jigsaw joint: a round head on a "
            "narrower neck on the lower-numbered piece, in a matching socket on the next. Slide them together "
            "across the jaw, from the side, never by pulling along it. Print one joint first to check the "
            "clearance on your machine. A drop of glue once it fits is fine; never force it."
        )
    lines.append("assembly.json records where each piece belongs in patient space.")
    return "\n".join(lines) + "\n"


def export_print_kit(folder, surface, build_mm, *, working_side="auto", cut_points=(), midline_x=None,
                     printer_name="Custom", supports=True, clearance_mm=0.3, frames=None):
    """Write a new, self-contained kit in ``folder`` and return its files.

    ``cut_points`` bound the resection, ``frames`` is the arch curve. No seam
    comes within ``PROTECT_MM`` of the resection or into the working side. An
    earlier kit in the folder is never overwritten.
    """
    import SimpleITK as sitk

    bed = np.asarray(build_mm, dtype=float)
    working_side = str(working_side).lower()
    if bed.shape != (3,) or not np.isfinite(bed).all() or np.min(bed) <= 10:
        raise ValueError("Enter three usable printer build dimensions greater than 10 mm.")
    if working_side not in ("auto", "left", "right"):
        raise ValueError("Choose Auto, Left or Right working side.")
    if not np.isfinite(clearance_mm) or not 0.1 <= clearance_mm <= 0.7:
        raise ValueError("Choose a joint clearance between 0.1 and 0.7 mm.")
    cut = np.asarray(cut_points, dtype=float).reshape(-1, 3)
    if not np.isfinite(cut).all():
        raise ValueError("Invalid resection coordinates.")
    target = Path(folder)
    if not target.is_dir():
        raise ValueError("Choose an existing print-kit destination folder.")
    if surface is None or not surface.GetNumberOfPolys():
        raise ValueError("There is no jaw surface to print.")
    vertices = vtk_to_numpy(surface.GetPoints().GetData()).astype(float)
    if not np.isfinite(vertices).all():
        raise ValueError("The model contains non-finite coordinates.")

    route = _route_of(frames, vertices)
    mid = float(0.5 * (vertices[:, 0].min() + vertices[:, 0].max()) if midline_x is None else midline_x)
    centre = vertices.mean(axis=0)
    cut_arcs = route.along(cut) if len(cut) else np.zeros(0)
    side, guard = _guard_span(route, working_side, cut_arcs, mid)

    step = GRID_MM
    solid, origin = _solid_grid(surface, step)
    if not solid.any():
        raise ValueError("The jaw surface does not enclose a solid. Check the threshold.")
    solid_index = np.nonzero(solid)
    stride = max(len(solid_index[0]) // 400_000, 1)
    solid_index = tuple(i[::stride] for i in solid_index)
    solid_points = _coordinates(solid_index, origin, step)
    solid_s = route.along(solid_points)
    every = max(len(solid_points) // 12_000, 1)
    sample, sample_s = solid_points[::every], solid_s[::every]

    seams = _plan_seams(route, sample, sample_s, solid_points, solid_s, bed, guard, supports, clearance_mm, centre)

    image = sitk.Cast(sitk.GetImageFromArray(solid.astype(np.uint8)), sitk.sitkFloat32)
    smooth = sitk.GetArrayFromImage(sitk.SmoothingRecursiveGaussian(image, SMOOTH_VOXELS)).astype(np.float32)
    # Position along the jaw of everything near the bone, on a coarser grid; a
    # voxel off the coarse bone takes the blank position, and is empty anyway.
    coarse = 4
    near = sitk.GetArrayFromImage(
        sitk.BinaryDilate(sitk.GetImageFromArray(solid.astype(np.uint8)), [coarse, coarse, coarse])
    )[::coarse, ::coarse, ::coarse] > 0
    along_coarse = np.zeros(near.shape, dtype=np.float32)
    index = np.nonzero(near)
    along_coarse[index] = route.along(_coordinates(index, origin, step * coarse))
    along = np.repeat(np.repeat(np.repeat(along_coarse, coarse, 0), coarse, 1), coarse, 2)[
        : solid.shape[0], : solid.shape[1], : solid.shape[2]
    ]

    limits = [None, *seams, None]
    pieces = []
    for i in range(len(seams) + 1):
        lower, upper = limits[i], limits[i + 1]
        lo_s = lower.s if lower else -np.inf
        hi_s = upper.s if upper else np.inf
        # The box of this piece, wide enough for its knob and its neighbour's.
        occupied = (solid_s >= lo_s - JOINT_REACH_MM) & (solid_s <= hi_s + JOINT_REACH_MM)
        pad = 6
        box = tuple(
            slice(max(int(indices[occupied].min()) - pad, 0), min(int(indices[occupied].max()) + pad + 1, solid.shape[axis]))
            for axis, indices in enumerate(solid_index)
        )
        mesh = _piece_mesh(smooth, along, origin, step, lower, upper, clearance_mm, box)
        pieces.append((_tidy(mesh, f"Piece {i + 1}"), lo_s, hi_s))

    joints = np.array([seam.centre for seam in seams]).reshape(-1, 3)
    avoid = np.vstack([cut, joints])

    report = {
        "printer": printer_name,
        "build_mm": bed.tolist(),
        "bed_margin_mm": BED_MARGIN,
        "working_side": side if side in ("left", "right") else "unspecified",
        "protected_arc_mm": list(guard) if guard else None,
        "protected_margin_mm": PROTECT_MM,
        "midline_x_mm": mid,
        "clearance_mm": clearance_mm,
        "joins": [seam.report() for seam in seams],
        "parts": [],
        "units": "mm; source coordinates DICOM LPS",
    }
    with tempfile.TemporaryDirectory(prefix=".mandiplan-", dir=target) as tmp:
        staged = Path(tmp) / "kit"
        staged.mkdir()
        for i, (part, lo_s, hi_s) in enumerate(pieces, 1):
            points = vtk_to_numpy(part.GetPoints().GetData()).astype(float)
            at = 0.5 * (max(lo_s, float(route.s[0])) + min(hi_s, float(route.s[-1])))
            inside = (cut_arcs >= lo_s) & (cut_arcs <= hi_s)
            holds_cut = bool(inside.any())
            if holds_cut:
                at = float(np.median(cut_arcs[inside]))
            rotation = _pose_for(points, bed, _ups(route.outward(at, centre)), supports)
            room = bed - np.array([2.0 * BED_MARGIN, 2.0 * BED_MARGIN, SUPPORT_LIFT if supports else 0.0])
            if rotation is None:
                raise ValueError(f"Piece {i} does not fit this printer. Choose a larger printer.")
            oriented = points @ rotation.T
            low, high = oriented.min(axis=0), oriented.max(axis=0)
            if np.any(high - low > room + 0.05):
                raise ValueError(f"Piece {i} does not fit this printer. Choose a larger printer.")
            shift = np.r_[(bed[:2] - (high[:2] - low[:2])) / 2.0, SUPPORT_LIFT if supports else 0.0] - low
            printed, matrix = _transform(part, rotation, shift)
            name = f"piece-{i:02d}.stl"
            write_surface_stl(staged / name, printed)
            spec = {
                "file": name,
                "world_to_print": matrix.tolist(),
                "print_to_world": np.linalg.inv(matrix).tolist(),
                "size_mm": (high - low).tolist(),
                "up_in_patient_lps": rotation[2].tolist(),
                "holds_resection": holds_cut,
                "arc_from_mm": float(lo_s) if np.isfinite(lo_s) else None,
                "arc_to_mm": float(hi_s) if np.isfinite(hi_s) else None,
            }
            if supports:
                stems, count = _stems(printed, rotation, shift, avoid)
                bounds = np.asarray(stems.GetBounds()).reshape(3, 2)
                if np.any(bounds[:, 0] < -1e-4) or np.any(bounds[:, 1] > bed + 1e-4):
                    raise ValueError("Stems would leave the bed. Switch stems off or choose a larger printer.")
                stem_name = f"piece-{i:02d}-stems.stl"
                write_surface_stl(staged / stem_name, stems)
                spec.update(stems_file=stem_name, breakaway_stems=count)
            report["parts"].append(spec)
        (staged / "assembly.json").write_text(json.dumps(report, indent=2) + "\n")
        (staged / "PRINT-AND-ASSEMBLE.txt").write_text(_instructions(len(pieces), bool(len(cut)), supports))
        # Publish only once everything is complete; never over an earlier kit.
        suffix = 1
        while True:
            final = target / ("MandiPlan-print-kit" if suffix == 1 else f"MandiPlan-print-kit-{suffix}")
            if final.exists():
                suffix += 1
                continue
            try:
                os.rename(staged, final)
                break
            except FileExistsError:
                suffix += 1
        return sorted(final.iterdir())
