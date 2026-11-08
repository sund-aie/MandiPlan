"""The mandible as a set of printable parts that connect with pegs.

All inputs are patient millimetres. The jaw is split only if the chosen printer
needs it, along the arch and not along a straight axis, so each part is a run of
jaw that lies well on a bed. The stretch that matters stays whole in one part
and no cut comes near it: the working side and the resection on it, or, on a
printer too small for a whole side, the resection and the bone either side of it
that the plate is fixed to.

A cut is flat, square to the jaw, and two parts join face to face. One part has
two round pegs standing out of its face, the other two holes to take them: of
different sizes, so they only go together one way. The pegs go on the part
further from the working section, so a broken peg costs a small reprint.

An isosurface of real bone is hollow: thin cortex round a darker marrow, so
there is nothing for a peg to stand in. The parts are therefore built from the
jaw as a solid, filled on a grid and contoured again, and every vertex of the
result is then moved back onto the scan's own surface. The outside of the part
is the scan's surface, to within the grid's facets, and its inside is solid.

Each part is laid with the side to keep best facing up: on the resection's part,
the outer face at the resection, which then never touches a support and is
finished as a top surface. Stems meet it from below. They keep off the outer face
of the working section altogether, and clear of the resection and of the cut
faces. Nothing is written unless the whole kit checks.
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

#: Build volumes, mm, as published for each machine. "Custom" is always there
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
    "ELEGOO Saturn 4 Ultra 12K": (218.88, 122.88, 220.0),
    "ELEGOO Saturn 4 Ultra 16K": (211.68, 118.37, 220.0),
}
#: Kept clear round the bed's edge, mm.
BED_MARGIN = 2.0
#: How far a seam keeps from the protected stretch, along the jaw, mm.
PROTECT_MM = 6.0
#: Height of the part above the bed on stems, mm.
SUPPORT_LIFT = 3.0
#: Grid the solid is built on, mm, and the smoothing of its surface, voxels. The
#: part's outside is then put back on the scan's surface, so this sets the facets.
GRID_MM = 0.3
SMOOTH_VOXELS = 1.1
#: Pores in the cortex up to about twice this are closed to find the ring round
#: the marrow, voxels; and what lies this far below that ring goes solid, mm.
MARROW_PORE_VOXELS = 3
MARROW_DEPTH_MM = 0.6
#: Within this of a cut the flat face decides which part bone goes to, mm; beyond
#: it, on the far side of the arch that the same plane also crosses, only the
#: position along the jaw does.
SEAM_NEAR_MM = 40.0
#: Stems keep this far from the resection, and from a cut face, mm.
STEM_CLEAR_MM = 8.0
#: Pieces shorter than this along the jaw are not worth having, mm.
MIN_PIECE_MM = 14.0
MAX_PIECES = 10
#: Loose bits under this are dropped from a piece, mm3.
MIN_FRAGMENT_MM3 = 5.0
#: How far a peg reaches past the end of a part, along the jaw, at most, mm.
JOINT_REACH_MM = 9.0
#: The connectors. Radii of the large and small peg, largest that the bone there
#: will take; how long they stand, how much deeper the hole, the chamfer on the
#: tip that lets it find the hole, and the least bone left round a hole, mm.
PEG_RADII_MM = ((2.0, 1.5), (1.5, 1.1), (1.1, 0.8))
PEG_LENGTH_MM = 6.0
HOLE_EXTRA_DEPTH_MM = 0.5
PEG_LEAD_IN_MM = 0.4
PEG_WALL_MM = 1.0
#: The fit test print: clearances tried, and the size of its blocks, mm.
FIT_TEST_CLEARANCES_MM = (0.1, 0.2, 0.3)
FIT_TEST_LENGTH_MM = 10.0
#: On a printer too small for the whole working side, the resection and this much
#: bone either side of it, along the jaw, stay in one part, mm.
RESECTION_FIXATION_MM = 25.0
#: A vertex this close to the scan's surface is put on it, mm; further off it is
#: where the solid was filled, and stays. And it is left alone this close to a
#: cut, to keep the face flat, mm.
SNAP_REACH_MM = 0.6
SNAP_CUT_CLEAR_MM = 1.0


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
    """A flat cut across the jaw at arc position ``s``, with two pegs across it."""

    s: float
    centre: np.ndarray
    along: np.ndarray  # along the jaw, towards higher s: the way the pegs point
    height: np.ndarray  # the long way of the jaw's section
    across: np.ndarray  # the short way
    pegs: list[tuple[float, float, float]]  # (along height, along across, radius), mm
    clearance: float
    pegs_on_lower: bool = True  # the part at lower s carries the pegs

    def _fields(self, points: np.ndarray):
        """``(t, pegs, holes)``: distance past the face towards the holes' part,
        and the two solids, each negative inside."""
        rel = points - self.centre
        n = rel @ self.along
        t = n if self.pegs_on_lower else -n
        v, w = rel @ self.height, rel @ self.across
        pegs, holes = np.full(len(t), np.inf), np.full(len(t), np.inf)
        for pv, pw, radius in self.pegs:
            radial = np.hypot(v - pv, w - pw)
            taper = radius - np.maximum(t - (PEG_LENGTH_MM - PEG_LEAD_IN_MM), 0.0)
            pegs = np.minimum(pegs, np.maximum.reduce((radial - taper, t - PEG_LENGTH_MM, -t - 1.0)))
            holes = np.minimum(
                holes,
                np.maximum.reduce(
                    (radial - (radius + self.clearance), t - (PEG_LENGTH_MM + HOLE_EXTRA_DEPTH_MM), -t - 1.0)
                ),
            )
        return t, pegs, holes

    def lower_field(self, points: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Negative in the part before the cut."""
        t, pegs, holes = self._fields(points)
        field = np.minimum(t, pegs) if self.pegs_on_lower else np.maximum(-t, -holes)
        far = np.linalg.norm(points - self.centre, axis=1) > SEAM_NEAR_MM
        field[far] = np.where(s[far] < self.s, -6.0, 6.0)
        return field.astype(np.float32)

    def upper_field(self, points: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Negative in the part after the cut."""
        t, pegs, holes = self._fields(points)
        field = np.maximum(-t, -holes) if self.pegs_on_lower else np.minimum(t, pegs)
        far = np.linalg.norm(points - self.centre, axis=1) > SEAM_NEAR_MM
        field[far] = np.where(s[far] > self.s, -6.0, 6.0)
        return field.astype(np.float32)

    def report(self) -> dict:
        return {
            "arc_position_mm": float(self.s),
            "centre_mm": self.centre.tolist(),
            "cut_normal": self.along.tolist(),
            "peg_diameters_mm": [2.0 * radius for _, _, radius in self.pegs],
            "peg_length_mm": PEG_LENGTH_MM,
            "hole_clearance_mm": self.clearance,
            "pegs_on": "lower-numbered part" if self.pegs_on_lower else "higher-numbered part",
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
    pad = MARROW_PORE_VOXELS + 4
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

    # Marrow runs the length of the jaw and opens to the outside somewhere, so it
    # is no closed hole in three dimensions. But a section across the jaw is a ring
    # of cortex round it, once the pores in the ring are closed. So each slice is
    # filled, along all three axes: a jaw encloses nothing else in any one slice.
    # Only what lies below the surface goes solid: pores stay on the outside.
    closed = sitk.GetArrayFromImage(
        sitk.BinaryMorphologicalClosing(
            sitk.GetImageFromArray(solid.astype(np.uint8)), [MARROW_PORE_VOXELS] * 3, sitk.sitkBall
        )
    ) > 0
    filled = closed.copy()
    for axis in range(3):
        occupied = np.flatnonzero(closed.any(axis=tuple(a for a in range(3) if a != axis)))
        for index in occupied:
            where = [slice(None)] * 3
            where[axis] = int(index)
            where = tuple(where)
            section = sitk.GetImageFromArray(closed[where].astype(np.uint8))
            filled[where] |= sitk.GetArrayFromImage(sitk.BinaryFillhole(section)) > 0
    image = sitk.GetImageFromArray((~filled).astype(np.uint8))
    image.SetSpacing([step, step, step])
    depth = sitk.GetArrayFromImage(
        sitk.SignedMaurerDistanceMap(image, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
    )
    solid |= depth >= MARROW_DEPTH_MM
    return sitk.GetArrayFromImage(sitk.BinaryFillhole(sitk.GetImageFromArray(solid.astype(np.uint8)))) > 0, origin


def _coordinates(indices, origin, step) -> np.ndarray:
    return origin + np.column_stack([indices[2], indices[1], indices[0]]) * step


def _make_seam(route, grid, solid_points, solid_s, s0: float, clearance: float) -> _Seam | None:
    """A cut at ``s0`` with its two pegs, or None if the bone there cannot take them.

    ``grid`` is ``(solid, origin, step)``. The pegs stand in solid bone for their
    whole length either way across the face, a wall of bone left round each.
    """
    import SimpleITK as sitk

    solid, origin, step = grid
    near = np.abs(solid_s - s0) < 1.5
    if near.sum() < 30:
        return None
    tangent = route.tangent(s0)
    section = solid_points[near]
    centre = section.mean(axis=0)
    # Keep to the section the cut's own plane crosses, not a far part of the jaw.
    section = section[np.abs((section - centre) @ tangent) < 2.0]
    if len(section) < 30:
        return None
    centre = section.mean(axis=0)
    flat = (section - centre) - np.outer((section - centre) @ tangent, tangent)
    _, _, axes = np.linalg.svd(flat[:: max(len(flat) // 4000, 1)], full_matrices=False)
    height = axes[0] - (axes[0] @ tangent) * tangent
    height = height / np.linalg.norm(height)
    across = np.cross(tangent, height)
    extent = max(float(np.ptp(flat @ height)), float(np.ptp(flat @ across)))

    # Where bone runs right through, from the depth of a hole on one side of the
    # face to the depth of a hole on the other.
    raster = 0.3
    half = 0.5 * extent + 3.0
    vs = np.arange(-half, half + raster, raster)
    ws = np.arange(-half, half + raster, raster)
    vv, ww = np.meshgrid(vs, ws, indexing="ij")
    through = np.ones(vv.shape, dtype=bool)
    depth = PEG_LENGTH_MM + HOLE_EXTRA_DEPTH_MM + PEG_WALL_MM
    for n in np.arange(-depth, depth + 0.1, 1.0):
        world = centre + vv[..., None] * height + ww[..., None] * across + n * tangent
        index = np.round((world - origin) / step).astype(int)[..., ::-1]
        inside = np.all((index >= 0) & (index < np.array(solid.shape)), axis=-1)
        here = np.zeros(vv.shape, dtype=bool)
        here[inside] = solid[index[inside][:, 0], index[inside][:, 1], index[inside][:, 2]]
        through &= here
    if not through.any():
        return None
    image = sitk.GetImageFromArray(through.astype(np.uint8))
    room = -sitk.GetArrayFromImage(
        sitk.SignedMaurerDistanceMap(image, insideIsPositive=False, squaredDistance=False, useImageSpacing=False)
    ) * raster

    for large, small in PEG_RADII_MM:
        fits_large = room >= large + clearance + PEG_WALL_MM
        fits_small = room >= small + clearance + PEG_WALL_MM
        if not fits_large.any() or not fits_small.any():
            continue
        apart = large + small + 2.0 * clearance + PEG_WALL_MM
        # As far apart as the bone allows, so they hold against turning: the large
        # one where the section has most room, the small one furthest from it.
        at = np.unravel_index(int(np.argmax(np.where(fits_large, room, -1.0))), room.shape)
        separation = np.hypot(vv - vv[at], ww - ww[at])
        candidates = fits_small & (separation >= apart)
        if not candidates.any():
            continue
        other = np.unravel_index(int(np.argmax(np.where(candidates, separation, -1.0))), room.shape)
        pegs = [(float(vv[at]), float(ww[at]), large), (float(vv[other]), float(ww[other]), small)]
        return _Seam(float(s0), centre, tangent, height, across, pegs, float(clearance))
    return None


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


def _plan_seams(route, sample, sample_s, grid, solid_points, solid_s, bed, guard, lifted, clearance, centre):
    """Where to split: arc positions, none in ``guard``, so every piece fits."""
    lo, hi = float(route.s[0]) - 1.0, float(route.s[-1]) + 1.0

    def fits(a, b, reach=JOINT_REACH_MM):
        # A part also holds the pegs reaching past its end. The working section's
        # part has holes, not pegs, and ``reach`` 0.
        chosen = sample[(sample_s >= a - reach) & (sample_s <= b + reach)]
        if not len(chosen):
            return True
        return _pose_for(chosen, bed, _ups(route.outward(0.5 * (a + b), centre)), lifted) is not None

    whole = _pose_for(sample, bed, _ups(route.outward(0.5 * (lo + hi), centre)), lifted)
    if whole is not None:
        return []
    if guard is None:
        raise ValueError("This jaw needs splitting for this printer. Choose the working side to protect first.")
    if not fits(*guard, reach=0.0):
        raise ValueError("The working section does not fit this printer in one part. Choose a larger printer.")

    def seam_at(s):
        return _make_seam(route, grid, solid_points, solid_s, float(s), clearance)

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
                    "No place on the jaw can take a cut with pegs for a part this printer can hold. "
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
        raise ValueError("No cut with pegs fits between the working section and the rest. Choose a larger printer.")

    # The guarded part grows as far as it can each way, to leave fewer seams.
    a, b = guard
    while a - 2.0 > lo and fits(a - 2.0, b, reach=0.0):
        a -= 2.0
    while b + 2.0 < hi and fits(a, b + 2.0, reach=0.0):
        b += 2.0
    plan = []
    if a > lo + MIN_PIECE_MM:
        seam = first_seam(a, 1.0, guard[0])
        plan = out_to(seam.s, lo, -1.0, plan + [seam])
    if b < hi - MIN_PIECE_MM:
        seam = first_seam(b, -1.0, guard[1])
        plan = out_to(seam.s, hi, 1.0, plan + [seam])
    return sorted(plan, key=lambda item: item.s)


def _piece_mesh(smooth, along, origin, step, lower, upper, box):
    """One piece of the jaw, its surface in patient space, from its fields."""
    zz, yy, xx = np.meshgrid(*(np.arange(b.start, b.stop) for b in box), indexing="ij")
    points = origin + np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]) * step
    s = along[box].ravel()
    field = smooth[box].ravel() - 0.5
    if lower is not None:
        field = np.minimum(field, -lower.upper_field(points, s))
    if upper is not None:
        field = np.minimum(field, -upper.lower_field(points, s))
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


def _stems(part, rotation, shift, avoid, route, working, centre):
    """Stems under ``part`` (in print pose), off the working surface.

    A stem may touch only a face turned to the bed, clear of everything in
    ``avoid``, and never the outer face of the jaw along the arc span ``working``:
    the face a plate lies on and the site is judged by.
    """
    locator = vtk.vtkOBBTree()
    locator.SetDataSet(part)
    locator.BuildLocator()
    bounds = np.array(part.GetBounds()).reshape(3, 2)
    lo, hi = bounds[:, 0], bounds[:, 1]
    candidates = []
    corners = vtk.vtkIdList()
    for x in np.arange(lo[0] + 2.0, hi[0] - 1.0, 3.0):
        for y in np.arange(lo[1] + 2.0, hi[1] - 1.0, 3.0):
            hits = vtk.vtkPoints()
            cells = vtk.vtkIdList()
            locator.IntersectWithLine((x, y, -1.0), (x, y, hi[2] + 1.0), hits, cells)
            if not hits.GetNumberOfPoints():
                continue
            heights = [hits.GetPoint(i)[2] for i in range(hits.GetNumberOfPoints())]
            lowest = int(np.argmin(heights))
            z = heights[lowest]
            part.GetCellPoints(cells.GetId(lowest), corners)
            a, b, c = (np.array(part.GetPoint(corners.GetId(i))) for i in range(3))
            normal = np.cross(b - a, c - a)
            length = float(np.linalg.norm(normal))
            if length < 1e-12:
                continue
            normal /= length
            if normal[2] > -0.2:
                continue  # not a face turned to the bed
            world = (np.array([x, y, z]) - shift) @ rotation
            if len(avoid) and np.min(np.linalg.norm(avoid - world, axis=1)) < STEM_CLEAR_MM:
                continue
            if working is not None:
                arc = float(route.along(world[None])[0])
                if working[0] <= arc <= working[1] and normal @ rotation @ route.outward(arc, centre) > 0.0:
                    continue  # the outer face of the working section
            candidates.append((z, x, y))
    picked = []
    for z, x, y in sorted(candidates):
        if all(np.hypot(x - xx, y - yy) >= 12.0 for _, xx, yy in picked):
            picked.append((z, x, y))
    if len(picked) < 3 or np.linalg.matrix_rank(np.array([[x, y, 1.0] for _, x, y in picked])) < 3:
        raise ValueError(
            "Too little safe underside for three stems on a part. Switch stems off and use the "
            "slicer's supports on the underside, or use a larger printer."
        )
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
    lo_end, hi_end = float(route.s[0]) - 1.0, float(route.s[-1]) + 1.0
    spans = []
    if side in ("left", "right"):
        # Patient left is +x in LPS. Which end of the run that is, the run says.
        first_is_left = x[0] > x[-1]
        lower_half = (side == "left") == first_is_left
        spans.append((lo_end, s_mid) if lower_half else (s_mid, hi_end))
    cut_lo = cut_hi = None
    if len(cut_arcs):
        cut_lo, cut_hi = float(np.min(cut_arcs)), float(np.max(cut_arcs))
        spans.append((cut_lo, cut_hi))
    options = []
    if spans:
        lo = max(min(span[0] for span in spans) - PROTECT_MM, lo_end)
        hi = min(max(span[1] for span in spans) + PROTECT_MM, hi_end)
        options.append((lo, hi))
    if cut_lo is not None:
        # All that must stay whole on a small printer: the section and its fixation.
        options.append((max(cut_lo - RESECTION_FIXATION_MM, lo_end), min(cut_hi + RESECTION_FIXATION_MM, hi_end)))
    return side, options


def _snapped_to_scan(part: vtk.vtkPolyData, surface: vtk.vtkPolyData, seams) -> vtk.vtkPolyData:
    """``part`` with its outside moved onto the scan's own surface.

    The grid rounds the surface off by a fraction of its size. Each vertex near
    the scan's surface goes back onto it along its normal. Cut faces, pegs and
    holes are too far from that surface to move, and stay as they are built.
    """
    for _ in range(2):
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(part)
        normals.SplittingOff()
        normals.ConsistencyOff()
        normals.AutoOrientNormalsOff()
        normals.ComputePointNormalsOn()
        normals.ComputeCellNormalsOff()
        normals.Update()
        with_normals = normals.GetOutput()
        distance = vtk.vtkDistancePolyDataFilter()
        distance.SetInputData(0, with_normals)
        distance.SetInputData(1, surface)
        distance.SignedDistanceOn()
        distance.ComputeSecondDistanceOff()
        distance.Update()
        away = vtk_to_numpy(distance.GetOutput().GetPointData().GetArray("Distance")).astype(float)
        points = vtk_to_numpy(with_normals.GetPoints().GetData()).astype(float)
        facing = vtk_to_numpy(with_normals.GetPointData().GetNormals()).astype(float)
        move = np.abs(away) < SNAP_REACH_MM
        for seam in seams:
            rel = points - seam.centre
            n = rel @ seam.along
            move &= np.abs(n) > SNAP_CUT_CLEAR_MM
            # Pegs and holes stand inside the bone, near the scan's inner surfaces.
            v, w = rel @ seam.height, rel @ seam.across
            for pv, pw, radius in seam.pegs:
                move &= ~(
                    (np.hypot(v - pv, w - pw) < radius + seam.clearance + SNAP_CUT_CLEAR_MM)
                    & (np.abs(n) < PEG_LENGTH_MM + HOLE_EXTRA_DEPTH_MM + SNAP_CUT_CLEAR_MM)
                )
        moved = points.copy()
        moved[move] -= away[move][:, None] * facing[move]
        result = vtk.vtkPolyData()
        result.SetPolys(part.GetPolys())
        new_points = vtk.vtkPoints()
        new_points.SetData(numpy_to_vtk(moved, deep=True))
        result.SetPoints(new_points)
        part = result
    return part


def _fit_test(seam: _Seam, clearances=FIT_TEST_CLEARANCES_MM) -> vtk.vtkPolyData:
    """Small blocks with this kit's pegs and holes at each clearance, to print first.

    Each pair lies apart, ready to try. Notches on the back edge count them.
    """
    step = 0.1
    large = max(radius for _, _, radius in seam.pegs)
    small = min(radius for _, _, radius in seam.pegs)
    length = FIT_TEST_LENGTH_MM
    gap = large + small + 2.0 * PEG_WALL_MM + 1.0
    width = gap + 2.0 * large + 6.0
    thick = 2.0 * large + 3.0
    meshes = vtk.vtkAppendPolyData()
    for index, clearance in enumerate(clearances):
        local = _Seam(
            0.0, np.array([0.0, 0.0, 0.5 * thick]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]), [(-0.5 * gap, 0.0, large), (0.5 * gap, 0.0, small)], clearance,
        )
        # Off the round numbers: no flat face may fall exactly on a grid plane.
        xs = np.arange(-length - 8.0, length + 8.0, step) + 0.0137
        ys = np.arange(-0.5 * width - 0.5, 0.5 * width + 0.5, step) + 0.0137
        zs = np.arange(-0.5, thick + 0.5, step) + 0.0137
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        arcs = np.zeros(len(points))
        block = np.maximum.reduce((np.abs(xx.ravel()) - length, np.abs(yy.ravel()) - 0.5 * width, zz.ravel() - thick, -zz.ravel()))
        for side, field in ((-1.0, local.lower_field(points, arcs)), (1.0, local.upper_field(points, arcs))):
            solid = np.maximum(block, field)
            # Count marks: a notch for each place in the row, on each block.
            for mark in range(index + 1):
                notch = np.maximum(
                    np.abs(xx.ravel() - side * (0.5 * length + 1.6 * mark)) - 0.5,
                    0.5 * width - 1.2 - yy.ravel(),
                )
                solid = np.maximum(solid, -notch)
            image = vtk.vtkImageData()
            image.SetDimensions(len(xs), len(ys), len(zs))
            image.SetSpacing(step, step, step)
            # Apart by more than a peg's length, or the pair would overlap.
            image.SetOrigin(
                float(xs[0]) + side * (0.5 * PEG_LENGTH_MM + 2.0), float(ys[0]) + index * (width + 4.0), float(zs[0])
            )
            image.GetPointData().SetScalars(numpy_to_vtk(solid.astype(np.float32), deep=True))
            contour = vtk.vtkFlyingEdges3D()
            contour.SetInputData(image)
            contour.SetValue(0, 0.0)
            contour.ComputeNormalsOff()
            contour.Update()
            piece = vtk.vtkPolyData()
            piece.DeepCopy(contour.GetOutput())
            meshes.AddInputData(piece)
    meshes.Update()
    return _tidy(meshes.GetOutput(), "The fit test")


def _surface_change(pieces, surface, avoid) -> dict:
    """How far the pieces' surface lies from the jaw's own, mm, away from the joints."""
    distance = vtk.vtkImplicitPolyDataDistance()
    distance.SetInput(surface)
    found = []
    for part, _, _ in pieces:
        points = vtk_to_numpy(part.GetPoints().GetData()).astype(float)
        sample = points[:: max(len(points) // 3000, 1)]
        if len(avoid):
            away = np.min(np.linalg.norm(sample[:, None, :] - avoid[None, :, :], axis=2), axis=1) > 15.0
            sample = sample[away]
        found.extend(abs(distance.EvaluateFunction(tuple(p))) for p in sample)
    if not found:
        return {}
    values = np.array(found)
    return {
        "median_mm": round(float(np.median(values)), 3),
        "p95_mm": round(float(np.percentile(values, 95.0)), 3),
        "max_mm": round(float(values.max()), 3),
    }


def _instructions(count: int, has_cut: bool, supports: bool, side_whole: bool) -> str:
    lines = ["Print at 100% scale, in millimetres, keeping each part's exported position and orientation."]
    if has_cut:
        lines.append(
            "The part holding the resection is laid with the outer face at the resection upward, so that "
            "face is finished as a top surface and no support touches it. Do not re-orient it, and print it "
            "at your machine's finest layer height. The other parts can be printed coarser."
        )
        if not side_whole:
            lines.append(
                "This printer cannot hold the whole working side in one part. The resection and the bone "
                "either side of it that the plate is fixed to are kept whole, and the cut lies beyond them."
            )
    if supports:
        lines.append(
            "Import each piece-NN-stems.stl with its part as parts of one object, keeping their relative "
            "positions; do not drop either to the bed separately. The stems have flared feet and narrow "
            "tips that snap off, and none touches the outer face of the working section. If your slicer "
            "asks for more supports, paint them off that face and keep them to the underside."
        )
    if count > 1:
        lines.append(
            f"The {count} parts are numbered along the jaw. Parts meet on a flat face: one has two pegs, the "
            "other two holes, of different sizes so they go together only one way. Push them straight "
            "together until the faces meet. A drop of glue once they fit is fine; never force them."
        )
        lines.append(
            "Print fit-test.stl first. It has this kit's pegs and holes at clearances of "
            + ", ".join(f"{c:.1f}" for c in FIT_TEST_CLEARANCES_MM)
            + " mm, one pair per row, with one, two and three notches on the back edge. Use the tightest "
            "that goes together by hand as the Peg clearance when you export the kit."
        )
    lines.append("assembly.json records where each part belongs in patient space.")
    return "\n".join(lines) + "\n"


def export_print_kit(folder, surface, build_mm, *, working_side="auto", cut_points=(), midline_x=None,
                     printer_name="Custom", supports=True, clearance_mm=0.2, frames=None):
    """Write a new, self-contained kit in ``folder`` and return its files.

    ``cut_points`` bound the resection, ``frames`` is the arch curve. No cut comes
    into the working side, or within ``PROTECT_MM`` of the resection; on a printer
    too small for the whole side, none within ``RESECTION_FIXATION_MM`` of it. An
    earlier kit in the folder is never overwritten.
    """
    import SimpleITK as sitk

    bed = np.asarray(build_mm, dtype=float)
    working_side = str(working_side).lower()
    if bed.shape != (3,) or not np.isfinite(bed).all() or np.min(bed) <= 10:
        raise ValueError("Enter three usable printer build dimensions greater than 10 mm.")
    if working_side not in ("auto", "left", "right"):
        raise ValueError("Choose Auto, Left or Right working side.")
    if not np.isfinite(clearance_mm) or not 0.05 <= clearance_mm <= 0.5:
        raise ValueError("Choose a peg clearance between 0.05 and 0.5 mm.")
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
    side, guards = _guard_span(route, working_side, cut_arcs, mid)

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

    grid = (solid, origin, step)
    guard = guards[0] if guards else None
    try:
        seams = _plan_seams(route, sample, sample_s, grid, solid_points, solid_s, bed, guard, supports, clearance_mm, centre)
    except ValueError:
        if len(guards) < 2:
            raise
        guard = guards[1]
        seams = _plan_seams(route, sample, sample_s, grid, solid_points, solid_s, bed, guard, supports, clearance_mm, centre)
    side_whole = guard is None or guard == guards[0]
    # A peg breaks more easily than a hole: on the part that is cheaper to reprint.
    if guard is not None:
        for seam in seams:
            seam.pegs_on_lower = seam.s < 0.5 * (guard[0] + guard[1])

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
        # The box of this part, wide enough for its pegs and its neighbour's.
        occupied = (solid_s >= lo_s - JOINT_REACH_MM) & (solid_s <= hi_s + JOINT_REACH_MM)
        pad = 6
        box = tuple(
            slice(max(int(indices[occupied].min()) - pad, 0), min(int(indices[occupied].max()) + pad + 1, solid.shape[axis]))
            for axis, indices in enumerate(solid_index)
        )
        mesh = _piece_mesh(smooth, along, origin, step, lower, upper, box)
        part = _snapped_to_scan(_tidy(mesh, f"Part {i + 1}"), surface, [t for t in (lower, upper) if t is not None])
        pieces.append((part, lo_s, hi_s))

    joints = np.array([seam.centre for seam in seams]).reshape(-1, 3)
    avoid = np.vstack([cut, joints])
    working = None
    if guard is not None:
        working = (guard[0], guard[1])

    report = {
        "surface_change_from_the_scan": _surface_change(pieces, surface, joints),
        "printer": printer_name,
        "build_mm": bed.tolist(),
        "bed_margin_mm": BED_MARGIN,
        "working_side": side if side in ("left", "right") else "unspecified",
        "kept_whole_arc_mm": list(guard) if guard else None,
        "whole_working_side_in_one_part": bool(side_whole),
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
            # A peg is printed better standing or lying than hanging down.
            ups = _ups(route.outward(at, centre))
            peg_ways = []
            if i >= 2 and not seams[i - 2].pegs_on_lower:
                peg_ways.append(-seams[i - 2].along)
            if i - 1 < len(seams) and seams[i - 1].pegs_on_lower:
                peg_ways.append(seams[i - 1].along)
            rotation = _pose_for(points, bed, [u for u in ups if all(u @ w > -0.5 for w in peg_ways)], supports)
            if rotation is None:
                rotation = _pose_for(points, bed, ups, supports)
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
                stems, count = _stems(printed, rotation, shift, avoid, route, working, centre)
                bounds = np.asarray(stems.GetBounds()).reshape(3, 2)
                if np.any(bounds[:, 0] < -1e-4) or np.any(bounds[:, 1] > bed + 1e-4):
                    raise ValueError("Stems would leave the bed. Switch stems off or choose a larger printer.")
                stem_name = f"piece-{i:02d}-stems.stl"
                write_surface_stl(staged / stem_name, stems)
                spec.update(stems_file=stem_name, breakaway_stems=count)
            report["parts"].append(spec)
        if seams:
            write_surface_stl(staged / "fit-test.stl", _fit_test(seams[0]))
        (staged / "assembly.json").write_text(json.dumps(report, indent=2) + "\n")
        (staged / "PRINT-AND-ASSEMBLE.txt").write_text(_instructions(len(pieces), bool(len(cut)), supports, side_whole))
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
