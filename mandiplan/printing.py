"""The mandible as a set of printable parts, joined by plug and socket, ready to print.

All inputs are patient millimetres. The jaw is split only if the chosen printer
needs it, along the arch and not along a straight axis, so each part is a run of
jaw that lies well on a bed. The stretch that matters stays whole in one part
and no cut comes near it: the working side and the resection on it, or, on a
printer too small for a whole side, the resection and the bone either side of it
that the plate is fixed to.

A cut is flat and two parts join face to face. One part runs on past its face as a
long plug, the jaw's own section drawn in by a wall of bone; the other has the socket
that plug fills. That is far more bone across the joint than any peg could be, and
pressing on the assembled jaw bends the plug as a whole. It has only the one way in,
and cannot turn. The plug goes on the part further from the working section, so a
broken one costs a small reprint.

An isosurface of real bone is hollow: thin cortex round a darker marrow, so
there is nothing for a plug or a socket to stand in. The parts are therefore built
from the jaw as a solid, filled on a grid and contoured again, and every vertex of
the result is then moved back onto the scan's own surface. The outside of the part
is the scan's surface, to within the grid's facets, and its inside is solid.

Each part is laid with the side to keep best facing up: on the resection's part,
the outer face at the resection, which then never touches a support and is
finished as a top surface. The rods and raft that hold it are in the file with it
(``render/supports.py``), ready for the printer. They keep off the outer face of the
working section altogether, and clear of the resection and of the joints' mating
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
from .render import supports as rods
from .render.bending_guide_mesh import thinned

#: Build volumes, mm, as published for each machine. "Custom" is always there
#: too; check your own machine's usable volume.
PRINTERS = {
    "Asiga MAX": (119.0, 67.0, 75.0),
    "Asiga MAX X35": (67.2, 37.8, 75.0),
    "Asiga PRO 4K45": (122.0, 69.0, 200.0),
    "Asiga PRO 4K65": (176.5, 99.0, 200.0),
    "Asiga PRO 4K80": (217.0, 122.0, 200.0),
    "Asiga ULTRA 32": (121.0, 68.0, 145.0),
    "Asiga ULTRA 50": (192.0, 108.0, 145.0),
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
#: Rods keep this far from the resection, mm, and this much clear of a joint's faces.
ROD_CLEAR_OF_CUT_MM = 8.0
ROD_CLEAR_OF_JOINT_MM = 1.5
#: A surface facing this much along the jaw's outward direction is its outer face.
OUTER_FACE_COS = 0.2
#: Pieces shorter than this along the jaw are not worth having, mm.
MIN_PIECE_MM = 14.0
MAX_PIECES = 10
#: Loose bits under this are dropped from a piece, mm3.
MIN_FRAGMENT_MM3 = 5.0
#: The joint: the jaw's own section carried on from one part as a long plug, into a
#: socket in the next. How long a plug is tried, longest first, and how much bone is
#: left round the socket, thickest first, mm.
PLUG_LENGTHS_MM = (16.0, 12.0, 9.0)
PLUG_WALLS_MM = (2.0, 1.5)
#: A longer plug is taken if it keeps this share of the strength of the strongest, and
#: a thicker wall if it keeps this share of the strength of a thinner one.
PLUG_STRENGTH_SHARE = 0.85
THICK_WALL_SHARE = 0.85
#: How far a joint's axis may lean from the chord of the jaw, to find the strongest
#: straight run through the bone: the tangents tried, each way.
AXIS_LEAN_TANGENTS = tuple(float(np.tan(np.radians(a))) for a in (0.0, -7.0, 7.0, -14.0, 14.0))
#: How much the plug's sides lean in, mm per mm: enough to find its way in and no more
#: than to stay long and full.
PLUG_TAPER = float(np.tan(np.radians(2.0)))
#: The chamfer round its tip, how much deeper the socket is than the plug is long, the
#: chamfer round the socket's mouth, and the bone left past its floor, mm.
PLUG_TIP_CHAMFER_MM = 0.6
SOCKET_EXTRA_DEPTH_MM = 0.6
SOCKET_MOUTH_CHAMFER_MM = 0.4
SOCKET_FLOOR_MM = 2.0
#: The least a plug may be, at its base, across its narrowest and in area, mm, mm2.
MIN_PLUG_HALF_WIDTH_MM = 1.5
MIN_PLUG_AREA_MM2 = 20.0
#: A place to split is preferred over the first that fits if it is this much stronger.
#: How finely the jaw is searched for one, mm.
PREFER_STRONGER_SHARE = 1.3
STRONGER_SEARCH_STEP_MM = 4.0
#: How far a plug reaches past the end of a part, along the jaw, at most, mm.
JOINT_REACH_MM = max(PLUG_LENGTHS_MM) + 1.0
#: The fit test print: clearances tried, how long its plugs stand, and how thick its
#: bases and the floors of its sockets are, mm.
FIT_TEST_CLEARANCES_MM = (0.1, 0.15, 0.2)
FIT_TEST_PLUG_MM = 6.0
FIT_TEST_BASE_MM = 2.5
FIT_TEST_GRID_MM = 0.15
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

    def at(self, s: float) -> np.ndarray:
        return np.array([np.interp(s, self.s, self.points[:, c]) for c in range(3)])

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
    """A flat cut across the jaw at arc position ``s``, one part running on as a plug.

    The plug has the jaw's own section at the cut, drawn in by the wall of bone left
    round it, so it is as big as this jaw allows there. The parts meet on the flat
    face, the plug standing in its socket with ``clearance`` all round: the faces take
    what presses them together, the plug what bends the joint, and its shape stops it
    turning or going in any way but one.
    """

    s: float
    centre: np.ndarray
    along: np.ndarray  # along the jaw, towards higher s: square to the face
    height: np.ndarray  # the long way of the jaw's section
    across: np.ndarray  # the short way
    clearance: float
    #: How far inside the socket's mouth, in the plane of the cut, on a raster starting
    #: at ``(v0, w0)`` along ``height`` and ``across``; negative outside it, mm.
    mouth: np.ndarray
    v0: float
    w0: float
    raster: float
    length: float  # how long the plug stands
    wall: float  # the least bone left round the socket
    plug_on_lower: bool = True  # the part at lower s carries the plug
    area_mm2: float = 0.0  # of the plug's section at its base
    #: The joint's strength: of whichever is weaker, plug or wall, for the worse way of
    #: bending it across the jaw.
    section_modulus_mm3: float = 0.0
    whole_section_modulus_mm3: float = 0.0  # of the jaw there, whole, for the same

    def inside_mouth(self, v: np.ndarray, w: np.ndarray) -> np.ndarray:
        """``mouth`` at these places, read between the raster's points."""
        at_v, at_w = (v - self.v0) / self.raster, (w - self.w0) / self.raster
        rows, cols = self.mouth.shape
        i = np.clip(np.floor(at_v).astype(int), 0, rows - 2)
        j = np.clip(np.floor(at_w).astype(int), 0, cols - 2)
        a, b = at_v - i, at_w - j
        out = (
            (1.0 - a) * (1.0 - b) * self.mouth[i, j]
            + a * (1.0 - b) * self.mouth[i + 1, j]
            + (1.0 - a) * b * self.mouth[i, j + 1]
            + a * b * self.mouth[i + 1, j + 1]
        )
        off = (at_v < 0.0) | (at_v > rows - 1) | (at_w < 0.0) | (at_w > cols - 1)
        out[off] = float(self.mouth.min())
        return out

    def _fields(self, points: np.ndarray):
        """``(t, plug, socket)``: distance past the face towards the socket's part,
        and the two solids, each negative inside."""
        rel = points - self.centre
        n = rel @ self.along
        t = n if self.plug_on_lower else -n
        plug = np.full(len(t), np.inf, dtype=float)
        socket = np.full(len(t), np.inf, dtype=float)
        deep = self.length + SOCKET_EXTRA_DEPTH_MM
        near = (t > -1.5) & (t < deep + 1.5)
        if near.any():
            here = t[near]
            room = self.inside_mouth(rel[near] @ self.height, rel[near] @ self.across)
            lean = PLUG_TAPER * np.maximum(here, 0.0)
            tip = np.maximum(here - (self.length - PLUG_TIP_CHAMFER_MM), 0.0)
            plug[near] = np.maximum.reduce((self.clearance + lean + tip - room, here - self.length, -here - 1.0))
            opening = np.maximum(SOCKET_MOUTH_CHAMFER_MM - here, 0.0)
            socket[near] = np.maximum.reduce((lean - opening - room, here - deep, -here - 1.0))
        return t, plug, socket

    def lower_field(self, points: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Negative in the part before the cut."""
        t, plug, socket = self._fields(points)
        field = np.minimum(t, plug) if self.plug_on_lower else np.maximum(-t, -socket)
        far = np.linalg.norm(points - self.centre, axis=1) > SEAM_NEAR_MM
        field[far] = np.where(s[far] < self.s, -6.0, 6.0)
        return field.astype(np.float32)

    def upper_field(self, points: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Negative in the part after the cut."""
        t, plug, socket = self._fields(points)
        field = np.maximum(-t, -socket) if self.plug_on_lower else np.minimum(t, plug)
        far = np.linalg.norm(points - self.centre, axis=1) > SEAM_NEAR_MM
        field[far] = np.where(s[far] > self.s, -6.0, 6.0)
        return field.astype(np.float32)

    def in_joint(self, points: np.ndarray, margin: float) -> np.ndarray:
        """Which points are on the joint's mating surfaces, to within ``margin``."""
        rel = points - self.centre
        n = rel @ self.along
        t = n if self.plug_on_lower else -n
        near = np.linalg.norm(rel, axis=1) < SEAM_NEAR_MM
        face = near & (np.abs(t) < margin)
        through = (t > -margin) & (t < self.length + SOCKET_EXTRA_DEPTH_MM + margin)
        room = self.inside_mouth(rel @ self.height, rel @ self.across)
        return face | (near & through & (room > -min(margin, 0.5 * self.wall)))

    def report(self) -> dict:
        return {
            "arc_position_mm": float(self.s),
            "centre_mm": self.centre.tolist(),
            "cut_normal": self.along.tolist(),
            "plug_length_mm": float(self.length),
            "plug_section_mm2": round(float(self.area_mm2), 1),
            "joint_section_modulus_mm3": round(float(self.section_modulus_mm3), 1),
            "whole_jaw_section_modulus_mm3": round(float(self.whole_section_modulus_mm3), 1),
            "wall_round_socket_mm": float(self.wall),
            "clearance_mm": float(self.clearance),
            "plug_on": "lower-numbered part" if self.plug_on_lower else "higher-numbered part",
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


def _section_moduli(vv: np.ndarray, ww: np.ndarray, region: np.ndarray, raster: float) -> np.ndarray:
    """Section moduli of a region of the plane, for bending each way across it, mm3."""
    area = raster * raster
    out = []
    for values in (vv[region], ww[region]):
        off = values - values.mean()
        inertia = float((off * off).sum()) * area
        out.append(inertia / max(float(np.abs(off).max()), 1e-6))
    return np.array(out)


def _bone_right_through(grid, centre, height, across, axis, sign, depth, half, raster, t_step):
    """Where bone runs unbroken from just behind the face to ``depth`` past it:
    ``(vv, ww, mask)`` on a raster of the plane."""
    solid, origin, step = grid
    vs = np.arange(-half, half + raster, raster)
    vv, ww = np.meshgrid(vs, vs, indexing="ij")
    through = np.ones(vv.shape, dtype=bool)
    plane = centre + vv[..., None] * height + ww[..., None] * across
    for t in np.arange(-1.5, depth + 0.1, t_step):
        world = plane + (sign * t) * axis
        index = np.round((world - origin) / step).astype(int)[..., ::-1]
        inside = np.all((index >= 0) & (index < np.array(solid.shape)), axis=-1)
        here = np.zeros(vv.shape, dtype=bool)
        here[inside] = solid[index[inside][:, 0], index[inside][:, 1], index[inside][:, 2]]
        through &= here
        if not through.any():
            break
    return vv, ww, through


def _distance_inside(mask: np.ndarray, raster: float) -> np.ndarray:
    """Signed distance to the edge of ``mask``, positive inside, mm."""
    import SimpleITK as sitk

    return -sitk.GetArrayFromImage(
        sitk.SignedMaurerDistanceMap(
            sitk.GetImageFromArray(mask.astype(np.uint8)), insideIsPositive=False,
            squaredDistance=False, useImageSpacing=False,
        )
    ) * raster


def _largest_piece(mask: np.ndarray) -> np.ndarray:
    import SimpleITK as sitk

    labels = sitk.GetArrayFromImage(
        sitk.RelabelComponent(sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8))))
    )
    return labels == 1


def _section_axes(section: np.ndarray, axis: np.ndarray):
    """The section's centre and long and short ways, seen along ``axis``."""
    centre = section.mean(axis=0)
    flat = (section - centre) - np.outer((section - centre) @ axis, axis)
    _, _, axes = np.linalg.svd(flat[:: max(len(flat) // 4000, 1)], full_matrices=False)
    height = axes[0] - (axes[0] @ axis) * axis
    height = height / np.linalg.norm(height)
    across = np.cross(axis, height)
    extent = max(float(np.ptp(flat @ height)), float(np.ptp(flat @ across)))
    return centre, height, across, extent


def _make_seam(route, grid, solid_points, solid_s, s0: float, clearance: float, plug_on_lower: bool = True) -> _Seam | None:
    """A cut at ``s0`` with its plug and socket, or None if the bone there cannot take them.

    ``grid`` is ``(solid, origin, step)``. A jaw curves and twists, and a plug is
    straight. So its axis starts along the chord of the jaw through the plug and leans
    from there to wherever the straight run through the bone is strongest. Bone must
    run right through the socket's whole depth, and a wall of it is left round the
    socket. Of the plugs and walls tried, the joint's strength is that of whichever is
    weaker, the plug or the wall round it, for bending each way across the jaw.
    """
    import SimpleITK as sitk

    near = np.abs(solid_s - s0) < 1.5
    if near.sum() < 30:
        return None
    raster = 0.3
    sign = 1.0 if plug_on_lower else -1.0
    found = []
    for length in PLUG_LENGTHS_MM:
        chord = route.at(s0 + sign * length) - route.at(s0)
        if np.linalg.norm(chord) < 0.5 * length:
            start_axis = route.tangent(s0)
        else:
            start_axis = sign * chord / np.linalg.norm(chord)
        section = solid_points[near]
        keep = np.abs((section - section.mean(axis=0)) @ start_axis) < 2.0
        if keep.sum() < 30:
            continue
        section = section[keep]
        centre, height, across, extent = _section_axes(section, start_axis)
        half = 0.5 * extent + 6.0
        depth = length + SOCKET_EXTRA_DEPTH_MM + SOCKET_FLOOR_MM

        # A coarse look at each lean of the axis for the strongest straight run.
        best_axis, best_score = None, 0.0
        for tilt_h in AXIS_LEAN_TANGENTS:
            for tilt_a in AXIS_LEAN_TANGENTS:
                axis = start_axis + tilt_h * height + tilt_a * across
                axis = axis / np.linalg.norm(axis)
                # Square to the leaning axis, or a lean would squash the section.
                up = height - (height @ axis) * axis
                up = up / np.linalg.norm(up)
                vv, ww, through = _bone_right_through(grid, centre, up, np.cross(axis, up), axis, sign, depth, half, 0.6, 2.0)
                if not through.any():
                    continue
                core = _distance_inside(through, 0.6) >= min(PLUG_WALLS_MM)
                if core.sum() < 4:
                    continue
                score = float(_section_moduli(vv, ww, _largest_piece(core), 0.6).min())
                # Other things equal, square to the jaw.
                score /= 1.0 + 0.5 * (tilt_h * tilt_h + tilt_a * tilt_a)
                if score > best_score:
                    best_axis, best_score = axis, score
        if best_axis is None:
            continue
        keep = np.abs((section - section.mean(axis=0)) @ best_axis) < 2.0
        if keep.sum() < 30:
            continue
        centre, height, across, extent = _section_axes(section[keep], best_axis)
        half = 0.5 * extent + 6.0
        vv, ww, through = _bone_right_through(grid, centre, height, across, best_axis, sign, depth, half, raster, 1.0)
        if not through.any():
            continue
        room = _distance_inside(through, raster)
        whole = _section_moduli(vv, ww, _largest_piece(through), raster)
        options = []
        for wall in PLUG_WALLS_MM:
            core = room >= wall
            if not core.any():
                continue
            # The largest piece of it: beside it, the plane may cross another process.
            core = _largest_piece(core)
            # Off the raster's stairs. Both surfaces are levels of this one field, so
            # the gap between them is the clearance.
            mouth = sitk.GetArrayFromImage(
                sitk.SmoothingRecursiveGaussian(
                    sitk.GetImageFromArray(_distance_inside(core, raster).astype(np.float32)), 1.0
                )
            ).astype(np.float32)
            plug = mouth >= clearance
            if not plug.any():
                continue
            area = float(plug.sum()) * raster * raster
            spare = float(mouth.max()) - clearance
            if area < MIN_PLUG_AREA_MM2 or spare < MIN_PLUG_HALF_WIDTH_MM:
                continue
            if spare - PLUG_TAPER * length - PLUG_TIP_CHAMFER_MM < 0.4:
                continue  # it would run out to nothing before its tip
            ring = through & (mouth < 0.0) & (mouth >= -(wall + 1.0))
            moduli = _section_moduli(vv, ww, plug, raster)
            if ring.any():
                moduli = np.minimum(moduli, _section_moduli(vv, ww, ring, raster))
            options.append(
                _Seam(
                    float(s0), centre, best_axis, height, across, float(clearance), mouth,
                    float(vv[0, 0]), float(ww[0, 0]), raster, float(length), float(wall), bool(plug_on_lower),
                    area, float(moduli.min()), float(whole.min()),
                )
            )
        if options:
            # A thicker wall, unless it costs real strength.
            top = max(option.section_modulus_mm3 for option in options)
            found.append(next(o for o in options if o.section_modulus_mm3 >= THICK_WALL_SHARE * top))
    if not found:
        return None
    strongest = max(seam.section_modulus_mm3 for seam in found)
    return next(seam for seam in found if seam.section_modulus_mm3 >= PLUG_STRENGTH_SHARE * strongest)


def _pose_for(points: np.ndarray, bed: np.ndarray, ups, lifted: bool):
    """A rotation that fits ``points`` on ``bed``: first of ``ups`` that does.

    Rows of the rotation are the print x, y, z axes in patient space.
    """
    limit = bed - np.array([2.0 * BED_MARGIN, 2.0 * BED_MARGIN, rods.lift_mm() if lifted else 0.0])
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
        # A part also holds the plug reaching past its end. The working section's
        # part has sockets, not plugs, and ``reach`` 0.
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

    middle = 0.5 * (guard[0] + guard[1])

    def seam_at(s):
        # The plug points towards the working section: a plug breaks more easily than
        # a socket, so it goes on the part that is cheaper to print again.
        return _make_seam(route, grid, solid_points, solid_s, float(s), clearance, float(s) < middle)

    def strongest_from(seam, sign, limit, rest_to):
        """The strongest place to split on the way from ``seam`` towards ``limit``,
        as long as what is left beyond it, out to ``rest_to``, still fits in one part.
        A jaw is far stronger in its body than high on its thin ramus."""
        best = seam
        at = seam.s + sign * STRONGER_SEARCH_STEP_MM
        while (limit - at) * sign >= 0.0:
            if fits(min(at, rest_to), max(at, rest_to)):
                other = seam_at(at)
                if other is not None and other.section_modulus_mm3 > PREFER_STRONGER_SHARE * best.section_modulus_mm3:
                    best = other
            at += sign * STRONGER_SEARCH_STEP_MM
        return best

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
                    "No place on the jaw can take a joint for a part this printer can hold. "
                    "Choose a larger printer."
                )
            seam = strongest_from(seam, -sign, here + sign * MIN_PIECE_MM, end)
            found.append(seam)
            here = seam.s
            if len(found) >= MAX_PIECES:
                raise ValueError("This printer would need too many pieces. Choose a larger printer.")
        return found

    def first_seam(s, sign, limit, rest_to):
        # From the far edge of the guarded piece, back towards the guard, until
        # the bone can hold a joint.
        while (limit - s) * sign >= 0:
            seam = seam_at(s)
            if seam is not None:
                return strongest_from(seam, sign, limit, rest_to)
            s += sign * 2.0
        raise ValueError("No joint fits between the working section and the rest. Choose a larger printer.")

    # The guarded part grows as far as it can each way, to leave fewer seams.
    a, b = guard
    while a - 2.0 > lo and fits(a - 2.0, b, reach=0.0):
        a -= 2.0
    while b + 2.0 < hi and fits(a, b + 2.0, reach=0.0):
        b += 2.0
    plan = []
    if a > lo + MIN_PIECE_MM:
        seam = first_seam(a, 1.0, guard[0], lo)
        plan = out_to(seam.s, lo, -1.0, plan + [seam])
    if b < hi - MIN_PIECE_MM:
        seam = first_seam(b, -1.0, guard[1], hi)
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


def _rod_rules(rotation, shift, route, centre, working, cut, seams):
    """``keep_off`` for the rods of a part in its print pose.

    No rod touches the outer face of the jaw along the arc span ``working``, which is
    the face a plate lies on and the site is judged by; none comes near the resection;
    none touches the flat faces, plugs or sockets of a joint, which have to fit.
    """

    def forbidden(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        world = (points - shift) @ rotation
        facing = normals @ rotation
        out = np.zeros(len(world), dtype=bool)
        if len(cut):
            out |= np.min(np.linalg.norm(world[:, None, :] - cut[None, :, :], axis=2), axis=1) < ROD_CLEAR_OF_CUT_MM
        if working is not None:
            arcs = route.along(world)
            for k in np.flatnonzero((arcs >= working[0]) & (arcs <= working[1])):
                if facing[k] @ route.outward(float(arcs[k]), centre) > OUTER_FACE_COS:
                    out[k] = True
        for seam in seams:
            out |= seam.in_joint(world, ROD_CLEAR_OF_JOINT_MM)
        return out

    return forbidden


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
    the scan's surface goes back onto it along its normal. Cut faces, plugs and
    sockets are built to fit each other, and stay as they are built.
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
            move &= ~seam.in_joint(points, SNAP_CUT_CLEAR_MM)
        moved = points.copy()
        moved[move] -= away[move][:, None] * facing[move]
        result = vtk.vtkPolyData()
        result.SetPolys(part.GetPolys())
        new_points = vtk.vtkPoints()
        new_points.SetData(numpy_to_vtk(moved, deep=True))
        result.SetPoints(new_points)
        part = result
    return part


def _contoured(field: np.ndarray, origin, step: float) -> vtk.vtkPolyData:
    """The surface of a solid given by its field on a grid, (z, y, x), negative inside."""
    image = vtk.vtkImageData()
    image.SetDimensions(field.shape[2], field.shape[1], field.shape[0])
    image.SetSpacing(step, step, step)
    image.SetOrigin(*(float(c) for c in origin))
    image.GetPointData().SetScalars(numpy_to_vtk(field.ravel().astype(np.float32), deep=True))
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, 0.0)
    contour.ComputeNormalsOff()
    contour.Update()
    piece = vtk.vtkPolyData()
    piece.DeepCopy(contour.GetOutput())
    return piece


def _fit_test(seam: _Seam, bed, clearances=FIT_TEST_CLEARANCES_MM):
    """This kit's real joint, short, at each clearance, standing on the bed: print it first.

    A pair for each clearance: a plug on a base, and a socket in a block, both of the
    jaw's own section at a joint. Holes in a small tag count which is which. Returns
    the mesh and the clearances it has: fewer, down to the middle one alone, if that
    is all the bed holds, and ``(None, ())`` if it cannot hold even one pair.
    """
    step = FIT_TEST_GRID_MM
    wall = seam.wall
    rows, cols = seam.mouth.shape
    vs = seam.v0 + np.arange(rows) * seam.raster
    ws = seam.w0 + np.arange(cols) * seam.raster
    outline = seam.mouth >= -wall
    v_on = vs[outline.any(axis=1)]
    w_on = ws[outline.any(axis=0)]
    v_low, v_high = float(v_on.min()), float(v_on.max())
    # The tag, at the end of the section along v.
    tag_w = float(ws[outline[int(np.flatnonzero(outline.any(axis=1))[-1])].nonzero()[0]].mean())
    w_low, w_high = min(float(w_on.min()), tag_w - 2.5), max(float(w_on.max()), tag_w + 2.5)
    tag_reach = 8.0
    v_top = v_high + tag_reach
    length = FIT_TEST_PLUG_MM
    deep = length + SOCKET_EXTRA_DEPTH_MM

    def grid(height):
        # Off the round numbers: no flat face may fall exactly on a grid plane.
        xs = np.arange(v_low - 1.0, v_top + 1.0, step) + 0.0137
        ys = np.arange(w_low - 1.0, w_high + 1.0, step) + 0.0137
        zs = np.arange(-0.5, height + 0.5, step) + 0.0137
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        return xx, yy, zz

    def tag(xx, yy, zz, count):
        centre_x = v_high + 0.5 * tag_reach - 0.5
        body = np.maximum.reduce(
            (np.abs(xx - centre_x) - (0.5 * tag_reach + 0.5), np.abs(yy - tag_w) - 2.5, -zz, zz - 1.5)
        )
        for k in range(count):
            hole = np.hypot(xx - (v_high + 1.5 + 2.0 * k), yy - tag_w) - 0.6
            body = np.maximum(body, -hole)
        return body

    def room_at(xx, yy):
        return seam.inside_mouth(xx.ravel(), yy.ravel()).reshape(xx.shape)

    def plug_piece(clearance, index):
        height = FIT_TEST_BASE_MM + length
        xx, yy, zz = grid(height)
        room = room_at(xx, yy)
        t = zz - FIT_TEST_BASE_MM
        base = np.maximum.reduce((-(room + wall), -zz, zz - FIT_TEST_BASE_MM))
        lean = PLUG_TAPER * np.maximum(t, 0.0)
        tip = np.maximum(t - (length - PLUG_TIP_CHAMFER_MM), 0.0)
        plug = np.maximum.reduce((clearance + lean + tip - room, t - length, -t - 1.0))
        return np.minimum.reduce((base, plug, tag(xx, yy, zz, index + 1))), xx, yy, zz

    def socket_piece(clearance, index):
        height = FIT_TEST_BASE_MM + deep
        xx, yy, zz = grid(height)
        room = room_at(xx, yy)
        t = height - zz
        block = np.maximum.reduce((-(room + wall), -zz, zz - height))
        lean = PLUG_TAPER * np.maximum(t, 0.0)
        opening = np.maximum(SOCKET_MOUTH_CHAMFER_MM - t, 0.0)
        socket = np.maximum.reduce((lean - opening - room, t - deep, -t - 1.0))
        solid = np.maximum(block, -socket)
        return np.minimum(solid, tag(xx, yy, zz, index + 1)), xx, yy, zz

    gap = 3.0
    size = np.array([v_top - v_low + gap, w_high - w_low + gap])
    room_xy = np.asarray(bed[:2], dtype=float) - 2.0 * BED_MARGIN + gap
    layouts = []
    for chosen in (tuple(clearances), (clearances[len(clearances) // 2],)):
        pieces = 2 * len(chosen)
        for columns in range(pieces, 0, -1):
            lines = int(np.ceil(pieces / columns))
            for swap in (False, True):
                cell = size[::-1] if swap else size
                if columns * cell[0] <= room_xy[0] and lines * cell[1] <= room_xy[1]:
                    layouts.append((chosen, columns, swap, cell))
        if layouts:
            break
    if not layouts:
        return None, ()
    chosen, columns, swap, cell = min(layouts, key=lambda item: abs(item[3][0] * item[1] - item[3][1] * np.ceil(2 * len(item[0]) / item[1])))
    meshes = vtk.vtkAppendPolyData()
    place = 0
    for clearance in chosen:
        index = list(clearances).index(clearance)
        for maker in (plug_piece, socket_piece):
            field, xx, yy, zz = maker(clearance, index)
            piece = _contoured(field, (xx[0, 0, 0], yy[0, 0, 0], zz[0, 0, 0]), step)
            points = vtk_to_numpy(piece.GetPoints().GetData()).astype(float)
            if swap:
                points = points[:, [1, 0, 2]]
            at = np.array([(place % columns) * cell[0] + BED_MARGIN, (place // columns) * cell[1] + BED_MARGIN, 0.0])
            points = points - points.min(axis=0) + at
            moved = vtk.vtkPoints()
            moved.SetData(numpy_to_vtk(points, deep=True))
            piece.SetPoints(moved)
            if swap:
                # A swap of two axes turns the solid inside out.
                turned = vtk.vtkReverseSense()
                turned.SetInputData(piece)
                turned.ReverseCellsOn()
                turned.ReverseNormalsOff()
                turned.Update()
                piece = vtk.vtkPolyData()
                piece.DeepCopy(turned.GetOutput())
            meshes.AddInputData(piece)
            place += 1
    meshes.Update()
    return thinned(_tidy(meshes.GetOutput(), "The fit test")), chosen


def _surface_change(pieces, surface, seams) -> dict:
    """How far the pieces' surface lies from the jaw's own, mm, away from the joints."""
    distance = vtk.vtkImplicitPolyDataDistance()
    distance.SetInput(surface)
    found = []
    for part, *_ in pieces:
        points = vtk_to_numpy(part.GetPoints().GetData()).astype(float)
        sample = points[:: max(len(points) // 3000, 1)]
        for seam in seams:
            sample = sample[~seam.in_joint(sample, SNAP_CUT_CLEAR_MM)]
        found.extend(abs(distance.EvaluateFunction(tuple(p))) for p in sample)
    if not found:
        return {}
    values = np.array(found)
    return {
        "median_mm": round(float(np.median(values)), 3),
        "p95_mm": round(float(np.percentile(values, 95.0)), 3),
        "max_mm": round(float(values.max()), 3),
    }


def _instructions(count: int, has_cut: bool, supports: bool, side_whole: bool, seams, tried=()) -> str:
    lines = ["Print at 100% scale, in millimetres, keeping each part's exported position and orientation."]
    if supports:
        lines.append(
            "Each part-NN-ready-to-print.stl is the part on its rods and raft, all in one file, ready for the "
            "printer's own software: place it on the build plate as it is, and add no more supports. Rods narrow "
            "to a point that snaps off. part-NN.stl is the part alone, in the same position, for those who "
            "prefer their slicer's supports."
        )
    if has_cut:
        lines.append(
            "The part holding the resection is laid with the outer face at the resection upward, so that face is "
            "finished as a top surface and no rod touches it. Do not re-orient it, and print it at your machine's "
            "finest layer height. The other parts can be printed coarser."
        )
        if not side_whole:
            lines.append(
                "This printer cannot hold the whole working side in one part. The resection and the bone "
                "either side of it that the plate is fixed to are kept whole, and the cut lies beyond them."
            )
    if count > 1:
        lines.append(
            f"The {count} parts are numbered along the jaw. Where two meet, one carries the jaw's own section as a "
            "long plug, and the other has the socket it fills, with bone left all round. There is one way in: "
            "push them straight together, tip into mouth, until the flat faces meet. Wash and cure both first, "
            "and take off any rod's stub. A drop of glue on the faces makes the joint permanent; do not force it. "
            "Plugs: " + "; ".join(
                f"{seam.length:.0f} mm long, {seam.area_mm2:.0f} mm2 in section" for seam in seams
            ) + "."
        )
        if tried:
            words = {1: "one hole", 2: "two holes", 3: "three holes"}
            marks = ", ".join(f"{words[list(FIT_TEST_CLEARANCES_MM).index(c) + 1]} for {c:.2f} mm" for c in tried)
            lines.append(
                "Print fit-test.stl first, flat on the plate with no supports. It has this kit's most slender joint, "
                f"short, and the tag on each piece says its clearance: {marks}. Use the tightest pair that goes fully "
                "home by hand as the Joint clearance when you export the kit. A socket lying on its side in a part "
                "prints a little tighter than one standing up: if in doubt take the next one up."
            )
    lines.append("assembly.json records where each part belongs in patient space.")
    return "\n".join(lines) + "\n"


def export_print_kit(folder, surface, build_mm, *, working_side="auto", cut_points=(), midline_x=None,
                     printer_name="Custom", supports=True, clearance_mm=0.15, frames=None):
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
        raise ValueError("Choose a joint clearance between 0.05 and 0.5 mm.")
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
        # The box of this part, wide enough for its plug and its neighbour's.
        occupied = (solid_s >= lo_s - JOINT_REACH_MM) & (solid_s <= hi_s + JOINT_REACH_MM)
        pad = 6
        box = tuple(
            slice(max(int(indices[occupied].min()) - pad, 0), min(int(indices[occupied].max()) + pad + 1, solid.shape[axis]))
            for axis, indices in enumerate(solid_index)
        )
        mesh = _piece_mesh(smooth, along, origin, step, lower, upper, box)
        part = _snapped_to_scan(_tidy(mesh, f"Part {i + 1}"), surface, [t for t in (lower, upper) if t is not None])
        pieces.append((part, lo_s, hi_s, [item for item in (lower, upper) if item is not None]))

    working = None
    if guard is not None:
        working = (guard[0], guard[1])

    report = {
        "surface_change_from_the_scan": _surface_change(pieces, surface, seams),
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
        lift = rods.lift_mm() if supports else 0.0
        for i, (part, lo_s, hi_s, own_seams) in enumerate(pieces, 1):
            points = vtk_to_numpy(part.GetPoints().GetData()).astype(float)
            at = 0.5 * (max(lo_s, float(route.s[0])) + min(hi_s, float(route.s[-1])))
            inside = (cut_arcs >= lo_s) & (cut_arcs <= hi_s)
            holds_cut = bool(inside.any())
            if holds_cut:
                at = float(np.median(cut_arcs[inside]))
            # A plug prints best standing up, with its underside no flatter than the
            # printer builds unaided, and a socket best not opening straight down.
            outward = route.outward(at, centre)
            ups = _ups(outward)
            plugs, sockets = [], []
            if i >= 2:
                (sockets if seams[i - 2].plug_on_lower else plugs).append(-seams[i - 2].along)
            if i - 1 < len(seams):
                (plugs if seams[i - 1].plug_on_lower else sockets).append(seams[i - 1].along)
            if plugs:
                ups = [outward + plugs[0], *ups]
            rotation = _pose_for(
                points, bed,
                [u for u in ups
                 if all(u @ way > 0.3 * np.linalg.norm(u) for way in plugs)
                 and all(u @ way > -0.5 * np.linalg.norm(u) for way in sockets)],
                supports,
            )
            if rotation is None:
                rotation = _pose_for(points, bed, ups, supports)
            room = bed - np.array([2.0 * BED_MARGIN, 2.0 * BED_MARGIN, lift])
            if rotation is None:
                raise ValueError(f"Piece {i} does not fit this printer. Choose a larger printer.")
            oriented = points @ rotation.T
            low, high = oriented.min(axis=0), oriented.max(axis=0)
            if np.any(high - low > room + 0.05):
                raise ValueError(f"Piece {i} does not fit this printer. Choose a larger printer.")
            shift = np.r_[(bed[:2] - (high[:2] - low[:2])) / 2.0, lift] - low
            printed, matrix = _transform(part, rotation, shift)
            name = f"part-{i:02d}.stl"
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
                margin = np.full(2, 0.1)
                held = rods.support_structure(
                    printed,
                    keep_off=_rod_rules(rotation, shift, route, centre, working, cut, own_seams),
                    limits=(margin, bed[:2] - margin),
                )
                ready = rods.merged(printed, held)
                bounds = np.asarray(ready.GetBounds()).reshape(3, 2)
                if np.any(bounds[:, 0] < -1e-4) or np.any(bounds[:, 1] > bed + 1e-4):
                    raise ValueError("The rods would leave the bed. Switch them off or choose a larger printer.")
                ready_name = f"part-{i:02d}-ready-to-print.stl"
                write_surface_stl(staged / ready_name, ready)
                spec.update(
                    ready_to_print_file=ready_name,
                    support_rods=held.rods,
                    rods_standing_on_the_part=held.on_part,
                    downward_surfaces_held_share=round(held.covered, 3),
                    rod_contacts_print_mm=np.round(held.contacts, 2).tolist(),
                )
            report["parts"].append(spec)
        # The joint most in need of the right clearance is the most slender one.
        slenderest = min(seams, key=lambda seam: seam.section_modulus_mm3) if seams else None
        fit_test, tried = _fit_test(slenderest, bed) if seams else (None, ())
        report["fit_test_clearances_mm"] = list(tried)
        if fit_test is not None:
            write_surface_stl(staged / "fit-test.stl", fit_test)
        (staged / "assembly.json").write_text(json.dumps(report, indent=2) + "\n")
        (staged / "PRINT-AND-ASSEMBLE.txt").write_text(
            _instructions(len(pieces), bool(len(cut)), supports, side_whole, seams, tried)
        )
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
