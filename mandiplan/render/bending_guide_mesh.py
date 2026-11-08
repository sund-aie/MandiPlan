"""The bending keys as printable meshes: each one closed solid.

The parts of a key (``geometry/bending_guide.py``) are simple solids that overlap
where they join. Their union is contoured from the minimum of their fields, so what
is printed is one watertight shell and not a heap of overlapping ones.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ..geometry.bending_guide import (
    BendingGuide,
    BendKey,
    gauge_parts,
    key_print_rotation,
    fit_test_key,
    pin_tab_parts,
    tool_parts,
)
from . import supports as rods

#: Contouring grid, for printing and for looking at, mm.
GRID_MM = 0.2
PREVIEW_GRID_MM = 0.4
#: A printed key's flat faces need few triangles. Its surface may move by this much
#: as they are thinned out, mm: well under what any printer can show.
THIN_OUT_ERROR_MM = 0.01
#: A piece this small next to the rest is a contouring sliver.
MAX_SLIVER_SHARE = 0.02
#: Space between keys on the bed, mm.
LAYOUT_GAP_MM = 3.0
BED_MARGIN_MM = 2.0
#: Rods touch a key only on a face turned this nearly straight down: that is its flat
#: side, not a face or a pin the plate has to fit.
ROD_FACE_COS = 0.95


def parts_polydata(parts, step: float) -> vtk.vtkPolyData:
    """The union of ``parts``, contoured at ``step`` mm."""
    placed = []
    for part in parts:
        bound = np.abs(part.frame) @ part.half + 2.0 * step
        placed.append((part, part.centre - bound, part.centre + bound))
    # Off the round numbers: no flat face may fall exactly on a grid plane.
    origin = np.min([p[1] for p in placed], axis=0) - step - 0.137 * step
    size = np.ceil((np.max([p[2] for p in placed], axis=0) - origin) / step).astype(int) + 2
    nx, ny, nz = (int(v) for v in size)
    slab = max(2, 5_000_000 // (nx * ny))
    merged = vtk.vtkAppendPolyData()
    for k in range(0, nz - 1, slab):
        last = min(nz - 1, k + slab)
        field = np.full((last - k + 1, ny, nx), 3.0 * step, np.float32)
        for part, low, high in placed:
            lo = np.maximum(np.floor((low - origin) / step).astype(int), [0, 0, k])
            hi = np.minimum(np.ceil((high - origin) / step).astype(int) + 1, [nx, ny, last + 1])
            if np.any(hi <= lo):
                continue
            x, y, z = (origin[i] + np.arange(lo[i], hi[i]) * step - part.centre[i] for i in range(3))
            x, y, z = x[None, None, :], y[None, :, None], z[:, None, None]
            local = [x * v[0] + y * v[1] + z * v[2] for v in part.frame.T]
            patch = field[lo[2] - k : hi[2] - k, lo[1] : hi[1], lo[0] : hi[0]]
            np.minimum(patch, part.field(local), out=patch)
        image = vtk.vtkImageData()
        image.SetExtent(0, nx - 1, 0, ny - 1, k, last)
        image.SetSpacing(step, step, step)
        image.SetOrigin(*origin)
        image.GetPointData().SetScalars(numpy_to_vtk(field.ravel(), deep=True))
        contour = vtk.vtkFlyingEdges3D()
        contour.SetInputData(image)
        contour.SetValue(0, 0.0)
        contour.ComputeNormalsOff()
        contour.Update()
        piece = vtk.vtkPolyData()
        piece.DeepCopy(contour.GetOutput())
        merged.AddInputData(piece)
    merged.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(merged.GetOutputPort())
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(1e-5)
    clean.ConvertPolysToLinesOff()
    clean.ConvertLinesToPointsOff()
    clean.Update()
    return _one_solid(_outward(clean.GetOutput()))


def _outward(mesh: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """Flying edges winds its triangles by the field's gradient, which here runs
    negative inside. Turn them to face outward if they do not."""
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    if not mesh.GetNumberOfPolys():
        return mesh
    faces = vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    centre = points.mean(axis=0)
    a, b, c = points[faces[:, 0]] - centre, points[faces[:, 1]] - centre, points[faces[:, 2]] - centre
    if float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) >= 0.0:
        return mesh
    reverse = vtk.vtkReverseSense()
    reverse.SetInputData(mesh)
    reverse.ReverseCellsOn()
    reverse.ReverseNormalsOff()
    reverse.Update()
    return reverse.GetOutput()


def _one_solid(mesh: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """The main body of a contour, without the odd sliver at a thin tip. Anything
    bigger than a sliver coming away is a solid that has fallen apart."""
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(mesh)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    sizes = vtk_to_numpy(connectivity.GetRegionSizes())
    if len(sizes) > 1:
        rest = np.delete(sizes, int(np.argmax(sizes)))
        if float(rest.max()) > MAX_SLIVER_SHARE * float(sizes.max()):
            raise ValueError("Part of this piece would come away from the rest.")
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    tidy = vtk.vtkCleanPolyData()
    tidy.SetInputConnection(connectivity.GetOutputPort())
    tidy.ConvertPolysToLinesOff()
    tidy.ConvertLinesToPointsOff()
    tidy.Update()
    result = vtk.vtkPolyData()
    result.DeepCopy(tidy.GetOutput())
    return result


def _is_closed(mesh: vtk.vtkPolyData) -> bool:
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(mesh)
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.Update()
    return edges.GetOutput().GetNumberOfCells() == 0


def thinned(mesh: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """The same solid in fewer triangles, within a hair of it. If that would cost it its
    closed surface, as now and then it does, the solid just as it was."""
    bounds = np.array(mesh.GetBounds()).reshape(3, 2)
    size = float(np.ptp(bounds, axis=1).max())
    thin = vtk.vtkDecimatePro()
    thin.SetInputData(mesh)
    thin.SetTargetReduction(0.9)
    thin.PreserveTopologyOn()
    thin.SplittingOff()
    thin.BoundaryVertexDeletionOff()
    thin.SetMaximumError(THIN_OUT_ERROR_MM / max(size, 1e-6))
    thin.Update()
    if not _is_closed(thin.GetOutput()):
        return mesh
    result = vtk.vtkPolyData()
    result.DeepCopy(thin.GetOutput())
    return result


def key_polydata(key: BendKey, preview: bool = False) -> vtk.vtkPolyData:
    """A key in its own frame: the plate's stop pose and the head can be drawn with it."""
    if preview:
        return parts_polydata(key.parts, PREVIEW_GRID_MM)
    return thinned(parts_polydata(key.parts, GRID_MM))


def tool_polydata(guide: BendingGuide, key: BendKey) -> vtk.vtkPolyData | None:
    """The bender's head at this key's stop, in the key's frame, for the rehearsal."""
    parts = tool_parts(guide, key)
    if not parts:
        return None
    append = vtk.vtkAppendPolyData()
    for part in parts:
        append.AddInputData(parts_polydata([part], PREVIEW_GRID_MM))
    append.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(append.GetOutput())
    return out


def _moved(mesh: vtk.vtkPolyData, rotation: np.ndarray, shift) -> vtk.vtkPolyData:
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    out = vtk.vtkPolyData()
    out.ShallowCopy(mesh)
    new = vtk.vtkPoints()
    new.SetData(numpy_to_vtk(points @ rotation.T + np.asarray(shift, dtype=float), deep=True))
    out.SetPoints(new)
    return out


def _lay_out(pieces, bed) -> list[list[vtk.vtkPolyData]]:
    """Solids in print pose, laid in rows on as many beds as they need, each standing
    at the height its supports want. Returns the beds' solids."""
    lift = rods.lift_mm()
    room_x = float(bed[0]) - 2.0 * BED_MARGIN_MM
    room_y = float(bed[1]) - 2.0 * BED_MARGIN_MM
    plates: list[list[vtk.vtkPolyData]] = [[]]
    x, y, row = 0.0, 0.0, 0.0
    for piece in pieces:
        bounds = np.array(piece.GetBounds()).reshape(3, 2)
        size = bounds[:, 1] - bounds[:, 0]
        if size[0] > room_x or size[1] > room_y or size[2] + lift > float(bed[2]):
            raise ValueError("A piece of this guide does not fit this printer. Choose a larger printer.")
        if x + size[0] > room_x + 1e-6:
            x, y, row = 0.0, y + row + LAYOUT_GAP_MM, 0.0
        if y + size[1] > room_y + 1e-6:
            plates.append([])
            x, y, row = 0.0, 0.0, 0.0
        shift = np.array([BED_MARGIN_MM + x, BED_MARGIN_MM + y, lift]) - bounds[:, 0]
        plates[-1].append(_moved(piece, np.eye(3), shift))
        x += size[0] + LAYOUT_GAP_MM
        row = max(row, size[1])
    return plates


def _flat_faces_only(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Which contacts a rod must not make: all but the flat side turned to the bed.
    The underside of a pin is turned to the bed too, but stands higher."""
    return (normals[:, 2] > -ROD_FACE_COS) | (points[:, 2] > rods.lift_mm() + 0.05)


def _ready_to_print(solids, bed) -> vtk.vtkPolyData:
    """One bed's solids with their rods and raft."""
    append = vtk.vtkAppendPolyData()
    for solid in solids:
        append.AddInputData(solid)
    append.Update()
    model = vtk.vtkPolyData()
    model.DeepCopy(append.GetOutput())
    limits = (np.full(2, 0.1), np.asarray(bed[:2], dtype=float) - 0.1)
    held = rods.support_structure(model, keep_off=_flat_faces_only, limits=limits, spacing=6.0)
    return rods.merged(model, held)


def key_solids(guide: BendingGuide) -> list[tuple[str, vtk.vtkPolyData]]:
    """Each key, closed and in its print pose, standing on the bed's plane: its name,
    and its solid, for a slicer's own supports."""
    out = []
    for key in guide.keys:
        solid = _moved(key_polydata(key), key_print_rotation(key), np.zeros(3))
        low = np.array(solid.GetBounds())[[0, 2, 4]]
        out.append((key.name, _moved(solid, np.eye(3), -low)))
    return out


def keys_ready_to_print(guide: BendingGuide, bed) -> list[vtk.vtkPolyData]:
    """Every key of the guide on the bed, with rods and a raft: one mesh per bed."""
    return keys_on_beds([solid for _, solid in key_solids(guide)], bed)


def keys_on_beds(solids, bed) -> list[vtk.vtkPolyData]:
    return [_ready_to_print(plate, bed) for plate in _lay_out(solids, bed)]


def fit_test_ready_to_print(guide: BendingGuide, bed) -> vtk.vtkPolyData:
    """The test print, for one bed: a key for one known bend, the gauge to check what
    it gave, and pins of three fits to try in the plate's real holes."""
    test = fit_test_key(guide)
    key = _moved(key_polydata(test), key_print_rotation(test), np.zeros(3))
    gauge = thinned(parts_polydata(gauge_parts(), GRID_MM))
    tab = thinned(parts_polydata(pin_tab_parts(guide), 0.1))
    plates = _lay_out([key, gauge, tab], bed)
    if len(plates) > 1:
        raise ValueError("The fit test does not fit this printer on one bed. Choose a larger printer.")
    return _ready_to_print(plates[0], bed)
