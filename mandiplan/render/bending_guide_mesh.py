"""The bending guide as a printable mesh: one closed solid.

The parts of the guide (``geometry/bending_guide.py``) are simple solids that
overlap where they join. Their union is contoured from the minimum of their
fields, in slabs so a long plate's guide stays within memory, so the printed
object is one watertight shell and not a heap of overlapping ones.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ..geometry.bending_guide import BendingGuide

#: Contouring grid, mm.
GRID_MM = 0.2


def guide_polydata(guide: BendingGuide, *, print_pose: bool = True) -> vtk.vtkPolyData:
    """The whole guide: base flat on z = 0, or where it stands over the patient."""
    step = min(GRID_MM, guide.tooth_mm / 5.0)
    placed = []
    for part in guide.parts:
        bound = np.abs(part.frame) @ part.half + 2.0 * step
        placed.append((part, part.centre - bound, part.centre + bound))
    origin = np.min([p[1] for p in placed], axis=0) - step
    # The base lies exactly on the bed.
    origin[2] = -2.0 * step
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
    result = vtk.vtkPolyData()
    result.DeepCopy(clean.GetOutput())
    points = vtk_to_numpy(result.GetPoints().GetData()).astype(float)
    if print_pose:
        points[:, 2] -= points[:, 2].min()
    else:
        points = guide.to_patient(points)
    result.GetPoints().SetData(numpy_to_vtk(points, deep=True))
    return result
