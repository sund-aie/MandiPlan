"""The fibula cutting guide as a printable mesh: one closed solid.

The guide (``geometry/fibula_guide.py``) is a signed field over a grid that also
holds the bone's distance; it is contoured where the field is zero.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk

from ..geometry.fibula_guide import GRID_MM, PREVIEW_GRID_MM, bone_distance, guide_field_on
from .bending_guide_mesh import _one_solid, _outward, thinned

#: The bone's distance is worked out this far round the guide's box, mm.
BOX_MARGIN_MM = 2.0


def guide_grid(guide, volume, fibula, threshold: float, preview: bool = False):
    """The bone's distance over the guide's box, at the printing or the preview step."""
    lo, hi = guide.bounds()
    step = PREVIEW_GRID_MM if preview else GRID_MM
    # Off the round numbers: no flat face may fall exactly on a grid plane.
    lo = lo - BOX_MARGIN_MM - 0.137 * step
    return bone_distance(volume, fibula, threshold, lo, hi + BOX_MARGIN_MM, step)


def guide_polydata(guide, bone, solid: np.ndarray | None = None, thin: bool = True) -> vtk.vtkPolyData:
    """The guide contoured on ``bone``'s grid: a closed, outward-facing solid."""
    if solid is None:
        solid = guide_field_on(guide, bone)
    nx, ny, nz = bone.shape_xyz
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(bone.step, bone.step, bone.step)
    image.SetOrigin(*bone.origin)
    image.GetPointData().SetScalars(numpy_to_vtk(np.ascontiguousarray(solid, dtype=np.float32).ravel(), deep=True))
    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, 0.0)
    contour.ComputeNormalsOff()
    contour.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(contour.GetOutputPort())
    clean.ToleranceIsAbsoluteOn()
    clean.SetAbsoluteTolerance(1e-5)
    clean.ConvertPolysToLinesOff()
    clean.ConvertLinesToPointsOff()
    clean.Update()
    mesh = _one_solid(_outward(clean.GetOutput()))
    return thinned(mesh) if thin else mesh


def clearance_to_bone(guide_mesh: vtk.vtkPolyData, bone_surface: vtk.vtkPolyData) -> float:
    """The least distance from the printed guide's surface to the bone's, mm,
    measured between the two meshes themselves."""
    from vtkmodules.util.numpy_support import vtk_to_numpy

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(bone_surface)
    locator.BuildLocator()
    points = vtk_to_numpy(guide_mesh.GetPoints().GetData()).astype(float)
    closest = [0.0, 0.0, 0.0]
    cell, sub = vtk.reference(0), vtk.reference(0)
    distance2 = vtk.reference(0.0)
    best = float("inf")
    for p in points:
        locator.FindClosestPoint(p.tolist(), closest, cell, sub, distance2)
        best = min(best, float(distance2))
    return float(np.sqrt(best))
