"""Conversions between MandiPlan's numpy geometry and VTK data objects."""

from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support


def volume_to_vtk(volume) -> vtk.vtkImageData:
    """Wrap a :class:`~mandiplan.geometry.volume.Volume` as ``vtkImageData``.

    Spacing and origin come straight from the volume, so VTK works in the same
    millimetre patient space as the geometry package.
    """
    nz, ny, nx = volume.array.shape
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(*(float(s) for s in volume.spacing))
    image.SetOrigin(*(float(o) for o in volume.origin))
    flat = np.ascontiguousarray(volume.array, dtype=np.float32).ravel()
    arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    arr.SetName("intensity")
    image.GetPointData().SetScalars(arr)
    return image


def polydata_points(polydata: vtk.vtkPolyData) -> np.ndarray:
    return numpy_support.vtk_to_numpy(polydata.GetPoints().GetData()).reshape(-1, 3)


def polydata_normals(polydata: vtk.vtkPolyData) -> np.ndarray | None:
    normals = polydata.GetPointData().GetNormals()
    if normals is None:
        return None
    return numpy_support.vtk_to_numpy(normals).reshape(-1, 3)


def triangles_to_polydata(points: np.ndarray, triangles: np.ndarray) -> vtk.vtkPolyData:
    """Build a triangle mesh ``vtkPolyData`` from numpy arrays."""
    pts = vtk.vtkPoints()
    pts.SetData(
        numpy_support.numpy_to_vtk(
            np.ascontiguousarray(points, dtype=np.float64), deep=True
        )
    )
    tris = np.ascontiguousarray(triangles, dtype=np.int64)
    cells_flat = np.hstack(
        [np.full((len(tris), 1), 3, dtype=np.int64), tris]
    ).ravel()
    cells = vtk.vtkCellArray()
    cells.SetCells(
        len(tris),
        numpy_support.numpy_to_vtkIdTypeArray(cells_flat, deep=True),
    )
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(pts)
    mesh.SetPolys(cells)
    return mesh


def polyline_to_polydata(points: np.ndarray) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    pts.SetData(
        numpy_support.numpy_to_vtk(
            np.ascontiguousarray(points, dtype=np.float64), deep=True
        )
    )
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(len(points))
    for i in range(len(points)):
        lines.InsertCellPoint(i)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)
    return pd
