"""Small VTK helpers used only by the 3-D view."""

from __future__ import annotations

import numpy as np
import vtk


def empty_polydata() -> vtk.vtkPolyData:
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk.vtkPoints())
    return pd


def points_to_polydata(points) -> vtk.vtkPolyData:
    """Vertex-only polydata, suitable as glyph input."""
    pts = vtk.vtkPoints()
    verts = vtk.vtkCellArray()
    for i, p in enumerate(np.asarray(points, dtype=float).reshape(-1, 3)):
        pts.InsertNextPoint(*p)
        verts.InsertNextCell(1)
        verts.InsertCellPoint(i)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetVerts(verts)
    return pd
