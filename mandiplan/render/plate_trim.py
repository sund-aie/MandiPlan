"""Remove the distal stock at an inter-hole bridge; no library mesh is modified."""
from dataclasses import replace

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ..geometry.mesh_io import Mesh, MeshLoadError
from ..plate_assets import load_asset_mesh
from .convert import triangles_to_polydata


def keep_proximal(mesh, centres, count, cap=True):
    """Cut halfway between neighbouring bores, even on a curved stock plate.

    Use the hole chain, not a global x plane. A wraparound plate may cross a
    cutting plane again on the opposite arm, which must stay intact.
    """
    def nearest_squared(holes):
        result = np.full(len(mesh.points), np.inf)
        for hole in holes:
            d = mesh.points - hole
            np.minimum(result, np.einsum("ij,ij->i", d, d), out=result)
        return result

    # Equal squared distances give the planar bisector in the cut bridge.
    field = nearest_squared(centres[count:]) - nearest_squared(centres[:count])
    poly = triangles_to_polydata(mesh.points, mesh.triangles)
    poly.GetPointData().SetScalars(numpy_to_vtk(field, deep=True))
    cut = vtk.vtkClipPolyData()
    cut.SetInputData(poly)
    cut.SetValue(0.0)
    cut.Update()
    if not cut.GetOutput().GetNumberOfCells():
        return Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(cut.GetOutputPort())
    last = clean
    if cap:
        fill = vtk.vtkFillHolesFilter()
        fill.SetInputConnection(clean.GetOutputPort())
        fill.SetHoleSize(float(np.ptp(mesh.points, axis=0).max()))
        orient = vtk.vtkPolyDataNormals()
        orient.SetInputConnection(fill.GetOutputPort())
        orient.SplittingOff()
        orient.ConsistencyOn()
        orient.AutoOrientNormalsOn()
        last = orient
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(last.GetOutputPort())
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    out = tri.GetOutput()
    if cap:
        edges = vtk.vtkFeatureEdges()
        edges.SetInputData(out)
        edges.FeatureEdgesOff()
        edges.ManifoldEdgesOff()
        edges.BoundaryEdgesOn()
        edges.NonManifoldEdgesOn()
        edges.Update()
        if edges.GetOutput().GetNumberOfCells():
            raise MeshLoadError("The bridge could not be cut into a closed plate at this hole.")
    return Mesh(vtk_to_numpy(out.GetPoints().GetData()).copy(),
                vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3).copy())


def trim_stock(stock, count):
    if not 2 <= count < stock.hole_count:
        raise ValueError("Retain at least two holes and cut between holes.")
    holes = stock.hole_centres_mm
    mesh = keep_proximal(load_asset_mesh(stock), holes, count)
    stations = np.asarray(stock.flat_hole_x_mm, dtype=float)
    if len(stations) != len(holes):
        stations = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(holes, axis=0), axis=1))]
    stations = stations - stations[0]
    margin = max(0.0, (stock.length_mm - stations[-1]) / 2)
    length = margin + (stations[count-1] + stations[count]) / 2
    active = replace(stock, name=f"{stock.name} (cut to {count} holes)",
                     hole_count=count, length_mm=float(length),
                     hole_centres_mm=holes[:count], hole_axes=stock.hole_axes[:count],
                     flat_hole_x_mm=tuple(stations[:count]), centreline_mm=holes[:count])
    return active, mesh
