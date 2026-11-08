"""Remove stock at an inter-hole bridge, from either end; no library mesh is modified."""
from dataclasses import replace

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ..geometry.mesh_io import Mesh, MeshLoadError
from ..plate_assets import load_asset_mesh
from .convert import triangles_to_polydata


def keep_section(mesh, centres, first, count, cap=True):
    """Keep holes ``first`` to ``first + count - 1``, cutting halfway between
    neighbouring bores, even on a curved stock plate.

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
    centres = np.asarray(centres, dtype=float)
    kept = centres[first : first + count]
    away = np.vstack([centres[:first], centres[first + count :]])
    field = nearest_squared(away) - nearest_squared(kept)
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
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(clean.GetOutputPort())
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    out = tri.GetOutput()
    points = vtk_to_numpy(out.GetPoints().GetData()).copy()
    triangles = vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3).copy()
    if cap:
        points, triangles = _capped(points, triangles)
    return Mesh(points, triangles)


def _capped(points: np.ndarray, triangles: np.ndarray):
    """The cut ends closed: a fan from the middle of each open loop, the section of a
    bridge being as near convex as makes no difference. Refuses what will not close."""
    edges = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    present = {(int(a), int(b)) for a, b in edges}
    following = {a: b for a, b in present if (b, a) not in present}
    new_points, new_triangles = [points], [triangles]
    count = len(points)
    while following:
        start = next(iter(following))
        loop = [start]
        while True:
            step = following.pop(loop[-1], None)
            if step is None:
                raise MeshLoadError("The bridge could not be cut into a closed plate at this hole.")
            if step == start:
                break
            loop.append(step)
        new_points.append(points[loop].mean(axis=0, keepdims=True))
        ring = np.array(loop)
        # Each edge of the ring is met the other way round by its cap, as a closed mesh's are.
        new_triangles.append(np.column_stack([np.roll(ring, -1), ring, np.full(len(ring), count)]))
        count += 1
    points = np.vstack(new_points)
    triangles = np.vstack(new_triangles)
    closed = np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    _, seen = np.unique(closed, axis=0, return_counts=True)
    if np.any(seen != 2):
        raise MeshLoadError("The bridge could not be cut into a closed plate at this hole.")
    return points, triangles


def trim_stock(stock, count, first=0):
    """The stock with only ``count`` holes from hole ``first`` left: ``(asset, mesh)``."""
    if not (2 <= count and first >= 0 and first + count <= stock.hole_count and count < stock.hole_count):
        raise ValueError("Retain at least two holes and cut between holes.")
    holes = stock.hole_centres_mm
    mesh = keep_section(load_asset_mesh(stock), holes, first, count)
    stations = np.asarray(stock.flat_hole_x_mm, dtype=float)
    if len(stations) != len(holes):
        stations = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(holes, axis=0), axis=1))]
    stations = stations - stations[0]
    margin = max(0.0, (stock.length_mm - stations[-1]) / 2)
    last = first + count - 1
    start = stations[first] - margin if first == 0 else (stations[first - 1] + stations[first]) / 2
    end = stations[last] + margin if last == stock.hole_count - 1 else (stations[last] + stations[last + 1]) / 2
    active = replace(stock, name=f"{stock.name} (cut to {count} holes)",
                     hole_count=count, length_mm=float(end - start),
                     hole_centres_mm=holes[first : last + 1], hole_axes=stock.hole_axes[first : last + 1],
                     flat_hole_x_mm=tuple(stations[first : last + 1]), centreline_mm=holes[first : last + 1])
    return active, mesh
