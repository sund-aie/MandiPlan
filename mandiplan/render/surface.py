"""Bone surface extraction, surface picking and plane clipping.

Segmentation here is threshold plus one morphological step (keep the largest
connected component).  No smoothing is applied: a smoothed isosurface no
longer coincides with the gray-value boundary that was thresholded, and every
distance in this application is taken off that surface.
"""

from __future__ import annotations

import numpy as np
import vtk

from ..geometry.resection import CutPlane
from .convert import volume_to_vtk


def extract_isosurface(
    volume, threshold: float, largest_component: bool = True
) -> vtk.vtkPolyData:
    """Isosurface of ``volume`` at ``threshold`` (native gray values)."""
    image = volume_to_vtk(volume)
    surface = vtk.vtkFlyingEdges3D()
    surface.SetInputData(image)
    surface.SetValue(0, float(threshold))
    surface.ComputeNormalsOn()
    surface.ComputeGradientsOff()
    surface.ComputeScalarsOff()
    surface.Update()
    result = surface.GetOutput()

    if largest_component and result.GetNumberOfPoints() > 0:
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputData(result)
        connectivity.SetExtractionModeToLargestRegion()
        connectivity.Update()
        result = connectivity.GetOutput()

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(result)
    triangles.PassLinesOff()
    triangles.PassVertsOff()
    triangles.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(triangles.GetOutput())
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.ComputePointNormalsOn()
    normals.Update()
    return normals.GetOutput()


def mesh_volume_mm3(polydata: vtk.vtkPolyData) -> float:
    """Enclosed volume of a closed triangle mesh, in mm³."""
    if polydata.GetNumberOfPolys() == 0:
        return 0.0
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(polydata)
    triangles.Update()
    mass = vtk.vtkMassProperties()
    mass.SetInputData(triangles.GetOutput())
    mass.Update()
    return float(mass.GetVolume())


def _vtk_plane(plane: CutPlane, invert: bool) -> vtk.vtkPlane:
    p = vtk.vtkPlane()
    p.SetOrigin(*(float(v) for v in plane.origin))
    normal = -plane.normal if invert else plane.normal
    p.SetNormal(*(float(v) for v in normal))
    return p


def clip_closed(
    polydata: vtk.vtkPolyData, planes: list[CutPlane], keep_resected: bool
) -> vtk.vtkPolyData:
    """Clip a closed surface with cut planes and cap the openings.

    ``CutPlane.normal`` points into the fragment being removed, and
    ``vtkClipClosedSurface`` keeps the half-space the plane normal points
    into, so the resected fragment is the intersection of the planes as given
    and the retained bone needs them flipped.

    With ``keep_resected=False`` and two planes the retained bone is the union
    of the two outer pieces, which is not a single half-space intersection; it
    is produced by clipping with each plane separately and appending.
    """
    if not planes:
        return polydata

    if keep_resected:
        collection = vtk.vtkPlaneCollection()
        for plane in planes:
            collection.AddItem(_vtk_plane(plane, invert=False))
        return _clip_with_collection(polydata, collection)

    pieces = vtk.vtkAppendPolyData()
    for plane in planes:
        collection = vtk.vtkPlaneCollection()
        collection.AddItem(_vtk_plane(plane, invert=True))
        pieces.AddInputData(_clip_with_collection(polydata, collection))
    pieces.Update()
    return pieces.GetOutput()


def _clip_with_collection(
    polydata: vtk.vtkPolyData, collection: vtk.vtkPlaneCollection
) -> vtk.vtkPolyData:
    clip = vtk.vtkClipClosedSurface()
    clip.SetInputData(polydata)
    clip.SetClippingPlanes(collection)
    clip.GenerateFacesOn()
    clip.TriangulationErrorDisplayOff()
    clip.Update()
    return clip.GetOutput()


class SurfaceProjector:
    """Projects arbitrary world points onto the bone surface, with normals."""

    def __init__(self, polydata: vtk.vtkPolyData):
        self.polydata = polydata
        self.locator = vtk.vtkCellLocator()
        self.locator.SetDataSet(polydata)
        self.locator.BuildLocator()
        normals = polydata.GetPointData().GetNormals()
        self._normals = normals

    def project(self, point) -> tuple[np.ndarray, np.ndarray]:
        """Closest point on the surface and the interpolated outward normal."""
        closest = [0.0, 0.0, 0.0]
        cell_id = vtk.reference(-1)
        sub_id = vtk.reference(0)
        dist2 = vtk.reference(0.0)
        self.locator.FindClosestPoint(
            [float(v) for v in point], closest, cell_id, sub_id, dist2
        )
        closest = np.asarray(closest, dtype=float)
        return closest, self._normal_at(int(cell_id), closest)

    def _normal_at(self, cell_id: int, point: np.ndarray) -> np.ndarray:
        cell = self.polydata.GetCell(cell_id)
        ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]
        pts = np.array([self.polydata.GetPoint(i) for i in ids])
        if self._normals is not None and len(ids) == 3:
            weights = _barycentric(pts, point)
            normals = np.array([self._normals.GetTuple3(i) for i in ids])
            n = weights @ normals
        else:
            n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        length = np.linalg.norm(n)
        return n / length if length > 0 else np.array([0.0, 0.0, 1.0])


def _barycentric(triangle: np.ndarray, point: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    v0, v1, v2 = b - a, c - a, point - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    return np.array([1.0 - v - w, v, w])
