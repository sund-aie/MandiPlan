"""Hand refinement of the reconstructed jaw."""

from __future__ import annotations

import numpy as np
import pytest
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from mandiplan.geometry.mesh_io import Mesh
from mandiplan.geometry.resection import CutPlane
from mandiplan.geometry.sculpt import SurfaceSculptor


def _sphere(radius=10.0, resolution=60):
    source = vtk.vtkSphereSource()
    source.SetRadius(radius)
    source.SetThetaResolution(resolution)
    source.SetPhiResolution(resolution)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(source.GetOutputPort())
    clean.Update()
    poly = clean.GetOutput()
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    triangles = vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return points, triangles


def _bumpy_sphere():
    points, triangles = _sphere()
    rng = np.random.default_rng(0)
    return points + rng.normal(0.0, 0.25, points.shape), triangles


def _roughness(points, centre, radius):
    """Spread of the distance to the sphere's centre, inside the brush."""
    near = np.linalg.norm(points - centre, axis=1) < radius
    return float(np.std(np.linalg.norm(points[near], axis=1)))


def test_the_smooth_brush_evens_the_surface_without_shrinking_it():
    points, triangles = _bumpy_sphere()
    sculptor = SurfaceSculptor(points, triangles)
    centre = np.array([0.0, 0.0, 10.0])
    before = _roughness(sculptor.points, centre, 4.0)
    volume = Mesh(sculptor.points, triangles).volume_mm3()
    for _ in range(5):
        sculptor.checkpoint()
        sculptor.stroke(centre, 5.0, 1.0, "smooth")
    assert _roughness(sculptor.points, centre, 4.0) < 0.5 * before
    # Taubin, not plain Laplacian: the patch does not sink.
    assert Mesh(sculptor.points, triangles).volume_mm3() == pytest.approx(volume, rel=0.005)
    # Nothing outside the brush moved.
    far = np.linalg.norm(points - centre, axis=1) > 5.5
    assert np.allclose(sculptor.points[far], points[far])


def test_fill_adds_bone_and_carve_takes_it_away():
    points, triangles = _sphere()
    sculptor = SurfaceSculptor(points, triangles)
    volume = Mesh(points, triangles).volume_mm3()
    sculptor.checkpoint()
    sculptor.stroke([10.0, 0.0, 0.0], 4.0, 1.0, "fill")
    grown = Mesh(sculptor.points, triangles).volume_mm3()
    assert grown > volume + 1.0
    sculptor.checkpoint()
    sculptor.stroke([10.0, 0.0, 0.0], 4.0, 1.0, "carve")
    sculptor.stroke([10.0, 0.0, 0.0], 4.0, 1.0, "carve")
    assert Mesh(sculptor.points, triangles).volume_mm3() < grown - 1.0
    # Still one closed, outward surface: only the points moved.
    edited = Mesh(sculptor.points, triangles)
    assert edited.is_consistently_oriented() and edited.is_outward()


def test_every_edit_can_be_undone_and_the_computed_surface_restored():
    points, triangles = _bumpy_sphere()
    sculptor = SurfaceSculptor(points, triangles)
    for brush in ("smooth", "fill", "carve"):
        sculptor.checkpoint()
        sculptor.stroke([0.0, 0.0, 10.0], 5.0, 0.8, brush)
    assert sculptor.edited and sculptor.max_change_mm() > 0.05
    assert sculptor.undo()
    sculptor.reset()
    assert not sculptor.edited
    assert sculptor.undo()  # the reset itself can be undone
    assert sculptor.edited


def test_junction_smoothing_stays_near_the_cuts():
    points, triangles = _bumpy_sphere()
    sculptor = SurfaceSculptor(points, triangles)
    plane = CutPlane(np.zeros(3), np.array([1.0, 0.0, 0.0]))
    moved = sculptor.smooth_near_planes([plane], band_mm=2.0)
    assert moved > 0
    changed = np.linalg.norm(sculptor.points - points, axis=1) > 1e-9
    assert np.all(np.abs(points[changed, 0]) < 2.0)
    assert sculptor.max_change_mm() < 1.0


def test_a_point_no_triangle_uses_never_moves():
    points, triangles = _sphere()
    lonely = np.vstack([points, [[0.0, 0.0, 10.2]]])
    sculptor = SurfaceSculptor(lonely, triangles)
    sculptor.stroke([0.0, 0.0, 10.0], 5.0, 1.0, "smooth")
    assert np.allclose(sculptor.points[-1], [0.0, 0.0, 10.2])
