"""Distal cutting changes the solid and the actual holes, without changing stock."""
import numpy as np
import pytest

from mandiplan.plate_assets import asset_by_id, load_assets, load_asset_mesh
from mandiplan.render.plate_trim import trim_stock


@pytest.mark.parametrize("stock", [
    "generic-recon-2.4-lp-8h",
    next(a.id for a in load_assets() if a.hole_count == 25 and a.brand == "Generic"),
    next(a.id for a in load_assets() if a.brand == "KLS Martin LevelOne" and a.hole_count >= 18),
])
def test_cut_stock_retains_full_proximal_holes_and_is_closed(stock):
    original = asset_by_id(stock)
    count = original.hole_count // 2
    active, mesh = trim_stock(original, count)
    assert np.array_equal(active.hole_centres_mm, original.hole_centres_mm[:count])
    assert 0 < active.length_mm < original.length_mm
    assert active.hole_count == count
    # Every retained hole is still a through-hole: genus counts handles.
    edges = np.sort(np.vstack([mesh.triangles[:, [0, 1]], mesh.triangles[:, [1, 2]],
                               mesh.triangles[:, [2, 0]]]), axis=1)
    unique, incidences = np.unique(edges, axis=0, return_counts=True)
    assert np.all(incidences == 2)
    assert len(mesh.points) - len(unique) + len(mesh.triangles) == 2 - 2 * count
    assert len(load_asset_mesh(original).triangles) > len(mesh.triangles)


def test_actual_jaw_projection_corrects_path_standoff_and_keeps_pitch():
    from mandiplan.geometry.plate_bend import seat_plate_chain
    from mandiplan.render.surface import SurfaceProjector
    import vtk
    # Deliberately hand the fitter a bad path proxy: it floats 1.2 mm above
    # a known flat jaw surface. Every retained stock hole must seat on bone.
    plane = vtk.vtkPlaneSource()
    plane.SetOrigin(-50, -25, 0)
    plane.SetPoint1(50, -25, 0)
    plane.SetPoint2(-50, 25, 0)
    plane.SetXResolution(20)
    plane.SetYResolution(10)
    tri = vtk.vtkTriangleFilter()
    tri.SetInputConnection(plane.GetOutputPort())
    tri.Update()
    projector = SurfaceProjector(tri.GetOutput())
    stock = np.column_stack([np.arange(6) * 8., np.zeros((6, 2))])
    up = np.tile([0., 0., 1.], (6, 1))
    path = stock + [0., 0., 2.9]   # desired midplane = 1.2 + 0.5 mm
    centres, axes = seat_plate_chain(projector, path, up, np.full(5, 8.), 2.4, .5, 8.)
    np.testing.assert_allclose(centres[:,2], 1.7, atol=.03)
    np.testing.assert_allclose(np.linalg.norm(np.diff(centres, axis=0),axis=1), 8., atol=.01)
    np.testing.assert_allclose(axes, up, atol=.001)


def _rippled_jaw(amplitude):
    """A curved jaw whose surface carries a ripple shorter than a hole pitch."""
    import vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk

    theta = np.radians(np.arange(-50, 50.01, 0.5))
    heights = np.arange(-15.0, 15.01, 0.5)
    tt, hh = np.meshgrid(theta, heights)
    radius = 50.0 + amplitude * np.sin(2 * np.pi * (50.0 * tt) / 5.0)
    grid = np.column_stack([(radius * np.sin(tt)).ravel(), (radius * np.cos(tt)).ravel(), hh.ravel()])
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(grid, deep=True))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    cells = vtk.vtkCellArray()
    nx = len(theta)
    for row in range(len(heights) - 1):
        for col in range(nx - 1):
            a = row * nx + col
            cells.InsertNextCell(3, [a, a + 1, a + nx])
            cells.InsertNextCell(3, [a + 1, a + nx + 1, a + nx])
    mesh.SetPolys(cells)
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(mesh)
    normals.SplittingOff()
    normals.AutoOrientNormalsOn()
    normals.Update()
    return normals.GetOutput()


def test_a_plate_on_a_rough_jaw_is_fair_exactly_spaced_and_seated():
    """The bone's texture must not reach the plate; its form must."""
    from mandiplan.geometry.plate_bend import seat_plate_chain
    from mandiplan.render.surface import SurfaceProjector

    projector = SurfaceProjector(_rippled_jaw(0.4))
    # A hand-drawn route: seven clicks scattered across the bone by a millimetre.
    t = np.radians(np.linspace(-40, 40, 7))
    scatter = np.array([0.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.0])
    path = np.column_stack([50 * np.sin(t), 50 * np.cos(t), scatter])
    normals = np.column_stack([np.sin(t), np.cos(t), np.zeros(7)])
    spacing = np.full(9, 8.0)
    centres, axes = seat_plate_chain(projector, path, normals, spacing, 2.4, 0.5, 8.0)

    np.testing.assert_allclose(np.linalg.norm(np.diff(centres, axis=0), axis=1), 8.0, atol=1e-3)
    # Neighbouring holes turn together, by what the jaw's own curve asks: 8 mm
    # on a 50 mm radius is about 9 degrees, not the tens a ripple would give.
    turns = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", axes[:-1], axes[1:]), -1, 1)))
    assert turns.max() < 12.0
    # The scatter across the bone is gone.
    assert np.ptp(centres[:, 2]) < 0.8
    # No hole is in the bone, and none stands off it by more than the ripple.
    radii = np.hypot(centres[:, 0], centres[:, 1])
    assert radii.min() > 50.0 + 1.2
    assert radii.max() < 50.0 + 1.2 + 0.5 + 1.2


def test_plate_holes_past_the_drawn_route_still_follow_the_jaw():
    from mandiplan.geometry.plate_bend import seat_plate_chain
    from mandiplan.render.surface import SurfaceProjector

    projector = SurfaceProjector(_rippled_jaw(0.0))
    t = np.radians(np.array([-10.0, 0.0, 10.0]))
    path = np.column_stack([50 * np.sin(t), 50 * np.cos(t), np.zeros(3)])
    normals = np.column_stack([np.sin(t), np.cos(t), np.zeros(3)])
    centres, _ = seat_plate_chain(projector, path, normals, np.full(7, 8.0), 2.4, 0.5, 8.0)
    radii = np.hypot(centres[:, 0], centres[:, 1])
    np.testing.assert_allclose(radii, 50.0 + 1.7, atol=0.4)
