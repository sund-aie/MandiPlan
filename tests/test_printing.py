"""The print kit: pieces that fit the printer, lock together, and spare the working side."""

import json

import numpy as np
import pytest
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from mandiplan.printing import PRINTERS, export_print_kit


def box():
    c = vtk.vtkCubeSource()
    c.SetXLength(110)
    c.SetYLength(28)
    c.SetZLength(25)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(c.GetOutputPort())
    triangles.Update()
    return triangles.GetOutput()


def hollow_box():
    """Thin cortex round a cavity, as a real bone's isosurface is."""
    outer = box()
    inner = vtk.vtkCubeSource()
    inner.SetXLength(100)
    inner.SetYLength(20)
    inner.SetZLength(17)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(inner.GetOutputPort())
    triangles.Update()
    both = vtk.vtkAppendPolyData()
    both.AddInputData(outer)
    both.AddInputData(triangles.GetOutput())
    both.Update()
    return both.GetOutput()


def load_stl(path):
    r = vtk.vtkSTLReader()
    r.SetFileName(str(path))
    r.Update()
    return r.GetOutput()


def closed(mesh):
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(mesh)
    edges.ManifoldEdgesOff()
    edges.FeatureEdgesOff()
    edges.Update()
    return edges.GetOutput().GetNumberOfCells() == 0


def volume(mesh):
    mass = vtk.vtkMassProperties()
    mass.SetInputData(mesh)
    mass.Update()
    return mass.GetVolume()


def split_kit(tmp_path, surface=None, **options):
    options.setdefault("supports", False)
    paths = export_print_kit(
        tmp_path, surface or box(), (84, 52, 76), working_side="left", cut_points=[(30, 0, 0)], **options
    )
    folder = paths[0].parent
    return folder, json.loads((folder / "assembly.json").read_text())


def test_a_large_printer_gets_one_piece_and_the_whole_jaw(tmp_path):
    paths = export_print_kit(tmp_path, box(), PRINTERS["Original Prusa MINI+"], supports=False)
    spec = json.loads((paths[0].parent / "assembly.json").read_text())
    assert len(spec["parts"]) == 1 and not spec["joins"]
    mesh = load_stl(paths[0].parent / spec["parts"][0]["file"])
    assert closed(mesh)
    assert volume(mesh) == pytest.approx(110 * 28 * 25, rel=0.02)


def test_pieces_fit_the_printer_and_add_up_to_the_jaw(tmp_path):
    folder, spec = split_kit(tmp_path)
    assert len(spec["parts"]) == 2 and len(spec["joins"]) == 1
    total = 0.0
    for part in spec["parts"]:
        mesh = load_stl(folder / part["file"])
        bounds = np.array(mesh.GetBounds()).reshape(3, 2)
        assert np.all(bounds[:, 0] >= -1e-5) and np.all(bounds[:, 1] <= np.array([84, 52, 76]) + 1e-5)
        assert bounds[2, 0] == pytest.approx(0.0, abs=1e-5)
        assert closed(mesh)
        total += volume(mesh)
    # Short of the whole by the printing clearance over the seam, and no more.
    assert total == pytest.approx(110 * 28 * 25, rel=0.03)


def _in_patient(folder, part):
    mesh = load_stl(folder / part["file"])
    points = vtk_to_numpy(mesh.GetPoints().GetData()).astype(float)
    back = np.array(part["print_to_world"])
    moved = vtk.vtkPolyData()
    moved.ShallowCopy(mesh)
    new = vtk.vtkPoints()
    from vtkmodules.util.numpy_support import numpy_to_vtk

    new.SetData(numpy_to_vtk(points @ back[:3, :3].T + back[:3, 3], deep=True))
    moved.SetPoints(new)
    return moved


def test_the_seam_is_a_jigsaw_joint_that_locks(tmp_path):
    folder, spec = split_kit(tmp_path, clearance_mm=0.3)
    joint = spec["joins"][0]
    centre = np.array(joint["centre_mm"])
    first, second = (_in_patient(folder, p) for p in spec["parts"])
    inside = []
    for mesh in (first, second):
        distance = vtk.vtkImplicitPolyDataDistance()
        distance.SetInput(mesh)
        inside.append(lambda p, d=distance: d.EvaluateFunction(tuple(p)) < 0)
    head = joint["head_diameter_mm"]
    neck = joint["neck_width_mm"]
    assert neck < head  # a head wider than its neck is what locks
    along = np.array([1.0, 0.0, 0.0])
    height = np.array(joint["slide_together_along"])
    height = np.cross(along, height)
    # Hunt the head along the jaw, either way the seam was built.
    for sign in (1.0, -1.0):
        middle = centre + sign * along * 1.1 * head / 2.0
        if inside[0](middle) != inside[1](centre - sign * along * 8.0):
            break
    knob_owner, other = (inside[0], inside[1]) if inside[0](middle) else (inside[1], inside[0])
    assert knob_owner(middle) and not other(middle)
    # Beside the neck, the socket wall holds the head back from coming out.
    wall = centre + sign * along * 0.5 + height * (neck / 2.0 + 0.8)
    assert other(wall) and not knob_owner(wall)
    # There is a real gap between them at the head's edge.
    edge = middle + height * (head / 2.0)
    assert not (knob_owner(edge) and other(edge))


def test_no_seam_enters_the_working_side_or_comes_near_the_resection(tmp_path):
    _, spec = split_kit(tmp_path)
    x = spec["joins"][0]["centre_mm"][0]
    # Patient left is +x, and that is the side named: the seam is on the other.
    assert x < 0.0
    assert spec["working_side"] == "left"


def test_a_hollow_bone_is_solid_at_its_joints(tmp_path):
    folder, spec = split_kit(tmp_path, hollow_box())
    assert len(spec["parts"]) == 2
    total = sum(volume(load_stl(folder / p["file"])) for p in spec["parts"])
    assert total == pytest.approx(110 * 28 * 25, rel=0.03)


def test_the_resection_side_faces_up_with_stems_away_from_it(tmp_path):
    paths = export_print_kit(
        tmp_path, box(), PRINTERS["Original Prusa MINI+"], working_side="left", cut_points=[(30, 0, 0)]
    )
    folder = paths[0].parent
    part = json.loads((folder / "assembly.json").read_text())["parts"][0]
    assert part["holds_resection"] and part["breakaway_stems"] >= 3
    stems, model = load_stl(folder / part["stems_file"]), load_stl(folder / part["file"])
    assert model.GetBounds()[4] == pytest.approx(3.0)
    assert stems.GetBounds()[4] == pytest.approx(0.0) and stems.GetBounds()[5] > model.GetBounds()[4]
    matrix = np.array(part["world_to_print"])
    site = matrix[:3, :3] @ np.array([30.0, 0.0, 0.0]) + matrix[:3, 3]
    tips = vtk_to_numpy(stems.GetPoints().GetData())
    assert np.min(np.linalg.norm(tips[tips[:, 2] > 3.0] - site, axis=1)) >= 8.0


def test_each_export_is_a_new_folder(tmp_path):
    first = export_print_kit(tmp_path, box(), (180, 180, 180), supports=False)[0].parent
    second = export_print_kit(tmp_path, box(), (180, 180, 180), supports=False)[0].parent
    assert first != second and first.is_dir() and second.is_dir()


def test_a_printer_too_small_for_the_working_side_is_told_so_and_writes_nothing(tmp_path):
    keep = tmp_path / "kept.txt"
    keep.write_text("mine")
    with pytest.raises(ValueError, match="larger printer"):
        export_print_kit(tmp_path, box(), (30, 30, 30), working_side="left", cut_points=[(30, 0, 0)], supports=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["kept.txt"]
    assert keep.read_text() == "mine"


def test_a_split_without_a_side_asks_for_one(tmp_path):
    with pytest.raises(ValueError, match="working side"):
        export_print_kit(tmp_path, box(), (84, 52, 76), supports=False)


def test_bad_settings_are_refused(tmp_path):
    with pytest.raises(ValueError):
        export_print_kit(tmp_path, box(), (5, 5, 5))
    with pytest.raises(ValueError):
        export_print_kit(tmp_path, box(), (180, 180, 180), clearance_mm=5.0)
    with pytest.raises(ValueError):
        export_print_kit(tmp_path, box(), (180, 180, 180), working_side="both")
