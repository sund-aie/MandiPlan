"""The print kit: parts that fit the printer, connect with pegs, and spare the working section."""

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
    # Flat faces meeting: the whole, but for the clearance round the pegs.
    assert total == pytest.approx(110 * 28 * 25, rel=0.01)


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


def _insides(folder, spec):
    out = []
    for part in spec["parts"]:
        distance = vtk.vtkImplicitPolyDataDistance()
        distance.SetInput(_in_patient(folder, part))
        out.append(lambda p, d=distance: d.EvaluateFunction(tuple(p)) < 0)
    return out


def test_parts_meet_on_a_flat_face_and_connect_with_two_pegs(tmp_path):
    folder, spec = split_kit(tmp_path, clearance_mm=0.2)
    joint = spec["joins"][0]
    centre = np.array(joint["centre_mm"])
    along = np.array(joint["cut_normal"])
    first, second = _insides(folder, spec)
    # Two pegs, of different sizes, so they go together only one way.
    diameters = joint["peg_diameters_mm"]
    assert len(diameters) == 2 and diameters[0] != diameters[1]

    # A flat face: either side of the plane, away from the pegs, is one part or
    # the other, right across the section.
    assert np.abs(along @ np.array([1.0, 0.0, 0.0])) > 0.99
    plane_x = centre[0]
    sign = float(np.sign(along[0]))
    corner = np.array([plane_x, 12.0, 10.5])
    assert first(corner - sign * along * 0.2 * sign) != second(corner - sign * along * 0.2 * sign)
    assert first(corner + along * 0.3) != first(corner - along * 0.3)
    assert second(corner + along * 0.3) != second(corner - along * 0.3)


def test_a_peg_stands_in_its_hole_with_clearance_all_round(tmp_path):
    from mandiplan.printing import HOLE_EXTRA_DEPTH_MM, PEG_LENGTH_MM, PEG_RADII_MM, _make_seam, _Route, _solid_grid, _coordinates

    surface = box()
    solid, origin = _solid_grid(surface, 0.3)
    index = np.nonzero(solid)
    points = _coordinates(index, origin, 0.3)
    s = np.arange(-55.0, 55.5, 1.0)
    route = _Route(np.column_stack([s, np.zeros(len(s)), np.zeros(len(s))]), s + 55.0)
    seam = _make_seam(route, (solid, origin, 0.3), points, points[:, 0] + 55.0, 40.0, 0.2)
    assert seam is not None and len(seam.pegs) == 2
    large = max(r for _, _, r in seam.pegs)
    assert large == PEG_RADII_MM[0][0], "a block this big takes the largest pegs"
    v, w, radius = next(p for p in seam.pegs if p[2] == large)
    at = seam.centre + v * seam.height + w * seam.across
    samples = np.array(
        [
            at + seam.along * 3.0,  # on the peg's axis, past the face
            at + seam.along * 3.0 + seam.height * (radius + 0.1),  # in the clearance
            at + seam.along * 3.0 + seam.height * (radius + 0.5),  # in the other part's wall
            at - seam.along * 3.0 + seam.height * (radius + 0.5),  # the peg's own part
            at + seam.along * (PEG_LENGTH_MM + 0.2),  # past the peg's tip, in the hole's depth
            at + seam.along * (PEG_LENGTH_MM + HOLE_EXTRA_DEPTH_MM + 0.5),  # past the hole
        ]
    )
    arcs = np.full(len(samples), 40.0)
    lower, upper = seam.lower_field(samples, arcs), seam.upper_field(samples, arcs)
    assert lower[0] < 0 and upper[0] > 0, "the peg belongs to its part alone"
    assert lower[1] > 0 and upper[1] > 0, "a clearance round it"
    assert lower[2] > 0 and upper[2] < 0, "and the other part's wall beyond"
    assert lower[3] < 0 and upper[3] > 0
    assert lower[4] > 0 and upper[4] > 0, "the hole is deeper than the peg is long"
    assert lower[5] > 0 and upper[5] < 0


def test_the_pegs_are_on_the_part_further_from_the_working_section(tmp_path):
    _, spec = split_kit(tmp_path)
    joint = spec["joins"][0]
    # Working side is left, +x, the far end of the box's run: the pegs are lower.
    assert joint["pegs_on"] == "lower-numbered part"
    assert spec["parts"][1]["holds_resection"] and not spec["parts"][0]["holds_resection"]


def test_outside_the_part_is_the_scans_own_surface(tmp_path):
    """The grid must not cost the surface anything: its outside is put back on the scan."""
    from mandiplan.printing import _snapped_to_scan, _tidy

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(20.0)
    sphere.SetThetaResolution(120)
    sphere.SetPhiResolution(120)
    sphere.Update()
    grown = vtk.vtkSphereSource()
    grown.SetRadius(20.25)
    grown.SetThetaResolution(48)
    grown.SetPhiResolution(48)
    grown.Update()
    part = _tidy(grown.GetOutput(), "test")
    snapped = _snapped_to_scan(part, sphere.GetOutput(), [])
    radii = np.linalg.norm(vtk_to_numpy(snapped.GetPoints().GetData()), axis=1)
    assert np.abs(radii - 20.0).max() < 0.02


def test_no_seam_enters_the_working_side_or_comes_near_the_resection(tmp_path):
    _, spec = split_kit(tmp_path)
    x = spec["joins"][0]["centre_mm"][0]
    # Patient left is +x, and that is the side named: the cut is on the other.
    assert x < 0.0
    assert spec["working_side"] == "left"
    assert spec["whole_working_side_in_one_part"]


def test_on_a_small_printer_the_resected_section_still_stays_whole(tmp_path):
    """Too small for the whole side: the cut may enter it, but never the section."""
    wide = vtk.vtkCubeSource()
    wide.SetXLength(170)
    wide.SetYLength(28)
    wide.SetZLength(25)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(wide.GetOutputPort())
    triangles.Update()
    # The working side alone is 85 mm: the bed holds 70.
    paths = export_print_kit(
        tmp_path, triangles.GetOutput(), (74, 52, 76), working_side="left",
        cut_points=[(30, 0, 0), (45, 0, 0)], supports=False,
    )
    spec = json.loads((paths[0].parent / "assembly.json").read_text())
    assert not spec["whole_working_side_in_one_part"]
    assert len(spec["parts"]) >= 3
    for joint in spec["joins"]:
        x = joint["centre_mm"][0]
        assert not 30.0 - 6.0 < x < 45.0 + 6.0, "no cut through the section or against it"
    assert "cannot hold the whole working side" in (paths[0].parent / "PRINT-AND-ASSEMBLE.txt").read_text()
    # And the two cuts of the resection are in one part.
    holding = [p for p in spec["parts"] if p["holds_resection"]]
    assert len(holding) == 1


def test_a_hollow_bone_is_solid_at_its_joints(tmp_path):
    folder, spec = split_kit(tmp_path, hollow_box())
    assert len(spec["parts"]) == 2
    total = sum(volume(load_stl(folder / p["file"])) for p in spec["parts"])
    assert total == pytest.approx(110 * 28 * 25, rel=0.03)


def test_the_resection_side_faces_up_with_stems_off_the_working_surface(tmp_path):
    from mandiplan.geometry import cpr
    from mandiplan.geometry.spline import ArchCurve

    # A curved jaw with its arch curve, so it has an outer face to keep clear.
    theta = np.radians(np.linspace(-80.0, 80.0, 9))
    frames = cpr.build_frames(ArchCurve(np.column_stack([50 * np.sin(theta), -50 * np.cos(theta), np.zeros(9)])), 0.5)
    line = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    for p in frames.points[::4]:
        points.InsertNextPoint(*p)
    line.SetPoints(points)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(points.GetNumberOfPoints(), list(range(points.GetNumberOfPoints())))
    line.SetLines(cells)
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(line)
    tube.SetRadius(9.0)
    tube.SetNumberOfSides(32)
    tube.CappingOn()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(tube.GetOutputPort())
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    clean.Update()
    jaw = clean.GetOutput()
    site = frames.points[frames.index_of(0.25 * frames.length_mm)]
    paths = export_print_kit(
        tmp_path, jaw, PRINTERS["Original Prusa MINI+"], working_side="auto", cut_points=[site], frames=frames
    )
    folder = paths[0].parent
    spec = json.loads((folder / "assembly.json").read_text())
    part = spec["parts"][0]
    assert part["holds_resection"] and part["breakaway_stems"] >= 3
    stems, model = load_stl(folder / part["stems_file"]), load_stl(folder / part["file"])
    assert model.GetBounds()[4] == pytest.approx(3.0)
    assert stems.GetBounds()[4] == pytest.approx(0.0) and stems.GetBounds()[5] > model.GetBounds()[4]
    matrix = np.array(part["world_to_print"])
    site_print = matrix[:3, :3] @ site + matrix[:3, 3]
    tips = vtk_to_numpy(stems.GetPoints().GetData())
    tips = tips[tips[:, 2] > 3.0]
    assert np.min(np.linalg.norm(tips - site_print, axis=1)) >= 8.0

    # The jaw's outer face at the resection looks up, away from the bed.
    centre = frames.points.mean(axis=0)
    outward = site - centre
    outward[2] = 0.0
    outward /= np.linalg.norm(outward)
    assert matrix[:3, :3][2] @ outward > 0.6

    # No stem tip on the outer face of the working stretch.
    back = np.linalg.inv(matrix)
    world = tips @ back[:3, :3].T + back[:3, 3]
    guard = spec["kept_whole_arc_mm"]
    arcs = frames.s[np.argmin(np.linalg.norm(world[:, None] - frames.points[None], axis=2), axis=1)]
    on_curve = frames.points[np.argmin(np.linalg.norm(world[:, None] - frames.points[None], axis=2), axis=1)]
    rel = world - on_curve
    rel[:, 2] = 0.0
    outside = on_curve - centre
    outside[:, 2] = 0.0
    facing_out = np.einsum("ij,ij->i", rel, outside) > 0.0
    assert not np.any(facing_out & (arcs >= guard[0]) & (arcs <= guard[1]))


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


def test_a_split_kit_comes_with_a_fit_test_to_print_first(tmp_path):
    folder, spec = split_kit(tmp_path)
    fit_test = load_stl(folder / "fit-test.stl")
    assert closed(fit_test)
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(fit_test)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    assert connectivity.GetNumberOfExtractedRegions() == 6, "three pairs of blocks"
    assert "fit-test.stl" in (folder / "PRINT-AND-ASSEMBLE.txt").read_text()
    # One part, nothing to connect, no test print.
    paths = export_print_kit(tmp_path, box(), (180, 180, 180), supports=False)
    assert not (paths[0].parent / "fit-test.stl").exists()


def test_the_kit_says_how_far_its_surface_moved_from_the_scan(tmp_path):
    _, spec = split_kit(tmp_path)
    change = spec["surface_change_from_the_scan"]
    assert change["median_mm"] < 0.02
    assert change["p95_mm"] < 0.1

