"""The print kit: parts that fit the printer, join by plug and socket, come on rods, and spare the working section."""

import json

import numpy as np
import pytest
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from mandiplan.printing import (
    FIT_TEST_BASE_MM,
    FIT_TEST_CLEARANCES_MM,
    PLUG_LENGTHS_MM,
    PLUG_TAPER,
    PRINTERS,
    SOCKET_EXTRA_DEPTH_MM,
    export_print_kit,
)


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
    # Flat faces meeting: the whole, but for the clearance round the plug.
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


def test_parts_meet_on_a_flat_face_round_a_long_plug(tmp_path):
    folder, spec = split_kit(tmp_path, clearance_mm=0.2)
    joint = spec["joins"][0]
    centre = np.array(joint["centre_mm"])
    along = np.array(joint["cut_normal"])
    first, second = _insides(folder, spec)
    assert joint["plug_length_mm"] == max(PLUG_LENGTHS_MM), "a block this straight takes the longest"

    # A flat face: either side of the plane, by the outside of the section, is one
    # part or the other, right across it.
    assert np.abs(along @ np.array([1.0, 0.0, 0.0])) > 0.99
    plane_x = centre[0]
    corner = np.array([plane_x, 13.5, 11.5])
    assert first(corner + along * 0.3) != first(corner - along * 0.3)
    assert second(corner + along * 0.3) != second(corner - along * 0.3)
    assert first(corner - along * 0.3) != second(corner - along * 0.3)

    # In the middle of the section past the face: the plug's part goes on, into the other.
    plug_side, socket_side = (first, second) if joint["plug_on"] == "lower-numbered part" else (second, first)
    into = along if joint["plug_on"] == "lower-numbered part" else -along
    middle = centre + into * (0.5 * joint["plug_length_mm"])
    assert plug_side(middle) and not socket_side(middle)
    wall = centre + into * (0.5 * joint["plug_length_mm"]) + np.array([0.0, 13.0, 0.0])
    assert socket_side(wall) and not plug_side(wall)


def _block_seam(clearance=0.2, plug_on_lower=True, surface=None):
    from mandiplan.printing import _coordinates, _make_seam, _Route, _solid_grid

    surface = surface or box()
    solid, origin = _solid_grid(surface, 0.3)
    index = np.nonzero(solid)
    points = _coordinates(index, origin, 0.3)
    s = np.arange(-55.0, 55.5, 1.0)
    route = _Route(np.column_stack([s, np.zeros(len(s)), np.zeros(len(s))]), s + 55.0)
    return _make_seam(route, (solid, origin, 0.3), points, points[:, 0] + 55.0, 40.0, clearance, plug_on_lower)


def test_a_plug_stands_in_its_socket_with_clearance_all_round():
    seam = _block_seam(0.2)
    assert seam is not None
    length = seam.length
    # Across the plug's section, along its long way: find its side from the field.
    at = seam.centre + seam.along * 0.5 * length
    arcs = np.full(1, 40.0)
    v = np.arange(0.0, 16.0, 0.01)
    samples = at + np.outer(v, seam.height)
    arcs = np.full(len(samples), 40.0)
    lower, upper = seam.lower_field(samples, arcs), seam.upper_field(samples, arcs)
    plug = lower < 0
    socket_wall = upper < 0
    assert plug[0] and not socket_wall[0], "the plug belongs to its part alone"
    edge_plug = v[plug].max()
    edge_wall = v[socket_wall].min()
    assert edge_wall - edge_plug == pytest.approx(0.2, abs=0.02), "with the clearance between them"
    # Past the plug's tip the socket goes on a little: nothing there, then the floor.
    tip = seam.centre + seam.along * (length + 0.3)
    floor = seam.centre + seam.along * (length + SOCKET_EXTRA_DEPTH_MM + 0.5)
    lower, upper = seam.lower_field(np.array([tip, floor]), np.full(2, 40.0)), seam.upper_field(np.array([tip, floor]), np.full(2, 40.0))
    assert lower[0] > 0 and upper[0] > 0, "the socket is deeper than the plug is long"
    assert lower[1] > 0 and upper[1] < 0
    # Behind the face it is simply the plug's own part.
    behind = seam.centre - seam.along * 3.0 + seam.height * 11.0
    assert seam.lower_field(behind[None], np.full(1, 40.0))[0] < 0 and seam.upper_field(behind[None], np.full(1, 40.0))[0] > 0


def test_the_plug_leans_in_towards_its_tip_and_finds_its_way():
    seam = _block_seam(0.15)
    v = np.arange(0.0, 16.0, 0.01)

    def side(t):
        samples = seam.centre + seam.along * t + np.outer(v, seam.height)
        return v[seam.lower_field(samples, np.full(len(v), 40.0)) < 0].max()

    base, middle, tip = side(1.0), side(0.5 * seam.length), side(seam.length - 0.05)
    assert base > middle > tip
    assert base - middle == pytest.approx(PLUG_TAPER * (0.5 * seam.length - 1.0), abs=0.03)
    assert middle - tip > PLUG_TAPER * 0.5 * seam.length, "and is chamfered at its tip"


def test_the_joint_is_the_jaws_own_section_far_bigger_than_any_peg():
    seam = _block_seam(0.15)
    # The block is 28 by 25: all of it but the wall.
    wall = seam.wall
    assert seam.area_mm2 == pytest.approx((28.0 - 2 * wall - 0.3) * (25.0 - 2 * wall - 0.3), rel=0.08)
    # A 4 mm peg has a section modulus of 6.3 mm3.
    assert seam.section_modulus_mm3 > 100.0 * 6.3
    assert seam.section_modulus_mm3 < seam.whole_section_modulus_mm3
    assert wall == 2.0, "a section this big keeps the thicker wall"


def test_a_wall_of_bone_stays_round_the_socket():
    seam = _block_seam(0.15)
    # At the mouth of the socket, by the short way of the section.
    v = np.arange(0.0, 16.0, 0.02)
    samples = seam.centre + seam.along * 1.0 + np.outer(v, seam.across)
    upper = seam.upper_field(samples, np.full(len(v), 40.0))
    hole_edge = v[upper > 0].max()
    half_width = float(np.abs(np.array([0, 14.0, 12.5]) @ np.abs(seam.across)))
    assert half_width - hole_edge >= seam.wall - 0.2


def test_a_thin_section_gets_the_thinner_wall():
    from mandiplan.printing import PLUG_WALLS_MM

    plate = vtk.vtkCubeSource()
    plate.SetXLength(110)
    plate.SetYLength(30)
    plate.SetZLength(8)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(plate.GetOutputPort())
    triangles.Update()
    seam = _block_seam(0.15, surface=triangles.GetOutput())
    assert seam is not None and seam.wall == min(PLUG_WALLS_MM)
    # The short way of it is across the 8 mm, the wall taken off each side.
    assert seam.area_mm2 == pytest.approx((30.0 - 3.3) * (8.0 - 3.3), rel=0.1)


def test_a_section_too_small_for_a_joint_takes_none():
    little = vtk.vtkCubeSource()
    little.SetXLength(110)
    little.SetYLength(5)
    little.SetZLength(5)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(little.GetOutputPort())
    triangles.Update()
    assert _block_seam(0.15, surface=triangles.GetOutput()) is None


def test_on_a_curving_jaw_the_axis_keeps_the_plug_in_the_bone():
    from mandiplan.geometry import cpr
    from mandiplan.geometry.spline import ArchCurve
    from mandiplan.printing import _coordinates, _make_seam, _Route, _solid_grid

    theta = np.radians(np.linspace(-80.0, 80.0, 9))
    frames = cpr.build_frames(ArchCurve(np.column_stack([45 * np.sin(theta), -45 * np.cos(theta), np.zeros(9)])), 0.5)
    jaw = _tube(frames, 7.0)
    solid, origin = _solid_grid(jaw, 0.3)
    points = _coordinates(np.nonzero(solid), origin, 0.3)
    stride = 2
    route = _Route(frames.points[::stride], frames.s[::stride])
    arcs = route.along(points)
    seam = _make_seam(route, (solid, origin, 0.3), points, arcs, float(0.5 * frames.length_mm), 0.15, True)
    assert seam is not None
    tangent = route.tangent(float(0.5 * frames.length_mm))
    chord = route.at(float(0.5 * frames.length_mm) + seam.length) - route.at(float(0.5 * frames.length_mm))
    chord = chord / np.linalg.norm(chord)
    assert seam.along @ chord > seam.along @ tangent - 0.02
    # Every point of the socket's depth, at its mouth's outline, is still in the bone.
    tube_axis = frames.points
    v = np.linspace(-6.0, 6.0, 49)
    vv, ww = np.meshgrid(v, v, indexing="ij")
    mouth = seam.inside_mouth(vv.ravel(), ww.ravel()) >= 0.0
    plane = seam.centre + vv.ravel()[mouth, None] * seam.height + ww.ravel()[mouth, None] * seam.across
    for t in (0.0, 0.5 * seam.length, seam.length):
        at = plane + t * seam.along
        nearest = np.min(np.linalg.norm(at[:, None, :] - tube_axis[None, :, :], axis=2), axis=1)
        assert nearest.max() < 7.0 - seam.wall + 0.3


def test_the_plug_is_on_the_part_further_from_the_working_section(tmp_path):
    _, spec = split_kit(tmp_path)
    joint = spec["joins"][0]
    # Working side is left, +x, the far end of the box's run: the plug is lower.
    assert joint["plug_on"] == "lower-numbered part"
    assert spec["parts"][1]["holds_resection"] and not spec["parts"][0]["holds_resection"]


def _tube(frames, radius):
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
    tube.SetRadius(radius)
    tube.SetNumberOfSides(32)
    tube.CappingOn()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(tube.GetOutputPort())
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


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


def test_the_resection_side_faces_up_with_rods_off_the_working_surface(tmp_path):
    from mandiplan.geometry import cpr
    from mandiplan.geometry.spline import ArchCurve
    from mandiplan.render import supports as rods

    # A curved jaw with its arch curve, so it has an outer face to keep clear.
    theta = np.radians(np.linspace(-80.0, 80.0, 9))
    frames = cpr.build_frames(ArchCurve(np.column_stack([50 * np.sin(theta), -50 * np.cos(theta), np.zeros(9)])), 0.5)
    jaw = _tube(frames, 9.0)
    site = frames.points[frames.index_of(0.25 * frames.length_mm)]
    paths = export_print_kit(
        tmp_path, jaw, PRINTERS["Original Prusa MINI+"], working_side="auto", cut_points=[site], frames=frames
    )
    folder = paths[0].parent
    spec = json.loads((folder / "assembly.json").read_text())
    part = spec["parts"][0]
    assert part["holds_resection"] and part["support_rods"] >= 3
    ready, model = load_stl(folder / part["ready_to_print_file"]), load_stl(folder / part["file"])
    assert model.GetBounds()[4] == pytest.approx(rods.lift_mm())
    assert ready.GetBounds()[4] == pytest.approx(0.0, abs=1e-5), "the raft is on the plate"
    assert ready.GetNumberOfCells() > model.GetNumberOfCells()
    # The part in the ready file is just where it is in the model file.
    for axis in (0, 1, 5):
        assert abs(ready.GetBounds()[axis]) >= abs(model.GetBounds()[axis]) - 3.0
    assert ready.GetBounds()[5] == pytest.approx(model.GetBounds()[5])

    matrix = np.array(part["world_to_print"])
    site_print = matrix[:3, :3] @ site + matrix[:3, 3]
    tips = np.array(part["rod_contacts_print_mm"])
    assert np.min(np.linalg.norm(tips - site_print, axis=1)) >= 8.0

    # The jaw's outer face at the resection looks up, away from the bed.
    centre = frames.points.mean(axis=0)
    outward = site - centre
    outward[2] = 0.0
    outward /= np.linalg.norm(outward)
    assert matrix[:3, :3][2] @ outward > 0.6

    # No rod on the outer face of the working stretch.
    back = np.linalg.inv(matrix)
    world = tips @ back[:3, :3].T + back[:3, 3]
    guard = spec["kept_whole_arc_mm"]
    nearest = np.argmin(np.linalg.norm(world[:, None] - frames.points[None], axis=2), axis=1)
    arcs = frames.s[nearest]
    rel = world - frames.points[nearest]
    rel[:, 2] = 0.0
    outside = frames.points[nearest] - centre
    outside[:, 2] = 0.0
    facing_out = np.einsum("ij,ij->i", rel, outside) > 0.0
    assert not np.any(facing_out & (arcs >= guard[0]) & (arcs <= guard[1]))


def test_every_part_comes_ready_to_print_within_the_bed(tmp_path):
    from mandiplan.render import supports as rods

    bed = np.array([84.0, 52.0, 76.0])
    folder, spec = split_kit(tmp_path, supports=True)
    assert len(spec["parts"]) == 2
    for part in spec["parts"]:
        ready = load_stl(folder / part["ready_to_print_file"])
        bounds = np.array(ready.GetBounds()).reshape(3, 2)
        assert np.all(bounds[:, 0] >= -1e-4) and np.all(bounds[:, 1] <= bed + 1e-4)
        assert bounds[2, 0] == pytest.approx(0.0, abs=1e-5)
        model = load_stl(folder / part["file"])
        assert model.GetBounds()[4] == pytest.approx(rods.lift_mm())
        assert closed(model)
        assert part["support_rods"] >= 3
        assert 0.0 < part["downward_surfaces_held_share"] <= 1.0
    assert "ready-to-print" in (folder / "PRINT-AND-ASSEMBLE.txt").read_text()


def test_no_rod_touches_a_joints_faces_or_goes_into_its_socket(tmp_path):
    folder, spec = split_kit(tmp_path, supports=True)
    joint = spec["joins"][0]
    centre, along = np.array(joint["centre_mm"]), np.array(joint["cut_normal"])
    for part in spec["parts"]:
        tips = np.array(part["rod_contacts_print_mm"])
        back = np.array(part["print_to_world"])
        world = tips @ back[:3, :3].T + back[:3, 3]
        past = (world - centre) @ along
        assert np.abs(past).min() > 1.0
        # In line with the socket means inside it, on the part that has it.
        across = (world - centre) - np.outer(past, along)
        inner = np.max(np.abs(across), axis=1) < 8.0
        into = along if joint["plug_on"] == "lower-numbered part" else -along
        depth = (world - centre) @ into
        assert not np.any(inner & (depth > 0.0) & (depth < joint["plug_length_mm"] + 1.0))


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
    # This bed only has room for the middle clearance.
    assert connectivity.GetNumberOfExtractedRegions() == 2, "a pair of pieces"
    bounds = np.array(fit_test.GetBounds()).reshape(3, 2)
    assert bounds[2, 0] == pytest.approx(0.0, abs=0.02), "they stand on the bed"
    assert np.all(bounds[:2, 1] <= np.array([84.0, 52.0]) + 1e-4)
    assert bounds[2, 1] < 12.0, "and are short"
    text = (folder / "PRINT-AND-ASSEMBLE.txt").read_text()
    assert "fit-test.stl" in text and "Joint clearance" in text
    assert spec["fit_test_clearances_mm"] == [FIT_TEST_CLEARANCES_MM[1]]
    # One part, nothing to connect, no test print.
    paths = export_print_kit(tmp_path, box(), (180, 180, 180), supports=False)
    assert not (paths[0].parent / "fit-test.stl").exists()


def test_the_fit_tests_pieces_have_the_real_section_at_each_clearance():
    from mandiplan.printing import _fit_test

    seam = _block_seam(0.15)
    mesh, tried = _fit_test(seam, np.array([180.0, 180.0, 180.0]))
    assert closed(mesh) and tried == FIT_TEST_CLEARANCES_MM
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(mesh)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    assert connectivity.GetNumberOfExtractedRegions() == 6, "three pairs of pieces"
    # Each pair: the plug's base and the socket's block have the section with its wall.
    mass = vtk.vtkMassProperties()
    mass.SetInputData(mesh)
    mass.Update()
    section = 28.0 * 25.0
    expected_each_plug = section * FIT_TEST_BASE_MM
    assert mass.GetVolume() > 3 * expected_each_plug


def test_a_printer_too_small_even_for_a_pair_just_goes_without_a_fit_test():
    from mandiplan.printing import _fit_test

    assert _fit_test(_block_seam(0.15), np.array([20.0, 20.0, 20.0])) == (None, ())


def test_the_asiga_printers_are_there_and_come_first():
    names = list(PRINTERS)
    assert names[0] == "Asiga MAX"
    assert PRINTERS["Asiga MAX"] == (119.0, 67.0, 75.0)
    assert "Asiga PRO 4K80" in names and "Asiga ULTRA 50" in names


def test_the_kit_says_how_far_its_surface_moved_from_the_scan(tmp_path):
    _, spec = split_kit(tmp_path)
    change = spec["surface_change_from_the_scan"]
    assert change["median_mm"] < 0.02
    assert change["p95_mm"] < 0.1

