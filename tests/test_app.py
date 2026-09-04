"""End-to-end test of the application itself.

Loads the phantom through the real DICOM path, draws an arch curve, flattens
the volume, measures, resects, draws a plate path and exports both files —
the whole workflow the README describes, without a human.
"""

from __future__ import annotations

import numpy as np
import pytest
import vtk

from helpers import rel_error
from make_phantom import PhantomSpec, make_phantom, write_dicom_series

from mandiplan.constants import DISCLAIMER
from mandiplan.exporting import write_bend_csv, write_template_stl
from mandiplan.ui.modes import Mode


@pytest.fixture(scope="module")
def phantom_folder(tmp_path_factory):
    spec = PhantomSpec()
    folder = tmp_path_factory.mktemp("app_dicom")
    write_dicom_series(make_phantom(spec), folder)
    return spec, folder


@pytest.fixture(scope="module")
def window(qt_app, phantom_folder):
    from mandiplan.ui.main_window import MainWindow

    spec, folder = phantom_folder
    win = MainWindow()
    win.show()
    win.start()
    win.load_folder(str(folder))

    win.set_mode(Mode.ARCH)
    for point in spec.centre_line(9, extend_deg=12.0):
        win.session.add_arch_seed([point[0], point[1], spec.centre_z])
    yield win
    win.close()


def test_volume_loads_with_its_real_voxel_spacing(window, phantom_folder):
    spec, _ = phantom_folder
    volume = window.session.volume
    assert volume is not None
    assert np.allclose(volume.spacing, spec.spacing)
    assert "0.300 × 0.300 × 0.600 mm" in window.volume_panel.info.text()


def test_threshold_is_seeded_and_adjustable(window, phantom_folder):
    spec, _ = phantom_folder
    session = window.session
    seeded = session.threshold
    assert spec.air_value < seeded < spec.bone_value

    before = session.surface.GetNumberOfPoints()
    window.volume_panel.threshold_panel.set_value(seeded + 300.0)
    assert session.threshold == pytest.approx(seeded + 300.0)
    assert session.surface.GetNumberOfPoints() != before
    window.volume_panel.threshold_panel.set_value(spec.half_max_value)


def test_panoramic_view_is_built_on_millimetre_axes(window):
    session = window.session
    assert session.panoramic is not None
    assert session.panoramic.pixel_mm == pytest.approx(session.cpr.step_mm)
    assert "arc length" in session.panoramic.x_label
    assert window.panoramic_view.has_image()
    assert window.cross_view.has_image()


def test_cross_section_stepper_moves_along_the_curve(window):
    window._set_cross_section(10.0)
    assert window.session.cross_section_s == pytest.approx(10.0)
    window._set_cross_section(1e6)  # clamped to the end of the curve
    assert window.session.cross_section_s == pytest.approx(
        window.session.frames.length_mm
    )
    window._set_cross_section(window.session.frames.length_mm / 2)


def test_panoramic_measurement_reports_arc_length_honestly(window):
    window.set_mode(Mode.MEASURE)
    window._on_reformat_pick("panoramic", 20.0, -4.0)
    window._on_reformat_pick("panoramic", 65.0, 2.0)
    text = window.measure_label.text()
    assert "arc length along arch curve 45.00 mm" in text
    assert "not a straight line" in text
    # A transient status message must not hide the result.
    window.statusBar().showMessage("something happened")
    assert window.measure_label.isVisible()
    window.statusBar().clearMessage()
    window.set_mode(Mode.NAVIGATE)


def test_angle_measurement_reports_degrees(window):
    window.set_mode(Mode.ANGLE)
    window._on_reformat_pick("panoramic", 10.0, 0.0)
    window._on_reformat_pick("panoramic", 0.0, 0.0)
    window._on_reformat_pick("panoramic", 0.0, 10.0)
    assert "90.0°" in window.measure_label.text()
    window.set_mode(Mode.NAVIGATE)


def test_measuring_the_bone_in_the_cross_section(window, phantom_folder):
    """The height of the phantom measured through the UI, in millimetres."""
    spec, _ = phantom_folder
    window._set_cross_section(window.session.frames.length_mm / 2)
    cross = window.session.cross_section
    column = int(round(cross.mm_to_col(0.0)))
    profile = cross.image[:, column]
    level = 0.5 * (np.percentile(profile, 5) + np.percentile(profile, 95))
    rows = np.where(profile >= level)[0]
    height = (cross.row_to_mm(rows[-1]) - cross.row_to_mm(rows[0]))
    assert rel_error(height, 2 * spec.semi_axis_si_mm) < 0.02


def test_resection_preview_cut_and_readout(window, phantom_folder):
    spec, _ = phantom_folder
    session = window.session
    session.clear_planes()
    window.add_cut_plane()
    window.add_cut_plane()
    assert len(session.planes) == 2

    report = session.report
    assert np.isfinite(report.arc_length_mm)
    assert report.straight_length_mm < report.arc_length_mm

    session.execute_cut()
    assert session.cut_applied
    expected = np.pi * spec.semi_axis_bl_mm * spec.semi_axis_si_mm * report.arc_length_mm
    assert rel_error(session.report.fragment_volume_mm3, expected) < 0.03
    assert "Resected segment (arc length" in window.resection_panel.readout.text()

    session.undo_cut()
    assert not session.cut_applied


def test_the_workflow_bar_tracks_the_toolbox_and_the_plan(window):
    window.toolbox.setCurrentIndex(1)
    assert "Step 2 of 5" in window.workflow_bar.heading.text()
    # The bar and the panels are two views of the same position.
    window.workflow_bar.step_selected.emit(3)
    assert window.toolbox.currentIndex() == 3
    assert "Step 4 of 5" in window.workflow_bar.heading.text()
    window.toolbox.setCurrentIndex(0)
    assert "Done — Volume loaded" in window.workflow_bar.detail.text()


def test_placing_a_cut_by_numbers_from_the_panel(window):
    session = window.session
    panel = window.resection_panel
    session.clear_planes()
    window.add_cut_plane()
    window.add_cut_plane()
    assert panel.plane_box.count() == 2

    panel.plane_box.setCurrentIndex(1)
    target = session.frames.length_mm / 2.0 + 20.0
    panel.position.setValue(target)
    assert session.plane_arc_position(1) == pytest.approx(target, abs=0.3)

    # Angulation is measured against the curve at the cut, which is what the
    # yaw control is relative to.
    panel.yaw.setValue(20.0)
    tangent = session.frames.tangents[session.frames.index_of(target)]
    after = session.planes[1].normal
    turned = np.degrees(np.arccos(abs(np.clip(np.dot(tangent, after), -1, 1))))
    assert turned == pytest.approx(20.0, abs=0.5)
    assert after[2] == pytest.approx(0.0, abs=1e-6)

    panel.tilt.setValue(15.0)
    assert abs(session.planes[1].normal[2]) == pytest.approx(
        np.sin(np.radians(15.0)), abs=1e-3
    )

    # Moving the cut afterwards carries the angulation with it, measured
    # against the jaw: the world normal follows the curve, the obliquity
    # relative to the local arch does not change.
    def obliquity() -> float:
        tangent_here, _, _ = session.frames.frame_at(session.plane_arc_position(1))
        return float(
            np.degrees(
                np.arccos(
                    abs(np.clip(np.dot(tangent_here, session.planes[1].normal), -1, 1))
                )
            )
        )

    # Measured with both yaw and tilt applied, so it is the full obliquity.
    reference = obliquity()
    angled = session.planes[1].normal.copy()
    origin_before = session.planes[1].origin.copy()
    panel.position.setValue(target - 8.0)
    assert obliquity() == pytest.approx(reference, abs=0.5)
    assert session.placement(1).yaw_deg == pytest.approx(20.0)
    assert session.placement(1).tilt_deg == pytest.approx(15.0)
    assert not np.allclose(session.planes[1].normal, angled)
    assert not np.allclose(session.planes[1].origin, origin_before)

    panel.yaw.setValue(0.0)
    panel.tilt.setValue(0.0)


def test_a_third_cutting_plane_is_refused(window):
    session = window.session
    messages = []
    session.message.connect(messages.append)
    window.add_cut_plane()
    session.message.disconnect(messages.append)
    assert len(session.planes) == 2
    assert any("cutting planes" in m for m in messages)


def test_margin_points_report_signed_distances(window, phantom_folder):
    spec, _ = phantom_folder
    session = window.session
    session.clear_landmarks()
    session.add_landmark(spec.point_at_arc(spec.arc_length_mm / 2.0), "lesion")
    assert len(session.report.margins_mm) == 2
    assert "lesion" in window.resection_panel.readout.text()


def test_mirror_reconstruction_through_the_interface(window, phantom_folder):
    spec, _ = phantom_folder
    session = window.session
    session.clear_planes()

    # A lateral defect: entirely on one side, so mirroring covers all of it.
    mid = session.frames.length_mm / 2.0
    for s, sign in ((mid + 12.0, +1.0), (mid + 34.0, -1.0)):
        index = session.frames.index_of(s)
        session.add_plane(
            session.frames.points[index], sign * session.frames.tangents[index]
        )

    session.estimate_symmetry_plane()
    plane = session.symmetry_plane
    assert plane is not None
    assert abs(plane.point[0] - spec.centre_xy[0]) < 1.0
    assert plane.symmetry > 0.9
    assert "Symmetry score" in window.reconstruction_panel.plane_info.text()

    assert session.coverage is not None
    assert not session.coverage.crosses_midline

    session.build_graft()
    assert session.graft_surface is not None
    assert session.graft_volume_mm3 > 0
    # The graft should be about the size of the fragment it replaces.
    session.execute_cut()
    assert rel_error(session.graft_volume_mm3, session.report.fragment_volume_mm3) < 0.1
    session.undo_cut()
    assert "Mirrored graft volume" in window.reconstruction_panel.coverage_info.text()


def test_a_defect_across_the_midline_is_reported_as_partly_unmirrorable(window):
    session = window.session
    session.clear_planes()
    mid = session.frames.length_mm / 2.0
    for s, sign in ((mid - 10.0, +1.0), (mid + 25.0, -1.0)):
        index = session.frames.index_of(s)
        session.add_plane(
            session.frames.points[index], sign * session.frames.tangents[index]
        )
    if session.symmetry_plane is None:
        session.estimate_symmetry_plane()
    session.update_reconstruction()

    coverage = session.coverage
    assert coverage.crosses_midline
    assert coverage.uncovered_mm > 15.0
    assert "no healthy counterpart" in window.reconstruction_panel.coverage_info.text()

    session.build_graft()
    assert session.bridge_surface is not None
    assert session.bridge_surface.GetNumberOfPoints() > 0


def _buccal_path(spec, count: int = 12, half_span_deg: float = 60.0):
    """Points just outside the phantom's buccal cortex, as a user would click."""
    radius = spec.radius_mm + spec.semi_axis_bl_mm + 0.5
    centre = np.radians(spec.arch_centre_deg)
    half = np.radians(half_span_deg)
    for theta in np.linspace(centre - half, centre + half, count):
        yield [radius * np.cos(theta), radius * np.sin(theta), spec.centre_z]


def test_plate_path_bend_table_and_exports(window, phantom_folder, tmp_path):
    spec, _ = phantom_folder
    session = window.session
    session.clear_plate_path()
    window.set_mode(Mode.PLATE)
    for point in _buccal_path(spec):
        session.add_plate_point(point)

    plan = session.plate_plan
    assert plan is not None
    assert len(plan.nodes) >= 8
    assert window.plate_panel.table.rowCount() == len(plan.nodes)
    # Nodes sit one screw-hole pitch apart *along the path*, so the straight
    # segment between two of them is a chord: never longer, and here within 1%.
    assert np.all(plan.segment_lengths_mm <= plan.pitch_mm + 1e-9)
    assert rel_error(float(plan.segment_lengths_mm.mean()), plan.pitch_mm) < 0.01

    csv_path = write_bend_csv(
        tmp_path / "bends.csv", plan, session.plate.width_mm, session.plate.thickness_mm
    )
    text = csv_path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == f"# {DISCLAIMER}"
    assert "in_plane_bend_deg" in text
    assert len([ln for ln in text.splitlines() if not ln.startswith("#")]) == len(
        plan.nodes
    ) + 1

    stl_path = write_template_stl(
        tmp_path / "template.stl", plan, session.plate.width_mm, session.plate.thickness_mm
    )
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()
    mesh = reader.GetOutput()
    assert mesh.GetNumberOfPolys() > 0

    mass = vtk.vtkMassProperties()
    mass.SetInputData(mesh)
    mass.Update()
    expected = session.plate.width_mm * session.plate.thickness_mm * plan.total_length_mm
    assert rel_error(mass.GetVolume(), expected) < 0.05
    window.set_mode(Mode.NAVIGATE)


def test_changing_the_pitch_changes_the_table(window):
    session = window.session
    nodes_at_9mm = len(session.plate_plan.nodes)
    session.set_plate_settings(pitch_mm=5.0)
    assert len(session.plate_plan.nodes) > nodes_at_9mm
    lengths = session.plate_plan.segment_lengths_mm
    assert np.all(lengths <= 5.0 + 1e-9)
    assert rel_error(float(lengths.mean()), 5.0) < 0.01
    session.set_plate_settings(pitch_mm=9.0)


def test_choosing_a_plate_system_drives_the_pitch_and_the_fit(window, tmp_path):
    from mandiplan.exporting import write_steps_csv

    session = window.session
    panel = window.plate_panel
    index = panel.system_box.findData("recon-2.7-bar")
    panel.system_box.setCurrentIndex(index)

    assert session.plate_system.id == "recon-2.7-bar"
    assert session.plate.pitch_mm == pytest.approx(session.plate_system.hole_pitch_mm)
    assert session.plate_plan.pitch_mm == pytest.approx(session.plate_system.hole_pitch_mm)
    assert panel.pitch.value() == pytest.approx(session.plate_system.hole_pitch_mm)

    fit = session.fit
    assert fit is not None
    assert fit.option is not None
    assert fit.option.length_mm >= session.plate_plan.total_length_mm
    assert "-hole" in panel.fit_info.text()

    assert session.steps
    assert panel.steps_table.rowCount() == len(session.steps)
    assert session.steps[0].kind == "cut"

    path = write_steps_csv(
        tmp_path / "steps.csv",
        session.steps,
        session.plate_system,
        session.bending_kit,
        fit,
    )
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == f"# {DISCLAIMER}"
    assert "distance_from_cut_end_mm" in text
    assert session.plate_system.name in text

    panel.system_box.setCurrentIndex(panel.system_box.findData("recon-2.4-bar"))


def test_changing_the_bending_kit_changes_the_instructions(window):
    session = window.session
    panel = window.plate_panel
    panel.kit_box.setCurrentIndex(panel.kit_box.findData("bar-bending-press"))
    assert session.bending_kit.id == "bar-bending-press"
    bends = [s for s in session.steps if s.kind == "bend"]
    assert bends
    assert all(s.instrument == "bending press" for s in bends)
    panel.kit_box.setCurrentIndex(panel.kit_box.findData("bending-irons"))


def test_a_custom_plate_needs_no_bench_steps(window):
    session = window.session
    panel = window.plate_panel
    session.clear_planes()  # judge the plate on its own, with no defect to span
    panel.system_box.setCurrentIndex(panel.system_box.findData("custom-psi"))
    assert session.fit.fits
    assert "Custom plate" in panel.fit_info.text()
    assert len(session.steps) == 1
    assert "no bending steps" in session.steps[0].text
    panel.system_box.setCurrentIndex(panel.system_box.findData("recon-2.4-bar"))


def test_a_plate_that_stops_short_of_the_defect_is_refused(window, phantom_folder):
    """Screw purchase is checked, not just length."""
    spec, _ = phantom_folder
    session = window.session
    panel = window.plate_panel
    session.clear_plate_path()
    # A short path that runs out before it reaches retained bone distally.
    for point in _buccal_path(spec, count=8, half_span_deg=25.0):
        session.add_plate_point(point)
    mid = session.frames.length_mm / 2.0
    for s, sign in ((mid + 4.0, +1.0), (mid + 30.0, -1.0)):
        index = session.frames.index_of(s)
        session.add_plane(
            session.frames.points[index], sign * session.frames.tangents[index]
        )

    fit = session.fit
    assert not fit.fits
    assert any("retained bone" in problem for problem in fit.problems)
    assert "Problem:" in panel.fit_info.text()

    session.clear_planes()
    session.clear_plate_path()
    for point in _buccal_path(spec):
        session.add_plate_point(point)


def test_undo_restores_the_previous_plate_path(window):
    session = window.session
    before = len(session.plate_points)
    session.add_plate_point([38.0, 0.0, 0.0])
    assert len(session.plate_points) == before + 1
    session.undo()
    assert len(session.plate_points) == before


def test_refusing_a_bad_series_reports_it(qt_app, tmp_path):
    from mandiplan.ui.main_window import MainWindow

    window = MainWindow()
    empty = tmp_path / "not_dicom"
    empty.mkdir()
    messages = []
    window.show_message = messages.append
    from mandiplan.dicom_io import DicomLoadError

    with pytest.raises(DicomLoadError):
        window.session.load_dicom_folder(str(empty))
    window.close()


def test_the_plane_widget_draws_no_handles(window):
    """Nothing extraneous is rendered, but the widget itself still functions."""
    session = window.session
    session.clear_planes()
    window.add_cut_plane()
    widget = window.view3d._plane_widgets[0]
    rep = widget.GetRepresentation()

    for getter in (
        "GetNormalProperty",
        "GetSelectedNormalProperty",
        "GetEdgesProperty",
        "GetOutlineProperty",
        "GetSelectedOutlineProperty",
    ):
        assert getattr(rep, getter)().GetOpacity() == 0.0, getter
    assert not rep.GetDrawOutline()

    # The widget logic is intact: the plane still draws and still responds.
    assert rep.GetDrawPlane()
    assert widget.GetEnabled()
    before = session.placement(0)
    moved = np.asarray(rep.GetOrigin(), dtype=float) + np.array([2.0, 0.0, 0.0])
    window.view3d.plane_translated.emit(0, moved)
    assert np.allclose(session.planes[0].origin, moved)
    # A drag is a translation: the angular offsets are carried, not cleared,
    # and the normal is re-derived from the frame where the cut landed.
    after = session.placement(0)
    assert (after.yaw_deg, after.tilt_deg, after.roll_deg) == (
        before.yaw_deg,
        before.tilt_deg,
        before.roll_deg,
    )


# -- camera interaction ------------------------------------------------------


def _camera(window) -> np.ndarray:
    return np.array(window.view3d.renderer.GetActiveCamera().GetPosition())


def _drag(window, hold: bool) -> float:
    """Move the pointer across the 3-D view, with or without the button down."""
    iren = window.view3d.interactor._Iren
    before = _camera(window)
    iren.SetEventPosition(300, 200)
    if hold:
        iren.LeftButtonPressEvent()
    for x in (320, 350, 380):
        iren.SetEventPosition(x, 210)
        iren.MouseMoveEvent()
    if hold:
        iren.LeftButtonReleaseEvent()
    return float(np.linalg.norm(_camera(window) - before))


def test_the_camera_style_is_trackball_not_the_switch(window):
    """vtkInteractorStyleSwitch carries a joystick mode and a hidden j/t toggle."""
    style = window.view3d.interactor._Iren.GetInteractorStyle()
    assert isinstance(style, vtk.vtkInteractorStyleTrackballCamera)
    assert not isinstance(style, vtk.vtkInteractorStyleSwitch)


def test_rotation_needs_the_button_held(window):
    window.set_mode(Mode.NAVIGATE)
    assert _drag(window, hold=True) > 1e-3
    assert _drag(window, hold=False) == pytest.approx(0.0, abs=1e-9)


def test_rotation_stops_on_release(window):
    _drag(window, hold=True)
    assert _drag(window, hold=False) == pytest.approx(0.0, abs=1e-9)


def test_the_hidden_joystick_key_no_longer_switches_modes(window):
    iren = window.view3d.interactor._Iren
    iren.SetKeyEventInformation(0, 0, "j", 0, "j")
    iren.CharEvent()
    assert isinstance(
        iren.GetInteractorStyle(), vtk.vtkInteractorStyleTrackballCamera
    )
    assert _drag(window, hold=False) == pytest.approx(0.0, abs=1e-9)
