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


def test_the_banner_is_permanent(window):
    assert window.banner.text() == DISCLAIMER
    window.statusBar().showMessage("a transient message")
    assert window.banner.isVisible()
    assert window.banner.text() == DISCLAIMER
    window.statusBar().clearMessage()
    assert window.banner.text() == DISCLAIMER


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


def test_plate_path_bend_table_and_exports(window, tmp_path):
    session = window.session
    session.clear_plate_path()
    window.set_mode(Mode.PLATE)
    for theta in np.linspace(np.radians(-60), np.radians(60), 12):
        session.add_plate_point([38.5 * np.cos(theta), 38.5 * np.sin(theta), 0.0])

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
