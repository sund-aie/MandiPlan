"""End-to-end test of the application itself.

Loads the phantom through the real DICOM path, draws an arch curve, flattens
the volume, measures, resects, draws a plate path and exports both files —
the whole workflow the README describes, without a human.
"""

from __future__ import annotations

import numpy as np
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy
import vtk

from helpers import rel_error
from make_phantom import PhantomSpec, make_phantom, write_dicom_series

from mandiplan.constants import DISCLAIMER
from mandiplan.exporting import write_bend_csv, write_bending_guide, write_template_stl
from mandiplan.plate_assets import load_assets
from mandiplan.plate_catalog import load_kits
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
    assert session.panoramic.pixel_mm == pytest.approx(
        max(session.cpr.step_mm, float(np.min(session.volume.spacing)))
    )
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


def test_the_step_headers_carry_each_steps_state(window):
    """One list of steps: the page headers, ticked when done, with no second
    step bar repeating them."""
    toolbox = window.toolbox
    assert toolbox.count() == 7
    assert toolbox.itemText(0).startswith("1 · Scan and mandible")
    assert "\u2713" in toolbox.itemText(0)
    assert "\u2713" in toolbox.itemText(1)
    assert "Drawn by hand" in toolbox.itemToolTip(1)
    assert "\u2713" not in toolbox.itemText(5)
    assert not hasattr(window, "workflow_bar")


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
    assert "mirrors onto bone" in window.reconstruction_panel.plane_info.text()

    assert session.coverage is not None
    assert not session.coverage.crosses_midline

    session.build_graft()
    assert session.graft_surface is not None
    assert session.graft_volume_mm3 > 0
    # The graft should be about the size of the fragment it replaces.
    session.execute_cut()
    assert rel_error(session.graft_volume_mm3, session.report.fragment_volume_mm3) < 0.1
    session.undo_cut()
    text = window.reconstruction_panel.coverage_info.text()
    assert "Mirrored segment volume" in text
    assert "Junction 1" in text and "Junction 2" in text
    # The reconstruction is one piece that replaces the bone in the view.
    rebuilt = session.reconstructed_surface
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(rebuilt)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.Update()
    assert connectivity.GetNumberOfExtractedRegions() == 1
    assert window.view3d.graft_actor.GetVisibility()
    assert not window.view3d.bone_actor.GetVisibility()
    # Moving a cut makes the reconstruction stale; it is dropped, not kept.
    session.clear_planes()
    assert session.reconstruction is None
    assert not window.view3d.graft_actor.GetVisibility()


def test_the_reconstruction_can_be_refined_by_hand_and_exported(window, tmp_path):
    from mandiplan.exporting import write_surface_stl

    session = window.session
    session.clear_planes()
    mid = session.frames.length_mm / 2.0
    for s, sign in ((mid + 12.0, +1.0), (mid + 34.0, -1.0)):
        index = session.frames.index_of(s)
        session.add_plane(session.frames.points[index], sign * session.frames.tangents[index])
    session.build_reconstruction()
    panel = window.reconstruction_panel
    assert panel.refine_box.isEnabled()
    assert "As computed" in panel.edit_info.text()

    window.set_mode(Mode.SCULPT)
    assert window.view3d.interactor.GetInteractorStyle() is window.view3d.sculpt_style
    assert panel.sculpt_button.isChecked()

    surface = session.reconstructed_surface
    before = vtk_to_numpy(surface.GetPoints().GetData()).copy()
    target = before[np.argmax(before[:, 2])]  # a point on the jaw's upper border
    session.brush = "fill"
    session.sculpt(target)
    after = vtk_to_numpy(surface.GetPoints().GetData())
    assert np.max(np.linalg.norm(after - before, axis=1)) > 0.01
    assert "Edited by hand" in panel.edit_info.text()

    # The export is the edited surface.
    path = write_surface_stl(tmp_path / "jaw.stl", surface)
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(path))
    reader.Update()
    exported = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
    assert np.min(np.linalg.norm(exported - after[np.argmax(np.linalg.norm(after - before, axis=1))], axis=1)) < 1e-3

    session.smooth_junctions()
    session.undo_edit()
    session.reset_edits()
    assert np.allclose(vtk_to_numpy(surface.GetPoints().GetData()), before)
    assert "As computed" in panel.edit_info.text()

    # The plate is planned on the reconstructed jaw, across the rebuilt part.
    assert session.plate_projector.polydata is surface
    window.set_mode(Mode.NAVIGATE)
    assert window.view3d.interactor.GetInteractorStyle() is window.view3d.camera_style
    session.clear_planes()


def test_a_fibula_graft_is_planned_from_a_ct_of_the_legs(window, tmp_path):
    from make_leg_phantom import LegSpec, make_leg_phantom

    from mandiplan.constants import DISCLAIMER
    from mandiplan.exporting import write_fibula_guide, write_fibula_pieces
    from mandiplan.workflow import workflow_status

    session = window.session
    view = window.view3d
    panel = window.fibula_panel
    session.clear_planes()
    mid = session.frames.length_mm / 2.0
    for s, sign in ((mid + 5.0, +1.0), (mid + 35.0, -1.0)):
        index = session.frames.index_of(s)
        session.add_plane(session.frames.points[index], sign * session.frames.tangents[index])
    session.build_reconstruction()
    graft = session.fibula
    assert workflow_status(session)[5].skipped, "optional until a leg CT is opened"
    assert "\u2713" not in window.toolbox.itemText(5)

    graft.set_scan(make_leg_phantom(LegSpec(side="both", spacing=(0.8, 0.8, 1.0))))
    assert sorted(f.side for f in graft.fibulae) == ["left", "right"]
    graft.set_setting("min_piece_mm", 12.0)
    plan = graft.plan
    assert plan is not None, graft.problem
    # A tight little jaw: the 30 mm defect takes two pieces to follow it.
    assert len(plan.pieces) == 2 and plan.deviation_mm <= 2.0
    assert workflow_status(session)[5].done and "\u2713" in window.toolbox.itemText(5)
    # The pieces are shown in the jaw, the rebuilt shape see-through round them.
    assert view.fibula_pieces_actor.GetVisibility()
    assert view.fibula_pieces_mapper.GetInput().GetNumberOfPoints() > 100
    assert view.graft_actor.GetProperty().GetOpacity() < 1.0
    assert panel.table.rowCount() == 2
    assert "pieces along" in panel.summary.text()

    # The leg view: the fibula with its cuts, the jaw put away, and back.
    panel.leg_view.setChecked(True)
    assert view.leg_bone_actor.GetVisibility() and view.leg_cuts_actor.GetVisibility()
    assert not view.graft_actor.GetVisibility() and not view.fibula_pieces_actor.GetVisibility()
    panel.leg_view.setChecked(False)
    assert not view.leg_bone_actor.GetVisibility()
    assert view.graft_actor.GetVisibility() and view.fibula_pieces_actor.GetVisibility()

    # The cutting guide: made for the plan, followed in use, shown on the leg, written out.
    panel.guide_button.click()
    assert graft.guide_check is not None and graft.guide_check.ok, graft.guide_check.lines()
    assert "the guide works" in panel.guide_report.text()
    panel.leg_view.setChecked(True)
    assert view.leg_guide_actor.GetVisibility()
    panel.leg_view.setChecked(False)
    stl, table = write_fibula_guide(tmp_path / "guide.stl", graft)
    assert stl.stat().st_size > 10_000
    text = table.read_text()
    assert DISCLAIMER in text and "lower cut (mm up the fibula)" in text
    assert len([line for line in text.splitlines() if not line.startswith("#")]) == 1 + len(plan.pieces)
    assert write_fibula_pieces(tmp_path / "pieces.stl", graft).stat().st_size > 10_000
    # A change to the plan drops the guide: it was made for the plan as it was.
    graft.set_setting("roll_deg", 30.0)
    assert graft.guide is None and graft.guide_check is None
    graft.set_setting("roll_deg", 0.0)

    # The other leg gives the other fibula; the plan follows.
    other = "left" if graft.side == "right" else "right"
    graft.choose_side(other)
    assert graft.plan is not None and graft.fibula.side == other

    # Without the cuts there is nothing to plan for.
    session.clear_planes()
    assert graft.plan is None and "cuts" in graft.problem
    assert not view.fibula_pieces_actor.GetVisibility()
    graft.set_scan(None)


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

    session.build_reconstruction()
    report = session.reconstruction.report
    assert report.donorless_voxels > 0
    assert session.reconstructed_surface.GetNumberOfPoints() > 0
    assert "no mirror donor" in window.reconstruction_panel.coverage_info.text()


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
    assert window.plate_panel.table.rowCount() == len(session.fitted_hole_plan.nodes)
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

    # The keys are worked out for the plate as finally seated.
    guide = session.bending_guide()
    assert guide is not None
    np.testing.assert_allclose(guide.centres, session.bent_plate.hole_centres)
    assert len(guide.joints) == len(session.bent_plate.hole_centres) - 1
    assert any(joint.angle_deg > 0.5 for joint in guide.joints) and guide.keys
    guide_stl, guide_table = write_bending_guide(tmp_path / "guide.stl", guide, session.plate_asset)
    assert guide_stl.stat().st_size > 10_000
    assert "lies flat on the key" in guide_table.read_text(encoding="utf-8")
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


def test_choosing_a_plate_drives_the_pitch_and_the_fit(window, tmp_path):
    """The plate chosen sets the pitch the path is laid out at, and the fit."""
    from mandiplan.exporting import write_steps_csv

    session = window.session
    panel = window.plate_panel
    panel.family_box.setCurrentIndex(panel.family_box.findData("generic-recon-2.8"))

    asset = session.plate_asset
    assert asset.family == "generic-recon-2.8"
    assert session.plate_system.hole_pitch_mm == pytest.approx(asset.hole_pitch_mm)
    assert session.plate.pitch_mm == pytest.approx(asset.hole_pitch_mm)
    assert session.plate_plan.pitch_mm == pytest.approx(asset.hole_pitch_mm)
    # The lengths on offer are the lengths the library has in this family.
    assert set(session.plate_system.hole_counts) == {11, 17}

    fit = session.fit
    assert fit is not None
    assert fit.option is not None
    assert fit.option.length_mm >= session.plate_plan.total_length_mm
    assert "-hole" in panel.fit_info.text()
    assert panel.verdict.text()
    # A new family's length follows the path: the shortest that spans it,
    # and the summary names the plate actually used, not the path's spacing.
    assert not session.plate_length_chosen
    assert asset.hole_count == fit.option.holes
    assert f"{asset.hole_count}-hole plate" in panel.summary.text()

    assert session.steps
    assert panel.steps_table.rowCount() == len(session.steps)
    trimmed = session.active_plate_asset.hole_count < session.plate_asset.hole_count
    assert session.steps[0].kind == ("cut" if trimmed else "check")
    assert all(step.distance_origin == "hole 1" for step in session.steps)
    assert panel.steps_table.horizontalHeaderItem(2).text() == "from hole 1"

    path = write_steps_csv(
        tmp_path / "steps.csv",
        session.steps,
        session.plate_system,
        session.bending_kit,
        fit,
    )
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == f"# {DISCLAIMER}"
    assert "distance_from_hole_1_mm" in text
    assert session.plate_system.name in text

    panel.family_box.setCurrentIndex(panel.family_box.findData("generic-recon-2.4-lp"))


def test_the_panel_follows_a_plate_chosen_elsewhere(window):
    """Choosing a plate from code (or 'use the length that fits') shows in the panel."""
    session = window.session
    panel = window.plate_panel
    session.set_plate_asset("generic-recon-2.4-lock-16h")
    assert panel.family_box.currentData() == "generic-recon-2.4-lock"
    assert panel.model_box.currentData() == "generic-recon-2.4-lock-16h"
    session.set_plate_asset("generic-recon-2.4-lp-12h")
    assert panel.model_box.currentData() == "generic-recon-2.4-lp-12h"


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
    asset_id = session.plate_asset.id
    session.set_plate_system("custom-psi")
    assert session.fit.fits
    assert "Custom plate" in panel.fit_info.text()
    assert len(session.steps) == 1
    assert "no bending steps" in session.steps[0].text
    session.set_plate_asset(asset_id)


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
    # In a corner, clear of the bone and of any cut plane's widget, which
    # would take the drag for itself.
    iren.SetEventPosition(12, 12)
    if hold:
        iren.LeftButtonPressEvent()
    for x in (32, 62, 92):
        iren.SetEventPosition(x, 22)
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


def test_the_viewport_draws_the_fitted_plate(window, phantom_folder):
    """The plate actor shows the fitted plate's own geometry, not nothing.

    A mapper that is given input data after being connected to a filter drops
    the connection without complaint; the plate then vanishes from the view
    while every geometry test still passes.
    """
    session = window.session
    frames = session.frames
    session.clear_plate_path()
    for s_mm in np.linspace(frames.length_mm * 0.2, frames.length_mm * 0.8, 10):
        _, _, buccal = frames.frame_at(float(s_mm))
        session.add_plate_point(frames.point_at(float(s_mm)) + buccal * 12.0)
    placed = session.plate_mesh()
    assert placed is not None
    view = window.view3d
    view.refresh_plate()
    mapper = view.plate_mapper
    mapper.Update()
    drawn = mapper.GetInput()
    assert drawn is not None and drawn.GetNumberOfCells() == len(placed.triangles)
    assert view.plate_actor.GetVisibility()
    bounds = np.array(drawn.GetBounds()).reshape(3, 2)
    assert np.allclose(bounds[:, 0], placed.points.min(axis=0), atol=1e-4)
    assert np.allclose(bounds[:, 1], placed.points.max(axis=0), atol=1e-4)


def test_the_mandible_is_separated_on_opening_and_can_be_exported(window):
    session = window.session
    session.ensure_mandible()
    assert session.mandible is not None
    assert session.mandible_source in ("auto", "arch")
    window.export_panel.refresh()
    assert window.export_panel.buttons["mandible"].isEnabled()
    text = window.volume_panel.mandible_info.text()
    assert "cm³ of bone" in text
    assert ("along your curve" in text) == (session.mandible_source == "arch")


def test_dragging_the_view_never_adds_plate_points(window):
    """Reported: after drawing the plate path, going to Export and dragging
    the model to look at it added path points and moved the plate."""
    session, view = window.session, window.view3d
    iren = view.interactor
    window.toolbox.setCurrentIndex(4)
    window.set_mode(Mode.PLATE)
    view.reset_camera()
    renderer = view.renderer
    point = session.frames.point_at(session.frames.length_mm / 2.0)
    renderer.SetWorldPoint(*point, 1.0)
    renderer.WorldToDisplay()
    x, y, _ = (int(round(v)) for v in renderer.GetDisplayPoint())

    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest

    # Real mouse events on the widget, as a hand makes them: VTK counts y up.
    ratio = iren.devicePixelRatioF()
    at = QPoint(int(x / ratio), int(iren.height() - 1 - y / ratio))
    left = Qt.MouseButton.LeftButton

    def press_drag_release(d):
        QTest.mousePress(iren, left, Qt.KeyboardModifier.NoModifier, at)
        if d:
            QTest.mouseMove(iren, at + QPoint(d // 2, d // 2))
            QTest.mouseMove(iren, at + QPoint(d, d))
        QTest.mouseRelease(iren, left, Qt.KeyboardModifier.NoModifier, at + QPoint(d, d))

    before = len(session.plate_points)
    press_drag_release(40)  # a drag turns the view
    assert len(session.plate_points) == before
    view.reset_camera()
    press_drag_release(0)  # a click still draws
    assert len(session.plate_points) == before + 1
    session.remove_last_plate_point()

    # Moving on to Export puts the plate tool down.
    window.toolbox.setCurrentIndex(6)
    assert window.mode == Mode.NAVIGATE
    press_drag_release(0)
    assert len(session.plate_points) == before


def test_plates_are_chosen_by_brand_then_shape_then_length(window):
    """Brand, Plate (shape) and Length (holes) are three choices in that order."""
    panel = window.plate_panel
    brands = [panel.brand_box.itemText(i) for i in range(panel.brand_box.count())]
    assert brands[0] == "Generic" and "KLS Martin LevelOne" in brands
    assert "DePuy Synthes MatrixMANDIBLE" in brands
    panel.brand_box.setCurrentIndex(panel.brand_box.findData("KLS Martin LevelOne"))
    shapes = [panel.family_box.itemText(i) for i in range(panel.family_box.count())]
    assert window.session.plate_asset.brand == "KLS Martin LevelOne"
    # The catalogue's range: miniplates to double-angled reconstruction plates.
    assert any(s.startswith("Miniplate") for s in shapes)
    wanted = "Reconstruction plate 2.7, angled 6 + 20, 56\u00b0, left, 3.0 mm profile"
    assert wanted in shapes
    panel.family_box.setCurrentIndex(shapes.index(wanted))
    assert window.session.plate_asset.hole_count == 26
    lengths = [panel.model_box.itemText(i) for i in range(panel.model_box.count())]
    assert lengths and lengths[0].startswith("26 holes")
    assert window.session.plate_asset.colour != window.session.plate_asset.__class__.colour
    # A plate chosen elsewhere brings its brand with it.
    window.session.set_plate_asset("generic-recon-2.4-lock-16h")
    panel.refresh_library()
    assert panel.brand_box.currentData() == "Generic"
    assert panel.family_box.currentData() == "generic-recon-2.4-lock"


# -- the plate places itself -------------------------------------------------


def _fresh_plate_session(window):
    session = window.session
    session.clear_plate_path()
    session.plate_auto = False
    session.plate_length_chosen = False
    session.ensure_mandible()
    return session


def test_choosing_a_plate_is_all_it_takes(window):
    session = _fresh_plate_session(window)
    panel = window.plate_panel
    assert not session.plate_points or session.frames is not None
    session.set_plate_asset("generic-recon-2.4-lock-10h", length_chosen=True)
    assert session.plate_auto and session.plate_plan is not None and session.plate_mesh() is not None
    verdict = session.plate_verdict
    assert verdict is not None and verdict.holes
    assert len(verdict.holes) == len(session.plate_mesh().hole_centres)
    assert panel.sign.text() and verdict.headline in panel.sign.text()
    assert panel.screws_table.rowCount() == len(verdict.holes)
    # Its holes are on the bone: nothing here was clicked.
    projector = session.plate_projector
    for centre in session.plate_mesh().hole_centres:
        point, _ = projector.project(centre)
        assert np.linalg.norm(point - centre) < 5.0


def test_drawing_by_hand_takes_over_from_the_automatic_plate(window):
    session = _fresh_plate_session(window)
    session.set_plate_asset("generic-recon-2.4-lock-10h", length_chosen=True)
    automatic = len(session.plate_points)
    assert session.plate_auto and automatic >= 2
    session.add_plate_point(session.plate_points[0])
    assert not session.plate_auto and len(session.plate_points) == 1


def test_a_plate_that_will_not_do_shows_a_red_sign(window):
    session = _fresh_plate_session(window)
    panel = window.plate_panel
    session.clear_planes()
    # Cut out most of the jaw: the shortest plate cannot span it.
    frames = session.frames
    for s, flip in ((0.25 * frames.length_mm, False), (0.75 * frames.length_mm, True)):
        i = frames.index_of(s)
        session.add_plane(frames.points[i], (-1.0 if flip else 1.0) * frames.tangents[i])
    try:
        session.set_plate_asset("generic-miniplate-2.0-4h", length_chosen=True)
        verdict = session.plate_verdict
        assert verdict is not None and verdict.level == "bad" and verdict.headline == "Not applicable"
        assert "#c62828" in panel.sign.styleSheet()
        assert window.view3d.verdict_actor.GetVisibility()
        assert "Not applicable" in window.view3d.verdict_actor.GetInput()
        assert verdict.suggested_holes is None or not panel.use_suggested.isHidden()
    finally:
        session.clear_planes()


def test_the_bends_can_be_rehearsed_on_the_keys_from_a_fresh_plate(window):
    session = _fresh_plate_session(window)
    panel = window.plate_panel
    session.set_bending_kit("synthes-329-142")
    session.set_plate_asset("generic-recon-2.4-lock-10h", length_chosen=True)
    panel.guide_toggle.setChecked(True)
    assert session.guide_view_on
    assert not window.view3d.bone_actor.GetVisibility()
    assert "fresh plate" in panel.guide_text.text()
    assert not window.view3d.arch_actor.GetVisibility()
    assert all(not widget.GetEnabled() for widget in window.view3d._plane_widgets)
    assert not panel.guide_back.isEnabled()
    solid, head, fresh, _ = session.guide_preview()
    assert solid is None and head is None, "no key yet: the plate as it comes"
    assert not window.view3d.guide_actor.GetVisibility()
    assert window.view3d.guide_plate_actor.GetVisibility()
    # Each key in turn: a check tray on its own, a press key in the bender's head.
    seen = set()
    for _ in range(session.guide_step_count - 2):
        panel.guide_next.click()
        key = session._guide_steps[session.guide_step].key
        solid, head, on_key, _ = session.guide_preview()
        assert solid is not None and solid.GetNumberOfCells() > 100
        assert window.view3d.guide_actor.GetVisibility()
        assert "Key " in panel.guide_text.text()
        if key.check:
            assert head is None and not window.view3d.guide_tool_actor.GetVisibility()
        else:
            assert head is not None and window.view3d.guide_tool_actor.GetVisibility()
        seen.add(key.check)
    assert False in seen, "at least one key is pressed"
    for _ in range(session.guide_step_count):
        panel.guide_next.click()
    assert "All the bends are made" in panel.guide_text.text()
    assert not panel.guide_next.isEnabled()
    solid, _, finished, _ = session.guide_preview()
    assert solid is None
    # The plate came from straight to the shape it has on the jaw: the same plate,
    # wherever each is. Between any two of its points the distances agree.
    placed = session.plate_mesh().points
    take = np.linspace(0, len(placed) - 1, 60).astype(int)
    made = np.linalg.norm(finished[take][:, None] - finished[take][None, :], axis=2)
    wanted = np.linalg.norm(placed[take][:, None] - placed[take][None, :], axis=2)
    assert np.abs(made - wanted).max() < 2.0
    was = np.linalg.norm(fresh[take][:, None] - fresh[take][None, :], axis=2)
    assert np.abs(was - wanted).max() > 1.0
    panel.guide_toggle.setChecked(False)
    assert not session.guide_view_on
    assert window.view3d.bone_actor.GetVisibility() and not window.view3d.guide_actor.GetVisibility()


def _with_a_cut(session, at=0.3, wide=20.0):
    """A resection a good way along the jaw."""
    session.clear_planes()
    frames = session.frames
    for s, flip in ((at * frames.length_mm, False), (at * frames.length_mm + wide, True)):
        i = frames.index_of(s)
        session.add_plane(frames.points[i], (-1.0 if flip else 1.0) * frames.tangents[i])
    return session._defect_span()


def test_a_long_plate_runs_from_the_neck_and_its_excess_is_cut_off(window):
    session = _fresh_plate_session(window)
    panel = window.plate_panel
    defect = _with_a_cut(session)
    try:
        assert defect is not None
        stock = max(
            (a for a in load_assets() if a.family == "generic-recon-2.4-lp"), key=lambda a: a.hole_count
        )
        session.set_plate_asset(stock.id, length_chosen=True)
        active = session.active_plate_asset
        verdict = session.plate_verdict
        assert session.plate_course == "neck" and session.plate_auto
        placement = session.plate_placement
        assert placement.holes_after >= 3
        route = session.plate_route
        assert min(placement.arcs_mm[0], placement.arcs_mm[-1]) == pytest.approx(route.s[0], abs=0.5) or (
            max(placement.arcs_mm[0], placement.arcs_mm[-1]) == pytest.approx(route.s[-1], abs=0.5)
        )
        if active.hole_count < stock.hole_count:
            assert any("Cut off" in note for note in verdict.notes)
            assert panel.retained.value() == active.hole_count
            assert "Cut off" in panel.sign.text()
            assert session.steps[0].kind == "cut"
        assert any("holes past the cut" in note for note in verdict.notes)

        # The operator's own count stands.
        keep = stock.hole_count - 1
        panel.retained.setValue(keep)
        assert session.active_plate_asset.hole_count == keep
        session._replace_plate()
        assert session.active_plate_asset.hole_count == keep

        # And centred on the resection is an option.
        session.set_plate_asset(stock.id, length_chosen=True)
        panel.course_box.setCurrentIndex(panel.course_box.findData("centred"))
        assert session.plate_course == "centred"
        assert session.active_plate_asset.hole_count == stock.hole_count
        assert not any("holes past the cut" in note for note in session.plate_verdict.notes)
    finally:
        session.set_plate_course("neck")
        session.clear_planes()


def test_a_plate_that_cannot_reach_from_the_neck_says_so(window):
    session = _fresh_plate_session(window)
    _with_a_cut(session, at=0.45)
    try:
        session.set_plate_asset("generic-recon-2.4-lock-10h", length_chosen=True)
        assert any("centred on the resection" in note for note in session.plate_verdict.notes)
        assert session.active_plate_asset.hole_count == 10
    finally:
        session.clear_planes()


def test_the_bender_can_be_measured_and_the_keys_follow(window):
    session = _fresh_plate_session(window)
    session.set_plate_asset("generic-recon-2.4-lock-10h", length_chosen=True)
    panel = window.export_panel
    session.set_bending_kit("synthes-329-142")
    assert panel.head_spins["lobe_spacing_mm"].isEnabled()
    assert panel.head_spins["lobe_spacing_mm"].value() == pytest.approx(32.0)
    panel.head_spins["lobe_spacing_mm"].setValue(36.0)
    guide = session.bending_guide()
    assert guide.head.lobe_spacing_mm == pytest.approx(36.0) and "measured" in guide.head.source
    session.set_bending_kit("bending-irons")
    assert session.guide_head is None
    assert not panel.head_spins["lobe_spacing_mm"].isEnabled()
    session.set_bending_kit("synthes-329-142")


def test_pin_play_and_extra_overbend_reach_the_keys(window):
    session = _fresh_plate_session(window)
    session.set_plate_asset("generic-recon-2.4-lock-10h", length_chosen=True)
    session.set_bending_kit("synthes-329-142")
    before = session.bending_guide()
    session.set_guide_setting("guide_pin_play_mm", 0.2)
    session.set_guide_setting("guide_extra_overbend_deg", 2.0)
    try:
        after = session.bending_guide()
        assert after.pin_diameter_mm == pytest.approx(before.pin_diameter_mm - 0.2)
        pressed = [k for k in before.keys if not k.check]
        if pressed:
            after_key = next(k for k in after.keys if k.name == pressed[0].name)
            assert after_key.stop_angle_deg == pytest.approx(pressed[0].stop_angle_deg + 2.0, abs=0.05)
    finally:
        session.set_guide_setting("guide_pin_play_mm", 0.1)
        session.set_guide_setting("guide_extra_overbend_deg", 0.0)


def test_a_kit_with_no_key_keeps_its_plain_bench_steps(window):
    session = _fresh_plate_session(window)
    session.set_plate_asset("generic-recon-2.4-lock-10h", length_chosen=True)
    wire = next(kit for kit in load_kits() if kit.head.kind == "none")
    session.set_bending_kit(wire.id)
    try:
        assert session.steps
    finally:
        session.set_bending_kit("synthes-329-142")


def test_rehearsing_with_a_kit_that_has_no_keys_says_why(window):
    session = _fresh_plate_session(window)
    panel = window.plate_panel
    session.set_plate_asset("generic-recon-2.4-lock-10h", length_chosen=True)
    wire = next(kit for kit in load_kits() if kit.head.kind == "none")
    session.set_bending_kit(wire.id)
    said = []
    session.message.connect(said.append)
    try:
        panel.guide_toggle.setChecked(True)
        assert not session.guide_view_on and not panel.guide_toggle.isChecked()
        assert any("by hand" in text for text in said)
    finally:
        session.message.disconnect(said.append)
        session.set_bending_kit("synthes-329-142")


def test_a_cut_plane_is_a_square_just_covering_the_bone_it_cuts(window):
    from mandiplan.ui.view3d import PLANE_HALF_SIZE_MM, PLANE_MARGIN_MM

    session = _fresh_plate_session(window)
    view = window.view3d
    session.clear_planes()
    frames = session.frames
    i = frames.index_of(0.5 * frames.length_mm)
    session.add_plane(frames.points[i], frames.tangents[i])
    try:
        plane = session.planes[0]
        normal = np.asarray(plane.normal) / np.linalg.norm(plane.normal)
        source = view._plane_source(0)
        o, p1, p2 = (np.array(v, dtype=float) for v in (source.GetOrigin(), source.GetPoint1(), source.GetPoint2()))
        u, w = p1 - o, p2 - o
        assert np.linalg.norm(u) == pytest.approx(np.linalg.norm(w)) and abs(u @ w) < 1e-6, "a square"
        half = 0.5 * np.linalg.norm(u)
        u, w = u / np.linalg.norm(u), w / np.linalg.norm(w)
        centre = o + half * (u + w)
        assert abs((centre - np.asarray(plane.origin)) @ normal) < 1e-6, "in the cut's own plane"
        # Upright with the jaw: an edge runs up it.
        _, up, _ = frames.frame_at(float(frames.s[i]))
        assert max(abs(u @ up), abs(w @ up)) > 0.99
        outline = view._plane_section(np.asarray(plane.origin, dtype=float), normal)
        assert len(outline) > 20
        rel = outline - centre
        reach = np.maximum(np.abs(rel @ u), np.abs(rel @ w))
        assert reach.max() <= half + 1e-6, "it covers the bone it cuts"
        assert half <= max(reach.max() + PLANE_MARGIN_MM, PLANE_HALF_SIZE_MM) + 0.01, "and no more"
        # Out in the air it is its small self.
        _, _, _, small = view._plane_square(np.asarray(plane.origin) + 500.0, normal)
        assert small == PLANE_HALF_SIZE_MM
    finally:
        session.clear_planes()


def test_an_executed_cut_leaves_the_view_and_is_not_moved_by_a_touch(window):
    session = _fresh_plate_session(window)
    view = window.view3d
    session.clear_planes()
    frames = session.frames
    for s, flip in ((0.3 * frames.length_mm, False), (0.3 * frames.length_mm + 20.0, True)):
        k = frames.index_of(s)
        session.add_plane(frames.points[k], (-1.0 if flip else 1.0) * frames.tangents[k])
    try:
        assert all(widget.GetEnabled() for widget in view._plane_widgets)
        session.execute_cut()
        assert session.cut_applied
        assert not any(widget.GetEnabled() for widget in view._plane_widgets), "gone once the cut is made"
        before = session.placement(0).s_mm
        rep = view._plane_widgets[0].GetRepresentation()
        rep.SetOrigin(*(np.array(rep.GetOrigin()) + 15.0 * np.asarray(session.planes[0].normal)))
        view._on_plane_moved(0)
        assert session.cut_applied and session.placement(0).s_mm == pytest.approx(before)
        session.undo_cut()
        assert all(widget.GetEnabled() for widget in view._plane_widgets), "back to adjust after an undo"
    finally:
        session.clear_planes()
