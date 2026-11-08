"""Exercise printer selection and export of the intended model through the UI."""
from types import SimpleNamespace

import numpy as np
import vtk
from PyQt6.QtWidgets import QFileDialog

from mandiplan.geometry.resection import CutPlane
from mandiplan.ui.print_setup import PrintSetupDialog, protection_points, working_boundary


def sphere(radius):
    shape = vtk.vtkSphereSource()
    shape.SetRadius(radius)
    shape.SetPhiResolution(24); shape.SetThetaResolution(32)
    shape.Update()
    return shape.GetOutput()


def test_work_site_includes_planned_fragment_not_just_plane_origins():
    # An oblique cut's affected bone extends laterally beyond its centre.
    session = SimpleNamespace(surface=sphere(12), planes=[CutPlane([2,0,0], [1,1,0])],
                              landmarks=[], frames=None, cut_applied=False)
    pts = protection_points(session)
    assert pts[:,0].max() > 11.5
    assert pts[:,0].min() < 1.0


def test_tilted_midline_protects_entire_anatomical_working_side():
    from vtkmodules.util.numpy_support import vtk_to_numpy
    mesh = sphere(12)
    normal = np.array([1., .6, .3]); normal /= np.linalg.norm(normal)
    session = SimpleNamespace(symmetry_plane=SimpleNamespace(point=np.zeros(3), normal=normal))
    vertices = vtk_to_numpy(mesh.GetPoints().GetData())
    side, edge = working_boundary(session, mesh, 'auto', np.array([[8., 1., 2.]]))
    protected = vertices[vertices @ normal >= 0]
    assert side == 'left' and edge < -3
    assert np.all(protected[:, 0] >= edge)
    side, edge = working_boundary(session, mesh, 'right', np.empty((0, 3)))
    assert side == 'right' and np.all(vertices[vertices @ normal <= 0, 0] <= edge)


def test_print_dialog_exports_selected_model_and_dimensions(qt_app, tmp_path, monkeypatch):
    from mandiplan.ui.main_window import MainWindow
    from mandiplan import printing

    window = MainWindow()
    seen = {}
    original = sphere(12)
    rebuilt = sphere(10)
    sess = SimpleNamespace(mandible=True, surface=original, reconstructed_surface=rebuilt,
                           ensure_mandible=lambda: None, landmarks=[("marker", [4.,1.,2.])],
                           planes=[], symmetry_plane=None, note_export=lambda x: seen.update(export=x))
    binding = SimpleNamespace(session=sess, show_message=lambda x: seen.update(message=x),
                              set_busy=lambda x: seen.update(busy=x))
    # QWidget parenting requires a real window; supply the test session to it.
    window.session = sess
    window.set_busy = binding.set_busy
    window.show_message = binding.show_message
    def accept_setup(dialog):
        assert dialog.model.currentData() == 'reconstructed'
        dialog.printer.setCurrentIndex(dialog.printer.count()-1)
        for dim, value in zip(dialog.dims, (88.,90.,110.)):
            assert dim.isEnabled()
            dim.setValue(value)
        dialog.side.setCurrentIndex(dialog.side.findData('left'))
        dialog.supports.setChecked(False)
        dialog.clearance.setValue(.45)
        return 1
    def capture(folder, surface, dims, **options):
        assert folder == str(tmp_path) and surface is rebuilt
        assert dims == (88.,90.,110.)
        seen.update(options)
        return [tmp_path / "kit" / "piece.stl"]
    monkeypatch.setattr(PrintSetupDialog, 'exec', accept_setup)
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *_: str(tmp_path))
    monkeypatch.setattr(printing, 'export_print_kit', capture)
    try:
        window.export_print_kit()
        assert seen['export'] == 'print kit' and not seen['busy']
        assert seen['working_side'] == 'left' and not seen['supports']
        assert seen['clearance_mm'] == .45
        np.testing.assert_allclose(seen['cut_points'], [[4.,1.,2.]])
    finally:
        window.close()
