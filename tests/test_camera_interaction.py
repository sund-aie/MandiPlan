"""Rotating the 3-D view is click-hold-drag, and nothing else."""

from __future__ import annotations

import numpy as np
import pytest
import vtk


@pytest.fixture(scope="module")
def view(qt_app):
    from mandiplan.ui.session import Session
    from mandiplan.ui.view3d import View3D

    widget = View3D(Session())
    widget.resize(600, 400)
    widget.show()
    widget.start()
    source = vtk.vtkSphereSource()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    widget.renderer.AddActor(actor)
    widget.renderer.ResetCamera()
    widget.render()
    yield widget
    widget.shutdown()


def _camera(view) -> np.ndarray:
    return np.array(view.renderer.GetActiveCamera().GetPosition())


def _drag(view, hold: bool) -> float:
    """Move the pointer across the view, with or without the button down."""
    iren = view.interactor._Iren
    before = _camera(view)
    iren.SetEventPosition(300, 200)
    if hold:
        iren.LeftButtonPressEvent()
    for x in (320, 350, 380):
        iren.SetEventPosition(x, 210)
        iren.MouseMoveEvent()
    if hold:
        iren.LeftButtonReleaseEvent()
    return float(np.linalg.norm(_camera(view) - before))


def test_the_camera_style_is_trackball_not_the_switch(view):
    """vtkInteractorStyleSwitch carries a joystick mode and a hidden toggle."""
    style = view.interactor._Iren.GetInteractorStyle()
    assert isinstance(style, vtk.vtkInteractorStyleTrackballCamera)
    assert not isinstance(style, vtk.vtkInteractorStyleSwitch)


def test_dragging_with_the_button_held_rotates(view):
    assert _drag(view, hold=True) > 1e-3


def test_moving_without_the_button_does_nothing(view):
    assert _drag(view, hold=False) == pytest.approx(0.0, abs=1e-9)


def test_rotation_stops_when_the_button_is_released(view):
    _drag(view, hold=True)
    assert _drag(view, hold=False) == pytest.approx(0.0, abs=1e-9)


def test_the_hidden_joystick_key_no_longer_switches_modes(view):
    """'j' toggles joystick mode on the switch style; it must do nothing here."""
    iren = view.interactor._Iren
    iren.SetKeyEventInformation(0, 0, "j", 0, "j")
    iren.CharEvent()
    assert isinstance(
        iren.GetInteractorStyle(), vtk.vtkInteractorStyleTrackballCamera
    )
    assert _drag(view, hold=False) == pytest.approx(0.0, abs=1e-9)
