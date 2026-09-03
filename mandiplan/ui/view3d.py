"""The 3-D bone view: isosurface, cut planes, plate overlay and picking.

VTK pipelines are built once in :meth:`View3D._build_pipeline` and afterwards
fed with ``SetInputData``; nothing here rebuilds a mapper per frame.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import vtk
import vtkmodules.qt

vtkmodules.qt.PyQtImpl = "PyQt6"

# VTK's widget defaults to a plain QWidget that paints itself directly into a
# native child window, using WA_PaintOnScreen and a null paint engine. Qt
# supports that on X11 only; on macOS it leaves the window unpainted, which
# looks like the application starting into no window at all. The QOpenGLWidget
# base lets Qt composite the widget normally. Override with
# MANDIPLAN_VTK_WIDGET=QWidget if a machine prefers the other one.
vtkmodules.qt.QVTKRWIBase = os.environ.get(
    "MANDIPLAN_VTK_WIDGET",
    "QOpenGLWidget" if sys.platform == "darwin" else "QWidget",
)

from PyQt6.QtCore import Qt, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402
from vtkmodules.qt.QVTKRenderWindowInteractor import (  # noqa: E402
    QVTKRenderWindowInteractor,
)

from ..geometry.plate import ribbon_mesh  # noqa: E402
from ..geometry.resection import resected_mask  # noqa: E402
from .convert_helpers import empty_polydata, points_to_polydata  # noqa: E402
from .modes import Mode  # noqa: E402

BONE_COLOUR = (0.93, 0.90, 0.83)
RESECT_COLOUR = (0.85, 0.25, 0.25)
PLATE_COLOUR = (0.55, 0.70, 0.95)
ARCH_COLOUR = (0.35, 0.85, 0.60)
MARK_COLOUR = (1.0, 0.75, 0.2)
GRAFT_COLOUR = (0.45, 0.85, 0.70)
BRIDGE_COLOUR = (0.95, 0.78, 0.45)


class _VtkWidget(QVTKRenderWindowInteractor):
    """VTK's interactor widget with its X11-only paint settings undone.

    Only applied on the QOpenGLWidget base, where Qt owns the framebuffer and
    has to be allowed to composite.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        if vtkmodules.qt.QVTKRWIBase == "QOpenGLWidget":
            self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, False)

    def paintEngine(self):  # noqa: N802
        if vtkmodules.qt.QVTKRWIBase == "QOpenGLWidget":
            return super(QVTKRenderWindowInteractor, self).paintEngine()
        return None


def _hide_widget_handles(rep: vtk.vtkImplicitPlaneRepresentation) -> None:
    """Render the plane's origin sphere and normal arrow invisible.

    The widget keeps working — the plane itself is still draggable and every
    callback stays wired — but nothing extraneous is drawn over the bone.
    Angulation is set from the resection panel, so the rotation handles have
    no job left. Visibility and opacity are turned off rather than the actors
    removed, so no widget logic is deleted.
    """
    rep.SetDrawOutline(False)
    for getter in ("GetNormalProperty", "GetSelectedNormalProperty"):
        prop = getattr(rep, getter)()
        prop.SetOpacity(0.0)
        prop.SetRepresentationToPoints()
    for getter in ("GetEdgesProperty", "GetOutlineProperty", "GetSelectedOutlineProperty"):
        getattr(rep, getter)().SetOpacity(0.0)
    rep.GetPlaneProperty().SetLineWidth(0.0)
    # The tube and cone geometry that make up the arrow, sized to nothing.
    rep.SetTubing(False)


class View3D(QWidget):
    """3-D view of the mandible with interactive resection planes."""

    surface_picked = pyqtSignal(object)  # np.ndarray world point
    plane_translated = pyqtSignal(int, object)  # index, origin (never the normal)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.mode = Mode.NAVIGATE
        self._plane_widgets: list[vtk.vtkImplicitPlaneWidget2] = []
        self._syncing = False
        self._measure_points: list[np.ndarray] = []

        self.interactor = _VtkWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.interactor)

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.10, 0.11, 0.14)
        self.interactor.GetRenderWindow().AddRenderer(self.renderer)
        self._build_pipeline()

        # VTK's default is vtkInteractorStyleSwitch, which carries a joystick
        # camera mode that keeps moving the camera from the cursor position
        # rather than from a click-drag delta, plus hidden "j"/"t" keys that
        # flip between the two. Pinning the trackball style gives plain
        # click-hold-drag-release rotation and removes the accidental toggle.
        self.camera_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self.camera_style)

        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.005)
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_left_button, 1.0)

        session.surface_changed.connect(self.refresh_surface)
        session.resection_changed.connect(self.refresh_resection)
        session.plate_changed.connect(self.refresh_plate)
        session.arch_changed.connect(self.refresh_arch)
        session.reconstruction_changed.connect(self.refresh_reconstruction)

    def start(self) -> None:
        self.interactor.Initialize()

    def shutdown(self) -> None:
        for widget in self._plane_widgets:
            widget.Off()
        self._plane_widgets.clear()
        self.interactor.GetRenderWindow().Finalize()

    # -- pipeline ----------------------------------------------------------

    def _build_pipeline(self) -> None:
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(2)
        lut.SetTableValue(0, *BONE_COLOUR, 1.0)
        lut.SetTableValue(1, *RESECT_COLOUR, 1.0)
        lut.SetTableRange(0, 1)
        lut.Build()

        self.bone_mapper = vtk.vtkPolyDataMapper()
        self.bone_mapper.SetLookupTable(lut)
        self.bone_mapper.SetScalarRange(0, 1)
        self.bone_mapper.ScalarVisibilityOff()
        self.bone_actor = vtk.vtkActor()
        self.bone_actor.SetMapper(self.bone_mapper)
        self.bone_actor.GetProperty().SetColor(*BONE_COLOUR)
        self.bone_actor.GetProperty().SetSpecular(0.15)
        self.renderer.AddActor(self.bone_actor)

        self.fragment_mapper = vtk.vtkPolyDataMapper()
        self.fragment_actor = vtk.vtkActor()
        self.fragment_actor.SetMapper(self.fragment_mapper)
        self.fragment_actor.GetProperty().SetColor(*RESECT_COLOUR)
        self.fragment_actor.GetProperty().SetOpacity(0.35)
        self.fragment_actor.VisibilityOff()
        self.renderer.AddActor(self.fragment_actor)

        self.arch_tube = vtk.vtkTubeFilter()
        self.arch_tube.SetRadius(0.5)
        self.arch_tube.SetNumberOfSides(12)
        self.arch_mapper = vtk.vtkPolyDataMapper()
        self.arch_mapper.SetInputConnection(self.arch_tube.GetOutputPort())
        self.arch_actor = vtk.vtkActor()
        self.arch_actor.SetMapper(self.arch_mapper)
        self.arch_actor.GetProperty().SetColor(*ARCH_COLOUR)
        self.renderer.AddActor(self.arch_actor)

        self.plate_mapper = vtk.vtkPolyDataMapper()
        self.plate_actor = vtk.vtkActor()
        self.plate_actor.SetMapper(self.plate_mapper)
        self.plate_actor.GetProperty().SetColor(*PLATE_COLOUR)
        self.renderer.AddActor(self.plate_actor)

        self.node_glyph = vtk.vtkGlyph3D()
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(1.1)
        sphere.SetThetaResolution(12)
        sphere.SetPhiResolution(12)
        self.node_glyph.SetSourceConnection(sphere.GetOutputPort())
        self.node_glyph.SetScaleModeToDataScalingOff()
        self.node_mapper = vtk.vtkPolyDataMapper()
        self.node_mapper.SetInputConnection(self.node_glyph.GetOutputPort())
        self.node_actor = vtk.vtkActor()
        self.node_actor.SetMapper(self.node_mapper)
        self.node_actor.GetProperty().SetColor(0.25, 0.45, 0.9)
        self.renderer.AddActor(self.node_actor)

        self.mark_glyph = vtk.vtkGlyph3D()
        mark = vtk.vtkSphereSource()
        mark.SetRadius(1.6)
        self.mark_glyph.SetSourceConnection(mark.GetOutputPort())
        self.mark_glyph.SetScaleModeToDataScalingOff()
        self.mark_mapper = vtk.vtkPolyDataMapper()
        self.mark_mapper.SetInputConnection(self.mark_glyph.GetOutputPort())
        self.mark_actor = vtk.vtkActor()
        self.mark_actor.SetMapper(self.mark_mapper)
        self.mark_actor.GetProperty().SetColor(*MARK_COLOUR)
        self.renderer.AddActor(self.mark_actor)

        self.graft_mapper = vtk.vtkPolyDataMapper()
        self.graft_actor = vtk.vtkActor()
        self.graft_actor.SetMapper(self.graft_mapper)
        self.graft_actor.GetProperty().SetColor(*GRAFT_COLOUR)
        self.graft_actor.GetProperty().SetOpacity(0.55)
        self.graft_actor.VisibilityOff()
        self.renderer.AddActor(self.graft_actor)

        self.bridge_mapper = vtk.vtkPolyDataMapper()
        self.bridge_actor = vtk.vtkActor()
        self.bridge_actor.SetMapper(self.bridge_mapper)
        self.bridge_actor.GetProperty().SetColor(*BRIDGE_COLOUR)
        self.bridge_actor.GetProperty().SetOpacity(0.55)
        self.bridge_actor.VisibilityOff()
        self.renderer.AddActor(self.bridge_actor)

        self.measure_mapper = vtk.vtkPolyDataMapper()
        self.measure_actor = vtk.vtkActor()
        self.measure_actor.SetMapper(self.measure_mapper)
        self.measure_actor.GetProperty().SetColor(1.0, 0.9, 0.3)
        self.measure_actor.GetProperty().SetLineWidth(3)
        self.renderer.AddActor(self.measure_actor)

        for mapper in (
            self.bone_mapper,
            self.fragment_mapper,
            self.plate_mapper,
            self.graft_mapper,
            self.bridge_mapper,
            self.measure_mapper,
        ):
            mapper.SetInputData(empty_polydata())
        self.arch_tube.SetInputData(empty_polydata())
        self.node_glyph.SetInputData(empty_polydata())
        self.mark_glyph.SetInputData(empty_polydata())

    # -- refresh -----------------------------------------------------------

    def refresh_surface(self) -> None:
        surface = self.session.surface
        if surface is None:
            self.bone_mapper.SetInputData(empty_polydata())
        else:
            self.bone_mapper.SetInputData(surface)
            self.renderer.ResetCamera()
        self._sync_plane_widgets()
        self.refresh_resection()

    def refresh_resection(self) -> None:
        session = self.session
        if session.cut_applied and session.retained_surface is not None:
            self.bone_mapper.SetInputData(session.retained_surface)
            self.bone_mapper.ScalarVisibilityOff()
            self.fragment_mapper.SetInputData(session.fragment_surface)
            self.fragment_actor.VisibilityOn()
        else:
            self.fragment_actor.VisibilityOff()
            if session.surface is not None:
                self.bone_mapper.SetInputData(session.surface)
                self._colour_preview()
        self.mark_glyph.SetInputData(
            points_to_polydata([p for _, p in session.landmarks])
        )
        self._sync_plane_widgets()
        self.render()

    def _colour_preview(self) -> None:
        """Tint the fragment that the current planes would remove."""
        surface = self.session.surface
        # Once a graft is in place the tint would hide it, and the graft itself
        # already shows what is being replaced.
        if surface is None or not self.session.planes or self.session.graft_surface:
            self.bone_mapper.ScalarVisibilityOff()
            return
        from vtk.util import numpy_support

        points = numpy_support.vtk_to_numpy(surface.GetPoints().GetData())
        mask = resected_mask(self.session.planes, points).astype(np.float32)
        array = numpy_support.numpy_to_vtk(mask, deep=True)
        array.SetName("resected")
        surface.GetPointData().SetScalars(array)
        self.bone_mapper.ScalarVisibilityOn()
        self.bone_mapper.SetScalarModeToUsePointData()
        self.bone_mapper.SetColorModeToMapScalars()

    def refresh_reconstruction(self) -> None:
        for surface, mapper, actor in (
            (self.session.graft_surface, self.graft_mapper, self.graft_actor),
            (self.session.bridge_surface, self.bridge_mapper, self.bridge_actor),
        ):
            if surface is None:
                actor.VisibilityOff()
            else:
                mapper.SetInputData(surface)
                actor.VisibilityOn()
        self.render()

    def refresh_arch(self) -> None:
        curve = self.session.arch_curve
        if curve is None:
            self.arch_tube.SetInputData(empty_polydata())
        else:
            from ..render.convert import polyline_to_polydata

            self.arch_tube.SetInputData(polyline_to_polydata(curve.dense_points))
        self.render()

    def refresh_plate(self) -> None:
        plan = self.session.plate_plan
        if plan is None:
            self.plate_mapper.SetInputData(empty_polydata())
            self.node_glyph.SetInputData(points_to_polydata(self.session.plate_points))
        else:
            from ..render.convert import triangles_to_polydata

            points, triangles = ribbon_mesh(
                plan, self.session.plate.width_mm, self.session.plate.thickness_mm
            )
            self.plate_mapper.SetInputData(triangles_to_polydata(points, triangles))
            self.node_glyph.SetInputData(points_to_polydata(plan.nodes))
        self.render()

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode
        if mode not in (Mode.MEASURE, Mode.ANGLE):
            self.clear_measurement()

    def clear_measurement(self) -> None:
        self._measure_points.clear()
        self.measure_mapper.SetInputData(empty_polydata())
        self.render()

    def render(self) -> None:
        self.interactor.GetRenderWindow().Render()

    def reset_camera(self) -> None:
        self.renderer.ResetCamera()
        self.render()

    def set_view_direction(self, direction: str) -> None:
        camera = self.renderer.GetActiveCamera()
        focal = np.array(camera.GetFocalPoint())
        distance = camera.GetDistance()
        vectors = {
            "anterior": ((0, -1, 0), (0, 0, 1)),
            "left": ((1, 0, 0), (0, 0, 1)),
            "right": ((-1, 0, 0), (0, 0, 1)),
            "superior": ((0, 0, 1), (0, -1, 0)),
        }
        offset, up = vectors[direction]
        camera.SetPosition(*(focal + np.array(offset, dtype=float) * distance))
        camera.SetViewUp(*up)
        self.renderer.ResetCameraClippingRange()
        self.render()

    # -- plane widgets -----------------------------------------------------

    def _sync_plane_widgets(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            while len(self._plane_widgets) > len(self.session.planes):
                widget = self._plane_widgets.pop()
                widget.Off()
                widget.SetInteractor(None)
            bounds = (
                self.session.surface.GetBounds()
                if self.session.surface is not None
                else (-50, 50, -50, 50, -50, 50)
            )
            for index, plane in enumerate(self.session.planes):
                if index >= len(self._plane_widgets):
                    self._plane_widgets.append(self._make_plane_widget(index, bounds))
                rep = self._plane_widgets[index].GetRepresentation()
                rep.SetOrigin(*(float(v) for v in plane.origin))
                rep.SetNormal(*(float(v) for v in plane.normal))
                self._plane_widgets[index].SetEnabled(not self.session.cut_applied)
        finally:
            self._syncing = False

    def _make_plane_widget(self, index: int, bounds) -> vtk.vtkImplicitPlaneWidget2:
        rep = vtk.vtkImplicitPlaneRepresentation()
        rep.SetPlaceFactor(1.1)
        rep.PlaceWidget(bounds)
        rep.SetDrawPlane(True)
        rep.SetDrawOutline(False)
        rep.OutlineTranslationOff()
        rep.ScaleEnabledOff()
        rep.GetPlaneProperty().SetOpacity(0.35)
        rep.GetPlaneProperty().SetColor(*RESECT_COLOUR)
        _hide_widget_handles(rep)

        widget = vtk.vtkImplicitPlaneWidget2()
        widget.SetInteractor(self.interactor)
        widget.SetRepresentation(rep)
        widget.AddObserver(
            "InteractionEvent", lambda obj, evt, i=index: self._on_plane_moved(i)
        )
        widget.AddObserver(
            "EndInteractionEvent", lambda obj, evt, i=index: self._on_plane_moved(i)
        )
        widget.On()
        return widget

    def _on_plane_moved(self, index: int) -> None:
        if self._syncing or index >= len(self._plane_widgets):
            return
        rep = self._plane_widgets[index].GetRepresentation()
        # Deliberately does not read GetNormal(): dragging a plane along the jaw
        # translates it, and must not re-angle it. Angulation is set explicitly.
        origin = np.array(rep.GetOrigin(), dtype=float)
        self._syncing = True
        try:
            self.plane_translated.emit(index, origin)
        finally:
            self._syncing = False

    # -- picking -----------------------------------------------------------

    def _on_left_button(self, obj, event) -> None:
        if self.mode == Mode.NAVIGATE:
            return
        x, y = self.interactor.GetEventPosition()
        if not self.picker.Pick(x, y, 0, self.renderer):
            return
        actor = self.picker.GetActor()
        if actor not in (self.bone_actor, self.fragment_actor):
            return
        point = np.array(self.picker.GetPickPosition(), dtype=float)
        if self.mode in (Mode.MEASURE, Mode.ANGLE):
            self._add_measure_point(point)
        self.surface_picked.emit(point)

    def _add_measure_point(self, point: np.ndarray) -> None:
        limit = 3 if self.mode == Mode.ANGLE else 2
        if len(self._measure_points) >= limit:
            self._measure_points.clear()
        self._measure_points.append(point)
        if len(self._measure_points) >= 2:
            from ..render.convert import polyline_to_polydata

            self.measure_mapper.SetInputData(
                polyline_to_polydata(np.vstack(self._measure_points))
            )
        else:
            self.measure_mapper.SetInputData(points_to_polydata(self._measure_points))
        self.render()

    @property
    def measure_points(self) -> list[np.ndarray]:
        return list(self._measure_points)
