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

from PyQt6.QtCore import Qt, QTimer, pyqtSignal  # noqa: E402
from PyQt6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402
from vtkmodules.qt.QVTKRenderWindowInteractor import (  # noqa: E402
    QVTKRenderWindowInteractor,
)

from ..geometry.resection import resected_mask  # noqa: E402
from .convert_helpers import empty_polydata, points_to_polydata  # noqa: E402
from .modes import Mode  # noqa: E402
from .theme import VIEWPORT_BACKGROUND  # noqa: E402

# Anatomy and planning state read by hue, not by saturation: bone is the only
# opaque body, everything else is a muted translucent overlay on it.
BONE_COLOUR = (0.949, 0.929, 0.882)      # ivory, not harsh white
RESECT_COLOUR = (0.804, 0.416, 0.361)    # muted warm red: the defect
PLATE_COLOUR = (0.722, 0.733, 0.757)     # titanium-like neutral metallic grey
ARCH_COLOUR = (0.278, 0.600, 0.502)      # muted green: the planning curve
MARK_COLOUR = (0.902, 0.678, 0.200)      # amber, active editing state only
GRAFT_COLOUR = (0.600, 0.792, 0.776)     # pale teal: the mirrored segment
NO_DONOR_COLOUR = (0.839, 0.706, 0.478)  # muted sand: no mirror donor there
#: Half the side of a cut plane's square where it crosses no bone, mm.
PLANE_HALF_SIZE_MM = 11.0
#: The bone a cut goes through is read out to this far from its centre, which keeps
#: it from taking in the far side of the arch; outlines this close count as one, mm.
PLANE_SECTION_REACH_MM = 45.0
PLANE_SECTION_LINK_MM = 3.0
#: How far the drawn square runs past the bone, and how much its size must change to
#: be redrawn during a drag, mm.
PLANE_MARGIN_MM = 2.0
PLANE_RESIZE_STEP_MM = 1.0
GUIDE_COLOUR = (0.45, 0.6, 0.8)
#: The screw plan's hole markers: size, and what their colours mean.
ADVICE_MARKER_MM = 1.3
ADVICE_COLOURS = {
    "screw": (30, 170, 70),
    "spare": (120, 140, 170),
    "skip": (235, 150, 20),
    "no bone": (215, 40, 40),
}
VERDICT_COLOURS = {"good": (0.11, 0.55, 0.25), "caution": (0.8, 0.5, 0.0), "bad": (0.85, 0.12, 0.12)}


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

    #: Called after a left-button release has been passed to VTK. Watched
    #: here, not by a VTK observer: once a press starts a rotation the
    #: interactor style holds the focus, and observers never see the release.
    left_released = None

    def mouseReleaseEvent(self, event):  # noqa: N802
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self.left_released is not None:
            self.left_released()


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
        # Flat, not graduated: a gradient here reads as decoration and makes
        # the ivory bone sit differently against the top and bottom of frame.
        self.renderer.SetBackground(*VIEWPORT_BACKGROUND)
        self.renderer.GradientBackgroundOff()
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
        # A pick happens on release, and only for a click: a press that turns
        # into a drag is rotating the view and must not drop a point.
        self._press_at: tuple[int, int] | None = None
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_left_press, 1.0)
        self.interactor.left_released = self._on_left_release

        # Refining the jaw needs the left button for the brush. A style of its
        # own takes left press/drag/release (an observer on a style replaces
        # its default handler) and hands rotation to the right button.
        self.sculpt_style = vtk.vtkInteractorStyleTrackballCamera()
        self.sculpt_style.AddObserver("LeftButtonPressEvent", self._sculpt_press)
        self.sculpt_style.AddObserver("LeftButtonReleaseEvent", self._sculpt_release)
        self.sculpt_style.AddObserver("MouseMoveEvent", self._sculpt_move)
        self.sculpt_style.AddObserver("RightButtonPressEvent", lambda o, e: o.OnLeftButtonDown())
        self.sculpt_style.AddObserver("RightButtonReleaseEvent", lambda o, e: o.OnLeftButtonUp())
        self._depth_picker = vtk.vtkWorldPointPicker()
        self._sculpting = False
        self._last_dab: np.ndarray | None = None

        session.surface_changed.connect(self.refresh_surface)
        # A scan just opened is shown from the front, whatever the last one was.
        self._new_scan = True
        session.volume_changed.connect(lambda: setattr(self, "_new_scan", True))
        session.resection_changed.connect(self.refresh_resection)
        session.plate_changed.connect(self.refresh_plate)
        session.guide_view_changed.connect(self.refresh_guide_view)
        session.arch_changed.connect(self.refresh_arch)
        session.reconstruction_changed.connect(self.refresh_reconstruction)
        session.reconstruction_edited.connect(self.render)

    def start(self) -> None:
        self.interactor.Initialize()

    def shutdown(self) -> None:
        # Nothing may draw after this: the render window is finalised below,
        # and a redraw queued by the last interaction would crash into it.
        self._shut_down = True
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
        # Smooth shading across the bent faces, crisp at real edges (the seat
        # rims, notches and edge breaks meet at well over 35 degrees).
        self.plate_normals = vtk.vtkPolyDataNormals()
        self.plate_normals.SetFeatureAngle(35.0)
        self.plate_normals.SplittingOn()
        self.plate_normals.ConsistencyOff()
        self.plate_normals.SetInputData(empty_polydata())
        self.plate_mapper.SetInputConnection(self.plate_normals.GetOutputPort())
        self.plate_actor.GetProperty().SetColor(*PLATE_COLOUR)
        self.plate_actor.GetProperty().SetSpecular(0.45)
        self.plate_actor.GetProperty().SetSpecularPower(28)
        self.renderer.AddActor(self.plate_actor)

        self.show_hole_centres = True
        self.show_screw_trajectories = False
        self.show_marking = True

        # The etched mark on the plate face. Drawn dark and recessed rather
        # than cut out of the solid, so the plate stays watertight for export.
        self.marking_mapper = vtk.vtkPolyDataMapper()
        self.marking_actor = vtk.vtkActor()
        self.marking_actor.SetMapper(self.marking_mapper)
        self.marking_actor.GetProperty().SetColor(0.20, 0.21, 0.23)
        self.marking_actor.GetProperty().SetAmbient(0.35)
        self.marking_actor.VisibilityOff()
        self.renderer.AddActor(self.marking_actor)

        self.screw_mapper = vtk.vtkPolyDataMapper()
        self.screw_actor = vtk.vtkActor()
        self.screw_actor.SetMapper(self.screw_mapper)
        self.screw_actor.GetProperty().SetColor(*MARK_COLOUR)
        self.screw_actor.GetProperty().SetLineWidth(2)
        self.screw_actor.VisibilityOff()
        self.renderer.AddActor(self.screw_actor)

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

        # The screw plan: a coloured marker in each hole, and the verdict in words.
        self.advice_points = vtk.vtkPolyData()
        self.advice_glyph = vtk.vtkGlyph3D()
        advice_sphere = vtk.vtkSphereSource()
        advice_sphere.SetRadius(ADVICE_MARKER_MM)
        advice_sphere.SetThetaResolution(12)
        advice_sphere.SetPhiResolution(12)
        self.advice_glyph.SetSourceConnection(advice_sphere.GetOutputPort())
        self.advice_glyph.SetScaleModeToDataScalingOff()
        self.advice_glyph.SetColorModeToColorByScalar()
        self.advice_glyph.SetInputData(empty_polydata())
        self.advice_mapper = vtk.vtkPolyDataMapper()
        self.advice_mapper.SetInputConnection(self.advice_glyph.GetOutputPort())
        self.advice_mapper.SetColorModeToDirectScalars()
        self.advice_mapper.SetScalarModeToUsePointData()
        self.advice_actor = vtk.vtkActor()
        self.advice_actor.SetMapper(self.advice_mapper)
        self.renderer.AddActor(self.advice_actor)
        self.verdict_actor = vtk.vtkTextActor()
        self.verdict_actor.GetTextProperty().SetFontSize(26)
        self.verdict_actor.GetTextProperty().SetBold(True)
        self.verdict_actor.GetTextProperty().SetFrame(False)
        self.verdict_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.verdict_actor.GetPositionCoordinate().SetValue(0.02, 0.93)
        self.verdict_actor.VisibilityOff()
        self.renderer.AddViewProp(self.verdict_actor)

        # Rehearsing the bends: the printed key, the bender's head, and the plate
        # at this step between them.
        self.guide_mapper = vtk.vtkPolyDataMapper()
        self.guide_mapper.SetInputData(empty_polydata())
        self.guide_mapper.ScalarVisibilityOff()
        self.guide_actor = vtk.vtkActor()
        self.guide_actor.SetMapper(self.guide_mapper)
        self.guide_actor.GetProperty().SetColor(*GUIDE_COLOUR)
        self.guide_actor.VisibilityOff()
        self.renderer.AddActor(self.guide_actor)
        self.guide_tool_mapper = vtk.vtkPolyDataMapper()
        self.guide_tool_mapper.SetInputData(empty_polydata())
        self.guide_tool_mapper.ScalarVisibilityOff()
        self.guide_tool_actor = vtk.vtkActor()
        self.guide_tool_actor.SetMapper(self.guide_tool_mapper)
        self.guide_tool_actor.GetProperty().SetColor(0.62, 0.64, 0.68)
        self.guide_tool_actor.GetProperty().SetOpacity(0.35)
        self.guide_tool_actor.VisibilityOff()
        self.renderer.AddActor(self.guide_tool_actor)
        self.guide_plate_normals = vtk.vtkPolyDataNormals()
        self.guide_plate_normals.SplittingOn()
        self.guide_plate_normals.ConsistencyOff()
        self.guide_plate_normals.SetInputData(empty_polydata())
        self.guide_plate_mapper = vtk.vtkPolyDataMapper()
        self.guide_plate_mapper.SetInputConnection(self.guide_plate_normals.GetOutputPort())
        self.guide_plate_mapper.ScalarVisibilityOff()
        self.guide_plate_actor = vtk.vtkActor()
        self.guide_plate_actor.SetMapper(self.guide_plate_mapper)
        self.guide_plate_actor.GetProperty().SetSpecular(0.45)
        self.guide_plate_actor.VisibilityOff()
        self.renderer.AddActor(self.guide_plate_actor)
        self._guide_view_shown = False
        self._guide_view_step = None
        self._plane_surface = None
        self._plane_vertices = None
        self._plane_triangles = None
        self._plane_halves: dict[int, float] = {}
        self._plane_offsets: dict[int, np.ndarray] = {}
        self._saved_visibility: list[tuple] = []

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

        # The reconstructed mandible is one surface; its point scalars say
        # which part is retained bone, mirrored donor, or no-donor fill.
        region_colours = vtk.vtkLookupTable()
        region_colours.SetNumberOfTableValues(3)
        region_colours.SetTableRange(0.0, 2.0)
        region_colours.SetTableValue(0, *BONE_COLOUR, 1.0)
        region_colours.SetTableValue(1, *GRAFT_COLOUR, 1.0)
        region_colours.SetTableValue(2, *NO_DONOR_COLOUR, 1.0)
        region_colours.Build()
        self.graft_mapper = vtk.vtkPolyDataMapper()
        self.graft_mapper.SetLookupTable(region_colours)
        self.graft_mapper.SetScalarRange(0.0, 2.0)
        self.graft_mapper.SetScalarModeToUsePointData()
        self.graft_mapper.SetColorModeToMapScalars()
        self.graft_mapper.InterpolateScalarsBeforeMappingOn()
        self.graft_actor = vtk.vtkActor()
        self.graft_actor.SetMapper(self.graft_mapper)
        self.graft_actor.GetProperty().SetSpecular(0.15)
        self.graft_actor.VisibilityOff()
        self.renderer.AddActor(self.graft_actor)

        self.measure_mapper = vtk.vtkPolyDataMapper()
        # The brush: a faint sphere of the brush's radius under the cursor.
        self.brush_source = vtk.vtkSphereSource()
        self.brush_source.SetThetaResolution(32)
        self.brush_source.SetPhiResolution(16)
        brush_mapper = vtk.vtkPolyDataMapper()
        brush_mapper.SetInputConnection(self.brush_source.GetOutputPort())
        self.brush_actor = vtk.vtkActor()
        self.brush_actor.SetMapper(brush_mapper)
        self.brush_actor.GetProperty().SetColor(0.30, 0.55, 0.95)
        self.brush_actor.GetProperty().SetOpacity(0.25)
        self.brush_actor.PickableOff()
        self.brush_actor.VisibilityOff()
        self.renderer.AddActor(self.brush_actor)

        self.measure_actor = vtk.vtkActor()
        self.measure_actor.SetMapper(self.measure_mapper)
        self.measure_actor.GetProperty().SetColor(1.0, 0.9, 0.3)
        self.measure_actor.GetProperty().SetLineWidth(3)
        self.renderer.AddActor(self.measure_actor)

        # The plate mapper is fed by the normals filter above; giving it input
        # data here would silently sever that connection.
        for mapper in (
            self.bone_mapper,
            self.fragment_mapper,
            self.graft_mapper,
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
            if self._new_scan and surface.GetNumberOfPoints():
                self._new_scan = False
                self.set_view_direction("anterior")
            self.renderer.ResetCamera()
        self._sync_plane_widgets()
        self.refresh_resection()

    def refresh_resection(self) -> None:
        session = self.session
        if session.reconstructed_surface is not None:
            # A built reconstruction owns the view until the plan changes.
            self.bone_actor.VisibilityOff()
            self.fragment_actor.VisibilityOff()
            self._sync_plane_widgets()
            self.render()
            return
        self.graft_actor.VisibilityOff()
        self.bone_actor.VisibilityOn()
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
        """Show the reconstructed mandible in place of the pre-operative bone."""
        surface = self.session.reconstructed_surface
        if surface is None:
            self.graft_actor.VisibilityOff()
            self.bone_actor.VisibilityOn()
            self.refresh_resection()
            return
        self.graft_mapper.SetInputData(surface)
        self.graft_actor.VisibilityOn()
        # The reconstruction already contains the retained bone; drawing the
        # pre-operative surface and the fragment preview over it would hide
        # exactly the junctions it is meant to show.
        self.bone_actor.VisibilityOff()
        self.fragment_actor.VisibilityOff()
        self._sync_plane_widgets()
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
        """Draw the selected plate asset, fitted to the planned path.

        There is no proxy geometry here. If a plate is selected, what is drawn
        is that plate's own mesh under its placement transform; if none is
        selected the actor is emptied rather than filled with a stand-in.
        """
        from ..render.convert import polyline_to_polydata, triangles_to_polydata

        session = self.session
        fitted = session.plate_mesh()
        plan = session.plate_plan
        asset = session.plate_asset
        verdict = session.plate_verdict
        colour = asset.colour if asset is not None else PLATE_COLOUR
        # A plate that will not do looks it, whichever way the view is turned.
        self.plate_actor.GetProperty().SetColor(*(VERDICT_COLOURS["bad"] if verdict is not None and verdict.level == "bad" else colour))
        self._refresh_verdict_sign(verdict)

        if fitted is None:
            self.plate_normals.SetInputData(empty_polydata())
        else:
            self.plate_normals.SetInputData(
                triangles_to_polydata(fitted.points, fitted.triangles)
            )

        # Hole markers are the plate's real hole centres after fitting, and
        # they are an overlay: the hole geometry stays visible underneath.
        advised = (
            fitted is not None and self.show_hole_centres and verdict is not None
            and len(verdict.holes) == len(fitted.hole_centres)
        )
        self.advice_glyph.SetInputData(
            self._advice_markers(fitted, verdict) if advised else empty_polydata()
        )
        if advised:
            self.node_glyph.SetInputData(empty_polydata())
        elif fitted is not None and self.show_hole_centres:
            self.node_glyph.SetInputData(points_to_polydata(fitted.hole_centres))
        elif plan is None:
            self.node_glyph.SetInputData(points_to_polydata(session.plate_points))
        else:
            self.node_glyph.SetInputData(empty_polydata())

        marking = session.plate_marking
        if fitted is not None and marking is not None and self.show_marking:
            # The mark was moved and bent with the plate, so it sits in the
            # face it is etched into rather than being placed afterwards.
            self.marking_mapper.SetInputData(
                triangles_to_polydata(marking[0], marking[1])
            )
            self.marking_actor.VisibilityOn()
        else:
            self.marking_mapper.SetInputData(empty_polydata())
            self.marking_actor.VisibilityOff()

        if fitted is not None and self.show_screw_trajectories:
            segments = fitted.screw_trajectories()
            merged = []
            for start, end in segments:
                merged.append(polyline_to_polydata(np.vstack([start, end])))
            appended = vtk.vtkAppendPolyData()
            for piece in merged:
                appended.AddInputData(piece)
            appended.Update()
            self.screw_mapper.SetInputData(appended.GetOutput())
            self.screw_actor.VisibilityOn()
        else:
            self.screw_mapper.SetInputData(empty_polydata())
            self.screw_actor.VisibilityOff()
        self.render()

    def refresh_guide_view(self) -> None:
        """Swap the jaw for the key, the bender and the plate at this step, or back again."""
        from ..render.convert import triangles_to_polydata

        preview = self.session.guide_preview()
        hidden = [
            self.bone_actor, self.fragment_actor, self.graft_actor, self.plate_actor,
            self.node_actor, self.advice_actor, self.marking_actor, self.screw_actor,
            self.verdict_actor, self.arch_actor,
        ]
        if preview is None:
            self.guide_actor.VisibilityOff()
            self.guide_tool_actor.VisibilityOff()
            self.guide_plate_actor.VisibilityOff()
            if self._guide_view_shown:
                for actor, visibility in self._saved_visibility:
                    actor.SetVisibility(visibility)
                self._guide_view_shown = False
                self._sync_plane_widgets()
                self.renderer.ResetCamera()
            self.render()
            return
        if not self._guide_view_shown:
            self._saved_visibility = [(actor, actor.GetVisibility()) for actor in hidden]
            for actor in hidden:
                actor.VisibilityOff()
            for widget in self._plane_widgets:
                widget.SetEnabled(False)
            self._guide_view_shown = True
            entering = True
        else:
            entering = False
        solid, head, points, triangles = preview
        self.guide_mapper.SetInputData(solid if solid is not None else empty_polydata())
        self.guide_tool_mapper.SetInputData(head if head is not None else empty_polydata())
        plate = triangles_to_polydata(points, triangles)
        self.guide_plate_normals.SetInputData(plate)
        asset = self.session.plate_asset
        self.guide_plate_actor.GetProperty().SetColor(*(asset.colour if asset is not None else PLATE_COLOUR))
        self.guide_actor.SetVisibility(solid is not None)
        self.guide_tool_actor.SetVisibility(head is not None)
        self.guide_plate_actor.VisibilityOn()
        # Each step is a different pose of the plate: show what it is about.
        focus = solid if solid is not None else plate
        step = (solid is not None, self.session.guide_step)
        if entering or step != self._guide_view_step:
            self.renderer.ResetCamera(focus.GetBounds())
        self._guide_view_step = step
        self.renderer.ResetCameraClippingRange()
        self.render()

    def _advice_markers(self, fitted, verdict) -> vtk.vtkPolyData:
        """The holes as points, each carrying the colour of what it is for."""
        from vtkmodules.util.numpy_support import numpy_to_vtk

        markers = points_to_polydata(fitted.hole_centres)
        colours = np.array([ADVICE_COLOURS[hole.state] for hole in verdict.holes], dtype=np.uint8)
        array = numpy_to_vtk(colours, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        array.SetName("advice")
        markers.GetPointData().SetScalars(array)
        return markers

    def _refresh_verdict_sign(self, verdict) -> None:
        if verdict is None:
            self.verdict_actor.VisibilityOff()
            return
        # Plain characters: the view's font has no tick or cross.
        marks = {"good": "", "caution": "!  ", "bad": "X  "}
        self.verdict_actor.SetInput(f"{marks[verdict.level]}Plate: {verdict.headline}")
        self.verdict_actor.GetTextProperty().SetColor(*VERDICT_COLOURS[verdict.level])
        self.verdict_actor.VisibilityOn()

    def set_plate_overlays(
        self, plate: bool, holes: bool, screws: bool, marking: bool = True
    ) -> None:
        """Independent display toggles for the plate and its overlays."""
        self.show_hole_centres = holes
        self.show_screw_trajectories = screws
        self.show_marking = marking
        self.plate_actor.SetVisibility(bool(plate))
        self.refresh_plate()

    def set_mode(self, mode: Mode) -> None:
        self.mode = mode
        if mode not in (Mode.MEASURE, Mode.ANGLE):
            self.clear_measurement()
        sculpting = mode == Mode.SCULPT
        self.interactor.SetInteractorStyle(self.sculpt_style if sculpting else self.camera_style)
        if not sculpting:
            self.brush_actor.VisibilityOff()
            self.render()

    # -- refining the reconstruction -------------------------------------------

    def _brush_point(self) -> np.ndarray | None:
        """The point on the reconstructed jaw under the cursor, or None."""
        surface = self.session.reconstructed_surface
        if surface is None:
            return None
        x, y = self.interactor.GetEventPosition()
        self._depth_picker.Pick(x, y, 0, self.renderer)
        point = np.array(self._depth_picker.GetPickPosition(), dtype=float)
        bounds = np.array(surface.GetBounds()).reshape(3, 2)
        margin = self.session.brush_radius_mm
        if np.any(point < bounds[:, 0] - margin) or np.any(point > bounds[:, 1] + margin):
            return None  # the background, not the jaw
        return point

    def _sculpt_press(self, style, event) -> None:
        point = self._brush_point()
        if point is None:
            return
        self._sculpting = True
        self._last_dab = point
        self.session.sculpt(point, new_stroke=True)

    def _sculpt_release(self, style, event) -> None:
        self._sculpting = False
        self._last_dab = None

    def _sculpt_move(self, style, event) -> None:
        point = self._brush_point()
        if point is None:
            self.brush_actor.VisibilityOff()
        else:
            self.brush_source.SetCenter(*point)
            self.brush_source.SetRadius(self.session.brush_radius_mm)
            self.brush_actor.VisibilityOn()
        if self._sculpting and point is not None:
            # A dab every quarter radius along the drag, not every mouse event.
            if np.linalg.norm(point - self._last_dab) >= 0.25 * self.session.brush_radius_mm:
                self._last_dab = point
                self.session.sculpt(point, new_stroke=False)
        elif not self._sculpting:
            style.OnMouseMove()  # keep right-drag rotation and wheel zoom working
        self.render()

    def clear_measurement(self) -> None:
        self._measure_points.clear()
        self.measure_mapper.SetInputData(empty_polydata())
        self.render()

    _shut_down = False

    def render(self) -> None:
        if self._shut_down:
            return
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
            # Whether they show at all is decided first, so nothing below can leave a
            # cut on the screen, or draggable, once it has been executed.
            shown = not self.session.cut_applied and not self._guide_view_shown
            faded = self.session.reconstructed_surface is not None
            for index, plane in enumerate(self.session.planes):
                if index >= len(self._plane_widgets):
                    self._plane_widgets.append(self._make_plane_widget(index, self._plane_box(plane.origin)))
                widget = self._plane_widgets[index]
                widget.SetEnabled(shown)
                # Over a finished reconstruction the planes are reference lines, faint,
                # and not to be caught by accident: the panel's numbers move them.
                widget.SetProcessEvents(shown and not faded)
                rep = widget.GetRepresentation()
                rep.GetPlaneProperty().SetOpacity(0.08 if faded else 0.35)
                if not shown:
                    continue
                # The square is drawn round the middle of the bone it cuts, a point of
                # the same plane; what the widget reports is taken back to the cut.
                up, side, centre, half = self._plane_square(plane.origin, plane.normal)
                self._plane_offsets[index] = centre - np.asarray(plane.origin, dtype=float)
                rep.PlaceWidget(self._plane_box(centre))
                rep.SetOrigin(*(float(v) for v in centre))
                rep.SetNormal(*(float(v) for v in plane.normal))
                rep.BuildRepresentation()
                self._shape_plane(index, centre, up, side, half)
        finally:
            self._syncing = False

    @staticmethod
    def _plane_box(origin):
        return [float(c) + sign * PLANE_HALF_SIZE_MM for c in origin for sign in (-1.0, 1.0)]

    def _plane_square(self, origin, normal, centre=None):
        """The square to draw for a cut: ``(up, side, centre, half)``. Its edges run
        along the jaw's height and across it; it is centred on the outline of the bone
        the cut goes through, or on ``centre`` if given, and just covers that outline."""
        n = np.asarray(normal, dtype=float)
        n = n / max(float(np.linalg.norm(n)), 1e-9)
        o = np.asarray(origin, dtype=float)
        frames = self.session.frames
        up = np.array([0.0, 0.0, 1.0])
        if frames is not None:
            up = frames.frame_at(float(frames.s[int(np.argmin(np.linalg.norm(frames.points - o, axis=1)))]))[1]
        up = up - (up @ n) * n
        if np.linalg.norm(up) < 1e-6:
            up = np.cross(n, [1.0, 0.0, 0.0])
        up = up / np.linalg.norm(up)
        side = np.cross(n, up)
        middle = o if centre is None else np.asarray(centre, dtype=float)
        half = PLANE_HALF_SIZE_MM
        try:
            outline = self._plane_section(o, n)
            if len(outline):
                rel = outline - o
                a, b = rel @ up, rel @ side
                if centre is None:
                    middle = o + 0.5 * (a.min() + a.max()) * up + 0.5 * (b.min() + b.max()) * side
                rel = outline - middle
                half = max(half, float(np.max(np.maximum(np.abs(rel @ up), np.abs(rel @ side)))) + PLANE_MARGIN_MM)
        except Exception:  # noqa: BLE001 - a square of the usual size will do
            pass
        return up, side, middle, half

    def _plane_section(self, origin: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """Points on the outline of the bone the plane cuts at ``origin``: the outline
        nearest it, and any inside that one, as the marrow's is."""
        from vtkmodules.util.numpy_support import vtk_to_numpy

        surface = self.session.surface
        if surface is None or not surface.GetNumberOfPolys():
            return np.zeros((0, 3))
        if self._plane_surface is not surface:
            self._plane_surface = surface
            self._plane_vertices = vtk_to_numpy(surface.GetPoints().GetData()).astype(float)
            self._plane_triangles = vtk_to_numpy(surface.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        vertices, triangles = self._plane_vertices, self._plane_triangles
        rel = vertices - origin
        near = np.einsum("ij,ij->i", rel, rel) < PLANE_SECTION_REACH_MM**2
        side = (rel @ normal) > 0.0
        crossing = (side[triangles].any(axis=1) & ~side[triangles].all(axis=1)) & near[triangles].all(axis=1)
        if not crossing.any():
            return np.zeros((0, 3))
        # Where each edge of those triangles crosses the plane.
        tri = triangles[crossing]
        points = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = tri[:, i], tri[:, j]
            cuts = side[a] != side[b]
            da, db = rel[a[cuts]] @ normal, rel[b[cuts]] @ normal
            w = (da / (da - db))[:, None]
            points.append(vertices[a[cuts]] + w * (vertices[b[cuts]] - vertices[a[cuts]]))
        points = np.vstack(points)
        # The outline nearest the cut's centre, grown along the points it runs through.
        taken = np.zeros(len(points), dtype=bool)
        frontier = [int(np.argmin(np.linalg.norm(points - origin, axis=1)))]
        taken[frontier[0]] = True
        while frontier:
            here = points[frontier]
            close = np.zeros(len(points), dtype=bool)
            for chunk in range(0, len(here), 256):
                d = np.linalg.norm(points[None, :, :] - here[chunk : chunk + 256, None, :], axis=2)
                close |= (d < PLANE_SECTION_LINK_MM).any(axis=0)
            fresh = close & ~taken
            taken |= fresh
            frontier = list(np.flatnonzero(fresh))
        main = points[taken]
        low, high = main.min(axis=0), main.max(axis=0)
        inside = np.all((points >= low) & (points <= high), axis=1)
        return points[taken | inside]

    def _shape_plane(self, index: int, centre, up, side, half) -> None:
        """Draw the cut as its own square, upright with the jaw, sized to the bone."""
        source = self._plane_source(index)
        if source is None:
            return
        centre = np.asarray(centre, dtype=float)
        source.SetOrigin(*(centre - half * up - half * side))
        source.SetPoint1(*(centre + half * up - half * side))
        source.SetPoint2(*(centre - half * up + half * side))
        source.Update()
        self._plane_halves[index] = half

    def _plane_source(self, index: int):
        """The plane source a cut's widget draws its square from."""
        if index >= len(self._plane_widgets):
            return None
        props = vtk.vtkPropCollection()
        self._plane_widgets[index].GetRepresentation().GetActors(props)
        props.InitTraversal()
        for _ in range(props.GetNumberOfItems()):
            prop = props.GetNextProp()
            if prop is not None and prop.IsA("vtkActor"):
                algorithm = prop.GetMapper().GetInputAlgorithm()
                if algorithm is not None and algorithm.IsA("vtkPlaneSource"):
                    return algorithm
        return None

    def _make_plane_widget(self, index: int, bounds) -> vtk.vtkImplicitPlaneWidget2:
        rep = vtk.vtkImplicitPlaneRepresentation()
        rep.SetPlaceFactor(1.0)
        rep.CropPlaneToBoundingBoxOff()
        rep.ConstrainToWidgetBoundsOff()
        rep.OutsideBoundsOn()
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
            "EndInteractionEvent", lambda obj, evt, i=index: self._on_plane_released(i)
        )
        widget.On()
        return widget

    def _on_plane_released(self, index: int) -> None:
        self._on_plane_moved(index)
        # Re-centre the drawn plane on where the cut now is, once the drag
        # is over (not during it: the widget is still holding the plane).
        QTimer.singleShot(0, self._resync_planes)

    def _resync_planes(self) -> None:
        if self._shut_down:
            return
        self._sync_plane_widgets()
        self.render()

    def _on_plane_moved(self, index: int) -> None:
        if self._syncing or index >= len(self._plane_widgets):
            return
        rep = self._plane_widgets[index].GetRepresentation()
        # Deliberately does not read GetNormal(): dragging a plane along the jaw
        # translates it, and must not re-angle it. Angulation is set explicitly.
        if self.session.cut_applied or self.session.reconstructed_surface is not None:
            return  # an executed cut is not moved by touching where it was
        drawn = np.array(rep.GetOrigin(), dtype=float)
        origin = drawn - self._plane_offsets.get(index, np.zeros(3))
        self._syncing = True
        try:
            self.plane_translated.emit(index, origin)
            if index < len(self.session.planes):
                # Grow or shrink with the bone it is passing over, as it moves. The
                # square is redrawn in place: the drag itself is left alone.
                normal = self.session.planes[index].normal
                up, side, centre, half = self._plane_square(origin, normal, centre=drawn)
                if abs(self._plane_halves.get(index, 0.0) - half) > PLANE_RESIZE_STEP_MM:
                    self._shape_plane(index, centre, up, side, half)
        finally:
            self._syncing = False

    # -- picking -----------------------------------------------------------

    #: How far the mouse may move between press and release and still be a
    #: click rather than a drag, in pixels.
    CLICK_TOLERANCE_PX = 4

    def _on_left_press(self, obj, event) -> None:
        self._press_at = tuple(self.interactor.GetEventPosition())

    def _on_left_release(self) -> None:
        start, self._press_at = self._press_at, None
        if start is None:
            return
        x, y = self.interactor.GetEventPosition()
        if max(abs(x - start[0]), abs(y - start[1])) > self.CLICK_TOLERANCE_PX:
            return
        self._on_left_button(start)

    def _on_left_button(self, position) -> None:
        if self.mode in (Mode.NAVIGATE, Mode.SCULPT):
            return
        x, y = position
        # The reconstruction replaces the bone in the view once it exists, and
        # the plate is planned on it: clicks on it count as clicks on bone.
        # Only the surfaces a click can be about are picked from, so a click
        # on bone behind the plate reaches the bone, and a measurement can
        # start or end on the plate itself.
        targets = [self.bone_actor, self.fragment_actor, self.graft_actor]
        if self.mode in (Mode.MEASURE, Mode.ANGLE):
            targets.append(self.plate_actor)
        self.picker.InitializePickList()
        for actor in targets:
            self.picker.AddPickList(actor)
        self.picker.PickFromListOn()
        if not self.picker.Pick(x, y, 0, self.renderer):
            return
        if self.picker.GetActor() not in targets:
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
