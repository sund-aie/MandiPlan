"""The fibula graft: the leg's CT, its fibula, and the plan of pieces for the jaw.

Held beside the session (``session.fibula``), which owns the jaw: the cuts, the
reconstruction and the arch the plan is made against. The leg's scan is a
second scan in its own patient millimetres; nothing of it is moved into the
jaw's except the planned pieces, each by its own rigid transform.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from vtkmodules.util.numpy_support import vtk_to_numpy

from ..dicom_io import load_folder
from ..geometry.fibula import DEFAULT_THRESHOLD_HU, find_fibulae
from ..geometry.fibula_guide import REACH_MM, SLOT_REACH_MM, check_guide, guide_field_on, plan_guide
from ..geometry.fibula_plan import (
    COURSE_TOLERANCE_MM,
    DISTAL_KEEP_MM,
    MAX_PIECES,
    MIN_PIECE_MM,
    PROXIMAL_KEEP_MM,
    choose_joints,
    defect_course,
    plan_fibula,
    section_rim,
    vessels_in_jaw,
    with_guide,
)
from ..render.fibula_guide_mesh import clearance_to_bone, guide_grid, guide_polydata
from ..render.fibula_mesh import coloured, cut_faces, cut_piece, fibula_surface, moved

#: The settings the panel may change, and their kinds.
SETTINGS = {
    "distal_keep_mm": float,
    "proximal_keep_mm": float,
    "min_piece_mm": float,
    "piece_count": int,  # 0: chosen automatically
    "raise_mm": float,
    "roll_deg": float,
    "pedicle": str,  # "auto", "first" or "second"
    "guide_face": str,  # "lateral" or "anterior": the face the cutting guide sits on
}


class FibulaGraft(QObject):
    changed = pyqtSignal()
    leg_view_changed = pyqtSignal()

    def __init__(self, session):
        super().__init__(session)
        self.session = session
        self.scan = None
        self.geometry = None
        self.series = None
        self.folder = ""
        self.threshold = DEFAULT_THRESHOLD_HU
        self.fibulae = []
        self.side = ""
        self.surface = None  # the chosen fibula alone, in the leg's space (vtkPolyData)
        self._surface_points = None
        self.distal_keep_mm = DISTAL_KEEP_MM
        self.proximal_keep_mm = PROXIMAL_KEEP_MM
        self.min_piece_mm = MIN_PIECE_MM
        self.piece_count = 0
        self.raise_mm = 0.0
        self.roll_deg = 0.0
        self.pedicle = "auto"
        self.guide_face = "lateral"
        #: The cutting guide, made on demand for the plan as it stands: the guide,
        #: its mesh, and what following its use found. Dropped when the plan changes.
        self.guide = None
        self.guide_check = None
        self._guide_mesh = None
        self.course = None
        self.plan = None
        #: The guide's other face, and how much fibula the plan would take with it there, mm.
        self.other_face = None
        #: Why there is no plan, when there is none.
        self.problem = ""
        #: Where the vessels lie round the fibula in the jaw: (up, out), each -1 to 1.
        self.vessels = None
        self.pieces_in_leg = []
        self.pieces_in_jaw = []
        self.leg_view = False
        session.resection_changed.connect(self._jaw_changed)
        session.reconstruction_changed.connect(self._jaw_changed)
        session.reconstruction_edited.connect(self._jaw_changed)

    # -- the leg's scan ---------------------------------------------------------

    def load_leg_folder(self, folder: str, series_uid: str | None = None) -> None:
        """Open a CT of the legs. Nothing is cropped to the jaws in it."""
        self.session.busy.emit(True)
        try:
            volume, geometry, series = load_folder(folder, series_uid, find_jaws=False)
        finally:
            self.session.busy.emit(False)
        self.set_scan(volume, geometry, series, folder)

    def set_scan(self, volume, geometry=None, series=None, folder: str = "") -> None:
        self.scan = volume
        self.geometry = geometry
        self.series = series
        self.folder = folder
        self.side = ""
        self.find()

    def set_threshold(self, value: float) -> None:
        self.threshold = float(value)
        self.find()

    def find(self) -> None:
        """Find the fibulae at the threshold, keep the leg chosen, and replan."""
        self.fibulae = find_fibulae(self.scan, self.threshold) if self.scan is not None else []
        sides = [f.side for f in self.fibulae]
        if self.side not in sides:
            self.side = self._default_side(sides)
        self._make_surface()
        self.replan()

    def _default_side(self, sides: list[str]) -> str:
        if not sides:
            return ""
        if len(sides) == 1:
            return sides[0]
        # The leg on the side the vessels will be joined: for a defect of the
        # body the same side (Wei; Weyh), and the surgeon says otherwise.
        end = self._pedicle_point()
        if end is None:
            return sides[0]
        return "left" if end[0] > self._midline_x() else "right"

    def choose_side(self, side: str) -> None:
        if side == self.side or side not in [f.side for f in self.fibulae]:
            return
        self.side = side
        self._make_surface()
        self.replan()

    @property
    def fibula(self):
        for fibula in self.fibulae:
            if fibula.side == self.side:
                return fibula
        return None

    def _make_surface(self) -> None:
        fibula = self.fibula
        if fibula is None or self.scan is None:
            self.surface = None
            self._surface_points = None
            return
        self.surface = fibula_surface(self.scan, fibula, self.threshold)
        points = self.surface.GetPoints()
        self._surface_points = vtk_to_numpy(points.GetData()).astype(float) if points is not None else None

    # -- the plan ----------------------------------------------------------------

    def set_setting(self, name: str, value) -> None:
        if name not in SETTINGS:
            raise KeyError(name)
        setattr(self, name, SETTINGS[name](value))
        self.replan()

    def _jaw_changed(self) -> None:
        if self.fibula is not None:
            self.replan()

    def _jaw_vertices(self):
        session = self.session
        surface = session.reconstructed_surface
        if surface is None:
            surface = session.surface
        if surface is None or surface.GetNumberOfPoints() == 0:
            return None
        return vtk_to_numpy(surface.GetPoints().GetData()).astype(float)

    def _midline_x(self) -> float:
        plane = getattr(self.session, "symmetry_plane", None)
        if plane is not None:
            return float(plane.point[0])
        frames = self.session.frames
        return float(frames.points[:, 0].mean()) if frames is not None else 0.0

    def _pedicle_point(self):
        """The end of the defect the vessels leave from, in the jaw, if known."""
        session = self.session
        if session.frames is None or len(session.planes) != 2:
            return None
        end = self._pedicle_end_for(session.frames, session.planes)
        planes = sorted(session.planes, key=lambda p: _arc_of(session.frames, p.origin))
        return planes[0 if end == "first" else 1].origin

    def _pedicle_end_for(self, frames, planes) -> str:
        if self.pedicle in ("first", "second"):
            return self.pedicle
        # By default the end further round the jaw from the chin: towards the
        # angle, where the neck's vessels are reached.
        middle = 0.5 * frames.length_mm
        arcs = sorted(_arc_of(frames, p.origin) for p in planes)
        return "second" if abs(arcs[1] - middle) >= abs(arcs[0] - middle) else "first"

    def replan(self) -> None:
        self.guide = None
        self.guide_check = None
        self._guide_mesh = None
        self.course = None
        self.plan = None
        self.other_face = None
        self.vessels = None
        self.pieces_in_leg = []
        self.pieces_in_jaw = []
        self.problem = ""
        session = self.session
        fibula = self.fibula
        if self.scan is None:
            self.problem = "Open a CT of the legs."
        elif fibula is None:
            self.problem = (
                f"No fibula found at {self.threshold:.0f} HU. Check this is a CT of the "
                "lower legs, or move the threshold."
            )
        elif session.frames is None:
            self.problem = "The jaw's arch is needed: lay it in step 2."
        elif len(session.planes) != 2:
            self.problem = "Place the resection's two cuts first (step 3)."
        if self.problem:
            self.changed.emit()
            return
        vertices = self._jaw_vertices()
        if vertices is None:
            self.problem = "No jaw surface to plan against."
            self.changed.emit()
            return
        harvest = (fibula.s >= self.distal_keep_mm) & (fibula.s <= fibula.length_mm - self.proximal_keep_mm)
        radius = float(np.median(fibula.radius_mm[harvest] if harvest.any() else fibula.radius_mm))
        self.course = defect_course(vertices, session.frames, session.planes, radius, self.raise_mm)
        if self.course is None:
            self.problem = "The jaw between the cuts could not be read to lay the fibula along."
            self.changed.emit()
            return
        count = self.piece_count if self.piece_count > 0 else None
        fit = choose_joints(self.course.points, self.min_piece_mm, MAX_PIECES, COURSE_TOLERANCE_MM, count)
        if fit is None:
            self.problem = (
                f"{self.piece_count} pieces of at least {self.min_piece_mm:.0f} mm do not fit "
                f"the {self.course.length_mm:.0f} mm defect."
                if count
                else f"The defect, {self.course.length_mm:.0f} mm, is shorter than one piece of "
                f"{self.min_piece_mm:.0f} mm."
            )
            self.changed.emit()
            return
        joints, deviation = fit
        self.plan = self._plan_for(self.guide_face, joints, deviation)
        # What the guide on the other face would cost: which face takes less
        # fibula depends on which way the bends go.
        other = "anterior" if self.guide_face == "lateral" else "lateral"
        self.other_face = (other, self._plan_for(other, joints, deviation).harvest_length_mm)
        self.vessels = vessels_in_jaw(self.plan, fibula, self.course)
        if self.surface is not None:
            self.pieces_in_leg = [cut_piece(self.surface, piece) for piece in self.plan.pieces]
            self.pieces_in_jaw = [
                moved(mesh, piece.to_jaw) for mesh, piece in zip(self.pieces_in_leg, self.plan.pieces)
            ]
        self.changed.emit()

    def _plan_for(self, face: str, joints, deviation):
        """The plan with its pieces spaced for a cutting guide on ``face``."""
        fibula = self.fibula
        points = self._surface_points
        rim_at = None
        if points is not None and len(points):
            # The cuts of a joint kept apart through the bone, along each slot's
            # length past it, and through the guide's depth on its face.
            def rim_at(s, frame):
                return with_guide(
                    section_rim(points, fibula, s, frame),
                    fibula.side_at(s, face),
                    frame[:, 0],
                    REACH_MM,
                    SLOT_REACH_MM + 0.5,
                )
        return plan_fibula(
            self.course,
            fibula,
            joints,
            deviation,
            pedicle_end=self._pedicle_end_for(self.session.frames, self.session.planes),
            roll_deg=self.roll_deg,
            distal_keep_mm=self.distal_keep_mm,
            proximal_keep_mm=self.proximal_keep_mm,
            rim_at=rim_at,
            min_piece_mm=self.min_piece_mm,
        )

    # -- the cutting guide ---------------------------------------------------------

    def make_guide(self) -> bool:
        """Make the cutting guide for the plan, at printing resolution, and follow
        its use: seat it, lift it off, run a blade down every slot, drive every
        screw. Kept until the plan changes."""
        fibula = self.fibula
        if self.plan is None or fibula is None or self.surface is None:
            return False
        self.session.busy.emit(True)
        try:
            guide = plan_guide(self.plan, fibula, self._surface_points, self.guide_face)
            grid = guide_grid(guide, self.scan, fibula, self.threshold)
            solid = guide_field_on(guide, grid)
            mesh = guide_polydata(guide, grid, solid)
            clear = clearance_to_bone(mesh, self.surface)
            check = check_guide(guide, self.plan, grid, solid, clear)
        except ValueError as error:
            self.session.message.emit(f"The cutting guide could not be made: {error}")
            return False
        finally:
            self.session.busy.emit(False)
        self.guide, self._guide_mesh, self.guide_check = guide, mesh, check
        self.changed.emit()
        return True

    def guide_mesh(self):
        return self._guide_mesh

    # -- what the view shows -------------------------------------------------------

    def jaw_pieces_mesh(self):
        return coloured(self.pieces_in_jaw) if self.pieces_in_jaw else None

    def leg_pieces_mesh(self):
        return coloured(self.pieces_in_leg) if self.pieces_in_leg else None

    def cut_squares(self):
        return cut_faces(self.plan) if self.plan is not None else None

    def set_leg_view(self, shown: bool) -> None:
        shown = bool(shown) and self.surface is not None
        if shown == self.leg_view:
            return
        self.leg_view = shown
        self.leg_view_changed.emit()

    def end_names(self) -> tuple[str, str]:
        """What to call the defect's first and second ends: by the patient's side."""
        session = self.session
        if session.frames is None or len(session.planes) != 2:
            return "first cut", "second cut"
        planes = sorted(session.planes, key=lambda p: _arc_of(session.frames, p.origin))
        middle = self._midline_x()
        names = []
        for plane in planes:
            names.append("patient-left cut" if plane.origin[0] > middle else "patient-right cut")
        if names[0] == names[1]:
            # Both on one side: the one nearer the chin, and the one further back.
            centre = 0.5 * session.frames.length_mm
            arcs = [_arc_of(session.frames, p.origin) for p in planes]
            if abs(arcs[0] - centre) < abs(arcs[1] - centre):
                return "front cut", "back cut"
            return "back cut", "front cut"
        return names[0], names[1]


def _arc_of(frames, point) -> float:
    return float(frames.s[int(np.argmin(np.linalg.norm(frames.points - np.asarray(point), axis=1)))])
