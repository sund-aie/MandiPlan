"""Control panels for each planning step."""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..bending_steps import steps_as_rows
from ..constants import MAX_RESECTION_PLANES
from ..geometry import cpr
from ..geometry.measure import format_deg, format_mm
from ..geometry.plate import bend_table_rows
from ..plate_assets import assets_in_family
from ..plate_assets import families as plate_families
from ..plate_catalog import load_kits
from .histogram import ThresholdPanel
from .modes import Mode
from .theme import set_role


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    set_role(label, "hint")
    return label


class _Section(QWidget):
    """A titled section that opens and closes: detail on demand."""

    def __init__(self, title: str, content: QWidget, expanded: bool = False, parent=None):
        super().__init__(parent)
        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setAutoRaise(True)
        self.content = content
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toggle)
        layout.addWidget(content)
        self.toggle.toggled.connect(self._show)
        self._show(expanded)

    def _show(self, shown: bool) -> None:
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)
        self.content.setVisible(shown)


def _boxed(*widgets: QWidget) -> QWidget:
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(8, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    return holder


def _empty_state(text: str) -> QLabel:
    """What a panel says before it has anything to show."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    set_role(label, "empty")
    return label


class _NumericItem(QTableWidgetItem):
    """Table cell that sorts by its numeric value, not by its text."""

    def __init__(self, text: str, value: float):
        super().__init__(text)
        self.value = value
        self.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

    def __lt__(self, other):  # noqa: D105
        if isinstance(other, _NumericItem):
            mine = self.value if np.isfinite(self.value) else -np.inf
            theirs = other.value if np.isfinite(other.value) else -np.inf
            return mine < theirs
        return super().__lt__(other)


class VolumePanel(QWidget):
    load_requested = pyqtSignal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        self.load_button = QPushButton("Open DICOM folder…")
        self.load_button.setProperty("primary", True)
        self.info = _empty_state(
            "Import a CBCT scan to begin.\nFile \u203a Open DICOM folder, or the "
            "button above."
        )
        self.info.setWordWrap(True)
        self.threshold_panel = ThresholdPanel()
        self.separate = QCheckBox("Separate the mandible from the skull")
        self.separate.setChecked(session.separate_mandible)
        self.separate.setToolTip(
            "Once the arch curve is drawn, the lower jaw is cut free of the upper "
            "teeth and the skull so cuts, mirror and exports touch the mandible only"
        )
        self.surface_info = QLabel("")
        self.surface_info.setWordWrap(True)
        self.mandible_info = QLabel("")
        self.mandible_info.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.load_button)
        layout.addWidget(self.info)
        layout.addWidget(self.threshold_panel)
        layout.addWidget(self.surface_info)
        layout.addWidget(self.separate)
        layout.addWidget(self.mandible_info)
        layout.addStretch(1)

        self.load_button.clicked.connect(self.load_requested)
        self.threshold_panel.threshold_changed.connect(session.set_threshold)
        self.separate.toggled.connect(session.set_separate_mandible)
        session.volume_changed.connect(self.refresh_volume)
        session.surface_changed.connect(self.refresh_surface)
        session.mandible_changed.connect(self.refresh_mandible)
        session.arch_changed.connect(self.refresh_mandible)
        self.refresh_mandible()

    def refresh_mandible(self) -> None:
        session = self.session
        if not session.separate_mandible:
            self.mandible_info.setText("Showing all bone; cuts may reach the skull.")
            set_role(self.mandible_info, "warning")
        elif session.mandible is not None:
            self.mandible_info.setText(session.mandible.summary())
            set_role(self.mandible_info, "hint")
        elif session.frames is None:
            self.mandible_info.setText(
                "The mandible is separated automatically once the arch curve is drawn."
            )
            set_role(self.mandible_info, "empty")
        else:
            self.mandible_info.setText("Separating the mandible…")
            set_role(self.mandible_info, "hint")

    def refresh_volume(self) -> None:
        session = self.session
        if session.volume is None:
            self.info.setText(
                "Import a CBCT scan to begin.\nFile \u203a Open DICOM folder, or the "
                "button above."
            )
            set_role(self.info, "empty")
            return
        self.info.setText("\n".join(session.info.lines(session.volume)))
        set_role(self.info, "hint")
        counts, edges = session.volume.histogram()
        self.threshold_panel.configure(counts, edges, session.threshold)

    def refresh_surface(self) -> None:
        surface = self.session.surface
        if surface is None:
            self.surface_info.setText("")
            return
        self.surface_info.setText(
            f"Surface: {surface.GetNumberOfPoints():,} points, "
            f"enclosed volume {format_mm(self.session.surface_volume_mm3, 0)[:-3]}mm³"
        )


class ArchPanel(QWidget):
    mode_requested = pyqtSignal(object)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        self.place_button = QPushButton("Place arch points in the axial view")
        self.place_button.setCheckable(True)
        self.undo_seed = QPushButton("Remove last point")
        self.clear_seeds = QPushButton("Clear curve")
        self.count = QLabel("0 seed points")

        self.step = QDoubleSpinBox()
        self.step.setRange(0.05, 2.0)
        self.step.setSingleStep(0.05)
        self.step.setDecimals(2)
        self.step.setSuffix(" mm")
        self.step.setValue(session.cpr.step_mm)
        self.slab = QDoubleSpinBox()
        self.slab.setRange(1.0, 60.0)
        self.slab.setSingleStep(1.0)
        self.slab.setSuffix(" mm")
        self.slab.setValue(session.cpr.slab_mm)
        self.mode_box = QComboBox()
        self.mode_box.addItems(list(cpr.AGGREGATION_MODES))
        self.mode_box.setCurrentText(session.cpr.mode)
        self.width = QDoubleSpinBox()
        self.width.setRange(5.0, 120.0)
        self.width.setSingleStep(5.0)
        self.width.setSuffix(" mm")
        self.width.setValue(session.cpr.cross_width_mm)

        form = QFormLayout()
        form.addRow("Sample step", self.step)
        form.addRow("Slab thickness", self.slab)
        form.addRow("Aggregation", self.mode_box)
        form.addRow("Cross-section width", self.width)

        self.summary = QLabel("No arch curve.")
        self.summary.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.place_button)
        layout.addWidget(
            _hint(
                "Draw the curve along the buccal cortex, where the plate will sit. "
                "A curve through the dental arch gives an arc length that is not the "
                "plate length."
            )
        )
        row = QHBoxLayout()
        row.addWidget(self.undo_seed)
        row.addWidget(self.clear_seeds)
        layout.addLayout(row)
        layout.addWidget(self.count)
        box = QGroupBox("Reformat settings")
        box.setLayout(form)
        layout.addWidget(box)
        layout.addWidget(self.summary)
        layout.addStretch(1)

        self.place_button.toggled.connect(
            lambda on: self.mode_requested.emit(Mode.ARCH if on else Mode.NAVIGATE)
        )
        self.undo_seed.clicked.connect(session.remove_last_arch_seed)
        self.clear_seeds.clicked.connect(session.clear_arch)
        self.step.valueChanged.connect(lambda v: session.set_cpr_settings(step_mm=v))
        self.slab.valueChanged.connect(lambda v: session.set_cpr_settings(slab_mm=v))
        self.mode_box.currentTextChanged.connect(
            lambda v: session.set_cpr_settings(mode=v)
        )
        self.width.valueChanged.connect(
            lambda v: session.set_cpr_settings(cross_width_mm=v)
        )
        session.arch_changed.connect(self.refresh)

    def refresh(self) -> None:
        session = self.session
        self.count.setText(f"{len(session.arch_seeds)} seed points")
        if session.arch_curve is None:
            self.summary.setText("No arch curve (place at least 2 points).")
        else:
            self.summary.setText(
                f"Arch curve length: {format_mm(session.arch_curve.length_mm)}\n"
                "(arc length along the curve, not a straight-line distance)"
            )


class ResectionPanel(QWidget):
    mode_requested = pyqtSignal(object)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        self.add_plane = QPushButton("Add cutting plane")
        self.flip_1 = QPushButton("Flip cut 1")
        self.flip_2 = QPushButton("Flip cut 2")
        self.clear = QPushButton("Clear planes")
        self.execute = QPushButton("Execute cut")
        self.undo = QPushButton("Undo cut")
        self.plane_box = QComboBox()
        self.position = QDoubleSpinBox()
        self.position.setRange(0.0, 1000.0)
        self.position.setSuffix(" mm along curve")
        self.position.setSingleStep(1.0)
        self.yaw = QDoubleSpinBox()
        self.yaw.setRange(-89.0, 89.0)
        self.yaw.setSuffix("° yaw")
        self.tilt = QDoubleSpinBox()
        self.tilt.setRange(-89.0, 89.0)
        self.tilt.setSuffix("° tilt")
        self.roll = QDoubleSpinBox()
        self.roll.setRange(-180.0, 180.0)
        self.roll.setSuffix("° roll")
        self.offsets = {}
        for key, label in (("x", " mm L/R"), ("y", " mm A/P"), ("z", " mm S/I")):
            spin = QDoubleSpinBox()
            spin.setRange(-60.0, 60.0)
            spin.setSingleStep(0.5)
            spin.setSuffix(label)
            self.offsets[key] = spin

        self.landmark_button = QPushButton("Mark tumour margin point")
        self.landmark_button.setCheckable(True)
        self.clear_landmarks = QPushButton("Clear margin points")
        self.readout = QLabel("No cutting planes.")
        self.readout.setWordWrap(True)
        self.readout.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        grid = QGridLayout()
        grid.addWidget(self.add_plane, 0, 0, 1, 2)
        grid.addWidget(self.flip_1, 1, 0)
        grid.addWidget(self.flip_2, 1, 1)
        grid.addWidget(self.clear, 2, 0)
        grid.addWidget(self.execute, 2, 1)
        grid.addWidget(self.undo, 3, 0, 1, 2)

        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(
            _hint(
                f"Up to {MAX_RESECTION_PLANES} planes. Drag the handles in the 3-D "
                "view; the red bone is what the current planes would remove."
            )
        )
        numbers = QFormLayout()
        numbers.addRow("Cut", self.plane_box)
        numbers.addRow("Position", self.position)
        numbers.addRow("Obliquity", self.yaw)
        numbers.addRow("Inclination", self.tilt)
        numbers.addRow("Roll", self.roll)
        for key in ("x", "y", "z"):
            numbers.addRow("Offset" if key == "x" else "", self.offsets[key])
        numeric_box = QGroupBox("Place the cut by numbers")
        numeric_box.setLayout(numbers)
        layout.addWidget(numeric_box)
        layout.addWidget(
            _hint(
                "Position runs along the arch curve; the cut re-angles itself to "
                "the jaw as it moves. Yaw, tilt and roll are measured against the "
                "local mandibular frame at the cut — the arch tangent, transported "
                "superior, and buccolingual — so an obliquity you dial in here is "
                "kept relative to the anatomy wherever you move the cut to. The "
                "offsets shift it in patient axes. Dragging in 3-D updates these."
            )
        )
        layout.addWidget(self.landmark_button)
        layout.addWidget(self.clear_landmarks)
        layout.addWidget(self.readout)
        layout.addStretch(1)

        self.flip_1.clicked.connect(lambda: self._flip(0))
        self.flip_2.clicked.connect(lambda: self._flip(1))
        self.clear.clicked.connect(session.clear_planes)
        self.execute.clicked.connect(session.execute_cut)
        self.undo.clicked.connect(session.undo_cut)
        self.clear_landmarks.clicked.connect(session.clear_landmarks)
        self.landmark_button.toggled.connect(
            lambda on: self.mode_requested.emit(Mode.LANDMARK if on else Mode.NAVIGATE)
        )
        self.plane_box.currentIndexChanged.connect(self._load_plane_controls)
        # Translation and rotation are wired to different handlers on purpose:
        # moving a cut must never re-angle it.
        for spin in (self.position, *self.offsets.values()):
            spin.valueChanged.connect(self._apply_translation)
        for spin in (self.yaw, self.tilt, self.roll):
            spin.valueChanged.connect(self._apply_rotation)
        session.resection_changed.connect(self.refresh)
        session.arch_changed.connect(self.refresh)

    def _current_plane(self) -> int:
        return max(self.plane_box.currentIndex(), 0)

    def _load_plane_controls(self) -> None:
        """Show the selected plane's numbers without re-applying them."""
        session = self.session
        index = self._current_plane()
        if index >= len(session.planes) or session.frames is None:
            return
        placement = session.placement(index)
        self._loading = True
        try:
            self.position.setRange(0.0, session.frames.length_mm)
            self.position.setValue(placement.s_mm)
            self.yaw.setValue(placement.yaw_deg)
            self.tilt.setValue(placement.tilt_deg)
            self.roll.setValue(placement.roll_deg)
            for key, value in zip(("x", "y", "z"), placement.offset_mm):
                self.offsets[key].setValue(float(value))
        finally:
            self._loading = False

    def _apply_translation(self) -> None:
        if getattr(self, "_loading", False):
            return
        index = self._current_plane()
        if index >= len(self.session.planes):
            return
        self.session.translate_plane(
            index,
            self.position.value(),
            [self.offsets[key].value() for key in ("x", "y", "z")],
        )

    def _apply_rotation(self) -> None:
        if getattr(self, "_loading", False):
            return
        index = self._current_plane()
        if index >= len(self.session.planes):
            return
        self.session.rotate_plane(
            index, self.yaw.value(), self.tilt.value(), self.roll.value()
        )

    def _flip(self, index: int) -> None:
        if index < len(self.session.planes):
            self.session.flip_plane(index)

    def summary_rows(self) -> list[tuple[str, str]]:
        report = self.session.report
        if report is None:
            return []
        rows = [
            ("resected segment, arc length along arch curve (mm)", _plain(report.arc_length_mm)),
            ("resected segment, straight-line between cuts (mm)", _plain(report.straight_length_mm)),
            ("resected fragment volume (mm^3)", _plain(report.fragment_volume_mm3)),
        ]
        for plane_label, name, distance in report.margins_mm:
            rows.append(
                (f"margin: {name} to {plane_label} (mm, + = inside resected side)",
                 _plain(distance))
            )
        return rows

    def refresh(self) -> None:
        session = self.session
        enabled = bool(session.planes) and session.frames is not None
        for widget in (
            self.position,
            self.yaw,
            self.tilt,
            self.roll,
            *self.offsets.values(),
        ):
            widget.setEnabled(enabled)
        if self.plane_box.count() != len(session.planes):
            self._loading = True
            self.plane_box.clear()
            for plane in session.planes:
                self.plane_box.addItem(plane.label)
            self._loading = False
            self._load_plane_controls()
        if not session.planes:
            self.readout.setText("No cutting planes.")
            return
        report = session.report
        lines = [f"Cutting planes: {len(session.planes)}"]
        lines.append(
            f"Resected segment (arc length along arch curve): "
            f"{format_mm(report.arc_length_mm)}"
        )
        lines.append(
            f"Straight-line distance between cuts: {format_mm(report.straight_length_mm)}"
        )
        if np.isfinite(report.fragment_volume_mm3):
            lines.append(f"Resected fragment volume: {report.fragment_volume_mm3:.0f} mm³")
        if report.margins_mm:
            lines.append("Margins (positive = inside the resected side):")
            for plane_label, name, distance in report.margins_mm:
                lines.append(f"  {name} → {plane_label}: {format_mm(distance)}")
        if not session.cut_applied:
            lines.append("Cut not executed yet.")
        self.readout.setText("\n".join(lines))


class ReconstructionPanel(QWidget):
    """Rebuild the resected segment from the mirrored healthy side, flush."""

    mode_requested = pyqtSignal(object)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        self.mirror_button = QPushButton("Reconstruct from the healthy side")
        self.mirror_button.setProperty("primary", True)
        self.mirror_button.setToolTip(
            "Mirror the healthy side into the defect, register it to both "
            "stumps and blend it in: one flush surface"
        )
        self.estimate_button = QPushButton("Re-estimate the plane of symmetry")
        self.plane_info = QLabel("")
        self.plane_info.setWordWrap(True)
        self.coverage_info = QLabel("")
        self.coverage_info.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.mirror_button)
        layout.addWidget(self.coverage_info)
        layout.addWidget(self.plane_info)
        layout.addWidget(self.estimate_button)
        layout.addWidget(
            _hint(
                "The healthy side is mirrored in the patient's own plane of "
                "symmetry, then registered to each cut stump so the mirror meets "
                "the bone that stays, and blended into it before one surface is "
                "built. The junction figures below are how far apart mirror and "
                "stump were before and after that registration. Where the defect "
                "crosses the midline there is no healthy counterpart; that part "
                "is filled from the pre-operative contour and shown in sand."
            )
        )
        # -- optional hand refinement of the computed result ----------------
        self.smooth_junctions = QPushButton("Smooth the junctions")
        self.smooth_junctions.setToolTip(
            "Computed: rounds the surface off within 3 mm of each cut"
        )
        self.sculpt_button = QPushButton("Brush on the jaw")
        self.sculpt_button.setCheckable(True)
        self.brush_box = QComboBox()
        self.brush_box.addItem("Smooth", "smooth")
        self.brush_box.addItem("Fill (add bone)", "fill")
        self.brush_box.addItem("Carve (remove bone)", "carve")
        self.brush_radius = QDoubleSpinBox()
        self.brush_radius.setRange(1.0, 15.0)
        self.brush_radius.setSingleStep(0.5)
        self.brush_radius.setSuffix(" mm")
        self.brush_radius.setValue(session.brush_radius_mm)
        self.brush_strength = QSlider(Qt.Orientation.Horizontal)
        self.brush_strength.setRange(5, 100)
        self.brush_strength.setValue(int(round(session.brush_strength * 100)))
        self.undo_edit = QPushButton("Undo edit")
        self.reset_edits = QPushButton("Back to computed")
        self.edit_info = QLabel("")
        self.edit_info.setWordWrap(True)

        refine = QFormLayout()
        refine.addRow(self.smooth_junctions)
        refine.addRow(self.sculpt_button)
        refine.addRow("Brush", self.brush_box)
        refine.addRow("Size", self.brush_radius)
        refine.addRow("Strength", self.brush_strength)
        edits = QHBoxLayout()
        edits.addWidget(self.undo_edit)
        edits.addWidget(self.reset_edits)
        refine.addRow(edits)
        refine.addRow(self.edit_info)
        self.refine_box = QGroupBox("Refine the result (optional)")
        self.refine_box.setLayout(refine)
        layout.addWidget(self.refine_box)
        layout.addStretch(1)

        self.estimate_button.clicked.connect(session.estimate_symmetry_plane)
        self.mirror_button.clicked.connect(session.build_reconstruction)
        self.smooth_junctions.clicked.connect(session.smooth_junctions)
        self.sculpt_button.toggled.connect(
            lambda on: self.mode_requested.emit(Mode.SCULPT if on else Mode.NAVIGATE)
        )
        self.brush_box.currentIndexChanged.connect(
            lambda _i: setattr(session, "brush", self.brush_box.currentData())
        )
        self.brush_radius.valueChanged.connect(
            lambda v: setattr(session, "brush_radius_mm", float(v))
        )
        self.brush_strength.valueChanged.connect(
            lambda v: setattr(session, "brush_strength", v / 100.0)
        )
        self.undo_edit.clicked.connect(session.undo_edit)
        self.reset_edits.clicked.connect(session.reset_edits)
        session.reconstruction_changed.connect(self.refresh)
        session.reconstruction_edited.connect(self.refresh_edits)
        self.refresh()

    def refresh_edits(self) -> None:
        session = self.session
        ready = session.reconstruction is not None
        self.refine_box.setEnabled(ready)
        self.undo_edit.setEnabled(session.can_undo_edit)
        moved = session.reconstruction_edited_mm
        self.reset_edits.setEnabled(moved > 0.0)
        if not ready:
            self.edit_info.setText("Available once the jaw is reconstructed.")
            set_role(self.edit_info, "empty")
        elif moved > 0.0:
            self.edit_info.setText(
                f"Edited by hand: up to {moved:.2f} mm from the computed surface. "
                "Exports include the edits."
            )
            set_role(self.edit_info, "warning")
        else:
            self.edit_info.setText("As computed: no hand edits.")
            set_role(self.edit_info, "hint")

    def refresh(self) -> None:
        session = self.session
        self.refresh_edits()
        plane = session.symmetry_plane
        if plane is None:
            self.plane_info.setText(
                "The plane of symmetry is found automatically when you reconstruct."
            )
            set_role(self.plane_info, "hint")
        else:
            self.plane_info.setText(
                f"Plane of symmetry tilted {plane.tilt_deg:.2f}° from the "
                f"left-right axis; {plane.symmetry:.0%} of bone mirrors onto bone."
            )
            set_role(self.plane_info, "warning" if plane.symmetry < 0.75 else "hint")
        reconstruction = session.reconstruction
        lines = []
        if session.coverage is not None:
            lines.append(session.coverage.summary())
        if reconstruction is None:
            lines.append(
                "Reconstructing replaces the pre-operative bone in the view: ivory "
                "is retained bone, teal the mirrored segment."
                if session.planes
                else "Place the cuts, then reconstruct."
            )
            self.coverage_info.setText("\n".join(lines))
            set_role(self.coverage_info, "empty")
            return
        lines.extend(reconstruction.report.summary_lines())
        lines.append(f"Mirrored segment volume: {session.graft_volume_mm3:.0f} mm³")
        self.coverage_info.setText("\n".join(lines))
        set_role(
            self.coverage_info,
            "warning" if reconstruction.report.donorless_voxels else "hint",
        )


class PlatePanel(QWidget):
    mode_requested = pyqtSignal(object)
    #: plate mesh, hole centres, screw trajectories
    overlays_changed = pyqtSignal(bool, bool, bool, bool)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        # -- plate library -------------------------------------------------
        self.family_box = QComboBox()
        for family_id, label in plate_families():
            self.family_box.addItem(label, family_id)
        self.model_box = QComboBox()
        self.status_badge = QLabel("")
        self.clearance = QDoubleSpinBox()
        self.clearance.setRange(0.0, 5.0)
        self.clearance.setSingleStep(0.1)
        self.clearance.setDecimals(1)
        self.clearance.setSuffix(" mm clearance")
        self.clearance.setValue(session.plate_clearance_mm)
        self.clearance.setToolTip(
            "How far the plate's inner face stands off the bone surface"
        )
        self.show_distortion = QCheckBox("Show holes as bending leaves them")
        self.show_distortion.setChecked(True)
        self.show_distortion.setToolTip(
            "Screw holes do not stay round through contouring. Show the "
            "predicted oval, or the catalogue's nominal circle."
        )
        self.use_insets = QCheckBox("Bending insets fitted")
        self.use_insets.setTristate(True)
        self.use_insets.setCheckState(Qt.CheckState.PartiallyChecked)
        self.use_insets.setToolTip(
            "Insets fill the threaded holes while the plate is contoured and "
            "keep them round. Leave partially checked to follow the kit's own "
            "default."
        )
        self.hole_report = QLabel("")
        self.hole_report.setWordWrap(True)
        self.bend_plate = QCheckBox("Bend the plate to the path")
        self.bend_plate.setChecked(True)
        self.bend_plate.setToolTip(
            "Sweep the inter-hole bridges onto the path. Screw-hole "
            "neighbourhoods stay rigid, so hole diameter and plate thickness "
            "are unchanged."
        )
        self.show_plate = QCheckBox("Plate mesh")
        self.show_plate.setChecked(True)
        self.show_holes = QCheckBox("Hole centres")
        self.show_holes.setChecked(True)
        self.show_screws = QCheckBox("Screw trajectories")
        self.show_marking = QCheckBox("Etched marking")
        self.show_marking.setChecked(True)
        self.show_marking.setToolTip(
            "The laser mark on the plate face, at its real 100 um etch depth"
        )
        self.properties = QLabel("")
        self.properties.setWordWrap(True)
        self.fit_status = QLabel("")
        self.fit_status.setWordWrap(True)

        self.kit_box = QComboBox()
        for kit in load_kits():
            self.kit_box.addItem(kit.name, kit.id)

        self.draw_button = QPushButton("Draw plate path on the bone")
        self.draw_button.setCheckable(True)
        self.draw_button.setProperty("primary", True)
        self.undo_point = QPushButton("Remove last point")
        self.clear_path = QPushButton("Clear path")
        self.verdict = QLabel("")
        self.verdict.setWordWrap(True)
        self.use_length = QPushButton("")
        self.use_length.setVisible(False)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            [
                "node",
                "from start (mm)",
                "in-plane (°)",
                "out-of-plane (°)",
                "twist (°)",
                "segment (mm)",
            ]
        )
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)

        self.steps_table = QTableWidget(0, 5)
        self.steps_table.setHorizontalHeaderLabels(
            ["step", "kind", "from cut end", "angle", "instruction"]
        )
        self.steps_table.verticalHeader().setVisible(False)
        self.steps_table.setWordWrap(True)
        self.steps_table.horizontalHeader().setStretchLastSection(True)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.table, "Bend table")
        self.tabs.addTab(self.steps_table, "Bench steps")
        self.tabs.setMinimumHeight(220)

        self.summary = QLabel("No plate path.")
        self.summary.setWordWrap(True)
        self.fit_info = QLabel("")
        self.fit_info.setWordWrap(True)

        layout = QVBoxLayout(self)

        # 1. Draw the path: the one thing this step is for.
        layout.addWidget(self.draw_button)
        layout.addWidget(
            _hint(
                "Click along the outer face of the jaw where the plate will lie; "
                "after reconstructing, click across the rebuilt segment too."
            )
        )
        row = QHBoxLayout()
        row.addWidget(self.undo_point)
        row.addWidget(self.clear_path)
        layout.addLayout(row)

        # 2. Which plate.
        library = QFormLayout()
        library.addRow("Plate", self.family_box)
        library.addRow("Length", self.model_box)
        library.addRow("", self.status_badge)
        library_box = QGroupBox("Plate")
        library_box.setLayout(library)
        layout.addWidget(library_box)

        # 3. Does it fit, in one line; the reasons on demand.
        layout.addWidget(self.verdict)
        layout.addWidget(self.use_length)
        layout.addWidget(self.summary)
        self.details = _Section(
            "Why: fit, contact and screw holes",
            _boxed(self.fit_info, self.fit_status, self.hole_report),
        )
        layout.addWidget(self.details)

        # 4. Bending.
        bending = QFormLayout()
        bending.addRow("", self.bend_plate)
        bending.addRow("Standoff", self.clearance)
        bending.addRow("Bending kit", self.kit_box)
        bending_box = QGroupBox("Bending")
        bending_box.setLayout(bending)
        layout.addWidget(bending_box)
        layout.addWidget(
            _Section("Screw-hole options", _boxed(self.show_distortion, self.use_insets))
        )
        toggles = QGridLayout()
        toggles.addWidget(self.show_plate, 0, 0)
        toggles.addWidget(self.show_holes, 0, 1)
        toggles.addWidget(self.show_screws, 1, 0)
        toggles.addWidget(self.show_marking, 1, 1)
        show = QWidget()
        show.setLayout(toggles)
        layout.addWidget(_Section("Show", show))
        layout.addWidget(_Section("Plate properties", _boxed(self.properties)))
        layout.addWidget(self.tabs, 1)

        self.family_box.currentIndexChanged.connect(self._on_family)
        self.model_box.currentIndexChanged.connect(self._on_model)
        self.clearance.valueChanged.connect(session.set_plate_clearance)
        self.bend_plate.toggled.connect(session.set_plate_bending)
        self.show_distortion.toggled.connect(session.set_hole_distortion_shown)
        self.use_insets.stateChanged.connect(self._on_insets)
        for box in (
            self.show_plate,
            self.show_holes,
            self.show_screws,
            self.show_marking,
        ):
            box.toggled.connect(self._emit_overlays)
        self.kit_box.currentIndexChanged.connect(
            lambda i: session.set_bending_kit(self.kit_box.itemData(i))
        )
        self.draw_button.toggled.connect(
            lambda on: self.mode_requested.emit(Mode.PLATE if on else Mode.NAVIGATE)
        )
        self.undo_point.clicked.connect(session.remove_last_plate_point)
        self.clear_path.clicked.connect(session.clear_plate_path)
        self.use_length.clicked.connect(session.use_fitting_length)
        session.plate_changed.connect(self.refresh)
        self._reload_models()

    # -- plate library ---------------------------------------------------

    def _reload_models(self) -> None:
        """Repopulate the model list for the selected family."""
        family = self.family_box.currentData()
        self._loading_models = True
        try:
            self.model_box.clear()
            for asset in assets_in_family(family):
                self.model_box.addItem(
                    f"{asset.hole_count} holes, {asset.length_mm:.0f} mm", asset.id
                )
        finally:
            self._loading_models = False
        self._on_model()

    def _on_family(self) -> None:
        self._reload_models()

    def _on_model(self) -> None:
        if getattr(self, "_loading_models", False):
            return
        asset_id = self.model_box.currentData()
        if asset_id:
            self.session.set_plate_asset(asset_id)
        self.refresh_library()

    def _on_insets(self, state: int) -> None:
        """Tri-state: follow the kit, force insets on, or force them off."""
        value = {
            Qt.CheckState.PartiallyChecked.value: None,
            Qt.CheckState.Checked.value: True,
            Qt.CheckState.Unchecked.value: False,
        }.get(int(state))
        self.session.set_bending_insets(value)

    def _emit_overlays(self) -> None:
        self.overlays_changed.emit(
            self.show_plate.isChecked(),
            self.show_holes.isChecked(),
            self.show_screws.isChecked(),
            self.show_marking.isChecked(),
        )

    def _select_current_asset(self) -> None:
        """Show the plate the session is using, however it was chosen."""
        asset = self.session.plate_asset
        if asset is None:
            return
        if self.family_box.currentData() != asset.family:
            index = self.family_box.findData(asset.family)
            if index >= 0:
                self.family_box.blockSignals(True)
                self.family_box.setCurrentIndex(index)
                self.family_box.blockSignals(False)
                self._loading_models = True
                try:
                    self.model_box.clear()
                    for candidate in assets_in_family(asset.family):
                        self.model_box.addItem(
                            f"{candidate.hole_count} holes, {candidate.length_mm:.0f} mm",
                            candidate.id,
                        )
                finally:
                    self._loading_models = False
        index = self.model_box.findData(asset.id)
        if index >= 0 and index != self.model_box.currentIndex():
            self.model_box.blockSignals(True)
            self.model_box.setCurrentIndex(index)
            self.model_box.blockSignals(False)

    def refresh_library(self) -> None:
        """Mirror the selected asset's identity, status and fit into the panel."""
        self._select_current_asset()
        asset = self.session.plate_asset
        if asset is None:
            self.status_badge.setText("No plate selected")
            set_role(self.status_badge, "hint")
            self.properties.setText("Choose a plate model to see its properties.")
            self.fit_status.setText("")
            return

        self.status_badge.setText(asset.status_label)
        set_role(
            self.status_badge, "badge-exact" if asset.exact else "badge-generic"
        )
        self.properties.setText("\n".join(asset.summary_lines()))

        fitted = self.session.fitted_plate
        warnings = self.session.plate_fit_warnings
        if fitted is None:
            self.fit_status.setText("Draw a plate path to fit this plate to it.")
            set_role(self.fit_status, "hint")
            return
        bent = self.session.bent_plate
        problems = self.session.plate_fit_problems
        if bent is None:
            text = (
                f"Placed rigidly on {fitted.holes_used} of {asset.hole_count} "
                f"holes. Residual: max {fitted.max_residual_mm:.2f} mm, "
                f"rms {fitted.rms_residual_mm:.2f} mm."
            )
        else:
            finite = [a for a in bent.bend_angles_deg if a == a]
            worst = max(finite) if finite else 0.0
            text = (
                f"Bent onto the path. Screw holes held rigid; bridges swept. "
                f"Largest bend at a hole {worst:.1f}°."
            )
        contact = self.session.plate_contact
        if contact is not None and len(contact.clearance_mm):
            text += (
                f"\nBone clearance {contact.clearance_mm.min():.2f} to "
                f"{contact.clearance_mm.max():.2f} mm."
            )
        for line in problems:
            text += f"\nProblem: {line}"
        for line in warnings:
            text += f"\n{line}"
        self.fit_status.setText(text)
        set_role(
            self.fit_status,
            "danger" if problems else ("warning" if warnings else "hint"),
        )
        self._refresh_hole_report()

    def _refresh_hole_report(self) -> None:
        """What this bending plan does to the screw holes, hole by hole."""
        report = self.session.hole_distortion
        if report is None:
            self.hole_report.setText(
                "Draw a plate path and bend the plate to see what the bends "
                "do to its screw holes."
            )
            set_role(self.hole_report, "hint")
            return
        lines = list(report.summary_lines())
        compromised = [h for h in report.holes if not h.takes_locking_screw]
        for hole in compromised[:4]:
            lines.append(hole.describe())
        if len(compromised) > 4:
            lines.append(f"...and {len(compromised) - 4} more.")
        self.hole_report.setText("\n".join(lines))
        set_role(
            self.hole_report,
            "danger" if report.problems else ("warning" if compromised else "hint"),
        )

    def _refresh_fit(self) -> None:
        fit = self.session.fit
        self.steps_table.setRowCount(0)
        if fit is None:
            self.fit_info.setText("")
            return
        self._refresh_verdict(fit)
        lines = [fit.verdict]
        if fit.holes_proximal or fit.holes_distal:
            lines.append(
                f"Screw holes: {fit.holes_proximal} proximal, "
                f"{fit.holes_over_defect} over the defect, {fit.holes_distal} distal."
            )
        lines.extend(f"Problem: {p}" for p in fit.problems)
        lines.extend(f"Note: {w}" for w in fit.warnings[:4])
        self.fit_info.setText("\n".join(lines))
        set_role(self.fit_info, "danger" if fit.problems else "hint")

        rows = steps_as_rows(self.session.steps)
        self.steps_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                self.steps_table.setItem(r, c, QTableWidgetItem(text))
        self.steps_table.resizeColumnsToContents()
        self.steps_table.resizeRowsToContents()

    def _refresh_verdict(self, fit) -> None:
        session = self.session
        problems = list(fit.problems) + list(session.plate_fit_problems)
        warnings = list(session.plate_fit_warnings)
        if problems:
            text = f"Needs attention: {problems[0]}"
            if len(problems) > 1:
                text += f" (+{len(problems) - 1} more below)"
            role = "danger"
        elif warnings:
            text = f"Fits, with {len(warnings)} point(s) to check below."
            role = "warning"
        else:
            text = "Fits: the plate spans the plan with sound screw purchase each side."
            role = "hint"
        self.verdict.setText(text)
        set_role(self.verdict, role)
        asset = session.plate_asset
        option = fit.option
        if asset is not None and option is not None and option.holes != asset.hole_count:
            self.use_length.setText(
                f"Use the {option.holes}-hole length ({option.length_mm:.0f} mm), "
                "the shortest that spans the plan"
            )
            self.use_length.setVisible(True)
        else:
            self.use_length.setVisible(False)

    def refresh(self) -> None:
        self.refresh_library()
        plan = self.session.plate_plan
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self._refresh_fit()
        if plan is None or self.session.fit is None:
            self.verdict.setText("")
            self.use_length.setVisible(False)
        if plan is None:
            self.summary.setText(
                f"{len(self.session.plate_points)} path points — "
                "at least two, spanning one pitch, are needed."
            )
            self.table.setSortingEnabled(True)
            return

        rows = bend_table_rows(plan)
        self.table.setRowCount(len(rows))
        for r, (node, text_row) in enumerate(zip(plan.bends, rows)):
            values = [
                float(node.index),
                node.cumulative_mm,
                node.in_plane_deg,
                node.out_of_plane_deg,
                node.twist_deg,
                node.segment_length_mm,
            ]
            for c, (text, value) in enumerate(zip(text_row, values)):
                self.table.setItem(r, c, _NumericItem(text, value))
        self.table.sortItems(0, Qt.SortOrder.AscendingOrder)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()

        bends = [b for b in plan.bends if np.isfinite(b.in_plane_deg)]
        largest = max((abs(b.in_plane_deg) for b in bends), default=float("nan"))
        self.summary.setText(
            f"Plate length along the path: {format_mm(plan.total_length_mm)}\n"
            f"{len(plan.nodes)} screw holes at {format_mm(plan.pitch_mm)} pitch\n"
            f"Largest in-plane bend: {format_deg(largest)}"
        )


def _plain(value: float) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.3f}"


class ExportPanel(QWidget):
    """Every file the plan produces, in one place, each saying what it is for."""

    #: (key, button text, what it is for)
    ITEMS = (
        ("jaw", "Reconstructed jaw (STL)", "The mandible as rebuilt, to print as a model."),
        ("segment", "Mirrored segment only (STL)", "Just the part that fills the defect."),
        ("fragment", "Resected segment (STL)", "The bone the cuts remove."),
        ("plate", "Bent plate (STL)", "The plate as planned, to print and bend against."),
        ("guide", "Bending guide (STL + table)", "Clip-on guide that stops each bend at its angle."),
        ("bends", "Bend table (CSV)", "Angle at every screw hole."),
        ("steps", "Bench steps (CSV)", "The bending steps for the chosen kit, in order."),
        ("resection", "Resection summary (CSV)", "Cut positions, angles and margins."),
    )
    export_requested = pyqtSignal(str)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.buttons: dict[str, QPushButton] = {}
        layout = QVBoxLayout(self)
        layout.addWidget(
            _hint(
                "Every file carries the MandiPlan attribution; tables carry the "
                "disclaimer. Plates are generic approximations — check them "
                "against the plate in your hand."
            )
        )
        for key, text, purpose in self.ITEMS:
            button = QPushButton(text + "…")
            button.setToolTip(purpose)
            button.clicked.connect(lambda _c=False, k=key: self.export_requested.emit(k))
            self.buttons[key] = button
            layout.addWidget(button)
            layout.addWidget(_hint(purpose))
        layout.addStretch(1)
        for signal in (
            session.reconstruction_changed,
            session.resection_changed,
            session.plate_changed,
            session.volume_changed,
        ):
            signal.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        session = self.session
        ready = {
            "jaw": session.reconstruction is not None,
            "segment": session.reconstruction is not None,
            "fragment": bool(session.planes) and session.surface is not None,
            "plate": session.plate_mesh() is not None,
            "guide": session.bent_plate is not None,
            "bends": session.plate_plan is not None,
            "steps": bool(session.steps),
            "resection": session.report is not None,
        }
        for key, button in self.buttons.items():
            button.setEnabled(ready[key])
