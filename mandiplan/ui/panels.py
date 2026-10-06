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
from ..geometry.mirror import SYMMETRY_TOLERANCE_MM, SYMMETRY_WARNING
from ..geometry.measure import format_mm
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
            "The lower jaw is found and cut free of the upper teeth and the skull "
            "when the scan opens, so cuts, mirror and exports touch the mandible only"
        )
        self.surface_info = QLabel("")
        self.surface_info.setWordWrap(True)
        self.mandible_info = QLabel("")
        self.mandible_info.setWordWrap(True)
        self.separate_again = QPushButton("Separate again along my curve")
        self.separate_again.setToolTip(
            "Use the arch curve you drew (step 2) to find the mandible, if the "
            "automatic separation missed part of it"
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.load_button)
        layout.addWidget(self.info)
        layout.addWidget(self.threshold_panel)
        layout.addWidget(self.surface_info)
        layout.addWidget(self.separate)
        layout.addWidget(self.mandible_info)
        layout.addWidget(self.separate_again)
        layout.addStretch(1)

        self.load_button.clicked.connect(self.load_requested)
        self.threshold_panel.threshold_changed.connect(session.set_threshold)
        self.separate.toggled.connect(session.set_separate_mandible)
        self.separate_again.clicked.connect(session.separate_along_arch)
        session.volume_changed.connect(self.refresh_volume)
        session.surface_changed.connect(self.refresh_surface)
        session.mandible_changed.connect(self.refresh_mandible)
        session.volume_changed.connect(self.refresh_mandible)
        session.arch_changed.connect(self.refresh_mandible)
        self.refresh_mandible()

    def refresh_mandible(self) -> None:
        session = self.session
        self.separate.blockSignals(True)
        self.separate.setChecked(session.separate_mandible)
        self.separate.blockSignals(False)
        self.separate_again.setVisible(
            session.separate_mandible
            and session.volume is not None
            and session.frames is not None
            and session.arch_source == "drawn"
        )
        if session.volume is None:
            self.mandible_info.setText("The mandible is separated from the skull when a scan is opened.")
            set_role(self.mandible_info, "empty")
        elif not session.separate_mandible:
            self.mandible_info.setText("Showing all bone; cuts may reach the skull.")
            set_role(self.mandible_info, "warning")
        elif session.mandible is not None:
            how = "" if session.mandible_source == "auto" else " Separated along your curve."
            self.mandible_info.setText(f"{session.mandible.summary()}{how}")
            set_role(self.mandible_info, "hint")
        elif session.mandible_problem:
            self.mandible_info.setText(session.mandible_problem)
            set_role(self.mandible_info, "warning")
        else:
            self.mandible_info.setText("Separating the mandible from the skull…")
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
        if self.session.mandible is not None:
            # The mandible line below says how much bone there is.
            self.surface_info.setText("")
            return
        self.surface_info.setText(
            f"Bone at this threshold: {self.session.surface_volume_mm3 / 1000:.1f} cm³"
        )


class ArchPanel(QWidget):
    """The curve along the mandible that every reformat and cut follows."""

    mode_requested = pyqtSignal(object)

    #: Panoramic slab aggregation, as the operator reads it.
    AGGREGATION_LABELS = {"max": "Brightest bone (sharp outline)", "mean": "Average (like an OPG)"}

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.place_button = QPushButton("Draw my own curve")
        self.place_button.setCheckable(True)
        self.place_button.setToolTip(
            "Click along the jaw on the axial slice (Slices tab); the first click "
            "replaces the automatic curve"
        )
        self.automatic = QPushButton("Use the automatic curve")
        self.undo_seed = QPushButton("Remove last point")
        self.clear_seeds = QPushButton("Clear")

        self.slab = QDoubleSpinBox()
        self.slab.setRange(1.0, 60.0)
        self.slab.setSingleStep(1.0)
        self.slab.setSuffix(" mm")
        self.slab.setValue(session.cpr.slab_mm)
        self.mode_box = QComboBox()
        for mode in cpr.AGGREGATION_MODES:
            self.mode_box.addItem(self.AGGREGATION_LABELS.get(mode, mode), mode)
        self.mode_box.setCurrentIndex(max(self.mode_box.findData(session.cpr.mode), 0))
        self.width = QDoubleSpinBox()
        self.width.setRange(5.0, 120.0)
        self.width.setSingleStep(5.0)
        self.width.setSuffix(" mm")
        self.width.setValue(session.cpr.cross_width_mm)
        form = QFormLayout()
        form.addRow("Panoramic depth", self.slab)
        form.addRow("Panoramic shows", self.mode_box)
        form.addRow("Cross-section width", self.width)
        settings = QWidget()
        settings.setLayout(form)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.place_button)
        self.drawn_row = QWidget()
        row = QHBoxLayout(self.drawn_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.undo_seed)
        row.addWidget(self.clear_seeds)
        layout.addWidget(self.drawn_row)
        layout.addWidget(self.automatic)
        layout.addWidget(_Section("Panoramic view settings", settings))
        layout.addStretch(1)

        self.place_button.toggled.connect(
            lambda on: self.mode_requested.emit(Mode.ARCH if on else Mode.NAVIGATE)
        )
        self.automatic.clicked.connect(session.use_automatic_arch)
        self.undo_seed.clicked.connect(session.remove_last_arch_seed)
        self.clear_seeds.clicked.connect(session.clear_arch)
        self.slab.valueChanged.connect(lambda v: session.set_cpr_settings(slab_mm=v))
        self.mode_box.currentIndexChanged.connect(
            lambda _i: session.set_cpr_settings(mode=self.mode_box.currentData())
        )
        self.width.valueChanged.connect(
            lambda v: session.set_cpr_settings(cross_width_mm=v)
        )
        session.arch_changed.connect(self.refresh)
        session.volume_changed.connect(self.refresh)
        session.mandible_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        session = self.session
        drawn = session.arch_source == "drawn"
        curve = session.arch_curve
        if session.volume is None:
            text, role = "Open a scan first.", "empty"
        elif curve is None and drawn:
            text, role = (
                f"{len(session.arch_seeds)} point(s) so far: keep clicking along the "
                "jaw on the axial slice.",
                "hint",
            )
        elif curve is None:
            text, role = (
                "The curve is laid along the mandible, condyle to condyle, as soon "
                "as the mandible is found.",
                "empty",
            )
        elif drawn:
            text, role = (
                f"Your curve: {len(session.arch_seeds)} points, "
                f"{format_mm(curve.length_mm)} along the jaw.",
                "status",
            )
        else:
            text, role = (
                "Laid automatically along the middle of the mandible, condyle to "
                f"condyle: {format_mm(curve.length_mm)}. The panoramic and the "
                "cuts follow it.",
                "status",
            )
        self.status.setText(text)
        set_role(self.status, role)
        self.drawn_row.setVisible(drawn)
        self.automatic.setVisible(session.volume is not None and session.arch_source != "auto")


class ResectionPanel(QWidget):
    mode_requested = pyqtSignal(object)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        self.add_plane = QPushButton("Add cut")
        self.add_plane.setProperty("primary", True)
        self.add_plane.setToolTip(
            "A cutting plane across the jaw, placed on the right body; drag it "
            "in the 3-D view or set it below"
        )
        self.flip = QPushButton("Flip side")
        self.flip.setToolTip("Remove the bone on the other side of this cut")
        self.clear = QPushButton("Remove all cuts")
        self.execute = QPushButton("Execute cut")
        self.execute.setToolTip("Separate the segment the cuts remove from the jaw")
        self.undo = QPushButton("Undo cut")
        self.plane_box = QComboBox()
        self.position = QDoubleSpinBox()
        self.position.setRange(0.0, 1000.0)
        self.position.setSuffix(" mm")
        self.position.setSingleStep(1.0)
        self.yaw = QDoubleSpinBox()
        self.yaw.setRange(-89.0, 89.0)
        self.yaw.setSuffix("°")
        self.tilt = QDoubleSpinBox()
        self.tilt.setRange(-89.0, 89.0)
        self.tilt.setSuffix("°")
        self.roll = QDoubleSpinBox()
        self.roll.setRange(-180.0, 180.0)
        self.roll.setSuffix("°")
        self.offsets = {}
        for key, label in (("x", " mm left"), ("y", " mm back"), ("z", " mm up")):
            spin = QDoubleSpinBox()
            spin.setRange(-60.0, 60.0)
            spin.setSingleStep(0.5)
            spin.setSuffix(label)
            self.offsets[key] = spin

        self.landmark_button = QPushButton("Mark a margin point")
        self.landmark_button.setCheckable(True)
        self.landmark_button.setToolTip(
            "Click the tumour's edge on the bone; the distance from each cut to "
            "it is reported"
        )
        self.clear_landmarks = QPushButton("Clear")
        self.readout = QLabel("No cuts yet.")
        self.readout.setWordWrap(True)
        self.readout.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.readout)
        row = QHBoxLayout()
        row.addWidget(self.add_plane)
        row.addWidget(self.clear)
        layout.addLayout(row)
        layout.addWidget(
            _hint(
                f"Up to {MAX_RESECTION_PLANES} cuts. Drag a plane in the 3-D view to "
                "slide it along the jaw; the bone shown red is what comes out."
            )
        )
        layout.addWidget(self.execute)
        layout.addWidget(self.undo)

        numbers = QFormLayout()
        pick = QHBoxLayout()
        pick.addWidget(self.plane_box, 1)
        pick.addWidget(self.flip)
        numbers.addRow("Cut", pick)
        numbers.addRow("Along the jaw", self.position)
        numbers.addRow("Obliquity", self.yaw)
        numbers.addRow("Inclination", self.tilt)
        numbers.addRow("Roll", self.roll)
        for key in ("x", "y", "z"):
            numbers.addRow("Shift" if key == "x" else "", self.offsets[key])
        adjust = QWidget()
        adjust.setLayout(numbers)
        adjust.setToolTip(
            "Angles are measured against the jaw at the cut, so they stay the "
            "same relative to the bone wherever the cut is moved"
        )
        self.adjust = _Section("Adjust a cut by numbers", adjust, expanded=True)
        layout.addWidget(self.adjust)

        margins = QHBoxLayout()
        margins.addWidget(self.landmark_button, 1)
        margins.addWidget(self.clear_landmarks)
        holder = QWidget()
        holder.setLayout(margins)
        layout.addWidget(_Section("Tumour margins (optional)", holder))
        layout.addStretch(1)

        self.flip.clicked.connect(lambda: self._flip(self._current_plane()))
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
        session.volume_changed.connect(self.refresh)
        self.refresh()

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
            self.plane_box,
            self.flip,
            self.position,
            self.yaw,
            self.tilt,
            self.roll,
            *self.offsets.values(),
        ):
            widget.setEnabled(enabled)
        self.add_plane.setEnabled(
            session.volume is not None and len(session.planes) < MAX_RESECTION_PLANES
        )
        self.clear.setEnabled(bool(session.planes))
        self.execute.setVisible(not session.cut_applied)
        self.execute.setEnabled(bool(session.planes) and session.report is not None)
        self.undo.setVisible(session.cut_applied)
        if self.plane_box.count() != len(session.planes):
            self._loading = True
            self.plane_box.clear()
            for plane in session.planes:
                self.plane_box.addItem(plane.label)
            # The cut just added is the one to adjust.
            self.plane_box.setCurrentIndex(len(session.planes) - 1)
            self._loading = False
        # Always: a drag in the 3-D view moves the cut under these numbers.
        self._load_plane_controls()
        if not session.planes:
            self.readout.setText(
                "No cuts yet. Add cut places the first one on the right body of "
                "the mandible; the second closes the segment."
            )
            set_role(self.readout, "empty")
            return
        set_role(self.readout, "status")
        report = session.report
        lines = []
        if report is None or not np.isfinite(report.arc_length_mm):
            lines.append(f"{len(session.planes)} cut(s); add the second to close a segment.")
        else:
            lines.append(
                f"Resected segment (arc length along the jaw): "
                f"{format_mm(report.arc_length_mm)}, "
                f"{format_mm(report.straight_length_mm)} straight across."
            )
            if np.isfinite(report.fragment_volume_mm3):
                lines.append(f"Bone removed: {report.fragment_volume_mm3 / 1000:.2f} cm³.")
            lines.append("Cut executed." if session.cut_applied else "Cut not executed yet.")
        if report is not None and report.margins_mm:
            lines.append("Margins (positive = inside the resected side):")
            for plane_label, name, distance in report.margins_mm:
                lines.append(f"  {name} → {plane_label}: {format_mm(distance)}")
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
            "The healthy side is mirrored in the patient's own plane of symmetry, "
            "registered to each cut stump and blended into it: one flush surface. "
            "Where the defect crosses the midline there is no healthy counterpart; "
            "that part follows the pre-operative contour and is shown in sand."
        )
        self.estimate_button = QPushButton("Re-estimate the plane of symmetry")
        self.plane_info = QLabel("")
        self.plane_info.setWordWrap(True)
        self.coverage_info = QLabel("")
        self.coverage_info.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.mirror_button)
        layout.addWidget(self.coverage_info)
        layout.addWidget(
            _Section("Plane of symmetry", _boxed(self.plane_info, self.estimate_button))
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
        refine.setContentsMargins(8, 0, 0, 0)
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
        self.refine_box = QWidget()
        self.refine_box.setLayout(refine)
        layout.addWidget(_Section("Refine by hand (optional)", self.refine_box, expanded=True))
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
        session.resection_changed.connect(self.refresh)
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
                f"left-right axis; {plane.symmetry:.0%} of the bone mirrors onto bone "
                f"(within {SYMMETRY_TOLERANCE_MM:g} mm)."
            )
            set_role(self.plane_info, "warning" if plane.symmetry < SYMMETRY_WARNING else "hint")
        reconstruction = session.reconstruction
        lines = []
        if session.coverage is not None:
            lines.append(session.coverage.summary())
        self.mirror_button.setEnabled(bool(session.planes))
        if reconstruction is None:
            lines.append(
                "The rebuilt jaw replaces the bone in the 3-D view: ivory is the "
                "bone that stays, teal the mirrored segment."
                if session.planes
                else "Place the cuts (step 3) first."
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

        # 2. Does it fit, in one line; the reasons on demand.
        layout.addWidget(self.summary)
        layout.addWidget(self.verdict)
        layout.addWidget(self.use_length)

        # 3. Which plate. Its length follows the path until one is picked.
        library = QFormLayout()
        library.addRow("Plate", self.family_box)
        library.addRow("Length", self.model_box)
        library.addRow("", self.status_badge)
        layout.addLayout(library)
        self.details = _Section(
            "Why: fit, contact and screw holes",
            _boxed(self.fit_info, self.fit_status, self.hole_report),
        )
        layout.addWidget(self.details)

        # 4. Bending.
        bending = QFormLayout()
        bending.setContentsMargins(8, 0, 0, 0)
        bending.addRow("", self.bend_plate)
        bending.addRow("Standoff", self.clearance)
        bending.addRow("Bending kit", self.kit_box)
        bending_box = QWidget()
        bending_box.setLayout(bending)
        layout.addWidget(_Section("Bending", bending_box))
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
        self._on_family()

    # -- plate library ---------------------------------------------------

    def _on_family(self) -> None:
        # A new family: its length follows the path until one is picked.
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
        if self.model_box.count():
            self.session.set_plate_asset(self.model_box.itemData(0), length_chosen=False)
        self.refresh_library()

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
            count = len(self.session.plate_points)
            self.summary.setText(
                "No plate path yet." if count == 0 else
                f"{count} point so far: click at least one more along the bone."
            )
            set_role(self.summary, "empty")
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

        asset = self.session.plate_asset
        plate = (
            f"{asset.hole_count}-hole plate ({asset.length_mm:.0f} mm)"
            if asset is not None else "Plate"
        )
        self.summary.setText(
            f"{plate} on a {format_mm(plan.total_length_mm)} path along the bone."
        )
        set_role(self.summary, "status")


def _plain(value: float) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.3f}"


class ExportPanel(QWidget):
    """Every file the plan produces, in one place, each saying what it is for."""

    #: (key, button text, what it is for)
    ITEMS = (
        ("mandible", "Mandible (STL)", "The patient's mandible on its own, separated from the skull."),
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
            session.mandible_changed,
        ):
            signal.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        session = self.session
        ready = {
            "mandible": session.mandible is not None and session.surface is not None,
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
