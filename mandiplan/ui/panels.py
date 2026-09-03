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
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..bending_steps import steps_as_rows
from ..constants import MAX_RESECTION_PLANES
from ..geometry import cpr
from ..geometry.measure import format_deg, format_mm
from ..geometry.plate import bend_table_rows
from ..plate_catalog import load_kits, load_systems
from .histogram import ThresholdPanel
from .modes import Mode


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: #5f6368; font-size: 11px;")
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
        self.info = QLabel("No volume loaded.")
        self.info.setWordWrap(True)
        self.threshold_panel = ThresholdPanel()
        self.largest = QCheckBox("Keep largest connected component")
        self.largest.setChecked(True)
        self.surface_info = QLabel("")
        self.surface_info.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.load_button)
        layout.addWidget(self.info)
        layout.addWidget(self.threshold_panel)
        layout.addWidget(self.largest)
        layout.addWidget(self.surface_info)
        layout.addStretch(1)

        self.load_button.clicked.connect(self.load_requested)
        self.threshold_panel.threshold_changed.connect(session.set_threshold)
        self.largest.toggled.connect(self._on_largest)
        session.volume_changed.connect(self.refresh_volume)
        session.surface_changed.connect(self.refresh_surface)

    def _on_largest(self, checked: bool) -> None:
        self.session.keep_largest_component = checked
        self.session.rebuild_surface()

    def refresh_volume(self) -> None:
        session = self.session
        if session.volume is None:
            self.info.setText("No volume loaded.")
            return
        self.info.setText("\n".join(session.info.lines(session.volume)))
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
        for key in ("x", "y", "z"):
            numbers.addRow("Offset" if key == "x" else "", self.offsets[key])
        numeric_box = QGroupBox("Place the cut by numbers")
        numeric_box.setLayout(numbers)
        layout.addWidget(numeric_box)
        layout.addWidget(
            _hint(
                "Position runs along the arch curve. Yaw turns the cut about the "
                "superior axis, tilt about the buccolingual direction; the offsets "
                "shift it in patient axes. Dragging the handle updates these."
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
        for spin in (self.yaw, self.tilt):
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
        self._loading = True
        try:
            self.position.setRange(0.0, session.frames.length_mm)
            self.position.setValue(session.plane_arc_position(index))
            for key in ("x", "y", "z"):
                self.offsets[key].setValue(0.0)
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
        self.session.rotate_plane(index, self.yaw.value(), self.tilt.value())

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
        for widget in (self.position, self.yaw, self.tilt, *self.offsets.values()):
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
    """Mirror the healthy side into the defect, and say what it cannot reach."""

    export_graft_requested = pyqtSignal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        self.estimate_button = QPushButton("Estimate mid-sagittal plane")
        self.mirror_button = QPushButton("Mirror healthy side into the defect")
        self.export_button = QPushButton("Export reconstruction target (STL)…")
        self.plane_info = QLabel("No symmetry plane estimated.")
        self.plane_info.setWordWrap(True)
        self.coverage_info = QLabel("")
        self.coverage_info.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(self.estimate_button)
        layout.addWidget(self.plane_info)
        layout.addWidget(self.mirror_button)
        layout.addWidget(self.coverage_info)
        layout.addWidget(
            _hint(
                "The mirrored healthy side is the reconstruction target. Where the "
                "defect crosses the midline there is no healthy counterpart to "
                "mirror, and that span is estimated by blending this patient's own "
                "cross-sections across the gap — an interpolation, not a prediction "
                "from a population of mandibles."
            )
        )
        layout.addWidget(self.export_button)
        layout.addStretch(1)

        self.estimate_button.clicked.connect(session.estimate_symmetry_plane)
        self.mirror_button.clicked.connect(session.build_graft)
        self.export_button.clicked.connect(self.export_graft_requested)
        session.reconstruction_changed.connect(self.refresh)

    def refresh(self) -> None:
        session = self.session
        plane = session.symmetry_plane
        if plane is None:
            self.plane_info.setText("No symmetry plane estimated.")
        else:
            self.plane_info.setText(
                f"Mid-sagittal plane at x = {plane.point[0]:.2f} mm, tilted "
                f"{plane.tilt_deg:.2f}° from the left-right axis.\n"
                f"Symmetry score: {plane.symmetry:.0%} of bone mirrors onto bone."
            )
        lines = []
        if session.coverage is not None:
            lines.append(session.coverage.summary())
        if session.graft_surface is not None:
            lines.append(
                f"Mirrored graft volume: {session.graft_volume_mm3:.0f} mm³"
            )
        if session.bridge_surface is not None:
            lines.append("The un-mirrorable span is shown as an estimated segment.")
        self.coverage_info.setText("\n".join(lines))


class PlatePanel(QWidget):
    mode_requested = pyqtSignal(object)
    export_csv_requested = pyqtSignal()
    export_stl_requested = pyqtSignal()
    export_steps_requested = pyqtSignal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

        self.system_box = QComboBox()
        for system in load_systems():
            self.system_box.addItem(system.name, system.id)
        self.kit_box = QComboBox()
        for kit in load_kits():
            self.kit_box.addItem(kit.name, kit.id)

        self.draw_button = QPushButton("Draw plate path on the bone")
        self.draw_button.setCheckable(True)
        self.undo_point = QPushButton("Remove last point")
        self.clear_path = QPushButton("Clear path")

        self.pitch = QDoubleSpinBox()
        self.pitch.setRange(2.0, 40.0)
        self.pitch.setSingleStep(0.5)
        self.pitch.setSuffix(" mm")
        self.pitch.setValue(session.plate.pitch_mm)
        self.width = QDoubleSpinBox()
        self.width.setRange(2.0, 40.0)
        self.width.setSingleStep(0.5)
        self.width.setSuffix(" mm")
        self.width.setValue(session.plate.width_mm)
        self.thickness = QDoubleSpinBox()
        self.thickness.setRange(0.2, 10.0)
        self.thickness.setSingleStep(0.1)
        self.thickness.setSuffix(" mm")
        self.thickness.setValue(session.plate.thickness_mm)

        form = QFormLayout()
        form.addRow("Screw-hole pitch", self.pitch)
        form.addRow("Template width", self.width)
        form.addRow("Template thickness", self.thickness)

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
        self.export_csv = QPushButton("Export bend table (CSV)…")
        self.export_stl = QPushButton("Export bending template (STL)…")
        self.export_steps = QPushButton("Export bench steps (CSV)…")

        layout = QVBoxLayout(self)
        selection = QFormLayout()
        selection.addRow("Plate system", self.system_box)
        selection.addRow("Bending kit", self.kit_box)
        chooser = QGroupBox("Selection")
        chooser.setLayout(selection)
        layout.addWidget(chooser)
        layout.addWidget(
            _hint(
                "The catalogue holds generic profiles by size class, not a "
                "manufacturer's catalogue. Check the dimensions against the system "
                "you are holding and edit mandiplan/data/plate_systems.json to match."
            )
        )
        layout.addWidget(self.draw_button)
        layout.addWidget(
            _hint("Clicks are projected onto the bone surface.")
        )
        row = QHBoxLayout()
        row.addWidget(self.undo_point)
        row.addWidget(self.clear_path)
        layout.addLayout(row)
        box = QGroupBox("Plate dimensions")
        box.setLayout(form)
        layout.addWidget(box)
        layout.addWidget(self.summary)
        layout.addWidget(self.fit_info)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.export_csv)
        layout.addWidget(self.export_steps)
        layout.addWidget(self.export_stl)

        self.system_box.currentIndexChanged.connect(
            lambda i: self._on_system(self.system_box.itemData(i))
        )
        self.kit_box.currentIndexChanged.connect(
            lambda i: session.set_bending_kit(self.kit_box.itemData(i))
        )
        self.draw_button.toggled.connect(
            lambda on: self.mode_requested.emit(Mode.PLATE if on else Mode.NAVIGATE)
        )
        self.undo_point.clicked.connect(session.remove_last_plate_point)
        self.clear_path.clicked.connect(session.clear_plate_path)
        self.pitch.valueChanged.connect(lambda v: session.set_plate_settings(pitch_mm=v))
        self.width.valueChanged.connect(lambda v: session.set_plate_settings(width_mm=v))
        self.thickness.valueChanged.connect(
            lambda v: session.set_plate_settings(thickness_mm=v)
        )
        self.export_csv.clicked.connect(self.export_csv_requested)
        self.export_stl.clicked.connect(self.export_stl_requested)
        self.export_steps.clicked.connect(self.export_steps_requested)
        session.plate_changed.connect(self.refresh)

    def _on_system(self, system_id: str) -> None:
        self.session.set_plate_system(system_id)
        for spin, value in (
            (self.pitch, self.session.plate.pitch_mm),
            (self.width, self.session.plate.width_mm),
            (self.thickness, self.session.plate.thickness_mm),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def _refresh_fit(self) -> None:
        fit = self.session.fit
        self.steps_table.setRowCount(0)
        if fit is None:
            self.fit_info.setText("")
            return
        lines = [fit.verdict]
        if fit.holes_proximal or fit.holes_distal:
            lines.append(
                f"Screw holes: {fit.holes_proximal} proximal, "
                f"{fit.holes_over_defect} over the defect, {fit.holes_distal} distal."
            )
        lines.extend(f"Problem: {p}" for p in fit.problems)
        lines.extend(f"Note: {w}" for w in fit.warnings[:4])
        self.fit_info.setText("\n".join(lines))
        self.fit_info.setStyleSheet(
            "color: #c5221f;" if fit.problems else "color: #188038;"
        )

        rows = steps_as_rows(self.session.steps)
        self.steps_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                self.steps_table.setItem(r, c, QTableWidgetItem(text))
        self.steps_table.resizeColumnsToContents()
        self.steps_table.resizeRowsToContents()

    def refresh(self) -> None:
        plan = self.session.plate_plan
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self._refresh_fit()
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
