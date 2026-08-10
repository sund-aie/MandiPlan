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
    QVBoxLayout,
    QWidget,
)

from ..constants import MAX_RESECTION_PLANES
from ..geometry import cpr
from ..geometry.measure import format_deg, format_mm
from ..geometry.plate import bend_table_rows
from .histogram import ThresholdPanel
from .modes import Mode


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: #9aa0b0; font-size: 10px;")
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
        session.resection_changed.connect(self.refresh)

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


class PlatePanel(QWidget):
    mode_requested = pyqtSignal(object)
    export_csv_requested = pyqtSignal()
    export_stl_requested = pyqtSignal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session

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
        self.table.setMinimumHeight(180)

        self.summary = QLabel("No plate path.")
        self.summary.setWordWrap(True)
        self.export_csv = QPushButton("Export bend table (CSV)…")
        self.export_stl = QPushButton("Export bending template (STL)…")

        layout = QVBoxLayout(self)
        layout.addWidget(self.draw_button)
        layout.addWidget(
            _hint(
                "Clicks are projected onto the bone surface. The pitch is the "
                "screw-hole spacing of your plate system — set it before reading "
                "the table."
            )
        )
        row = QHBoxLayout()
        row.addWidget(self.undo_point)
        row.addWidget(self.clear_path)
        layout.addLayout(row)
        box = QGroupBox("Plate")
        box.setLayout(form)
        layout.addWidget(box)
        layout.addWidget(self.summary)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.export_csv)
        layout.addWidget(self.export_stl)

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
        session.plate_changed.connect(self.refresh)

    def refresh(self) -> None:
        plan = self.session.plate_plan
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
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
