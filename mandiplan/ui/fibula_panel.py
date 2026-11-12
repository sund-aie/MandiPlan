"""The fibula graft step: the leg's CT, the harvest, the pieces and their cuts."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..geometry.fibula_plan import MAX_PIECES
from .theme import set_role


def _spin(lo: float, hi: float, step: float, suffix: str, value: float, decimals: int = 0) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setSuffix(suffix)
    box.setValue(value)
    box.setKeyboardTracking(False)
    return box


def _say(label: QLabel, text: str, role: str) -> None:
    label.setText(text)
    label.setVisible(bool(text))
    set_role(label, role)


class FibulaPanel(QWidget):
    """Plan a fibula graft for the resected jaw, from a CT of the legs."""

    leg_requested = pyqtSignal()

    COLUMNS = ("Piece", "Length", "Lower end, up from the ankle tip", "Ends lean", "Joint to next")

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        graft = session.fibula
        self.graft = graft

        self.open_button = QPushButton("Open the leg CT…")
        self.open_button.setProperty("primary", True)
        self.open_button.setToolTip(
            "A CT (or CT angiogram) of the lower legs, its own DICOM folder. Each fibula in it "
            "is found by itself; nothing of the leg is moved into the jaw's scan."
        )
        self.scan_info = QLabel("")
        self.scan_info.setWordWrap(True)

        self.leg_box = QComboBox()
        self.threshold = _spin(-500.0, 3000.0, 10.0, " HU", graft.threshold)
        self.threshold.setToolTip("Bone in the leg's CT: cortex is far above it, muscle below.")
        self.distal_keep = _spin(30.0, 150.0, 5.0, " mm", graft.distal_keep_mm)
        self.distal_keep.setToolTip(
            "Fibula left above the lateral malleolus for the ankle: Materialise keeps the graft "
            "70 mm from its tip, KLS Martin 80; the surgical atlases say 50 to 60."
        )
        self.proximal_keep = _spin(30.0, 150.0, 5.0, " mm", graft.proximal_keep_mm)
        self.proximal_keep.setToolTip(
            "Fibula left below the head, for the knee and the common peroneal nerve at the neck. "
            "No figure is published; bone above the graft is freed to lengthen the vessels anyway."
        )

        self.count_box = QComboBox()
        self.count_box.addItem("Automatic", 0)
        for count in range(1, MAX_PIECES + 2):
            self.count_box.addItem(str(count), count)
        self.min_piece = _spin(10.0, 60.0, 1.0, " mm", graft.min_piece_mm)
        self.min_piece.setToolTip(
            "No piece shorter: under 2 cm a piece more often has no periosteal vessel of its own."
        )
        self.raise_mm = _spin(0.0, 25.0, 1.0, " mm", graft.raise_mm)
        self.raise_mm.setToolTip(
            "0: the fibula's lower edge on the jaw's lower border. Raised, it sits nearer the "
            "teeth for implants; up to about 10 mm above the border hardly shows."
        )
        self.pedicle_box = QComboBox()
        self.roll = QSpinBox()
        self.roll.setRange(-180, 180)
        self.roll.setSingleStep(15)
        self.roll.setSuffix("°")
        self.roll.setKeyboardTracking(False)
        self.roll.setToolTip(
            "Turns the fibula about its own line. At 0° its lateral side faces out of the jaw; "
            "the pieces keep the same turn, so each joint is a plain wedge."
        )
        self.face_box = QComboBox()
        self.face_box.addItem("Lateral face", "lateral")
        self.face_box.addItem("Anterior face", "anterior")
        self.face_box.setToolTip(
            "The face of the fibula the cutting guide sits on. Each wedge narrows towards the "
            "side of the fibula that faces out of the jaw, its lateral side unless it is turned; "
            "a guide on that face needs its cuts further apart, so the graft takes more fibula."
        )
        self.guide_button = QPushButton("Make the cutting guide")
        self.guide_button.setToolTip(
            "Builds the guide at printing resolution and follows its use on the bone: seated, "
            "lifted off, a blade run down every slot, every screw driven. Takes a few seconds."
        )
        self.guide_report = QLabel("")
        self.guide_report.setWordWrap(True)
        self.leg_view = QPushButton("Show the leg")
        self.leg_view.setCheckable(True)
        self.leg_view.setToolTip("The fibula as it lies in the leg, with the cuts and the pieces.")

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.vessels = QLabel("")
        self.vessels.setWordWrap(True)
        self.warnings = QLabel("")
        self.warnings.setWordWrap(True)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)

        form = QFormLayout()
        form.addRow("Leg", self.leg_box)
        form.addRow("Bone from", self.threshold)
        form.addRow("Keep at the ankle", self.distal_keep)
        form.addRow("Keep at the knee", self.proximal_keep)
        form.addRow("Pieces", self.count_box)
        form.addRow("Shortest piece", self.min_piece)
        form.addRow("Above the lower border", self.raise_mm)
        form.addRow("Vessels leave from", self.pedicle_box)
        form.addRow("Turn about its line", self.roll)
        form.addRow("Guide sits on", self.face_box)
        self.settings = QWidget()
        self.settings.setLayout(form)

        layout = QVBoxLayout(self)
        layout.addWidget(self.open_button)
        layout.addWidget(self.scan_info)
        layout.addWidget(self.settings)
        layout.addWidget(self.summary)
        layout.addWidget(self.vessels)
        layout.addWidget(self.warnings)
        layout.addWidget(self.table)
        layout.addWidget(self.guide_button)
        layout.addWidget(self.guide_report)
        layout.addWidget(self.leg_view)
        layout.addStretch(1)

        self.open_button.clicked.connect(self.leg_requested)
        self.leg_box.activated.connect(lambda _i: graft.choose_side(self.leg_box.currentData()))
        self.threshold.valueChanged.connect(graft.set_threshold)
        self.distal_keep.valueChanged.connect(lambda v: graft.set_setting("distal_keep_mm", v))
        self.proximal_keep.valueChanged.connect(lambda v: graft.set_setting("proximal_keep_mm", v))
        self.count_box.activated.connect(
            lambda _i: graft.set_setting("piece_count", self.count_box.currentData())
        )
        self.min_piece.valueChanged.connect(lambda v: graft.set_setting("min_piece_mm", v))
        self.raise_mm.valueChanged.connect(lambda v: graft.set_setting("raise_mm", v))
        self.pedicle_box.activated.connect(
            lambda _i: graft.set_setting("pedicle", self.pedicle_box.currentData())
        )
        self.roll.valueChanged.connect(lambda v: graft.set_setting("roll_deg", v))
        self.face_box.activated.connect(
            lambda _i: graft.set_setting("guide_face", self.face_box.currentData())
        )
        self.guide_button.clicked.connect(self._make_guide)
        self.leg_view.toggled.connect(graft.set_leg_view)
        graft.changed.connect(self.refresh)
        graft.leg_view_changed.connect(self._refresh_leg_button)
        self.refresh()

    def _refresh_leg_button(self) -> None:
        self.leg_view.blockSignals(True)
        self.leg_view.setChecked(self.graft.leg_view)
        self.leg_view.blockSignals(False)

    def _fill_choices(self) -> None:
        graft = self.graft
        self.leg_box.blockSignals(True)
        self.leg_box.clear()
        for fibula in graft.fibulae:
            self.leg_box.addItem(f"{fibula.side.capitalize()} ({fibula.length_mm:.0f} mm)", fibula.side)
        index = self.leg_box.findData(graft.side)
        if index >= 0:
            self.leg_box.setCurrentIndex(index)
        self.leg_box.setEnabled(len(graft.fibulae) > 1)
        self.leg_box.blockSignals(False)
        first, second = graft.end_names()
        self.pedicle_box.blockSignals(True)
        self.pedicle_box.clear()
        self.pedicle_box.addItem("Automatic: the end towards the angle", "auto")
        self.pedicle_box.addItem(f"The {first}", "first")
        self.pedicle_box.addItem(f"The {second}", "second")
        index = self.pedicle_box.findData(graft.pedicle)
        self.pedicle_box.setCurrentIndex(max(index, 0))
        self.pedicle_box.blockSignals(False)
        self.count_box.blockSignals(True)
        self.count_box.setCurrentIndex(max(self.count_box.findData(graft.piece_count), 0))
        self.count_box.blockSignals(False)
        self.face_box.blockSignals(True)
        self.face_box.setCurrentIndex(max(self.face_box.findData(graft.guide_face), 0))
        self.face_box.blockSignals(False)

    def refresh(self) -> None:
        graft = self.graft
        self._fill_choices()
        self._refresh_leg_button()
        self.settings.setEnabled(graft.scan is not None)
        self.leg_view.setEnabled(graft.surface is not None)
        if graft.scan is None:
            _say(self.scan_info, "Open a CT of the lower legs to plan a fibula graft for the "
                 "resected jaw. It is optional: without one, plan the plate on the rebuilt jaw.", "empty")
        else:
            spacing = graft.scan.spacing
            found = ", ".join(f"{f.side} {f.length_mm:.0f} mm" for f in graft.fibulae) or "none"
            _say(
                self.scan_info,
                f"Leg CT at {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm. "
                f"Fibulae found: {found} (from the tip of each, as far as it is bone at "
                f"{graft.threshold:.0f} HU).",
                "hint",
            )
        plan = graft.plan
        self.guide_button.setEnabled(plan is not None)
        self._refresh_guide_report()
        if plan is None:
            _say(self.summary, graft.problem if graft.scan is not None else "", "warning" if graft.problem else "hint")
            _say(self.vessels, "", "hint")
            _say(self.warnings, "", "warning")
            self.table.setRowCount(0)
            return
        fibula = graft.fibula
        taken = plan.harvest_length_mm
        many = len(plan.pieces) != 1
        _say(
            self.summary,
            f"{len(plan.pieces)} piece{'s' if many else ''} along "
            f"{graft.course.length_mm:.0f} mm of jaw, keeping within {plan.deviation_mm:.1f} mm of "
            f"the rebuilt shape. {'They take' if many else 'It takes'} {taken:.0f} mm of the "
            f"{fibula.side} fibula, from {plan.harvest_mm[0]:.0f} to {plan.harvest_mm[1]:.0f} mm "
            f"above the ankle tip; {plan.above_graft_mm:.0f} mm more is free above "
            f"{'them' if many else 'it'} for the vessels."
            + (
                f" With the guide on the {graft.other_face[0]} face they would take "
                f"{graft.other_face[1]:.0f} mm."
                if graft.other_face is not None and len(plan.pieces) > 1
                else ""
            ),
            "hint",
        )
        if graft.vessels is not None:
            up, out = graft.vessels
            height = "low" if up < -0.35 else "high" if up > 0.35 else "half-way up"
            side = "inner" if out < -0.35 else "outer" if out > 0.35 else "lower or upper"
            _say(
                self.vessels,
                f"The peroneal vessels run down the back of the fibula: in the jaw they lie "
                f"{height}, on its {side} side. Turn the fibula or take the other leg to move them.",
                "hint",
            )
        _say(self.warnings, "\n".join(plan.warnings), "warning")
        self.table.setRowCount(len(plan.pieces))
        for row, piece in enumerate(plan.pieces):
            joint = plan.joint_angles_deg[row] if row < len(plan.joint_angles_deg) else None
            cells = (
                str(piece.number),
                f"{piece.length_mm:.1f} mm",
                f"{piece.s_distal:.0f} mm",
                f"{piece.distal_face.lean_deg:.0f}° / {piece.proximal_face.lean_deg:.0f}°",
                f"{joint:.0f}°" if joint is not None else "",
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()

    def _make_guide(self) -> None:
        self.graft.make_guide()

    def _refresh_guide_report(self) -> None:
        graft = self.graft
        check = graft.guide_check
        if check is None:
            text = (
                "The cutting guide is made for the plan as it stands; change the plan and it is "
                "made again." if graft.plan is not None else ""
            )
            _say(self.guide_report, text, "hint")
            return
        lines = check.lines()
        head = "Followed in use, the guide works:" if check.ok else "Followed in use, the guide has a fault:"
        _say(self.guide_report, head + "\n" + "\n".join(lines), "hint" if check.ok else "warning")
