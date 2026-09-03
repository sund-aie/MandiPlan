"""A strip above the planning panels showing where you are in the workflow."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..workflow import workflow_status

DONE = QColor(24, 128, 56)
CURRENT = QColor(26, 115, 232)
BLOCKED = QColor(218, 220, 224)


class _Dots(QWidget):
    """One dot per step: filled when done, ringed for the step you are on."""

    step_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(18)
        self.statuses = []
        self.current = 0

    def set_statuses(self, statuses, current: int) -> None:
        self.statuses = statuses
        self.current = current
        self.update()

    def _spacing(self) -> float:
        return self.width() / max(len(self.statuses), 1)

    def paintEvent(self, event):  # noqa: N802
        if not self.statuses:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        spacing = self._spacing()
        for status in self.statuses:
            x = spacing * (status.index + 0.5)
            colour = DONE if status.done else (BLOCKED if status.blocked_by else CURRENT)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(colour)
            painter.drawEllipse(int(x - 5), 4, 10, 10)
            if status.index == self.current:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QColor(26, 115, 232))
                painter.drawEllipse(int(x - 8), 1, 16, 16)

    def mousePressEvent(self, event):  # noqa: N802
        if self.statuses:
            index = int(event.position().x() // self._spacing())
            self.step_clicked.emit(max(0, min(index, len(self.statuses) - 1)))


class WorkflowBar(QWidget):
    """Step counter, progress dots, Back/Next, and the next thing to do."""

    step_selected = pyqtSignal(int)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._page = 0

        self.heading = QLabel("Step 1 of 5")
        self.heading.setStyleSheet("font-weight: bold;")
        self.back = QPushButton("◀ Back")
        self.next = QPushButton("Next ▶")
        for button in (self.back, self.next):
            button.setMaximumWidth(84)
        self.dots = _Dots()
        self.detail = QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color: #5f6368; font-size: 11px;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        row = QHBoxLayout()
        row.addWidget(self.heading, 1)
        row.addWidget(self.back)
        row.addWidget(self.next)
        layout.addLayout(row)
        layout.addWidget(self.dots)
        layout.addWidget(self.detail)

        self.back.clicked.connect(lambda: self.step_selected.emit(self._page - 1))
        self.next.clicked.connect(lambda: self.step_selected.emit(self._page + 1))
        self.dots.step_clicked.connect(self.step_selected)
        for signal in (
            session.volume_changed,
            session.surface_changed,
            session.arch_changed,
            session.resection_changed,
            session.reconstruction_changed,
            session.plate_changed,
        ):
            signal.connect(self.refresh)
        self.refresh()

    def set_page(self, page: int) -> None:
        self._page = page
        self.refresh()

    def refresh(self) -> None:
        statuses = workflow_status(self.session)
        self._page = max(0, min(self._page, len(statuses) - 1))
        status = statuses[self._page]
        done = sum(1 for s in statuses if s.done)
        self.heading.setText(
            f"Step {self._page + 1} of {len(statuses)} — {done} complete"
        )
        self.dots.set_statuses(statuses, self._page)
        self.back.setEnabled(self._page > 0)
        self.next.setEnabled(self._page < len(statuses) - 1)
        if status.blocked_by:
            self.detail.setText(f"Needs {status.blocked_by}. {status.summary}")
        elif status.done:
            self.detail.setText(f"Done — {status.summary}")
        else:
            self.detail.setText(f"Next: {status.summary}")
