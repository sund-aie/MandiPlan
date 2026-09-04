"""Gray-value histogram with a live bone-threshold control.

CBCT gray values are not Hounsfield units, so the threshold is seeded from
Otsu's method over this volume's own histogram and then left to the user.
Nothing in the application assumes a fixed bone value.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .theme import ACCENT, DANGER, SURFACE_SUNKEN, set_role

_SLIDER_STEPS = 2000


class HistogramView(QWidget):
    """Log-scaled histogram with a draggable threshold marker."""

    threshold_dragged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(90)
        self._counts: np.ndarray | None = None
        self._edges: np.ndarray | None = None
        self._threshold = 0.0

    def set_histogram(self, counts: np.ndarray, edges: np.ndarray) -> None:
        self._counts = np.asarray(counts, dtype=float)
        self._edges = np.asarray(edges, dtype=float)
        self.update()

    def set_threshold(self, value: float) -> None:
        self._threshold = float(value)
        self.update()

    def _value_to_x(self, value: float) -> float:
        lo, hi = self._edges[0], self._edges[-1]
        return (value - lo) / max(hi - lo, 1e-9) * self.width()

    def _x_to_value(self, x: float) -> float:
        lo, hi = self._edges[0], self._edges[-1]
        return lo + np.clip(x / max(self.width(), 1), 0.0, 1.0) * (hi - lo)

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(SURFACE_SUNKEN))
        if self._counts is None or self._counts.sum() == 0:
            return
        log_counts = np.log10(self._counts + 1.0)
        peak = max(float(log_counts.max()), 1e-6)
        n = len(log_counts)
        w = self.width() / n
        painter.setPen(Qt.PenStyle.NoPen)
        bars = QColor(ACCENT)
        bars.setAlpha(110)
        painter.setBrush(bars)
        for i, value in enumerate(log_counts):
            h = value / peak * (self.height() - 6)
            painter.drawRect(int(i * w), int(self.height() - h), max(int(w) + 1, 1), int(h))

        pen = QPen(QColor(DANGER))
        pen.setWidth(2)
        painter.setPen(pen)
        x = int(self._value_to_x(self._threshold))
        painter.drawLine(x, 0, x, self.height())

    def mousePressEvent(self, event):  # noqa: N802
        self._drag(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._drag(event)

    def _drag(self, event) -> None:
        if self._edges is None:
            return
        self.threshold_dragged.emit(float(self._x_to_value(event.position().x())))


class ThresholdPanel(QWidget):
    """Histogram, slider and numeric entry for the bone threshold."""

    threshold_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._range = (0.0, 1.0)
        self._value = 0.0
        self._updating = False

        self.histogram = HistogramView()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, _SLIDER_STEPS)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(1)
        self.spin.setSingleStep(5.0)
        self.spin.setSuffix(" gray value")
        self.otsu_button = QPushButton("Reset to Otsu estimate")
        self.note = QLabel(
            "CBCT gray values are not Hounsfield units — set this by eye on the bone."
        )
        self.note.setWordWrap(True)
        set_role(self.note, "hint")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self.histogram)
        layout.addWidget(self.slider)
        row = QHBoxLayout()
        row.addWidget(QLabel("Bone threshold"))
        row.addWidget(self.spin, 1)
        layout.addLayout(row)
        layout.addWidget(self.otsu_button)
        layout.addWidget(self.note)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.histogram.threshold_dragged.connect(self.set_value)

    def configure(self, counts, edges, value: float) -> None:
        self._range = (float(edges[0]), float(edges[-1]))
        self.histogram.set_histogram(counts, edges)
        self.spin.setRange(*self._range)
        self.set_value(value, emit=False)

    def value(self) -> float:
        return self._value

    def set_value(self, value: float, emit: bool = True) -> None:
        lo, hi = self._range
        value = float(np.clip(value, lo, hi))
        if self._updating:
            return
        self._updating = True
        self._value = value
        self.histogram.set_threshold(value)
        self.spin.setValue(value)
        self.slider.setValue(int(round((value - lo) / max(hi - lo, 1e-9) * _SLIDER_STEPS)))
        self._updating = False
        if emit:
            self.threshold_changed.emit(value)

    def _on_slider(self, position: int) -> None:
        if self._updating:
            return
        lo, hi = self._range
        self.set_value(lo + position / _SLIDER_STEPS * (hi - lo))

    def _on_spin(self, value: float) -> None:
        if self._updating:
            return
        self.set_value(value)
