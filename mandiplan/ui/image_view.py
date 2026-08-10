"""A 2-D image view whose coordinate system is millimetres.

Every image shown in MandiPlan (orthogonal slices, the panoramic reformat,
buccolingual cross-sections) arrives with a pixel size in millimetres and the
world coordinate of its first pixel.  This widget keeps that mapping and
converts screen positions back to millimetres before anything is measured, so
no measurement path through the UI ever sees pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget


@dataclass
class Overlay:
    """A polyline or point set drawn on top of the image, in millimetres."""

    points: np.ndarray  # (N, 2) in the view's (x, y) millimetre axes
    colour: QColor
    kind: str = "line"  # "line", "points", "crosshair"
    width: float = 1.5
    radius: float = 4.0
    label: str = ""


class ImageView(QWidget):
    """Displays a millimetre-calibrated 2-D image with rulers and picking."""

    picked = pyqtSignal(float, float)  # x_mm, y_mm in the view's own axes
    hovered = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._image: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self.pixel_mm_x = 1.0
        self.pixel_mm_y = 1.0
        self.x0 = 0.0
        self.y0 = 0.0
        self.x_label = "x (mm)"
        self.y_label = "y (mm)"
        self.row_axis_up = False
        self.title = ""

        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._fit_pending = True
        self._panning_from: QPointF | None = None
        self.overlays: list[Overlay] = []

    # -- data -------------------------------------------------------------

    def set_image(
        self,
        image: np.ndarray,
        pixel_mm_x: float,
        pixel_mm_y: float,
        x0: float,
        y0: float,
        x_label: str = "x (mm)",
        y_label: str = "y (mm)",
        row_axis_up: bool = False,
        keep_view: bool = False,
    ) -> None:
        """Show ``image`` with a separate millimetre size for each axis.

        Anisotropic voxels give coronal and sagittal slices non-square pixels;
        keeping the two sizes apart is what stops them being drawn distorted.
        """
        self._image = np.asarray(image)
        self.pixel_mm_x = float(pixel_mm_x)
        self.pixel_mm_y = float(pixel_mm_y)
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.x_label = x_label
        self.y_label = y_label
        self.row_axis_up = row_axis_up
        self._pixmap = self._to_pixmap(self._image)
        if not keep_view:
            self._fit_pending = True
        self.update()

    def set_reformat(self, reformat, keep_view: bool = False) -> None:
        """Convenience for :class:`mandiplan.geometry.cpr.Reformat`."""
        self.set_image(
            reformat.image,
            reformat.pixel_mm,
            reformat.pixel_mm,
            reformat.x0,
            reformat.y0,
            reformat.x_label,
            reformat.y_label,
            row_axis_up=reformat.row_axis_up,
            keep_view=keep_view,
        )

    def clear(self) -> None:
        self._image = None
        self._pixmap = None
        self.overlays = []
        self.update()

    def has_image(self) -> bool:
        return self._image is not None

    @staticmethod
    def _to_pixmap(image: np.ndarray) -> QPixmap:
        finite = image[np.isfinite(image)]
        lo = float(np.percentile(finite, 1.0)) if finite.size else 0.0
        hi = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        if hi <= lo:
            hi = lo + 1.0
        scaled = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
        grey = np.ascontiguousarray((scaled * 255).astype(np.uint8))
        h, w = grey.shape
        qimage = QImage(grey.data, w, h, w, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimage.copy())

    # -- coordinate mapping -----------------------------------------------

    @property
    def _extent_mm(self) -> tuple[float, float]:
        rows, cols = self._image.shape
        return cols * self.pixel_mm_x, rows * self.pixel_mm_y

    def _display_y(self, y_mm: float) -> float:
        return -y_mm if self.row_axis_up else y_mm

    def _scale(self) -> float:
        return self._zoom

    def mm_to_widget(self, x_mm: float, y_mm: float) -> QPointF:
        s = self._scale()
        return QPointF(x_mm * s + self._pan.x(), self._display_y(y_mm) * s + self._pan.y())

    def widget_to_mm(self, pos: QPointF) -> tuple[float, float]:
        s = self._scale()
        x = (pos.x() - self._pan.x()) / s
        y = (pos.y() - self._pan.y()) / s
        return x, (-y if self.row_axis_up else y)

    def fit(self) -> None:
        if self._image is None:
            return
        width_mm, height_mm = self._extent_mm
        margin = 46.0
        avail_w = max(self.width() - margin - 8, 20)
        avail_h = max(self.height() - margin - 8, 20)
        self._zoom = min(avail_w / width_mm, avail_h / height_mm)
        cx = self.x0 - 0.5 * self.pixel_mm_x + width_mm / 2.0
        cy_mm = self.y0 - 0.5 * self.pixel_mm_y + height_mm / 2.0
        cy = self._display_y(cy_mm)
        centre = QPointF(margin + avail_w / 2.0, avail_h / 2.0 + 4)
        self._pan = QPointF(
            centre.x() - cx * self._zoom, centre.y() - cy * self._zoom
        )
        self._fit_pending = False
        self.update()

    # -- events ------------------------------------------------------------

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._fit_pending:
            self.fit()

    def wheelEvent(self, event):  # noqa: N802
        if self._image is None:
            return
        factor = 1.15 ** (event.angleDelta().y() / 120.0)
        cursor = event.position()
        before = self.widget_to_mm(cursor)
        self._zoom = float(np.clip(self._zoom * factor, 0.05, 400.0))
        after = self.mm_to_widget(*before)
        self._pan += cursor - after
        self.update()

    def mousePressEvent(self, event):  # noqa: N802
        if self._image is None:
            return
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self._panning_from = event.position()
        elif event.button() == Qt.MouseButton.LeftButton:
            x, y = self.widget_to_mm(event.position())
            self.picked.emit(x, y)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._image is None:
            return
        if self._panning_from is not None:
            self._pan += event.position() - self._panning_from
            self._panning_from = event.position()
            self.update()
            return
        x, y = self.widget_to_mm(event.position())
        self.hovered.emit(x, y)

    def mouseReleaseEvent(self, event):  # noqa: N802
        self._panning_from = None

    # -- painting ----------------------------------------------------------

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(18, 18, 22))
        if self._pixmap is None:
            painter.setPen(QColor(150, 150, 160))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "no image"
            )
            return
        if self._fit_pending:
            self.fit()

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self._draw_image(painter)
        self._draw_overlays(painter)
        self._draw_axes(painter)
        self._draw_scale_bar(painter)
        if self.title:
            painter.setPen(QColor(220, 220, 230))
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(52, 18, self.title)

    def _draw_image(self, painter: QPainter) -> None:
        rows, cols = self._image.shape
        left = self.x0 - 0.5 * self.pixel_mm_x
        right = self.x0 + (cols - 0.5) * self.pixel_mm_x
        bottom = self.y0 - 0.5 * self.pixel_mm_y
        top = self.y0 + (rows - 0.5) * self.pixel_mm_y

        top_left = self.mm_to_widget(left, top if self.row_axis_up else bottom)
        bottom_right = self.mm_to_widget(right, bottom if self.row_axis_up else top)
        target = QRectF(top_left, bottom_right)
        pixmap = self._pixmap
        if self.row_axis_up:
            pixmap = self._pixmap.transformed(
                self._flip_transform(), Qt.TransformationMode.FastTransformation
            )
        painter.drawPixmap(target, pixmap, QRectF(pixmap.rect()))

    @staticmethod
    def _flip_transform():
        from PyQt6.QtGui import QTransform

        return QTransform().scale(1.0, -1.0)

    def _draw_overlays(self, painter: QPainter) -> None:
        for item in self.overlays:
            if len(item.points) == 0:
                continue
            pen = QPen(item.colour)
            pen.setWidthF(item.width)
            pen.setCosmetic(True)
            painter.setPen(pen)
            pts = [self.mm_to_widget(float(p[0]), float(p[1])) for p in item.points]
            if item.kind == "line" and len(pts) > 1:
                painter.drawPolyline(*pts)
            elif item.kind == "points":
                painter.setBrush(item.colour)
                for p in pts:
                    painter.drawEllipse(p, item.radius, item.radius)
                painter.setBrush(Qt.BrushStyle.NoBrush)
            elif item.kind == "crosshair":
                for p in pts:
                    painter.drawLine(
                        QPointF(p.x() - 9, p.y()), QPointF(p.x() + 9, p.y())
                    )
                    painter.drawLine(
                        QPointF(p.x(), p.y() - 9), QPointF(p.x(), p.y() + 9)
                    )
            if item.label and pts:
                painter.drawText(pts[-1] + QPointF(8, -8), item.label)

    def _draw_axes(self, painter: QPainter) -> None:
        painter.setPen(QColor(160, 160, 175))
        font = QFont(painter.font())
        font.setPointSizeF(8.0)
        painter.setFont(font)
        painter.drawText(52, self.height() - 4, self.x_label)
        painter.save()
        painter.translate(11, self.height() - 52)
        painter.rotate(-90)
        painter.drawText(0, 0, self.y_label)
        painter.restore()

    def _draw_scale_bar(self, painter: QPainter) -> None:
        target_px = 90.0
        raw_mm = target_px / self._scale()
        magnitude = 10.0 ** np.floor(np.log10(max(raw_mm, 1e-6)))
        for multiple in (1.0, 2.0, 5.0, 10.0):
            length_mm = multiple * magnitude
            if length_mm * self._scale() >= 55.0:
                break
        length_px = length_mm * self._scale()
        y = self.height() - 22
        x = 56
        pen = QPen(QColor(235, 235, 245))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(int(x), int(y), int(x + length_px), int(y))
        painter.drawLine(int(x), int(y - 4), int(x), int(y + 4))
        painter.drawLine(int(x + length_px), int(y - 4), int(x + length_px), int(y + 4))
        painter.drawText(int(x), int(y - 8), f"{length_mm:g} mm")
