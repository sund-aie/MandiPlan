"""Draw the MandiPlan application icon.

A mandible arch seen from below with a reconstruction bar running along its
inferior border. Bold shapes only, so it stays readable at 32 px.

    python3 packaging/make_icon.py            # writes packaging/icon.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PyQt6.QtGui import (  # noqa: E402
    QBrush,
    QColor,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)

SIZE = 1024
BACKGROUND_TOP = QColor(32, 38, 50)
BACKGROUND_BOTTOM = QColor(14, 17, 24)
BONE = QColor(238, 230, 214)
BONE_SHADOW = QColor(196, 186, 168)
PLATE = QColor(126, 166, 221)
SCREW = QColor(30, 44, 66)


def _arch_path(width: float, depth: float, top: float, bottom: float) -> QPainterPath:
    """A U-shaped mandible outline: two rami down to a rounded symphysis."""
    half = width / 2.0
    path = QPainterPath()
    path.moveTo(QPointF(SIZE / 2 - half, top))
    path.cubicTo(
        QPointF(SIZE / 2 - half - depth * 0.05, top + depth * 0.62),
        QPointF(SIZE / 2 - half * 0.72, bottom),
        QPointF(SIZE / 2, bottom),
    )
    path.cubicTo(
        QPointF(SIZE / 2 + half * 0.72, bottom),
        QPointF(SIZE / 2 + half + depth * 0.05, top + depth * 0.62),
        QPointF(SIZE / 2 + half, top),
    )
    return path


def draw_icon() -> QImage:
    image = QImage(SIZE, SIZE, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    gradient = QLinearGradient(0, 0, 0, SIZE)
    gradient.setColorAt(0.0, BACKGROUND_TOP)
    gradient.setColorAt(1.0, BACKGROUND_BOTTOM)
    plate_rect = QRectF(0, 0, SIZE, SIZE)
    painter.setBrush(QBrush(gradient))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(plate_rect, SIZE * 0.22, SIZE * 0.22)

    # Mandible body: a thick stroked arch, with a darker inner edge so the
    # cortex reads as a solid bone rather than a wire.
    arch = _arch_path(width=580, depth=430, top=248, bottom=764)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(BONE_SHADOW, 132, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawPath(arch)
    painter.setPen(QPen(BONE, 108, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawPath(arch)

    # Condylar heads.
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(BONE)
    for x in (SIZE / 2 - 290, SIZE / 2 + 290):
        painter.drawEllipse(QPointF(x, 232), 84, 64)

    # Reconstruction bar along the outer border. It spans the body and angle
    # and stops short of the condyles, which is where a bar actually sits.
    bar = _arch_path(width=700, depth=430, top=402, bottom=852)
    span = [bar.pointAtPercent(t / 100.0) for t in range(14, 87)]
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(PLATE, 50, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
    painter.drawPolyline(*span)

    # Screw holes, spaced evenly along the bar.
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(SCREW)
    for fraction in (0.17, 0.29, 0.41, 0.5, 0.59, 0.71, 0.83):
        painter.drawEllipse(bar.pointAtPercent(fraction), 15, 15)

    painter.end()
    return image


def main() -> int:
    out = Path(__file__).resolve().parent / "icon.png"
    draw_icon().save(str(out))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
