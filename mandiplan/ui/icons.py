"""Toolbar icons, drawn with QPainter.

Every glyph is stroked from code on a 24x24 grid, so there are no image files
to ship, nothing to fetch, and no emoji anywhere. Icons follow the current
text colour by default and the accent colour when the action is active, and
they are rendered at the widget's device pixel ratio so they stay sharp on a
Retina or scaled display.

Add a glyph by adding a ``_draw_*`` function and an entry in ``_GLYPHS``.
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

from .theme import TOKENS

_GRID = 24.0
_STROKE = 1.9


def _path(*segments: list[tuple[float, float]]) -> QPainterPath:
    """A path made of one or more open polylines given in grid coordinates."""
    path = QPainterPath()
    for points in segments:
        path.moveTo(QPointF(*points[0]))
        for point in points[1:]:
            path.lineTo(QPointF(*point))
    return path


def _draw_import(p: QPainter) -> None:
    p.drawPath(_path([(12, 3), (12, 14)], [(8, 10), (12, 14), (16, 10)]))
    p.drawPath(_path([(4, 16), (4, 20), (20, 20), (20, 16)]))


def _draw_export(p: QPainter) -> None:
    p.drawPath(_path([(12, 14), (12, 3)], [(8, 7), (12, 3), (16, 7)]))
    p.drawPath(_path([(4, 16), (4, 20), (20, 20), (20, 16)]))


def _draw_threshold(p: QPainter) -> None:
    """Half-filled circle: a gray-value split, i.e. the bone threshold."""
    p.drawEllipse(QRectF(3.5, 3.5, 17, 17))
    wedge = QPainterPath()
    wedge.moveTo(12, 3.5)
    wedge.arcTo(QRectF(3.5, 3.5, 17, 17), 90, -180)
    wedge.closeSubpath()
    p.fillPath(wedge, p.pen().color())


def _draw_arch(p: QPainter) -> None:
    curve = QPainterPath()
    curve.moveTo(3, 7)
    curve.cubicTo(QPointF(6, 20), QPointF(18, 20), QPointF(21, 7))
    p.drawPath(curve)
    for x, y in ((3, 7), (12, 16.4), (21, 7)):
        p.drawEllipse(QPointF(x, y), 1.7, 1.7)


def _draw_mirror(p: QPainter) -> None:
    pen = p.pen()
    dashed = QPen(pen)
    dashed.setDashPattern([2.0, 2.0])
    p.setPen(dashed)
    p.drawPath(_path([(12, 2.5), (12, 21.5)]))
    p.setPen(pen)
    p.drawPath(_path([(9, 5), (3, 12), (9, 19), (9, 5)]))
    p.drawPath(_path([(15, 5), (21, 12), (15, 19), (15, 5)]))


def _draw_resection(p: QPainter) -> None:
    """A mandibular arc with two cut planes across it."""
    body = QPainterPath()
    body.moveTo(3, 6)
    body.cubicTo(QPointF(6, 19), QPointF(18, 19), QPointF(21, 6))
    p.drawPath(body)
    pen = p.pen()
    dashed = QPen(pen)
    dashed.setDashPattern([2.4, 1.8])
    p.setPen(dashed)
    p.drawPath(_path([(8, 3.5), (8, 20.5)], [(16, 3.5), (16, 20.5)]))
    p.setPen(pen)


def _draw_plate(p: QPainter) -> None:
    """A lobed bar with three drilled holes — the shape of the real thing."""
    bar = QPainterPath()
    bar.addRoundedRect(QRectF(2.5, 8.5, 19, 7), 3.5, 3.5)
    p.drawPath(bar)
    for x in (6.5, 12.0, 17.5):
        p.drawEllipse(QPointF(x, 12), 1.7, 1.7)


def _draw_measure(p: QPainter) -> None:
    p.drawPath(_path([(3, 15), (9, 9)], [(9, 9), (15, 3)]))
    p.save()
    p.translate(3, 15)
    p.rotate(-45)
    p.drawRect(QRectF(0, -4.6, 17, 4.6))
    p.restore()
    p.drawPath(_path([(16.5, 16.5), (21, 21)]))


def _draw_angle(p: QPainter) -> None:
    p.drawPath(_path([(4, 20), (20, 20)], [(4, 20), (17, 6)]))
    arc = QPainterPath()
    arc.moveTo(12.5, 20)
    arc.arcTo(QRectF(-4.5, 11.5, 17, 17), 0, 47)
    p.drawPath(arc)


def _draw_undo(p: QPainter) -> None:
    arc = QPainterPath()
    arc.moveTo(4, 11)
    arc.arcTo(QRectF(4, 6, 16, 12), 180, -230)
    p.drawPath(arc)
    p.drawPath(_path([(4, 5.5), (4, 11), (9.5, 11)]))


def _draw_redo(p: QPainter) -> None:
    arc = QPainterPath()
    arc.moveTo(20, 11)
    arc.arcTo(QRectF(4, 6, 16, 12), 0, 230)
    p.drawPath(arc)
    p.drawPath(_path([(20, 5.5), (20, 11), (14.5, 11)]))


def _draw_fit(p: QPainter) -> None:
    p.drawPath(
        _path([(3, 8.5), (3, 3), (8.5, 3)], [(15.5, 3), (21, 3), (21, 8.5)]),
    )
    p.drawPath(
        _path([(21, 15.5), (21, 21), (15.5, 21)], [(8.5, 21), (3, 21), (3, 15.5)]),
    )


def _draw_panel(p: QPainter) -> None:
    p.drawRect(QRectF(3, 4.5, 18, 15))
    p.drawPath(_path([(15, 4.5), (15, 19.5)]))


def _draw_help(p: QPainter) -> None:
    p.drawEllipse(QRectF(3.5, 3.5, 17, 17))
    mark = QPainterPath()
    mark.moveTo(9.2, 9.6)
    mark.cubicTo(QPointF(9.4, 6.6), QPointF(14.8, 6.6), QPointF(14.6, 9.8))
    mark.cubicTo(QPointF(14.4, 12.2), QPointF(12, 12.2), QPointF(12, 15))
    p.drawPath(mark)
    p.drawEllipse(QPointF(12, 18), 1.0, 1.0)


def _draw_navigate(p: QPainter) -> None:
    """Four-way arrows: orbit, pan and zoom the view."""
    p.drawPath(_path([(12, 3), (12, 21)], [(3, 12), (21, 12)]))
    p.drawPath(_path([(9, 6), (12, 3), (15, 6)]))
    p.drawPath(_path([(9, 18), (12, 21), (15, 18)]))
    p.drawPath(_path([(6, 9), (3, 12), (6, 15)]))
    p.drawPath(_path([(18, 9), (21, 12), (18, 15)]))


def _draw_landmark(p: QPainter) -> None:
    """A map pin: a marked tumour margin point."""
    pin = QPainterPath()
    pin.moveTo(12, 21.5)
    pin.cubicTo(QPointF(12, 16), QPointF(5.5, 14.2), QPointF(5.5, 9.2))
    pin.cubicTo(QPointF(5.5, 2.6), QPointF(18.5, 2.6), QPointF(18.5, 9.2))
    pin.cubicTo(QPointF(18.5, 14.2), QPointF(12, 16), QPointF(12, 21.5))
    p.drawPath(pin)
    p.drawEllipse(QPointF(12, 9.2), 2.3, 2.3)


_GLYPHS = {
    "navigate": _draw_navigate,
    "landmark": _draw_landmark,
    "import": _draw_import,
    "export": _draw_export,
    "threshold": _draw_threshold,
    "arch": _draw_arch,
    "mirror": _draw_mirror,
    "resection": _draw_resection,
    "plate": _draw_plate,
    "measure": _draw_measure,
    "angle": _draw_angle,
    "undo": _draw_undo,
    "redo": _draw_redo,
    "fit": _draw_fit,
    "panel": _draw_panel,
    "help": _draw_help,
}


def glyph_names() -> tuple[str, ...]:
    return tuple(_GLYPHS)


def _pixmap(name: str, size: int, colour: str, ratio: float) -> QPixmap:
    pixmap = QPixmap(int(size * ratio), int(size * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    # QPainter already applies the pixmap's device pixel ratio, so scale by
    # the logical size only; folding ``ratio`` in again would over-scale.
    painter.scale(size / _GRID, size / _GRID)
    pen = QPen(QColor(colour))
    pen.setWidthF(_STROKE)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    _GLYPHS[name](painter)
    painter.end()
    return pixmap


def icon(name: str, size: int = 20, ratio: float = 2.0) -> QIcon:
    """A themed icon with normal, active/selected and disabled renderings.

    ``ratio`` is the device pixel ratio to rasterise at; 2.0 covers Retina and
    150-200% Windows scaling without a visible penalty at 100%.
    """
    if name not in _GLYPHS:
        raise KeyError(f"no icon glyph named {name!r}")
    result = QIcon()
    result.addPixmap(
        _pixmap(name, size, TOKENS["text_secondary"], ratio),
        QIcon.Mode.Normal,
        QIcon.State.Off,
    )
    for mode, state in (
        (QIcon.Mode.Normal, QIcon.State.On),
        (QIcon.Mode.Selected, QIcon.State.On),
        (QIcon.Mode.Active, QIcon.State.On),
    ):
        result.addPixmap(_pixmap(name, size, TOKENS["accent"], ratio), mode, state)
    result.addPixmap(
        _pixmap(name, size, TOKENS["text_primary"], ratio),
        QIcon.Mode.Active,
        QIcon.State.Off,
    )
    result.addPixmap(
        _pixmap(name, size, TOKENS["text_disabled"], ratio),
        QIcon.Mode.Disabled,
        QIcon.State.Off,
    )
    return result
