"""The application's visual layer. Styling only — no interaction logic here.

A light, flat, low-chrome look: an off-white ground rather than stark white,
one accent colour used only for primary actions, soft shadows instead of hard
borders, 10 px corners, and a system sans-serif stack. The 3-D viewport stays
the dominant element and the controls read as an overlay on it.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QGraphicsDropShadowEffect
from PyQt6.QtGui import QColor

# Palette
BACKGROUND = "#f1f3f4"       # off-white ground, not stark white
SURFACE = "#ffffff"
SURFACE_SUNKEN = "#f8f9fa"
BORDER = "#e3e6ea"
TEXT = "#202124"
TEXT_MUTED = "#5f6368"
TEXT_FAINT = "#80868b"
ACCENT = "#1a73e8"           # used sparingly, for primary actions only
ACCENT_HOVER = "#1b66c9"
ACCENT_SOFT = "#e8f0fe"
DANGER = "#c5221f"
DANGER_SOFT = "#fce8e6"
SUCCESS = "#188038"

RADIUS = 10
FONT_STACK = (
    '"Google Sans", system-ui, -apple-system, "Segoe UI", Roboto, Inter, '
    '"Helvetica Neue", Arial, sans-serif'
)

# The 3-D viewport background, kept light so the viewport belongs to the same
# surface as the rest of the interface rather than sitting in a dark hole.
VIEWPORT_BACKGROUND = (0.898, 0.914, 0.929)
VIEWPORT_BACKGROUND_TOP = (0.965, 0.973, 0.980)
VIEWPORT_IMAGE_BACKGROUND = "#e9ecef"


def stylesheet() -> str:
    return f"""
    QWidget {{
        background: {BACKGROUND};
        color: {TEXT};
        font-family: {FONT_STACK};
        font-size: 13px;
    }}
    QMainWindow::separator {{ background: transparent; width: 8px; height: 8px; }}

    QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none; }}
    QDockWidget::title {{
        background: transparent;
        padding: 10px 12px 4px 12px;
        font-weight: 500;
        color: {TEXT_MUTED};
    }}

    QToolBar {{
        background: {SURFACE};
        border: none;
        border-radius: {RADIUS}px;
        padding: 6px;
        spacing: 2px;
    }}
    QToolBar QToolButton {{
        background: transparent;
        border: none;
        border-radius: {RADIUS - 2}px;
        padding: 7px 13px;
        color: {TEXT_MUTED};
        font-weight: 500;
    }}
    QToolBar QToolButton:hover {{ background: {SURFACE_SUNKEN}; color: {TEXT}; }}
    QToolBar QToolButton:checked {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
    QToolBar::separator {{ background: {BORDER}; width: 1px; margin: 6px 8px; }}

    QMenuBar {{ background: {BACKGROUND}; border: none; padding: 2px 6px; }}
    QMenuBar::item {{ background: transparent; padding: 6px 11px; border-radius: 8px; }}
    QMenuBar::item:selected {{ background: {SURFACE_SUNKEN}; }}
    QMenu {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: {RADIUS}px;
        padding: 6px;
    }}
    QMenu::item {{ padding: 7px 22px; border-radius: 7px; }}
    QMenu::item:selected {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}

    QTabWidget::pane {{ border: none; background: transparent; }}
    QTabBar::tab {{
        background: transparent;
        color: {TEXT_MUTED};
        padding: 9px 18px;
        margin-right: 3px;
        border: none;
        border-radius: 8px;
        font-weight: 500;
    }}
    QTabBar::tab:hover {{ background: {SURFACE_SUNKEN}; }}
    QTabBar::tab:selected {{ background: {SURFACE}; color: {ACCENT}; }}

    QToolBox {{ background: transparent; }}
    QToolBox::tab {{
        background: {SURFACE};
        border: none;
        border-radius: {RADIUS}px;
        color: {TEXT_MUTED};
    }}
    QToolBox::tab:selected {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
    QToolBox QToolButton {{
        background: transparent;
        border: none;
        color: {TEXT_MUTED};
        font-weight: 500;
        text-align: left;
        padding: 11px 14px;
    }}
    QToolBox QScrollArea {{ border: none; background: transparent; }}

    QPushButton {{
        background: {SURFACE};
        border: none;
        border-radius: {RADIUS - 2}px;
        padding: 9px 15px;
        color: {TEXT};
        font-weight: 500;
    }}
    QPushButton:hover {{ background: {SURFACE_SUNKEN}; }}
    QPushButton:pressed {{ background: {BORDER}; }}
    QPushButton:checked {{ background: {ACCENT}; color: white; }}
    QPushButton:disabled {{ color: {TEXT_FAINT}; background: {SURFACE_SUNKEN}; }}
    QPushButton[primary="true"] {{ background: {ACCENT}; color: white; }}
    QPushButton[primary="true"]:hover {{ background: {ACCENT_HOVER}; }}
    QPushButton[danger="true"] {{ background: {DANGER_SOFT}; color: {DANGER}; }}

    QGroupBox {{
        background: {SURFACE};
        border: none;
        border-radius: {RADIUS}px;
        margin-top: 16px;
        padding: 12px 12px 10px 12px;
        font-weight: 500;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 4px;
        padding: 0 2px;
        color: {TEXT_MUTED};
    }}

    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
        background: {SURFACE_SUNKEN};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 6px 9px;
        selection-background-color: {ACCENT_SOFT};
        selection-color: {TEXT};
    }}
    QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
        border: 1px solid {ACCENT};
        background: {SURFACE};
    }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox QAbstractItemView {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
        selection-background-color: {ACCENT_SOFT};
        selection-color: {TEXT};
        padding: 4px;
    }}
    QDoubleSpinBox::up-button, QSpinBox::up-button,
    QDoubleSpinBox::down-button, QSpinBox::down-button {{ width: 16px; border: none; }}

    QSlider::groove:horizontal {{
        height: 4px;
        background: {BORDER};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: {ACCENT};
        width: 14px;
        height: 14px;
        margin: -6px 0;
        border-radius: 7px;
    }}

    QCheckBox {{ spacing: 8px; }}
    QCheckBox::indicator {{
        width: 16px; height: 16px;
        border: 2px solid {TEXT_FAINT};
        border-radius: 4px;
        background: {SURFACE};
    }}
    QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

    QTableWidget, QTableView {{
        background: {SURFACE};
        alternate-background-color: {SURFACE_SUNKEN};
        border: none;
        border-radius: {RADIUS}px;
        gridline-color: {BORDER};
        selection-background-color: {ACCENT_SOFT};
        selection-color: {TEXT};
    }}
    QHeaderView::section {{
        background: {SURFACE};
        border: none;
        border-bottom: 1px solid {BORDER};
        padding: 7px 8px;
        color: {TEXT_MUTED};
        font-weight: 500;
    }}

    QScrollBar:vertical, QScrollBar:horizontal {{
        background: transparent;
        width: 10px; height: 10px;
        margin: 2px;
    }}
    QScrollBar::handle {{ background: #dadce0; border-radius: 5px; min-height: 28px; }}
    QScrollBar::handle:hover {{ background: #bdc1c6; }}
    QScrollBar::add-line, QScrollBar::sub-line,
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; border: none; }}

    QStatusBar {{ background: {BACKGROUND}; border: none; }}
    QStatusBar::item {{ border: none; }}
    QLabel {{ background: transparent; }}
    QToolTip {{
        background: {TEXT};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 6px 9px;
    }}
    """


def card(widget, radius: int = RADIUS) -> None:
    """Make a widget read as a raised surface: white, rounded, softly shadowed."""
    widget.setStyleSheet(
        f"background: {SURFACE}; border-radius: {radius}px;"
    )
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(18)
    shadow.setXOffset(0)
    shadow.setYOffset(2)
    shadow.setColor(QColor(60, 64, 67, 38))
    widget.setGraphicsEffect(shadow)
