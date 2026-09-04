"""The application's visual layer. Styling only — no interaction logic here.

A clinical workspace rather than a dashboard: a cool neutral ground, white
surfaces, one accent blue reserved for the active tool and primary actions,
1 px low-contrast borders instead of shadows, and 8 px corners. The 3-D
viewport is the subject of the window and is deliberately the quietest thing
in it — near-white, flat, unbordered, so bone reads against it.

Structure
---------
``TOKENS`` is the single source of truth. Everything below is a small
``_qss_*`` fragment consuming those tokens, and :func:`stylesheet` joins them.
Add a component by adding a fragment, not by growing one blob.

Type scale is expressed in points, not pixels, so it follows the operating
system's font scaling. Paddings and radii stay in pixels, which Qt already
scales by the device pixel ratio.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QGraphicsDropShadowEffect

#: Semantic design tokens. Name things by role, never by colour.
TOKENS: dict[str, str] = {
    "background": "#f8f9fa",
    "surface": "#ffffff",
    "surface_sunken": "#f1f3f4",
    "surface_hover": "#f1f3f4",
    "border": "#dadce0",
    "border_subtle": "#e8eaed",
    "text_primary": "#202124",
    "text_secondary": "#5f6368",
    "text_disabled": "#9aa0a6",
    "accent": "#1a73e8",
    "accent_hover": "#1765cc",
    "accent_soft": "#e8f0fe",
    "selection": "#d2e3fc",
    "warning": "#b06000",
    "warning_soft": "#fef7e0",
    "danger": "#c5221f",
    "danger_soft": "#fce8e6",
    "success": "#188038",
}

# Backwards-compatible flat names. Existing modules import these directly.
BACKGROUND = TOKENS["background"]
SURFACE = TOKENS["surface"]
SURFACE_SUNKEN = TOKENS["surface_sunken"]
BORDER = TOKENS["border"]
TEXT = TOKENS["text_primary"]
TEXT_MUTED = TOKENS["text_secondary"]
TEXT_FAINT = TOKENS["text_disabled"]
ACCENT = TOKENS["accent"]
ACCENT_HOVER = TOKENS["accent_hover"]
ACCENT_SOFT = TOKENS["accent_soft"]
DANGER = TOKENS["danger"]
DANGER_SOFT = TOKENS["danger_soft"]
SUCCESS = TOKENS["success"]
WARNING = TOKENS["warning"]
WARNING_SOFT = TOKENS["warning_soft"]

RADIUS = 8
RADIUS_SM = 6

#: Type scale, in points so that OS font scaling is respected.
FONT_SM = "8.5pt"    # secondary technical labels, units, provenance
FONT_MD = "9pt"      # section headers (semibold)
FONT_BASE = "9.75pt"  # tool labels, body
FONT_LG = "10.5pt"   # primary numeric values

FONT_STACK = (
    '"Google Sans", Inter, Roboto, system-ui, -apple-system, "Segoe UI", '
    '"SF Pro Text", "Helvetica Neue", Arial, sans-serif'
)
MONO_STACK = (
    '"Roboto Mono", "SF Mono", "JetBrains Mono", Menlo, Consolas, '
    '"DejaVu Sans Mono", monospace'
)

#: The 3-D viewport: a near-white cool neutral, flat. Bone is ivory, so the
#: ground is kept slightly cooler and slightly darker than the panels to give
#: the mesh a silhouette without boxing the viewport in a border.
VIEWPORT_BACKGROUND = (0.929, 0.937, 0.949)
#: Retained so a gradient can be reinstated if depth cueing ever needs it.
VIEWPORT_BACKGROUND_TOP = VIEWPORT_BACKGROUND
VIEWPORT_IMAGE_BACKGROUND = "#eceff1"


def _qss_base(t: dict[str, str]) -> str:
    return f"""
    QWidget {{
        background: {t['background']};
        color: {t['text_primary']};
        font-family: {FONT_STACK};
        font-size: {FONT_BASE};
    }}
    QWidget:disabled {{ color: {t['text_disabled']}; }}
    QMainWindow::separator {{ background: {t['border_subtle']}; width: 1px; height: 1px; }}
    QToolTip {{
        background: {t['text_primary']};
        color: #ffffff;
        border: none;
        border-radius: {RADIUS_SM}px;
        padding: 5px 8px;
        font-size: {FONT_SM};
    }}
    """


def _qss_chrome(t: dict[str, str]) -> str:
    """Menu bar, toolbar and the dock frame."""
    return f"""
    QMenuBar {{ background: {t['surface']}; border-bottom: 1px solid {t['border_subtle']}; }}
    QMenuBar::item {{ padding: 6px 10px; background: transparent; border-radius: {RADIUS_SM}px; }}
    QMenuBar::item:selected {{ background: {t['surface_hover']}; }}
    QMenu {{
        background: {t['surface']};
        border: 1px solid {t['border']};
        border-radius: {RADIUS}px;
        padding: 6px;
    }}
    QMenu::item {{ padding: 6px 22px 6px 12px; border-radius: {RADIUS_SM}px; }}
    QMenu::item:selected {{ background: {t['accent_soft']}; color: {t['accent']}; }}
    QMenu::separator {{ height: 1px; background: {t['border_subtle']}; margin: 5px 8px; }}

    QToolBar {{
        background: {t['surface']};
        border-bottom: 1px solid {t['border_subtle']};
        padding: 5px 8px;
        spacing: 2px;
    }}
    QToolBar::separator {{
        background: {t['border_subtle']};
        width: 1px;
        margin: 5px 7px;
    }}
    QToolButton {{
        background: transparent;
        color: {t['text_secondary']};
        border: 1px solid transparent;
        border-radius: {RADIUS_SM}px;
        padding: 5px 9px;
        font-size: {FONT_BASE};
    }}
    QToolButton:hover {{ background: {t['surface_hover']}; color: {t['text_primary']}; }}
    QToolButton:checked {{
        background: {t['accent_soft']};
        color: {t['accent']};
        font-weight: 600;
    }}
    QToolButton:disabled {{ color: {t['text_disabled']}; }}

    QDockWidget {{ titlebar-close-icon: none; titlebar-normal-icon: none; }}
    QDockWidget::title {{
        background: {t['background']};
        padding: 9px 12px 5px 14px;
        font-size: {FONT_MD};
        font-weight: 600;
        color: {t['text_secondary']};
    }}
    QStatusBar {{
        background: {t['surface']};
        border-top: 1px solid {t['border_subtle']};
        color: {t['text_secondary']};
        font-size: {FONT_SM};
    }}
    QStatusBar::item {{ border: none; }}
    """


def _qss_inspector(t: dict[str, str]) -> str:
    """The right-hand inspector: toolbox sections and group boxes."""
    return f"""
    QToolBox {{ background: transparent; border: none; }}
    QToolBox::tab {{
        background: {t['surface']};
        border: 1px solid {t['border_subtle']};
        border-radius: {RADIUS}px;
        margin-bottom: 4px;
    }}
    QToolBox::tab:selected {{
        background: {t['accent_soft']};
        border-color: {t['accent_soft']};
    }}
    /* The tab label is painted by an internal QToolButton; styling
       QToolBox::tab alone leaves it invisible. */
    QToolBox QToolButton {{
        color: {t['text_primary']};
        font-size: {FONT_MD};
        font-weight: 600;
        text-align: left;
        padding: 9px 12px;
        background: transparent;
        border: none;
    }}
    QToolBox QToolButton:hover {{ color: {t['accent']}; }}
    QGroupBox {{
        background: {t['surface']};
        border: 1px solid {t['border_subtle']};
        border-radius: {RADIUS}px;
        margin-top: 14px;
        padding: 10px 10px 8px 10px;
        font-size: {FONT_MD};
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: {t['text_secondary']};
    }}
    """


def _qss_controls(t: dict[str, str]) -> str:
    """Buttons, spin boxes, combo boxes, sliders, check boxes."""
    return f"""
    QPushButton {{
        background: {t['surface']};
        color: {t['text_primary']};
        border: 1px solid {t['border']};
        border-radius: {RADIUS_SM}px;
        padding: 6px 14px;
        font-size: {FONT_BASE};
    }}
    QPushButton:hover {{ background: {t['surface_hover']}; }}
    QPushButton:pressed {{ background: {t['selection']}; }}
    QPushButton:disabled {{
        color: {t['text_disabled']};
        background: {t['surface_sunken']};
        border-color: {t['border_subtle']};
    }}
    QPushButton[primary="true"] {{
        background: {t['accent']};
        color: #ffffff;
        border-color: {t['accent']};
        font-weight: 600;
    }}
    QPushButton[primary="true"]:hover {{ background: {t['accent_hover']}; }}
    QPushButton[primary="true"]:disabled {{
        background: {t['surface_sunken']};
        color: {t['text_disabled']};
        border-color: {t['border_subtle']};
    }}

    QAbstractSpinBox, QComboBox, QLineEdit {{
        background: {t['surface']};
        border: 1px solid {t['border']};
        border-radius: {RADIUS_SM}px;
        padding: 5px 8px;
        min-height: 18px;
        selection-background-color: {t['selection']};
        selection-color: {t['text_primary']};
    }}
    QAbstractSpinBox:focus, QComboBox:focus, QLineEdit:focus {{
        border-color: {t['accent']};
    }}
    QAbstractSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled {{
        background: {t['surface_sunken']};
        color: {t['text_disabled']};
        border-color: {t['border_subtle']};
    }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    /* Without explicit rules the framed border bleeds into the stepper area
       and paints a stray line down the right edge of every spin box. */
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
        subcontrol-origin: border;
        background: transparent;
        border: none;
        width: 15px;
    }}
    QAbstractSpinBox::up-button {{ subcontrol-position: top right; }}
    QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; }}
    QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
        background: {t['surface_hover']};
    }}
    QComboBox QAbstractItemView {{
        background: {t['surface']};
        border: 1px solid {t['border']};
        border-radius: {RADIUS_SM}px;
        selection-background-color: {t['accent_soft']};
        selection-color: {t['accent']};
        outline: none;
    }}

    QCheckBox, QRadioButton {{ spacing: 7px; background: transparent; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; }}
    QCheckBox::indicator {{
        border: 1px solid {t['border']};
        border-radius: 3px;
        background: {t['surface']};
    }}
    QCheckBox::indicator:checked {{
        background: {t['accent']};
        border-color: {t['accent']};
    }}
    QCheckBox::indicator:disabled {{
        background: {t['surface_sunken']};
        border-color: {t['border_subtle']};
    }}

    QSlider::groove:horizontal {{
        height: 3px;
        background: {t['border']};
        border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: {t['accent']};
        width: 13px;
        height: 13px;
        margin: -6px 0;
        border-radius: 7px;
    }}
    QSlider::handle:horizontal:disabled {{ background: {t['text_disabled']}; }}
    """


def _qss_data(t: dict[str, str]) -> str:
    """Tables, lists, scroll bars, tabs, progress."""
    return f"""
    QTableWidget, QTableView, QListWidget, QTreeWidget {{
        background: {t['surface']};
        border: 1px solid {t['border_subtle']};
        border-radius: {RADIUS}px;
        gridline-color: {t['border_subtle']};
        selection-background-color: {t['accent_soft']};
        selection-color: {t['text_primary']};
        font-size: {FONT_SM};
        outline: none;
    }}
    QHeaderView::section {{
        background: {t['surface_sunken']};
        color: {t['text_secondary']};
        border: none;
        border-bottom: 1px solid {t['border_subtle']};
        padding: 6px 8px;
        font-size: {FONT_SM};
        font-weight: 600;
    }}
    QTableWidget::item {{ padding: 4px 6px; }}

    QTabWidget::pane {{
        border: none;
        background: {t['background']};
        top: -1px;
    }}
    QTabBar::tab {{
        background: transparent;
        color: {t['text_secondary']};
        border: none;
        border-radius: {RADIUS_SM}px;
        padding: 7px 14px;
        margin: 3px 2px;
        font-size: {FONT_BASE};
    }}
    QTabBar::tab:hover {{ background: {t['surface_hover']}; }}
    QTabBar::tab:selected {{
        background: {t['accent_soft']};
        color: {t['accent']};
        font-weight: 600;
    }}

    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle {{ background: {t['border']}; border-radius: 5px; min-height: 28px; }}
    QScrollBar::handle:hover {{ background: {t['text_disabled']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollArea {{ border: none; background: transparent; }}

    QProgressBar {{
        background: {t['surface_sunken']};
        border: none;
        border-radius: 3px;
        height: 4px;
        text-align: center;
    }}
    QProgressBar::chunk {{ background: {t['accent']}; border-radius: 3px; }}
    QSplitter::handle {{ background: {t['border_subtle']}; }}
    """


def _qss_semantic(t: dict[str, str]) -> str:
    """Role classes set with ``widget.setProperty("role", ...)``."""
    return f"""
    QLabel[role="section"] {{
        color: {t['text_secondary']};
        font-size: {FONT_MD};
        font-weight: 600;
        background: transparent;
    }}
    QLabel[role="value"] {{
        color: {t['text_primary']};
        font-size: {FONT_LG};
        font-family: {MONO_STACK};
        background: transparent;
    }}
    QLabel[role="hint"], QLabel[role="empty"] {{
        color: {t['text_secondary']};
        font-size: {FONT_SM};
        background: transparent;
    }}
    QLabel[role="empty"] {{ padding: 18px 12px; }}
    QLabel[role="wordmark"] {{
        color: {t['text_primary']};
        font-size: {FONT_LG};
        font-weight: 600;
        padding: 0 12px 0 6px;
        background: transparent;
    }}
    QLabel[role="badge-generic"] {{
        color: {t['warning']};
        background: {t['warning_soft']};
        border-radius: {RADIUS_SM}px;
        padding: 3px 8px;
        font-size: {FONT_SM};
        font-weight: 600;
    }}
    QLabel[role="badge-exact"] {{
        color: {t['success']};
        background: #e6f4ea;
        border-radius: {RADIUS_SM}px;
        padding: 3px 8px;
        font-size: {FONT_SM};
        font-weight: 600;
    }}
    QLabel[role="warning"] {{
        color: {t['warning']};
        background: {t['warning_soft']};
        border-radius: {RADIUS_SM}px;
        padding: 6px 10px;
        font-size: {FONT_SM};
    }}
    QLabel[role="danger"] {{
        color: {t['danger']};
        background: {t['danger_soft']};
        border-radius: {RADIUS_SM}px;
        padding: 6px 10px;
        font-size: {FONT_SM};
    }}
    """


_FRAGMENTS = (
    _qss_base,
    _qss_chrome,
    _qss_inspector,
    _qss_controls,
    _qss_data,
    _qss_semantic,
)


def stylesheet(tokens: dict[str, str] | None = None) -> str:
    """The application stylesheet, composed from the fragments above."""
    t = dict(TOKENS)
    if tokens:
        t.update(tokens)
    return "\n".join(fragment(t) for fragment in _FRAGMENTS)


def card(widget, radius: int = RADIUS) -> None:
    """Make a widget read as a distinct surface.

    A 1 px border rather than a shadow: shadows are reserved for things that
    genuinely float above the workspace, which a docked panel does not.
    """
    widget.setStyleSheet(
        f"background: {SURFACE}; border: 1px solid {TOKENS['border_subtle']};"
        f" border-radius: {radius}px;"
    )
    widget.setGraphicsEffect(None)


def elevate(widget, radius: int = RADIUS) -> None:
    """Light elevation for genuinely floating surfaces (menus, popovers)."""
    widget.setStyleSheet(f"background: {SURFACE}; border-radius: {radius}px;")
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(16)
    shadow.setXOffset(0)
    shadow.setYOffset(2)
    shadow.setColor(QColor(60, 64, 67, 40))
    widget.setGraphicsEffect(shadow)


def set_role(widget, role: str) -> None:
    """Tag a widget with a semantic role and re-polish it so QSS reapplies."""
    widget.setProperty("role", role)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
