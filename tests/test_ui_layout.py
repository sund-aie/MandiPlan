"""Layout and theme regressions for the clinical workspace shell.

Structural assertions, not pixel comparisons: a screenshot diff across Qt
versions, platforms and font packages fails for reasons that have nothing to
do with the interface being wrong. What is asserted here is what the redesign
actually promised — the viewport dominates, the inspector collapses cleanly,
nothing is unlabelled, and no text is clipped.
"""

from __future__ import annotations

import re

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QToolBar, QWidgetAction

from mandiplan.ui import theme
from mandiplan.ui.icons import glyph_names, icon

WINDOW_SIZE = (1440, 900)


@pytest.fixture(scope="module")
def shell(qt_app):
    """A bare window at laptop resolution. No DICOM: layout only."""
    from mandiplan.ui.main_window import MainWindow

    win = MainWindow()
    win.resize(*WINDOW_SIZE)
    win.show()
    qt_app.processEvents()
    yield win
    win.close()
    qt_app.processEvents()


# -- the viewport is the subject of the window ---------------------------


def test_the_viewport_is_the_largest_region(shell, qt_app):
    qt_app.processEvents()
    viewport = shell.centralWidget().width()
    inspector = shell.planning_dock.width()
    assert viewport > inspector, (viewport, inspector)
    assert viewport / WINDOW_SIZE[0] > 0.55


def test_the_inspector_is_on_the_right(shell):
    assert (
        shell.dockWidgetArea(shell.planning_dock)
        == Qt.DockWidgetArea.RightDockWidgetArea
    )


def test_the_inspector_is_width_limited_and_resizeable(shell):
    assert shell.planning_dock.minimumWidth() >= 320
    assert shell.planning_dock.maximumWidth() <= 640
    assert shell.planning_dock.features() & (
        shell.planning_dock.DockWidgetFeature.DockWidgetMovable
    )


def test_the_inspector_hides_and_restores_without_corruption(shell, qt_app):
    before = shell.centralWidget().width()

    shell.panel_action.setChecked(False)
    qt_app.processEvents()
    assert not shell.planning_dock.isVisible()
    collapsed = shell.centralWidget().width()
    assert collapsed > before

    shell.panel_action.setChecked(True)
    qt_app.processEvents()
    assert shell.planning_dock.isVisible()
    assert shell.centralWidget().width() == before
    assert shell.planning_dock.width() > 0


# -- nothing unlabelled, nothing clipped ---------------------------------


def test_every_icon_only_action_has_a_tooltip(shell):
    for toolbar in shell.findChildren(QToolBar):
        for action in toolbar.actions():
            # Widget actions are the wordmark and the flexible spacer; they
            # carry no icon and are not controls.
            if action.isSeparator() or action.text() or isinstance(
                action, QWidgetAction
            ):
                continue
            assert action.toolTip(), f"icon-only action with no tooltip: {action}"


def test_tool_tooltips_name_their_keyboard_shortcut(shell):
    for mode, action in shell.mode_actions.items():
        assert action.shortcut().toString(), f"{mode} has no shortcut"
        assert action.shortcut().toString() in action.toolTip(), mode


def test_no_visible_label_is_clipped(shell, qt_app):
    qt_app.processEvents()
    for label in shell.findChildren(QLabel):
        if not label.isVisible() or not label.text() or label.wordWrap():
            continue
        needed = label.sizeHint().width()
        assert label.width() + 1 >= needed, (
            f"clipped label {label.text()!r}: {label.width()}px < {needed}px"
        )


def test_the_toolbox_tabs_keep_their_text(shell):
    for index in range(shell.toolbox.count()):
        assert shell.toolbox.itemText(index).strip()


def test_the_wordmark_is_present(shell):
    texts = {label.text() for label in shell.findChildren(QLabel)}
    assert "MandiPlan" in texts


# -- theme tokens ---------------------------------------------------------

REQUIRED_ROLES = (
    "background",
    "surface",
    "surface_hover",
    "border",
    "text_primary",
    "text_secondary",
    "accent",
    "warning",
    "danger",
    "selection",
)


def test_theme_defines_every_semantic_role():
    missing = [role for role in REQUIRED_ROLES if role not in theme.TOKENS]
    assert not missing, missing


def test_every_token_is_a_hex_colour():
    for name, value in theme.TOKENS.items():
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", value), (name, value)


def test_the_stylesheet_has_no_unresolved_placeholders():
    qss = theme.stylesheet()
    assert "{t[" not in qss
    assert "None" not in qss


def test_font_sizes_scale_with_the_system_font():
    """Font sizes in points follow OS scaling; pixel sizes do not."""
    sizes = re.findall(r"font-size:\s*([^;]+);", theme.stylesheet())
    assert sizes
    bad = [size for size in sizes if size.strip().endswith("px")]
    assert not bad, f"hard-coded pixel font sizes: {bad}"


def test_a_caller_can_override_tokens():
    qss = theme.stylesheet({"accent": "#ff0000"})
    assert "#ff0000" in qss


# -- icons ----------------------------------------------------------------


def test_every_glyph_renders_without_a_file_on_disk(qt_app):
    for name in glyph_names():
        rendered = icon(name, size=20)
        assert not rendered.isNull(), name
        assert not rendered.pixmap(20, 20).isNull(), name


def test_icons_are_rasterised_above_one_to_one_for_hidpi(qt_app):
    """A 20 px icon at ratio 2 holds 40 real pixels, so it stays sharp.

    ``QIcon.pixmap`` hands back a device-ratio-1 copy, so the source pixmap is
    measured directly.
    """
    from mandiplan.ui.icons import _pixmap

    pixmap = _pixmap("plate", 20, "#000000", 2.0)
    assert pixmap.width() == 40
    assert pixmap.devicePixelRatio() == 2.0


def test_an_unknown_glyph_is_refused(qt_app):
    with pytest.raises(KeyError):
        icon("not-a-glyph")
