"""No control, click or sequence of actions may take the application down.

Every control is exercised in the empty state and with a full plan loaded,
mouse clicks are driven into every view in every mode, known user mistakes are
replayed (adding a cut before the arch exists, double-clicking an arch point),
and a seeded fuzz runs random sequences of real actions. Any exception that
escapes — including one raised inside a Qt signal handler, which PyQt6 would
otherwise turn into an abort — fails the test.
"""

from __future__ import annotations

import random
import sys
import traceback

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QAction, QMouseEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QToolBox,
)

from make_phantom import PhantomSpec, make_phantom, write_dicom_series

from mandiplan.ui.modes import Mode


@pytest.fixture(scope="module")
def errors():
    """Record exceptions from signal handlers instead of aborting."""
    caught: list[str] = []
    previous = sys.excepthook
    sys.excepthook = lambda t, e, tb: caught.append(
        "".join(traceback.format_exception(t, e, tb))
    )
    yield caught
    sys.excepthook = previous


@pytest.fixture(scope="module")
def dialogs(tmp_path_factory):
    """Stub every modal dialog so nothing blocks the run."""
    out = tmp_path_factory.mktemp("exports")
    saved = {
        "dir": QFileDialog.getExistingDirectory,
        "save": QFileDialog.getSaveFileName,
        "info": QMessageBox.information,
        "warn": QMessageBox.warning,
        "crit": QMessageBox.critical,
    }
    counter = {"n": 0}

    def save_name(*_a, **_k):
        counter["n"] += 1
        return str(out / f"export_{counter['n']}.dat"), ""

    QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")
    QFileDialog.getSaveFileName = staticmethod(save_name)
    for name in ("information", "warning", "critical"):
        setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
    yield out
    QFileDialog.getExistingDirectory = saved["dir"]
    QFileDialog.getSaveFileName = saved["save"]
    QMessageBox.information = saved["info"]
    QMessageBox.warning = saved["warn"]
    QMessageBox.critical = saved["crit"]


@pytest.fixture(scope="module")
def phantom_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("robust_dicom")
    write_dicom_series(make_phantom(PhantomSpec()), folder)
    return folder


@pytest.fixture(scope="module")
def win(qt_app, dialogs, errors):
    from mandiplan.ui.main_window import MainWindow

    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    window.start()
    qt_app.processEvents()
    yield window
    window.close()
    qt_app.processEvents()


def _run(qt_app, errors, label, fn, failures):
    before = len(errors)
    try:
        fn()
        qt_app.processEvents()
    except Exception:  # noqa: BLE001 - recording is the point
        failures.append(f"{label}\n{traceback.format_exc()}")
    for detail in errors[before:]:
        failures.append(f"{label} (inside a signal handler)\n{detail}")


def _sweep(win, qt_app, errors, phase) -> list[str]:
    failures: list[str] = []
    run = lambda label, fn: _run(qt_app, errors, f"[{phase}] {label}", fn, failures)  # noqa: E731

    for action in win.findChildren(QAction):
        text = action.text() or action.toolTip() or "<unnamed>"
        if "Quit" in text:
            continue
        run(f"action {text}", action.trigger)
    for button in win.findChildren(QPushButton):
        if button.isEnabled():
            run(f"button {button.text() or button.toolTip()}", button.click)
    for box in win.findChildren(QCheckBox):
        for state in (not box.isChecked(), box.isChecked()):
            run(f"checkbox {box.text()}={state}", lambda b=box, s=state: b.setChecked(s))
    for combo in win.findChildren(QComboBox):
        for i in range(combo.count()):
            run(f"combo {combo.itemText(i)}", lambda c=combo, i=i: c.setCurrentIndex(i))
    for spin in win.findChildren((QDoubleSpinBox, QSpinBox)):
        lo, hi = spin.minimum(), spin.maximum()
        for value in (lo, (lo + hi) / 2, hi):
            run(f"spin {spin.suffix()}={value}", lambda s=spin, v=value: s.setValue(v))
    for slider in win.findChildren(QSlider):
        for value in (slider.minimum(), slider.maximum()):
            run(f"slider={value}", lambda s=slider, v=value: s.setValue(v))
    for toolbox in win.findChildren(QToolBox):
        for i in range(toolbox.count()):
            run(f"page {toolbox.itemText(i)}", lambda t=toolbox, i=i: t.setCurrentIndex(i))
    for tabs in win.findChildren(QTabWidget):
        for i in range(tabs.count()):
            run(f"tab {tabs.tabText(i)}", lambda t=tabs, i=i: t.setCurrentIndex(i))
    return failures


def _load_plan(win, qt_app, folder):
    spec = PhantomSpec()
    win.load_folder(str(folder))
    for point in spec.centre_line(9, extend_deg=12.0):
        win.session.add_arch_seed([point[0], point[1], spec.centre_z])
    win.add_cut_plane()
    win.add_cut_plane()
    frames = win.session.frames
    for s_mm in np.linspace(frames.length_mm * 0.15, frames.length_mm * 0.85, 10):
        _, _, buccolingual = frames.frame_at(float(s_mm))
        win.session.add_plate_point(frames.point_at(float(s_mm)) + buccolingual * 12.0)
    qt_app.processEvents()


def _click(qt_app, widget, x, y):
    local = QPointF(x, y)
    global_pos = QPointF(widget.mapToGlobal(QPoint(int(x), int(y))))
    for kind in (QMouseEvent.Type.MouseButtonPress, QMouseEvent.Type.MouseButtonRelease):
        event = QMouseEvent(
            kind,
            local,
            global_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        qt_app.sendEvent(widget, event)
    qt_app.processEvents()


# -- the tests -----------------------------------------------------------


def test_every_control_in_the_empty_state(win, qt_app, errors):
    failures = _sweep(win, qt_app, errors, "empty")
    assert not failures, "\n\n".join(failures[:5])


def test_known_mistakes_do_not_crash(win, qt_app, errors, phantom_dir):
    failures: list[str] = []
    run = lambda label, fn: _run(qt_app, errors, label, fn, failures)  # noqa: E731
    session = win.session

    run("load a folder that does not exist", lambda: win.load_folder("/no/such/folder"))
    run("load the real scan", lambda: win.load_folder(str(phantom_dir)))
    # The two crashes found by hand: both were one click away.
    run("add a cut before drawing the arch", win.add_cut_plane)
    run("double-click an arch point", lambda: (
        session.add_arch_seed([10.0, 20.0, 5.0]),
        session.add_arch_seed([10.0, 20.0, 5.0]),
    ))
    run("shaky double-click, 0.1 um apart", lambda: (
        session.add_arch_seed([30.0, 20.0, 5.0]),
        session.add_arch_seed([30.0000001, 20.0, 5.0]),
    ))
    run("undo past the beginning", lambda: [session.undo() for _ in range(30)])
    run("execute a cut with no planes", session.execute_cut)
    run("build a graft with nothing to mirror", session.build_graft)
    run("flip a plane that does not exist", lambda: session.flip_plane(5))
    run("remove points from empty lists", lambda: (
        session.remove_last_arch_seed(),
        session.remove_last_plate_point(),
    ))
    assert not failures, "\n\n".join(failures[:5])


def test_every_control_with_a_full_plan(win, qt_app, errors, phantom_dir):
    _load_plan(win, qt_app, phantom_dir)
    failures = _sweep(win, qt_app, errors, "loaded")
    assert not failures, "\n\n".join(failures[:5])


def test_clicks_in_every_view_in_every_mode(win, qt_app, errors, phantom_dir):
    from mandiplan.ui.image_view import ImageView

    _load_plan(win, qt_app, phantom_dir)
    failures: list[str] = []
    views = [*win.findChildren(ImageView), win.view3d.interactor]
    for mode in Mode:
        for index, view in enumerate(views):
            _run(
                qt_app,
                errors,
                f"view {index} in {mode.value}",
                lambda m=mode, v=view: (
                    win.set_mode(m),
                    [_click(qt_app, v, 60 + 37 * i, 40 + 19 * i) for i in range(5)],
                ),
                failures,
            )
    assert not failures, "\n\n".join(failures[:5])


def test_random_sequences_of_real_actions(win, qt_app, errors, phantom_dir):
    from mandiplan.plate_assets import load_assets
    from mandiplan.plate_catalog import load_kits, load_systems

    _load_plan(win, qt_app, phantom_dir)
    session = win.session
    spec = PhantomSpec()
    rng = random.Random(20260924)
    actions = [
        ("arch seed", lambda: session.add_arch_seed(
            [rng.uniform(-40, 40), rng.uniform(-40, 40), spec.centre_z])),
        ("remove arch seed", session.remove_last_arch_seed),
        ("clear arch", session.clear_arch),
        ("add cut", win.add_cut_plane),
        ("clear cuts", session.clear_planes),
        ("flip", lambda: session.flip_plane(rng.randint(0, 2))),
        ("translate", lambda: session.translate_plane(rng.randint(0, 2), rng.uniform(-50, 200))),
        ("rotate", lambda: session.rotate_plane(
            rng.randint(0, 2), rng.uniform(-90, 90), rng.uniform(-90, 90))),
        ("asset", lambda: session.set_plate_asset(rng.choice(load_assets()).id)),
        ("kit", lambda: session.set_bending_kit(rng.choice(load_kits()).id)),
        ("system", lambda: session.set_plate_system(rng.choice(load_systems()).id)),
        ("clearance", lambda: session.set_plate_clearance(rng.uniform(0, 5))),
        ("bending", lambda: session.set_plate_bending(rng.random() > 0.5)),
        ("distortion", lambda: session.set_hole_distortion_shown(rng.random() > 0.5)),
        ("insets", lambda: session.set_bending_insets(rng.choice([True, False, None]))),
        ("remove plate point", session.remove_last_plate_point),
        ("clear plate path", session.clear_plate_path),
        ("undo", session.undo),
        ("execute cut", session.execute_cut),
        ("undo cut", session.undo_cut),
        ("graft", session.build_graft),
        ("mode", lambda: win.set_mode(rng.choice(list(Mode)))),
    ]
    failures: list[str] = []
    for step in range(260):
        name, action = rng.choice(actions)
        _run(qt_app, errors, f"step {step}: {name}", action, failures)
    assert not failures, "\n\n".join(failures[:5])


def test_the_safety_net_keeps_the_app_alive(qt_app):
    """An error escaping a handler is logged and reported, not fatal."""
    from mandiplan import safety

    reported: list[str] = []
    safety.set_reporter(reported.append)
    try:
        raise ValueError("deliberate test failure")
    except ValueError:
        safety.handle_exception(*sys.exc_info())
    safety.set_reporter(None)
    assert reported and "deliberate test failure" in reported[0]
    assert "plan is unchanged" in reported[0]
    assert "deliberate test failure" in safety.recent_errors[-1]
