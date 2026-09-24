"""Keep one failed action from taking the whole application down.

PyQt6 treats an exception escaping a signal handler — a button click, a menu
action, a mouse pick — as fatal and aborts the process. For a planning tool
that is the worst possible failure: one bad click and the plan is gone. This
module installs a handler that records the error, tells the operator in plain
words, and keeps the application running.

It is a safety net, not a substitute for fixing the fault. Every error it
catches is written with its full traceback to the log below so it can be
found and fixed at the root.

Hard crashes in native code (VTK, Qt) cannot be caught from Python at all;
``faulthandler`` writes their Python stack to the same log so there is at least
a trace of where it happened.
"""

from __future__ import annotations

import datetime as _dt
import faulthandler
import os
import sys
import traceback
from pathlib import Path


def log_dir() -> Path:
    """Where MandiPlan keeps its error log, per platform."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    path = base / "MandiPlan"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_path() -> Path:
    return log_dir() / "mandiplan-errors.log"


_reporter = None
_fault_file = None
#: Most recent errors, newest last, for tests and the About dialog.
recent_errors: list[str] = []


def set_reporter(callback) -> None:
    """Route the one-line error summary to the interface, e.g. the status bar."""
    global _reporter
    _reporter = callback


def _write(text: str) -> None:
    try:
        with log_path().open("a", encoding="utf-8") as handle:
            handle.write(text)
    except OSError:
        # Logging must never be the thing that fails.
        pass


def handle_exception(exc_type, exc, tb) -> None:
    """Record an unhandled error and keep running."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    detail = "".join(traceback.format_exception(exc_type, exc, tb))
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    _write(f"\n=== {stamp} ===\n{detail}")
    recent_errors.append(detail)
    del recent_errors[:-20]
    sys.__stderr__.write(detail)

    summary = f"{exc_type.__name__}: {exc}" if str(exc) else exc_type.__name__
    message = (
        f"That action failed and was cancelled ({summary}). The rest of your "
        f"plan is unchanged. Details: {log_path()}"
    )
    if _reporter is not None:
        try:
            _reporter(message)
        except Exception:  # noqa: BLE001 - the reporter itself must not raise
            pass


def install() -> None:
    """Install the handler and native fault tracing. Safe to call twice."""
    global _fault_file
    sys.excepthook = handle_exception
    if _fault_file is None:
        try:
            _fault_file = log_path().open("a", encoding="utf-8")
            faulthandler.enable(file=_fault_file, all_threads=True)
        except OSError:
            faulthandler.enable()
    _quiet_vtk()


def _quiet_vtk() -> None:
    """Send VTK's warnings to the log instead of a pop-up window.

    On Windows VTK opens its own output window for every warning, which looks
    like a crash dialog and steals focus.
    """
    try:
        import vtk

        output = vtk.vtkFileOutputWindow()
        output.SetFileName(str(log_dir() / "vtk-output.log"))
        output.SetAppend(True)
        vtk.vtkOutputWindow.SetInstance(output)
    except Exception:  # noqa: BLE001 - optional nicety
        pass
