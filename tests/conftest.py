"""Shared fixtures.  The phantom is built once per session; it takes seconds."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


def _ensure_display() -> None:
    """Make sure the GUI tests have an OpenGL-capable display.

    macOS and any desktop Linux session already have one.  On a headless
    Linux box VTK's 3-D widgets abort under Qt's ``offscreen`` platform
    plugin, so Xvfb is started instead when it is installed.
    """
    if os.environ.get("DISPLAY") or not sys.platform.startswith("linux"):
        return
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        return
    for number in range(99, 130):
        socket = Path(f"/tmp/.X11-unix/X{number}")
        if socket.exists():
            continue
        process = subprocess.Popen(
            [xvfb, f":{number}", "-screen", "0", "1600x1000x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=_die_with_parent,
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if socket.exists():
                os.environ["DISPLAY"] = f":{number}"
                return
            if process.poll() is not None:
                break
            time.sleep(0.05)
        process.terminate()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _die_with_parent() -> None:
    """Ask the kernel to stop Xvfb once pytest exits.

    Killing it from an atexit handler instead would pull the X server out from
    under VTK's still-open connection, and VTK aborts the interpreter when its
    display disappears — turning a green run into a non-zero exit code.
    """
    import ctypes
    import signal

    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


_ensure_display()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_phantom import PhantomSpec, make_phantom  # noqa: E402

from mandiplan.geometry import cpr  # noqa: E402
from mandiplan.geometry.spline import ArchCurve  # noqa: E402


@pytest.fixture(scope="session")
def spec() -> PhantomSpec:
    """Default phantom: anisotropic 0.3 x 0.3 x 0.6 mm voxels."""
    return PhantomSpec()


@pytest.fixture(scope="session")
def phantom(spec):
    return make_phantom(spec)


@pytest.fixture(scope="session")
def skewed_spec() -> PhantomSpec:
    """A second phantom whose three voxel dimensions all differ."""
    return PhantomSpec(spacing=(0.25, 0.5, 0.8), seed=7)


@pytest.fixture(scope="session")
def skewed_phantom(skewed_spec):
    return make_phantom(skewed_spec)


@pytest.fixture(scope="session")
def arch_curve(spec) -> ArchCurve:
    """Arch curve fitted through 7 seed points on the phantom centre-line."""
    return ArchCurve(spec.centre_line(7))


@pytest.fixture(scope="session")
def arch_frames(arch_curve):
    return cpr.build_frames(arch_curve, step_mm=0.2)


@pytest.fixture(scope="session")
def wide_arch_frames(spec):
    """Arch curve drawn past both ends of the bone, as a user would draw it."""
    curve = ArchCurve(spec.centre_line(9, extend_deg=15.0))
    return cpr.build_frames(curve, step_mm=0.2)


@pytest.fixture(scope="session")
def qt_app():
    """One QApplication for the whole session; Qt allows only one."""
    import PyQt6.sip as sip
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    app.closeAllWindows()
    app.processEvents()
    # Close Qt's display connection here.  If it were still open when the
    # temporary X server is shut down, Qt would abort the interpreter.
    sip.delete(app)


@pytest.fixture(scope="session")
def bone_surface(phantom, spec):
    from mandiplan.render.surface import extract_isosurface

    return extract_isosurface(phantom, spec.half_max_value)
