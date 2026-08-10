"""Find out which layer of the graphics stack is failing.

Run it from the repository, with the virtual environment active:

    python3 tools/check_display.py

It opens three windows in turn, each for a few seconds, and prints what it
built. Watch the screen and note which of the three you actually see; that
tells us exactly where the problem is:

    stage 1 only        Qt works, VTK's widget does not
    stages 1 and 3      the QOpenGLWidget base is the one to use
    nothing at all      Qt itself is not putting windows on screen

Paste the printed output back along with which windows appeared.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SECONDS = 5


def _environment_report() -> None:
    import PyQt6.QtCore as qtcore
    import vtk
    from PyQt6.QtWidgets import QApplication

    app = QApplication([])
    screen = app.primaryScreen()
    print(f"  python           {sys.version.split()[0]} ({sys.platform})")
    print(f"  Qt               {qtcore.QT_VERSION_STR} via PyQt {qtcore.PYQT_VERSION_STR}")
    print(f"  Qt platform      {app.platformName()}")
    print(f"  VTK              {vtk.VTK_VERSION}")
    print(f"  screen           {screen.size().width()} x {screen.size().height()} points")
    available = screen.availableGeometry()
    print(f"  usable area      {available.width()} x {available.height()} points")
    print(f"  device pixels    {screen.devicePixelRatio():g}x")
    app.quit()


def _stage(number: int) -> int:
    """One stage, in its own process: the VTK base class is fixed at import."""
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget

    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle(f"MandiPlan display check — stage {number}")
    central = QWidget()
    layout = QVBoxLayout(central)
    label = QLabel(f"STAGE {number}\nyou should see this window")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setStyleSheet("font-size: 26px; padding: 20px;")
    layout.addWidget(label)

    if number == 1:
        description = "plain Qt window, no VTK"
    else:
        import vtkmodules.qt

        vtkmodules.qt.PyQtImpl = "PyQt6"
        vtkmodules.qt.QVTKRWIBase = "QWidget" if number == 2 else "QOpenGLWidget"
        description = f"Qt + VTK widget on the {vtkmodules.qt.QVTKRWIBase} base"

        import vtk
        from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

        interactor = QVTKRenderWindowInteractor(central)
        layout.addWidget(interactor, 1)
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.1, 0.15, 0.3)
        interactor.GetRenderWindow().AddRenderer(renderer)
        source = vtk.vtkConeSource()
        source.SetResolution(32)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer.AddActor(actor)
        renderer.ResetCamera()

    window.setCentralWidget(central)
    window.resize(560, 420)
    window.show()
    window.raise_()
    window.activateWindow()
    if number != 1:
        interactor.Initialize()
        interactor.GetRenderWindow().Render()

    print(f"    built: {description}")
    print(f"    window.isVisible() = {window.isVisible()}")
    geometry = window.geometry()
    print(
        f"    window at ({geometry.x()}, {geometry.y()}) "
        f"size {geometry.width()} x {geometry.height()}"
    )
    QTimer.singleShot(SECONDS * 1000, app.quit)
    app.exec()
    return 0


def main() -> int:
    if "--stage" in sys.argv:
        return _stage(int(sys.argv[sys.argv.index("--stage") + 1]))

    print("Environment")
    _environment_report()
    print()
    print(f"Three windows follow, {SECONDS} seconds each. Note which ones you see.")
    for number in (1, 2, 3):
        print(f"\nStage {number}:")
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--stage", str(number)],
            capture_output=True,
            text=True,
        )
        print(result.stdout.rstrip() or "    (no output)")
        if result.returncode != 0:
            print(f"    stage {number} exited with code {result.returncode}")
            tail = result.stderr.strip().splitlines()[-6:]
            for line in tail:
                print(f"    | {line}")
    print("\nDone. Tell me which stage windows appeared on screen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
