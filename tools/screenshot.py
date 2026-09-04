"""Repeatable application screenshots.

Captures the whole window and, separately, the 3-D viewport through VTK's own
render window, because ``QWidget.grab()`` cannot read back an OpenGL surface
and would otherwise leave the viewport black.

    python3 tools/screenshot.py --out artifacts/screenshots

Runs headless under Xvfb when there is no display. Makes no network calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def capture(window, path: Path, viewport_path: Path | None = None) -> list[Path]:
    """Save the window, and optionally the 3-D viewport, as PNG."""
    from PyQt6.QtWidgets import QApplication

    written: list[Path] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    QApplication.processEvents()
    window.grab().save(str(path))
    written.append(path)

    if viewport_path is not None:
        import vtk

        render_window = window.view3d.interactor.GetRenderWindow()
        render_window.Render()
        grabber = vtk.vtkWindowToImageFilter()
        grabber.SetInput(render_window)
        grabber.ReadFrontBufferOff()
        grabber.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(viewport_path))
        writer.SetInputConnection(grabber.GetOutputPort())
        writer.Write()
        written.append(viewport_path)
    return written


def build_window(width: int, height: int, phantom: bool):
    from mandiplan.ui.main_window import MainWindow
    from mandiplan.ui.theme import stylesheet
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(stylesheet())
    window = MainWindow()
    window.resize(width, height)
    window.show()
    app.processEvents()
    if phantom:
        _load_phantom(window)
    return app, window


def _load_phantom(window) -> None:
    """Load the test phantom so the screenshot shows a populated workspace."""
    sys.path.insert(0, str(ROOT / "tests"))
    from make_phantom import (  # type: ignore[import-not-found]
        PhantomSpec,
        make_phantom,
        write_dicom_series,
    )
    from PyQt6.QtWidgets import QApplication

    folder = ROOT / "artifacts" / "phantom"
    if not list(folder.glob("*.dcm")):
        write_dicom_series(make_phantom(PhantomSpec()), folder)
    window.load_folder(str(folder))
    QApplication.processEvents()

    # Lay an arch curve so the reformats and the resection tools have
    # something to work with in the captured frame.
    spec = PhantomSpec()
    for point in spec.centre_line(9, extend_deg=12.0):
        window.session.add_arch_seed([point[0], point[1], spec.centre_z])
    QApplication.processEvents()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/screenshots", type=Path)
    parser.add_argument("--name", default="workspace")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--phantom",
        action="store_true",
        help="load the test phantom first, for a populated workspace",
    )
    args = parser.parse_args(argv)

    _, window = build_window(args.width, args.height, args.phantom)
    written = capture(
        window,
        Path(args.out) / f"{args.name}.png",
        Path(args.out) / f"{args.name}-viewport.png",
    )
    for path in written:
        print(path)
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
