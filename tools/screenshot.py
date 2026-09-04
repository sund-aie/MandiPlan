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


def sweep_cut(window, out_dir: Path, yaw_deg: float = 20.0) -> list[Path]:
    """Capture one oblique cut at several positions along the mandible.

    The frames are the evidence for the local-frame convention: the plane's
    world orientation changes at every station while the angle it makes with
    the arch stays exactly ``yaw_deg``.
    """
    import numpy as np
    from PyQt6.QtWidgets import QApplication

    session = window.session
    frames = session.frames
    if frames is None:
        raise RuntimeError("lay an arch curve before sweeping a cut")

    index = frames.index_of(frames.length_mm / 2.0)
    session.add_plane(frames.points[index], frames.tangents[index])
    session.rotate_plane(0, yaw_deg=yaw_deg, tilt_deg=0.0)
    window.view3d.set_view_direction("superior")

    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, fraction in (("body", 0.2), ("angle", 0.5), ("ramus", 0.82)):
        session.translate_plane(0, frames.length_mm * fraction)
        QApplication.processEvents()
        window.view3d.reset_camera()
        tangent, _, _ = frames.frame_at(session.plane_arc_position(0))
        normal = session.planes[0].normal
        obliquity = np.degrees(
            np.arccos(abs(np.clip(np.dot(tangent, normal), -1.0, 1.0)))
        )
        print(
            f"  {label:>6}: s = {session.plane_arc_position(0):6.1f} mm  "
            f"normal = [{normal[0]:6.3f} {normal[1]:6.3f} {normal[2]:6.3f}]  "
            f"obliquity to arch = {obliquity:.2f} deg"
        )
        written.extend(
            capture(
                window,
                out_dir / f"cut-{label}.png",
                out_dir / f"cut-{label}-viewport.png",
            )
        )
    return written


def fit_plate(window, out_dir: Path, asset_id: str | None = None) -> list[Path]:
    """Draw a plate path along the bone and capture the fitted plate."""
    import numpy as np
    from PyQt6.QtWidgets import QApplication

    session = window.session
    frames = session.frames
    if frames is None or session.surface is None:
        raise RuntimeError("load a volume and lay an arch curve first")

    if asset_id:
        session.set_plate_asset(asset_id)
    # Walk the arch and drop path points on the bone beneath each station.
    for s_mm in np.linspace(
        frames.length_mm * 0.12, frames.length_mm * 0.88, 12
    ):
        point = frames.point_at(float(s_mm))
        session.add_plate_point(point)
    QApplication.processEvents()

    asset = session.plate_asset
    fitted = session.fitted_plate
    if fitted is None:
        raise RuntimeError("the plate did not fit; nothing to capture")
    print(
        f"  plate: {asset.name}\n"
        f"  status: {asset.status_label}\n"
        f"  holes: {asset.hole_count} at {asset.hole_pitch_mm:.1f} mm pitch, "
        f"{asset.hole_diameter_mm:.1f} mm diameter\n"
        f"  rigid placement residual: max {fitted.max_residual_mm:.2f} mm, "
        f"rms {fitted.rms_residual_mm:.2f} mm"
    )
    window.view3d.set_plate_overlays(True, False, False)
    window.view3d.set_view_direction("superior")
    # Three-quarter view: from directly above the plate is edge-on, and the
    # screw holes are the whole point of showing it.
    camera = window.view3d.renderer.GetActiveCamera()
    camera.Elevation(-38)
    camera.Azimuth(18)
    camera.OrthogonalizeViewUp()
    window.view3d.reset_camera()
    QApplication.processEvents()
    return capture(
        window, out_dir / "plate-fitted.png", out_dir / "plate-fitted-viewport.png"
    )


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
    parser.add_argument(
        "--cut-sweep",
        action="store_true",
        help=(
            "place one oblique cut and capture it at several positions along "
            "the jaw, to show it re-angling with the local anatomy"
        ),
    )
    parser.add_argument(
        "--plate",
        action="store_true",
        help="draw a plate path and capture the fitted plate asset",
    )
    parser.add_argument("--asset", default=None, help="plate asset id to fit")
    args = parser.parse_args(argv)

    _, window = build_window(
        args.width, args.height, args.phantom or args.cut_sweep or args.plate
    )
    if args.plate:
        for path in fit_plate(window, Path(args.out), args.asset):
            print(path)
        window.close()
        return 0
    if args.cut_sweep:
        for path in sweep_cut(window, Path(args.out)):
            print(path)
        window.close()
        return 0

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
