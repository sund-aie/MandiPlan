"""The MandiPlan main window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QSplitter,
    QTabWidget,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from ..constants import APP_NAME, DISCLAIMER
from ..dicom_io import DicomLoadError
from ..exporting import (
    write_bend_csv,
    write_plan_summary_csv,
    write_steps_csv,
    write_surface_stl,
    write_template_stl,
)
from ..geometry.cpr import cross_section_world_point
from ..geometry.measure import angle_deg, distance_mm, format_mm
from .image_view import ImageView, Overlay
from .modes import Mode
from .panels import (
    ArchPanel,
    PlatePanel,
    ReconstructionPanel,
    ResectionPanel,
    VolumePanel,
)
from .session import Session
from .view3d import View3D
from .workflow_bar import WorkflowBar

SEED_COLOUR = QColor(90, 220, 150)
CURVE_COLOUR = QColor(60, 190, 130)
MEASURE_COLOUR = QColor(255, 225, 80)
CUT_COLOUR = QColor(230, 90, 90)
CURSOR_COLOUR = QColor(120, 170, 255)

AXIS_LABELS = {
    "axial": ("x — patient left (mm)", "y — posterior (mm)"),
    "coronal": ("x — patient left (mm)", "z — superior (mm)"),
    "sagittal": ("y — posterior (mm)", "z — superior (mm)"),
}


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self._size_to_screen()

        self.session = Session(self)
        self.mode = Mode.NAVIGATE
        self._measure_points: dict[str, list[tuple[float, float]]] = {}
        self._last_measurement = ""

        self._build_views()
        self._build_docks()
        self._build_actions()
        self._build_status_bar()
        self._connect_session()
        self.set_mode(Mode.NAVIGATE)

    def _size_to_screen(self) -> None:
        """Open at a usable size that still fits the display.

        A hard-coded 1500 x 950 is larger than the usable area of a 13-inch
        laptop, which leaves the window hanging off the screen.
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1280, 860)
            return
        available = screen.availableGeometry()
        self.resize(
            min(1500, int(available.width() * 0.95)),
            min(950, int(available.height() * 0.95)),
        )
        self.move(available.center() - self.rect().center())

    # -- construction ------------------------------------------------------

    def _build_views(self) -> None:
        self.tabs = QTabWidget()
        self.view3d = View3D(self.session)
        self.tabs.addTab(self.view3d, "3-D bone")

        self.slice_views: dict[str, ImageView] = {}
        self.slice_sliders: dict[str, QSlider] = {}
        self.slice_labels: dict[str, QLabel] = {}
        slices = QWidget()
        grid = QGridLayout(slices)
        for column, name in enumerate(("axial", "coronal", "sagittal")):
            view = ImageView()
            view.title = name.capitalize()
            slider = QSlider(Qt.Orientation.Horizontal)
            label = QLabel("—")
            label.setStyleSheet("color: #9aa0b0; font-size: 10px;")
            grid.addWidget(view, 0, column)
            grid.addWidget(slider, 1, column)
            grid.addWidget(label, 2, column)
            self.slice_views[name] = view
            self.slice_sliders[name] = slider
            self.slice_labels[name] = label
            view.picked.connect(lambda x, y, n=name: self._on_slice_pick(n, x, y))
            slider.valueChanged.connect(lambda _v, n=name: self.refresh_slices())
        self.tabs.addTab(slices, "Slices")

        self.panoramic_view = ImageView()
        self.panoramic_view.title = "Panoramic reformat"
        self.cross_view = ImageView()
        self.cross_view.title = "Buccolingual cross-section"
        self.s_slider = QSlider(Qt.Orientation.Horizontal)
        self.s_spin = QDoubleSpinBox()
        self.s_spin.setSuffix(" mm along curve")
        self.s_spin.setDecimals(1)
        self.s_spin.setSingleStep(1.0)

        cpr_tab = QWidget()
        cpr_layout = QVBoxLayout(cpr_tab)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.panoramic_view)
        splitter.addWidget(self.cross_view)
        splitter.setSizes([500, 380])
        cpr_layout.addWidget(splitter, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Cross-section at"))
        row.addWidget(self.s_spin)
        row.addWidget(self.s_slider, 1)
        cpr_layout.addLayout(row)
        cpr_layout.addWidget(
            QLabel(
                "← / → step the cross-section along the curve (Shift for 5 mm steps)."
            )
        )
        self.tabs.addTab(cpr_tab, "Panoramic (CPR)")

        self.panoramic_view.picked.connect(
            lambda x, y: self._on_reformat_pick("panoramic", x, y)
        )
        self.cross_view.picked.connect(
            lambda x, y: self._on_reformat_pick("cross", x, y)
        )
        self.s_slider.valueChanged.connect(
            lambda v: self._set_cross_section(v / 10.0)
        )
        self.s_spin.valueChanged.connect(self._set_cross_section)
        self.setCentralWidget(self.tabs)

    def _build_docks(self) -> None:
        self.toolbox = QToolBox()
        self.volume_panel = VolumePanel(self.session)
        self.arch_panel = ArchPanel(self.session)
        self.resection_panel = ResectionPanel(self.session)
        self.reconstruction_panel = ReconstructionPanel(self.session)
        self.plate_panel = PlatePanel(self.session)
        self.toolbox.addItem(self.volume_panel, "1 · Volume and bone threshold")
        self.toolbox.addItem(self.arch_panel, "2 · Arch curve and reformat")
        self.toolbox.addItem(self.resection_panel, "3 · Resection planning")
        self.toolbox.addItem(self.reconstruction_panel, "4 · Mirror reconstruction")
        self.toolbox.addItem(self.plate_panel, "5 · Plate path and bends")

        self.workflow_bar = WorkflowBar(self.session)
        planning = QWidget()
        planning_layout = QVBoxLayout(planning)
        planning_layout.setContentsMargins(0, 0, 0, 0)
        planning_layout.addWidget(self.workflow_bar)
        planning_layout.addWidget(self.toolbox, 1)

        dock = QDockWidget("Planning", self)
        dock.setWidget(planning)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setMinimumWidth(400)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

        self.toolbox.currentChanged.connect(self.workflow_bar.set_page)
        self.workflow_bar.step_selected.connect(self.toolbox.setCurrentIndex)
        self.volume_panel.load_requested.connect(self.open_dicom_folder)
        for panel in (self.arch_panel, self.resection_panel, self.plate_panel):
            panel.mode_requested.connect(self.set_mode)
        self.plate_panel.export_csv_requested.connect(self.export_bend_csv)
        self.plate_panel.export_stl_requested.connect(self.export_template_stl)
        self.plate_panel.export_steps_requested.connect(self.export_steps_csv)
        self.reconstruction_panel.export_graft_requested.connect(self.export_graft_stl)

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_action = QAction("&Open DICOM folder…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_dicom_folder)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        for text, slot in (
            ("Export plate bend table (CSV)…", self.export_bend_csv),
            ("Export bending template (STL)…", self.export_template_stl),
            ("Export bench steps (CSV)…", self.export_steps_csv),
            ("Export reconstruction target (STL)…", self.export_graft_stl),
            ("Export resection summary (CSV)…", self.export_resection_csv),
            ("Export resected fragment (STL)…", self.export_fragment_stl),
        ):
            action = QAction(text, self)
            action.triggered.connect(slot)
            file_menu.addAction(action)
        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        edit_menu = self.menuBar().addMenu("&Edit")
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.session.undo)
        edit_menu.addAction(undo_action)

        toolbar = self.addToolBar("Mode")
        toolbar.setMovable(False)
        self.mode_actions: dict[Mode, QAction] = {}
        group = QActionGroup(self)
        group.setExclusive(True)
        for mode in Mode:
            action = QAction(mode.value, self)
            action.setCheckable(True)
            action.triggered.connect(lambda _c, m=mode: self.set_mode(m))
            group.addAction(action)
            toolbar.addAction(action)
            self.mode_actions[mode] = action
        toolbar.addSeparator()
        for label, direction in (
            ("Front", "anterior"),
            ("Left", "left"),
            ("Right", "right"),
            ("Top", "superior"),
        ):
            action = QAction(label, self)
            action.triggered.connect(
                lambda _c, d=direction: self.view3d.set_view_direction(d)
            )
            toolbar.addAction(action)
        add_plane = QAction("Add cutting plane", self)
        add_plane.triggered.connect(self.add_cut_plane)
        toolbar.addSeparator()
        toolbar.addAction(add_plane)
        self.resection_panel.add_plane.clicked.connect(self.add_cut_plane)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction("About MandiPlan", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)

    def _build_status_bar(self) -> None:
        self.hint_label = QLabel("")
        self.measure_label = QLabel("")
        self.measure_label.setStyleSheet("color: #ffe150;")
        # The disclaimer is a permanent status-bar widget: transient messages
        # are shown in the temporary area and never cover or replace it.
        self.banner = QLabel(DISCLAIMER)
        self.banner.setStyleSheet(
            "color: #ffffff; background: #a03030; padding: 2px 10px; font-weight: bold;"
        )
        bar = self.statusBar()
        # The hint shares its slot with transient messages, which is fine; the
        # measurement and the disclaimer are permanent, because Qt hides normal
        # status-bar widgets for as long as a message is showing.
        bar.addWidget(self.hint_label, 1)
        bar.addPermanentWidget(self.measure_label)
        bar.addPermanentWidget(self.banner)
        bar.setSizeGripEnabled(False)

    def _connect_session(self) -> None:
        session = self.session
        session.volume_changed.connect(self.refresh_slices)
        session.volume_changed.connect(self._reset_slice_sliders)
        session.arch_changed.connect(self.refresh_slices)
        session.reformat_changed.connect(self.refresh_reformats)
        session.resection_changed.connect(self.refresh_reformats)
        session.message.connect(self.show_message)
        self.view3d.surface_picked.connect(self._on_surface_pick)
        self.view3d.plane_moved.connect(self._on_plane_moved)

    def start(self) -> None:
        self.view3d.start()

    def closeEvent(self, event):  # noqa: N802
        self.view3d.shutdown()
        super().closeEvent(event)

    # -- modes and messages ------------------------------------------------

    def set_mode(self, mode: Mode) -> None:
        if getattr(self, "_setting_mode", False):
            return
        self._setting_mode = True
        try:
            self.mode = mode
            self.view3d.set_mode(mode)
            self._measure_points.clear()
            self.mode_actions[mode].setChecked(True)
            # Un-checking a panel's toggle button would otherwise bounce back
            # into set_mode and reset the mode that was just chosen.
            for button, owner in (
                (self.arch_panel.place_button, Mode.ARCH),
                (self.plate_panel.draw_button, Mode.PLATE),
                (self.resection_panel.landmark_button, Mode.LANDMARK),
            ):
                button.blockSignals(True)
                button.setChecked(mode == owner)
                button.blockSignals(False)
            self.hint_label.setText(f"{mode.value}: {mode.hint}")
        finally:
            self._setting_mode = False

    def show_message(self, text: str) -> None:
        self.statusBar().showMessage(text, 8000)

    def show_about(self) -> None:
        QMessageBox.information(
            self,
            f"About {APP_NAME}",
            f"{APP_NAME} — CBCT mandibular resection and reconstruction-plate "
            "planning.\n\n"
            f"{DISCLAIMER}\n\n"
            "All measurements are derived from DICOM voxel spacing in millimetres. "
            "The application makes no network connections.",
        )

    # -- loading -----------------------------------------------------------

    def open_dicom_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open DICOM folder")
        if not folder:
            return
        self.load_folder(folder)

    def load_folder(self, folder: str) -> None:
        try:
            self.session.load_dicom_folder(folder)
        except DicomLoadError as error:
            QMessageBox.critical(self, "Cannot load this series", str(error))
            return
        self.show_message(f"Loaded {Path(folder).name}")

    # -- slice views -------------------------------------------------------

    def _reset_slice_sliders(self) -> None:
        volume = self.session.volume
        if volume is None:
            return
        for name, axis in (("axial", 2), ("coronal", 1), ("sagittal", 0)):
            slider = self.slice_sliders[name]
            slider.blockSignals(True)
            slider.setRange(0, int(volume.size_xyz[axis]) - 1)
            slider.setValue(int(volume.size_xyz[axis]) // 2)
            slider.blockSignals(False)
        self.refresh_slices()

    def _slice_world_coord(self, name: str) -> float:
        volume = self.session.volume
        axis = {"axial": 2, "coronal": 1, "sagittal": 0}[name]
        return float(
            volume.origin[axis] + self.slice_sliders[name].value() * volume.spacing[axis]
        )

    def refresh_slices(self) -> None:
        volume = self.session.volume
        if volume is None:
            return
        sx, sy, sz = (float(v) for v in volume.spacing)
        ox, oy, oz = (float(v) for v in volume.origin)
        for name in ("axial", "coronal", "sagittal"):
            index = self.slice_sliders[name].value()
            view = self.slice_views[name]
            x_label, y_label = AXIS_LABELS[name]
            if name == "axial":
                image, px, py, x0, y0, up = (
                    volume.orthogonal_slice(2, index), sx, sy, ox, oy, False,
                )
            elif name == "coronal":
                image, px, py, x0, y0, up = (
                    volume.orthogonal_slice(1, index), sx, sz, ox, oz, True,
                )
            else:
                image, px, py, x0, y0, up = (
                    volume.orthogonal_slice(0, index), sy, sz, oy, oz, True,
                )
            view.set_image(
                image, px, py, x0, y0, x_label, y_label, row_axis_up=up, keep_view=True
            )
            self.slice_labels[name].setText(
                f"slice {index} · {self._slice_world_coord(name):.2f} mm"
            )
        self._update_slice_overlays()

    def _update_slice_overlays(self) -> None:
        session = self.session
        axial = self.slice_views["axial"]
        overlays: list[Overlay] = []
        if session.arch_seeds:
            overlays.append(
                Overlay(
                    points=np.vstack(session.arch_seeds)[:, :2],
                    colour=SEED_COLOUR,
                    kind="points",
                )
            )
        if session.arch_curve is not None:
            overlays.append(
                Overlay(
                    points=session.arch_curve.dense_points[:, :2],
                    colour=CURVE_COLOUR,
                    kind="line",
                    width=2.0,
                )
            )
        overlays.extend(self._measure_overlays("axial"))
        axial.overlays = overlays
        axial.update()
        for name in ("coronal", "sagittal"):
            view = self.slice_views[name]
            view.overlays = self._measure_overlays(name)
            view.update()

    # -- reformats ---------------------------------------------------------

    def refresh_reformats(self) -> None:
        session = self.session
        if session.panoramic is None:
            self.panoramic_view.clear()
            self.cross_view.clear()
            return
        self.panoramic_view.set_reformat(session.panoramic, keep_view=True)
        if session.cross_section is not None:
            self.cross_view.set_reformat(session.cross_section, keep_view=True)

        length = session.frames.length_mm
        self.s_slider.blockSignals(True)
        self.s_spin.blockSignals(True)
        self.s_slider.setRange(0, int(length * 10))
        self.s_slider.setValue(int(session.cross_section_s * 10))
        self.s_spin.setRange(0.0, length)
        self.s_spin.setValue(session.cross_section_s)
        self.s_slider.blockSignals(False)
        self.s_spin.blockSignals(False)

        overlays = self._measure_overlays("panoramic")
        top = session.panoramic.y0 + session.panoramic.height_mm
        overlays.append(
            Overlay(
                points=np.array(
                    [[session.cross_section_s, session.panoramic.y0],
                     [session.cross_section_s, top]]
                ),
                colour=CURSOR_COLOUR,
                kind="line",
                width=1.5,
            )
        )
        report = session.report
        if report is not None and np.isfinite(report.arc_length_mm):
            for s in (report.entry_s_mm, report.exit_s_mm):
                overlays.append(
                    Overlay(
                        points=np.array([[s, session.panoramic.y0], [s, top]]),
                        colour=CUT_COLOUR,
                        kind="line",
                        width=2.0,
                    )
                )
        self.panoramic_view.overlays = overlays
        self.panoramic_view.update()
        self.cross_view.overlays = self._measure_overlays("cross")
        self.cross_view.update()

    def _set_cross_section(self, s_mm: float) -> None:
        if self.session.frames is None:
            return
        self.session.update_cross_section(s_mm)
        self.refresh_reformats()

    def keyPressEvent(self, event):  # noqa: N802
        if self.tabs.currentIndex() == 2 and self.session.frames is not None:
            step = 5.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            if event.key() == Qt.Key.Key_Left:
                self._set_cross_section(self.session.cross_section_s - step)
                return
            if event.key() == Qt.Key.Key_Right:
                self._set_cross_section(self.session.cross_section_s + step)
                return
        super().keyPressEvent(event)

    # -- picking and measurement -------------------------------------------

    def _on_slice_pick(self, name: str, x_mm: float, y_mm: float) -> None:
        volume = self.session.volume
        if volume is None:
            return
        if self.mode == Mode.ARCH and name == "axial":
            self.session.add_arch_seed(
                [x_mm, y_mm, self._slice_world_coord("axial")]
            )
            return
        if self.mode in (Mode.MEASURE, Mode.ANGLE):
            self._record_measure_point(name, x_mm, y_mm)

    def _on_reformat_pick(self, name: str, x_mm: float, y_mm: float) -> None:
        session = self.session
        if self.mode in (Mode.MEASURE, Mode.ANGLE):
            self._record_measure_point(name, x_mm, y_mm)
        elif self.mode == Mode.LANDMARK and name == "cross" and session.frames is not None:
            session.add_landmark(
                cross_section_world_point(
                    session.frames, session.cross_section_s, x_mm, y_mm
                )
            )
        elif name == "panoramic":
            self._set_cross_section(x_mm)

    def _record_measure_point(self, name: str, x_mm: float, y_mm: float) -> None:
        wanted = 3 if self.mode == Mode.ANGLE else 2
        points = self._measure_points.setdefault(name, [])
        if len(points) >= wanted:
            points.clear()
        points.append((x_mm, y_mm))
        if len(points) == wanted:
            if wanted == 3:
                self._report_2d_angle(name, points)
            else:
                self._report_2d_measurement(name, points)
        self.refresh_slices()
        self.refresh_reformats()

    def _report_2d_measurement(self, name: str, points) -> None:
        (x0, y0), (x1, y1) = points
        dx, dy = x1 - x0, y1 - y0
        planar = float(np.hypot(dx, dy))
        if name == "panoramic":
            text = (
                f"arc length along arch curve {format_mm(abs(dx))} · "
                f"superior-inferior {format_mm(abs(dy))} · "
                f"flattened distance {format_mm(planar)} "
                "(x-axis is arc length, not a straight line)"
            )
        else:
            text = f"straight-line distance in this plane: {format_mm(planar)}"
        self._last_measurement = text
        self.measure_label.setText(text)

    def _report_2d_angle(self, name: str, points) -> None:
        a, b, c = (np.array([x, y, 0.0]) for x, y in points)
        text = f"angle at the middle point: {angle_deg(a, b, c):.1f}°"
        if name == "panoramic":
            text += " (in the flattened view, whose x-axis is arc length)"
        self._last_measurement = text
        self.measure_label.setText(text)

    def _measure_overlays(self, name: str) -> list[Overlay]:
        points = self._measure_points.get(name, [])
        if not points:
            return []
        array = np.array(points, dtype=float)
        overlays = [Overlay(points=array, colour=MEASURE_COLOUR, kind="points", radius=4)]
        if len(points) >= 2:
            overlays.append(
                Overlay(points=array, colour=MEASURE_COLOUR, kind="line", width=2.0)
            )
        return overlays

    def _on_surface_pick(self, point) -> None:
        if self.mode == Mode.PLATE:
            self.session.add_plate_point(point)
        elif self.mode == Mode.LANDMARK:
            self.session.add_landmark(point)
        elif self.mode in (Mode.MEASURE, Mode.ANGLE):
            picks = self.view3d.measure_points
            text = ""
            if self.mode == Mode.MEASURE and len(picks) == 2:
                text = (
                    "straight-line distance in 3-D: "
                    f"{format_mm(distance_mm(picks[0], picks[1]))}"
                )
            elif self.mode == Mode.ANGLE and len(picks) == 3:
                text = f"angle at the middle point: {angle_deg(*picks):.1f}°"
            if text:
                self._last_measurement = text
                self.measure_label.setText(text)

    # -- resection ---------------------------------------------------------

    def add_cut_plane(self) -> None:
        session = self.session
        if session.surface is None:
            self.show_message("Load a volume first.")
            return
        bounds = session.surface.GetBounds()
        centre = np.array(
            [
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            ]
        )
        normal = np.array([1.0, 0.0, 0.0])
        origin = centre
        if session.frames is not None:
            length = session.frames.length_mm
            fraction = 0.35 if not session.planes else 0.65
            index = session.frames.index_of(length * fraction)
            origin = session.frames.points[index].copy()
            normal = session.frames.tangents[index].copy()
            if session.planes:
                normal = -normal
        session.add_plane(origin, normal)
        self.view3d.render()

    def _on_plane_moved(self, index: int, origin, normal) -> None:
        self.session.set_plane(index, origin, normal)

    # -- export ------------------------------------------------------------

    def _save_path(self, title: str, filters: str, suggested: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(self, title, suggested, filters)
        return path or None

    def export_bend_csv(self) -> None:
        plan = self.session.plate_plan
        if plan is None:
            self.show_message("Draw a plate path first.")
            return
        path = self._save_path(
            "Export bend table", "CSV files (*.csv)", "plate_bends.csv"
        )
        if path:
            write_bend_csv(
                path, plan, self.session.plate.width_mm, self.session.plate.thickness_mm
            )
            self.show_message(f"Bend table written to {path}")

    def export_template_stl(self) -> None:
        plan = self.session.plate_plan
        if plan is None:
            self.show_message("Draw a plate path first.")
            return
        path = self._save_path(
            "Export bending template", "STL files (*.stl)", "bending_template.stl"
        )
        if path:
            write_template_stl(
                path, plan, self.session.plate.width_mm, self.session.plate.thickness_mm
            )
            self.show_message(f"Bending template written to {path}")

    def export_steps_csv(self) -> None:
        if not self.session.steps:
            self.show_message("Draw a plate path first.")
            return
        path = self._save_path(
            "Export bench steps", "CSV files (*.csv)", "plate_bending_steps.csv"
        )
        if path:
            write_steps_csv(
                path,
                self.session.steps,
                self.session.plate_system,
                self.session.bending_kit,
                self.session.fit,
            )
            self.show_message(f"Bench steps written to {path}")

    def export_graft_stl(self) -> None:
        surface = self.session.graft_surface
        if surface is None:
            self.show_message("Mirror the healthy side first.")
            return
        path = self._save_path(
            "Export reconstruction target", "STL files (*.stl)", "reconstruction.stl"
        )
        if path:
            write_surface_stl(path, surface)
            self.show_message(f"Reconstruction target written to {path}")

    def export_resection_csv(self) -> None:
        rows = self.resection_panel.summary_rows()
        if not rows:
            self.show_message("Place a cutting plane first.")
            return
        if self._last_measurement:
            rows.append(("last measurement", self._last_measurement))
        path = self._save_path(
            "Export resection summary", "CSV files (*.csv)", "resection_summary.csv"
        )
        if path:
            write_plan_summary_csv(path, rows, "MandiPlan resection summary")
            self.show_message(f"Resection summary written to {path}")

    def export_fragment_stl(self) -> None:
        if self.session.fragment_surface is None:
            self.show_message("Execute the cut first.")
            return
        path = self._save_path(
            "Export resected fragment", "STL files (*.stl)", "resected_fragment.stl"
        )
        if path:
            write_surface_stl(path, self.session.fragment_surface)
            self.show_message(f"Fragment written to {path}")
