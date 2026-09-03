"""Application state: the volume, the arch curve, the cuts and the plate.

The widgets read from here and call into here; nothing in this module knows
what a widget looks like.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from .. import constants
from ..bending_steps import BendStep, generate_steps
from ..dicom_io import SeriesGeometry, SeriesInfo, load_folder
from ..geometry import cpr
from ..geometry.mirror import (
    MidSagittalPlane,
    MirrorCoverage,
    bridge_mesh,
    estimate_midsagittal_plane,
    estimate_profile,
    mirror_coverage,
)
from ..geometry.plate import PlatePlan, compute_plate_plan
from ..geometry.resection import CutPlane, ResectionReport, build_report, plane_from_frame
from ..geometry.spline import ArchCurve
from ..geometry.threshold import estimate_bone_threshold
from ..geometry.volume import Volume
from ..plate_catalog import FitReport, fit_check, kit_by_id, load_kits, load_systems, system_by_id
from ..render.convert import triangles_to_polydata
from ..render.surface import (
    SurfaceExtractor,
    SurfaceProjector,
    clip_closed,
    mesh_volume_mm3,
    reflect_polydata,
)


@dataclass
class CprSettings:
    step_mm: float = constants.CPR_STEP_MM
    slab_mm: float = constants.CPR_SLAB_MM
    mode: str = constants.CPR_MODE
    cross_width_mm: float = constants.CROSS_SECTION_WIDTH_MM


@dataclass
class PlateSettings:
    pitch_mm: float = constants.PLATE_PITCH_MM
    width_mm: float = constants.PLATE_WIDTH_MM
    thickness_mm: float = constants.PLATE_THICKNESS_MM


@dataclass
class _Snapshot:
    arch_seeds: list
    planes: list
    landmarks: list
    plate_points: list
    plate_normals: list
    cut_applied: bool


@dataclass
class VolumeInfo:
    series: SeriesInfo
    geometry: SeriesGeometry
    folder: str

    def lines(self, volume: Volume) -> list[str]:
        nx, ny, nz = (int(v) for v in volume.size_xyz)
        sx, sy, sz = volume.spacing
        return [
            f"Series: {self.series.description or '(no description)'}",
            f"Slices: {self.series.n_files}",
            f"Matrix: {nx} × {ny} × {nz} voxels",
            f"Voxel size: {sx:.3f} × {sy:.3f} × {sz:.3f} mm",
            f"Field of view: {volume.extent_mm[0]:.1f} × {volume.extent_mm[1]:.1f}"
            f" × {volume.extent_mm[2]:.1f} mm",
        ]


class Session(QObject):
    volume_changed = pyqtSignal()
    surface_changed = pyqtSignal()
    arch_changed = pyqtSignal()
    reformat_changed = pyqtSignal()
    resection_changed = pyqtSignal()
    reconstruction_changed = pyqtSignal()
    plate_changed = pyqtSignal()
    message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume: Volume | None = None
        self.info: VolumeInfo | None = None
        self.threshold: float = 0.0
        self.surface = None
        self._extractor: SurfaceExtractor | None = None
        self.projector: SurfaceProjector | None = None
        self.keep_largest_component = True

        self.arch_seeds: list[np.ndarray] = []
        self.arch_curve: ArchCurve | None = None
        self.frames: cpr.ArchFrames | None = None
        self.cpr = CprSettings()
        self.panoramic: cpr.Reformat | None = None
        self.cross_section: cpr.Reformat | None = None
        self.cross_section_s: float = 0.0

        self.planes: list[CutPlane] = []
        self.landmarks: list[tuple[str, np.ndarray]] = []
        self.cut_applied = False
        self.report: ResectionReport | None = None
        self.fragment_surface = None
        self.retained_surface = None

        self.symmetry_plane: MidSagittalPlane | None = None
        self.coverage: MirrorCoverage | None = None
        self.graft_surface = None
        self.bridge_surface = None

        self.plate_points: list[np.ndarray] = []
        self.plate_normals: list[np.ndarray] = []
        self.plate = PlateSettings()
        self.plate_plan: PlatePlan | None = None
        self.plate_system = load_systems()[0]
        self.bending_kit = load_kits()[0]
        self.fit: FitReport | None = None
        self.steps: list[BendStep] = []

        self._undo: list[_Snapshot] = []

    # -- volume -----------------------------------------------------------

    def load_dicom_folder(self, folder: str, series_uid: str | None = None) -> None:
        volume, geometry, series = load_folder(folder, series_uid)
        self.volume = volume
        self._extractor = SurfaceExtractor(volume)
        self.info = VolumeInfo(series=series, geometry=geometry, folder=folder)
        self.threshold = estimate_bone_threshold(volume)
        self.arch_seeds.clear()
        self.arch_curve = None
        self.frames = None
        self.panoramic = None
        self.cross_section = None
        self.planes.clear()
        self.landmarks.clear()
        self.plate_points.clear()
        self.plate_normals.clear()
        self.plate_plan = None
        self.cut_applied = False
        self.report = None
        self.symmetry_plane = None
        self.coverage = None
        self.graft_surface = None
        self.bridge_surface = None
        self._undo.clear()
        self.volume_changed.emit()
        for warning in geometry.warnings:
            self.message.emit(warning)
        self.rebuild_surface()

    def set_threshold(self, value: float) -> None:
        if self.volume is None:
            return
        self.threshold = float(value)
        self.rebuild_surface()

    def rebuild_surface(self) -> None:
        if self.volume is None:
            return
        self.surface = self._extractor.update(
            self.threshold, largest_component=self.keep_largest_component
        )
        self.projector = (
            SurfaceProjector(self.surface) if self.surface.GetNumberOfPoints() else None
        )
        self.cut_applied = False
        self.fragment_surface = None
        self.retained_surface = None
        self.graft_surface = None
        self.bridge_surface = None
        self.surface_changed.emit()
        self.update_plate_plan()

    @property
    def surface_volume_mm3(self) -> float:
        return mesh_volume_mm3(self.surface) if self.surface is not None else float("nan")

    # -- arch curve --------------------------------------------------------

    def add_arch_seed(self, point) -> None:
        self._push_undo()
        self.arch_seeds.append(np.asarray(point, dtype=float).reshape(3))
        self._rebuild_arch()

    def remove_last_arch_seed(self) -> None:
        if not self.arch_seeds:
            return
        self._push_undo()
        self.arch_seeds.pop()
        self._rebuild_arch()

    def clear_arch(self) -> None:
        self._push_undo()
        self.arch_seeds.clear()
        self._rebuild_arch()

    def _rebuild_arch(self) -> None:
        if len(self.arch_seeds) >= 2:
            self.arch_curve = ArchCurve(np.vstack(self.arch_seeds))
            self.frames = cpr.build_frames(self.arch_curve, self.cpr.step_mm)
        else:
            self.arch_curve = None
            self.frames = None
            self.panoramic = None
            self.cross_section = None
        self.arch_changed.emit()
        self.update_reformats()

    def set_cpr_settings(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self.cpr, key, value)
        if self.arch_curve is not None:
            self.frames = cpr.build_frames(self.arch_curve, self.cpr.step_mm)
        self.update_reformats()

    def update_reformats(self) -> None:
        if self.volume is None or self.frames is None:
            self.panoramic = None
            self.cross_section = None
            self.reformat_changed.emit()
            return
        self.panoramic = cpr.build_panoramic(
            self.volume,
            self.frames,
            slab_mm=self.cpr.slab_mm,
            mode=self.cpr.mode,
        )
        self.cross_section_s = float(
            np.clip(self.cross_section_s, 0.0, self.frames.length_mm)
        )
        self.update_cross_section(self.cross_section_s)
        self.reformat_changed.emit()
        self.update_resection_report()

    def update_cross_section(self, s_mm: float) -> None:
        if self.volume is None or self.frames is None:
            return
        self.cross_section_s = float(np.clip(s_mm, 0.0, self.frames.length_mm))
        self.cross_section = cpr.build_cross_section(
            self.volume,
            self.frames,
            self.cross_section_s,
            width_mm=self.cpr.cross_width_mm,
        )

    # -- resection ---------------------------------------------------------

    def add_plane(self, origin, normal, label: str | None = None) -> None:
        if len(self.planes) >= constants.MAX_RESECTION_PLANES:
            self.message.emit(
                f"MandiPlan plans up to {constants.MAX_RESECTION_PLANES} cutting planes."
            )
            return
        self._push_undo()
        index = len(self.planes) + 1
        self.planes.append(
            CutPlane(origin=origin, normal=normal, label=label or f"Cut {index}")
        )
        self.cut_applied = False
        self.update_resection_report()

    def set_plane(self, index: int, origin, normal) -> None:
        plane = self.planes[index]
        self.planes[index] = CutPlane(origin=origin, normal=normal, label=plane.label)
        self.update_resection_report()

    def translate_plane(
        self, index: int, s_mm: float, offset_mm=(0.0, 0.0, 0.0)
    ) -> None:
        """Move a cut plane along the jaw. Its angulation is left alone.

        Sliding a cut to a new position is not a request to re-angle it. The
        normal is deliberately not recomputed from the arch tangent here, so a
        cut you have angled by hand keeps that angle as you move it.
        """
        if self.frames is None or index >= len(self.planes):
            return
        origin, _ = plane_from_frame(self.frames, s_mm, 0.0, 0.0, offset_mm)
        plane = self.planes[index]
        self._replace_plane(index, origin, plane.normal)

    def set_plane_origin(self, index: int, origin) -> None:
        """Move a cut plane to a world point, keeping its angulation."""
        if index >= len(self.planes):
            return
        self._replace_plane(index, np.asarray(origin, dtype=float), self.planes[index].normal)

    def rotate_plane(self, index: int, yaw_deg: float, tilt_deg: float) -> None:
        """Re-angle a cut plane about its own position. Its origin is left alone."""
        if self.frames is None or index >= len(self.planes):
            return
        plane = self.planes[index]
        s_mm = self.plane_arc_position(index)
        _, normal = plane_from_frame(self.frames, s_mm, yaw_deg, tilt_deg)
        # Keep the side the plane removes: the frame normal follows the curve,
        # so flip it when this plane was facing the other way.
        if np.dot(normal, plane.normal) < 0:
            normal = -normal
        self._replace_plane(index, plane.origin, normal)

    def _replace_plane(self, index: int, origin, normal) -> None:
        label = self.planes[index].label
        self.planes[index] = CutPlane(origin=origin, normal=normal, label=label)
        self.cut_applied = False
        self.fragment_surface = None
        self.retained_surface = None
        self.update_resection_report()

    def plane_arc_position(self, index: int) -> float:
        """Where a cut plane sits along the arch curve, in mm."""
        if self.frames is None or index >= len(self.planes):
            return float("nan")
        d = np.linalg.norm(self.frames.points - self.planes[index].origin, axis=1)
        return float(self.frames.s[int(np.argmin(d))])

    def flip_plane(self, index: int) -> None:
        self._push_undo()
        plane = self.planes[index]
        self.planes[index] = CutPlane(
            origin=plane.origin, normal=-plane.normal, label=plane.label
        )
        self.update_resection_report()

    def clear_planes(self) -> None:
        self._push_undo()
        self.planes.clear()
        self.cut_applied = False
        self.fragment_surface = None
        self.retained_surface = None
        self.update_resection_report()

    def add_landmark(self, point, name: str | None = None) -> None:
        self._push_undo()
        name = name or f"Margin point {len(self.landmarks) + 1}"
        self.landmarks.append((name, np.asarray(point, dtype=float).reshape(3)))
        self.update_resection_report()

    def clear_landmarks(self) -> None:
        self._push_undo()
        self.landmarks.clear()
        self.update_resection_report()

    def update_resection_report(self) -> None:
        if self.planes:
            self.report = build_report(self.frames, self.planes, self.landmarks)
        else:
            self.report = ResectionReport()
        if self.cut_applied and self.fragment_surface is not None:
            self.report.fragment_volume_mm3 = mesh_volume_mm3(self.fragment_surface)
        self.resection_changed.emit()
        self.update_reconstruction()
        # Moving a cut changes how many screw holes land on retained bone, so
        # the plate verdict has to be recomputed and re-shown with it.
        self._update_fit()
        self.plate_changed.emit()

    def execute_cut(self) -> None:
        if self.surface is None or not self.planes:
            return
        self._push_undo()
        self.fragment_surface = clip_closed(self.surface, self.planes, keep_resected=True)
        self.retained_surface = clip_closed(self.surface, self.planes, keep_resected=False)
        self.cut_applied = True
        self.update_resection_report()

    def undo_cut(self) -> None:
        self.cut_applied = False
        self.fragment_surface = None
        self.retained_surface = None
        self.update_resection_report()

    # -- reconstruction ----------------------------------------------------

    def estimate_symmetry_plane(self) -> None:
        """Find the patient's mid-sagittal plane from the bone itself."""
        if self.volume is None:
            return
        self.symmetry_plane = estimate_midsagittal_plane(self.volume, self.threshold)
        if self.symmetry_plane.symmetry < 0.75:
            self.message.emit(
                f"The best symmetry plane only maps {self.symmetry_plane.symmetry:.0%} "
                "of the bone onto bone. Check it before mirroring."
            )
        self.update_reconstruction()

    def update_reconstruction(self) -> None:
        """Mirror the healthy side into the defect and measure what it misses."""
        self.coverage = None
        if (
            self.symmetry_plane is None
            or self.frames is None
            or self.report is None
            or not np.isfinite(self.report.arc_length_mm)
        ):
            self.reconstruction_changed.emit()
            return
        self.coverage = mirror_coverage(
            self.frames,
            self.symmetry_plane,
            self.report.entry_s_mm,
            self.report.exit_s_mm,
        )
        self.reconstruction_changed.emit()

    def build_graft(self) -> None:
        """Build the mirrored graft, and estimate any span it cannot reach."""
        if self.surface is None or self.symmetry_plane is None or not self.planes:
            self.message.emit(
                "Place the cutting planes and estimate the mid-sagittal plane first."
            )
            return
        retained = self.retained_surface
        if retained is None:
            retained = clip_closed(self.surface, self.planes, keep_resected=False)
        mirrored = reflect_polydata(retained, self.symmetry_plane)
        self.graft_surface = clip_closed(mirrored, self.planes, keep_resected=True)

        self.bridge_surface = None
        if self.coverage is not None and self.coverage.uncovered_spans:
            self.bridge_surface = self._build_bridge(self.coverage.uncovered_spans[0])
        self.reconstruction_changed.emit()

    def _build_bridge(self, span: tuple[float, float]):
        """Estimate the bone across a span that mirroring cannot cover."""
        if self.volume is None or self.frames is None:
            return None
        margin = 1.0
        ends = (
            max(span[0] - margin, 0.0),
            min(span[1] + margin, self.frames.length_mm),
        )
        profiles = []
        for s in ends:
            cross = cpr.build_cross_section(
                self.volume, self.frames, s, width_mm=self.cpr.cross_width_mm
            )
            profile = estimate_profile(cross, self.threshold, s)
            if profile is None:
                self.message.emit(
                    f"No bone found at {s:.1f} mm along the curve, so the missing "
                    "segment cannot be estimated there."
                )
                return None
            profiles.append(profile)
        points, triangles = bridge_mesh(self.frames, profiles[0], profiles[1])
        return triangles_to_polydata(points, triangles)

    @property
    def graft_volume_mm3(self) -> float:
        return (
            mesh_volume_mm3(self.graft_surface)
            if self.graft_surface is not None
            else float("nan")
        )

    # -- plate -------------------------------------------------------------

    def set_plate_system(self, system_id: str) -> None:
        self.plate_system = system_by_id(system_id)
        self.plate.pitch_mm = self.plate_system.hole_pitch_mm
        self.plate.width_mm = self.plate_system.width_mm
        self.plate.thickness_mm = self.plate_system.thickness_mm
        self.update_plate_plan()

    def set_bending_kit(self, kit_id: str) -> None:
        self.bending_kit = kit_by_id(kit_id)
        self.update_plate_plan()

    def _update_fit(self) -> None:
        self.fit = None
        self.steps = []
        if self.plate_plan is None:
            return
        span = None
        if self.report is not None and np.isfinite(self.report.arc_length_mm):
            span = (self.report.entry_s_mm, self.report.exit_s_mm)
        self.fit = fit_check(
            self.plate_plan, self.plate_system, defect_span=span, frames=self.frames
        )
        self.steps = generate_steps(
            self.plate_plan, self.plate_system, self.bending_kit, self.fit
        )

    def add_plate_point(self, world_point) -> None:
        if self.projector is None:
            self.message.emit("Extract a bone surface before drawing a plate path.")
            return
        self._push_undo()
        point, normal = self.projector.project(world_point)
        self.plate_points.append(point)
        self.plate_normals.append(normal)
        self.update_plate_plan()

    def remove_last_plate_point(self) -> None:
        if not self.plate_points:
            return
        self._push_undo()
        self.plate_points.pop()
        self.plate_normals.pop()
        self.update_plate_plan()

    def clear_plate_path(self) -> None:
        self._push_undo()
        self.plate_points.clear()
        self.plate_normals.clear()
        self.update_plate_plan()

    def set_plate_settings(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self.plate, key, value)
        self.update_plate_plan()

    def update_plate_plan(self) -> None:
        self.plate_plan = None
        if len(self.plate_points) >= 2:
            path = np.vstack(self.plate_points)
            normals = np.vstack(self.plate_normals)
            total = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if total >= self.plate.pitch_mm:
                self.plate_plan = compute_plate_plan(path, normals, self.plate.pitch_mm)
            else:
                self.message.emit(
                    f"Plate path is {total:.1f} mm, shorter than one "
                    f"{self.plate.pitch_mm:.1f} mm screw-hole pitch."
                )
        self._update_fit()
        self.plate_changed.emit()

    # -- undo --------------------------------------------------------------

    def _push_undo(self) -> None:
        self._undo.append(
            _Snapshot(
                arch_seeds=copy.deepcopy(self.arch_seeds),
                planes=copy.deepcopy(self.planes),
                landmarks=copy.deepcopy(self.landmarks),
                plate_points=copy.deepcopy(self.plate_points),
                plate_normals=copy.deepcopy(self.plate_normals),
                cut_applied=self.cut_applied,
            )
        )
        del self._undo[:-40]

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    def undo(self) -> None:
        if not self._undo:
            self.message.emit("Nothing to undo.")
            return
        state = self._undo.pop()
        self.arch_seeds = state.arch_seeds
        self.planes = state.planes
        self.landmarks = state.landmarks
        self.plate_points = state.plate_points
        self.plate_normals = state.plate_normals
        if not state.cut_applied:
            self.fragment_surface = None
            self.retained_surface = None
        self.cut_applied = state.cut_applied
        self._rebuild_arch()
        self.update_resection_report()
        self.update_plate_plan()
