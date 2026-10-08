"""Application state: the volume, the arch curve, the cuts and the plate.

The widgets read from here and call into here; nothing in this module knows
what a widget looks like.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from .. import constants
from ..bending_steps import BendStep, generate_steps
from ..dicom_io import SeriesGeometry, SeriesInfo, load_folder
from ..reference_cases import SAMPLE_SCAN, load_sample
from ..geometry import cpr
from ..geometry.mirror import (
    SYMMETRY_TOLERANCE_MM,
    SYMMETRY_WARNING,
    MidSagittalPlane,
    MirrorCoverage,
    estimate_midsagittal_plane,
    mirror_coverage,
)
from ..geometry.mandible import (
    MandibleIsolation,
    find_arch,
    frames_for,
    isolate_mandible,
    mandible_centreline,
    masked_bone_volume,
)
from ..geometry.sculpt import SurfaceSculptor, vertex_normals
from ..geometry.mesh_io import MeshLoadError
from ..geometry.plate import PlatePlan, compute_plate_plan, fair_plate_path
from ..geometry.hole_distortion import ovalise_mesh, predict_distortion
from ..geometry.plate_bend import (
    BentPlate,
    ClearanceReport,
    bend_to_path,
    clearance_report,
)
from ..geometry.plate_fit import (
    FittedPlate,
    extend_targets,
    plate_targets,
    rigid_fit,
)
from ..materials import Alloy, alloy_by_id
from ..plate_assets import PlateAsset, asset_by_id, assets_in_family, load_asset_mesh, load_assets
from ..geometry.resection import (
    CutPlane,
    PlanePlacement,
    ResectionReport,
    bound_resection,
    build_report,
    plane_from_placement,
)
from ..geometry.spline import ArchCurve
from ..geometry.threshold import estimate_bone_threshold
from ..geometry.volume import Volume
from ..plate_catalog import (
    FitReport,
    fit_check,
    kit_by_id,
    load_kits,
    load_systems,
    system_by_id,
    system_for_asset,
)
from ..render.reconstruct import Reconstruction, reconstruct
from ..render.surface import (
    SurfaceExtractor,
    SurfaceProjector,
    clip_closed,
    mesh_volume_mm3,
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
    arch_source: str
    planes: list
    placements: list
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
            f"Slices: {self.series.n_files or int(volume.size_xyz[2])}",
            f"Matrix: {nx} × {ny} × {nz} voxels",
            f"Voxel size: {sx:.3f} × {sy:.3f} × {sz:.3f} mm",
            # A CT is cut down to a box round the jaws (see dicom_io).
            ("Jaw box: " if getattr(self.geometry, "jaw_box", False) else "Field of view: ")
            + f"{volume.extent_mm[0]:.1f} × {volume.extent_mm[1]:.1f}"
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
    mandible_changed = pyqtSignal()
    #: The reconstructed surface was edited by hand; only its shape changed.
    reconstruction_edited = pyqtSignal()
    message = pyqtSignal(str)
    #: True while a slow computation runs, so the window can show it is busy.
    busy = pyqtSignal(bool)
    #: The arch curve was laid automatically; the height of the mandibular
    #: body, mm, so the axial slice can be shown there.
    arch_laid = pyqtSignal(float)

    #: Radius of bone a plate rests on around each path point, mm: half a
    #: reconstruction plate's width and a little more.
    PLATE_FOOTPRINT_MM = 5.0

    #: Pieces of bone smaller than this fraction of the largest are noise.
    SPECK_FRACTION = 0.05
    #: Wait this long after the last change to the arch or threshold before
    #: separating the mandible, so a run of clicks or a slider drag triggers
    #: it once.
    MANDIBLE_DELAY_MS = 900

    def __init__(self, parent=None):
        super().__init__(parent)
        self.volume: Volume | None = None
        self.info: VolumeInfo | None = None
        self.threshold: float = 0.0
        self.surface = None
        self._extractor: SurfaceExtractor | None = None
        self.projector: SurfaceProjector | None = None
        #: Separate the mandible from the skull once the arch curve exists.
        self.separate_mandible = True
        self.mandible: MandibleIsolation | None = None
        #: "auto" when the mandible was found in the scan by itself, "arch"
        #: when it was separated along the arch curve as drawn.
        self.mandible_source = ""
        #: Why the mandible could not be separated, when it could not.
        self.mandible_problem = ""
        self._auto_arch: np.ndarray | None = None
        self._auto_arch_scan = None
        self._separation_failed: set = set()
        #: The scan with the non-mandible bone removed; ``None`` when nothing
        #: had to be removed. See :attr:`bone_volume`.
        self._bone_volume: Volume | None = None
        self._mandible_timer = QTimer(self)
        self._mandible_timer.setSingleShot(True)
        self._mandible_timer.timeout.connect(self.ensure_mandible)

        self.arch_seeds: list[np.ndarray] = []
        #: "auto" when the arch curve was laid along the separated mandible,
        #: "drawn" when the operator clicked it on the axial slice.
        self.arch_source = ""
        self.arch_curve: ArchCurve | None = None
        self.frames: cpr.ArchFrames | None = None
        self.cpr = CprSettings()
        self.panoramic: cpr.Reformat | None = None
        self.cross_section: cpr.Reformat | None = None
        self.cross_section_s: float = 0.0

        self.planes: list[CutPlane] = []
        #: Authoritative cut state; ``planes`` is derived from this.
        self.placements: list[PlanePlacement] = []
        self.landmarks: list[tuple[str, np.ndarray]] = []
        self.cut_applied = False
        self.report: ResectionReport | None = None
        self.fragment_surface = None
        self.retained_surface = None

        self.symmetry_plane: MidSagittalPlane | None = None
        self.coverage: MirrorCoverage | None = None
        #: The reconstructed mandible: one flush surface, retained bone and
        #: mirrored donor together. ``graft_surface`` is the donor part alone.
        self.reconstruction: Reconstruction | None = None
        self._reconstruction_key = None
        self._graft_surface = None
        self._graft_stale = False
        #: Hand edits to the reconstruction (``geometry/sculpt.py``).
        self._sculptor: SurfaceSculptor | None = None
        self.brush = "smooth"
        self.brush_radius_mm = 5.0
        self.brush_strength = 0.5
        self._plate_projector: SurfaceProjector | None = None
        self._plate_projector_source = None

        self.plate_points: list[np.ndarray] = []
        self.plate_normals: list[np.ndarray] = []
        self.plate = PlateSettings()
        self.plate_plan: PlatePlan | None = None
        self.plate_system = load_systems()[0]
        self.bending_kit = load_kits()[0]
        self.fit: FitReport | None = None
        self.steps: list[BendStep] = []
        #: The selected plate asset and its placement on the planned path.
        #: When an asset is selected the viewport and every export show its
        #: real mesh; there is no proxy geometry left in either path.
        self.plate_asset: PlateAsset | None = None
        #: The operator picked the plate's length. Until then the shortest
        #: plate of the family that spans the drawn path is used.
        self.plate_length_chosen = False
        self.fitted_plate: FittedPlate | None = None
        self.bent_plate: BentPlate | None = None
        self.plate_contact: ClearanceReport | None = None
        self.plate_clearance_mm: float = 0.5
        self.plate_bending_enabled: bool = True
        #: Show the screw holes as bending would actually leave them, rather
        #: than perfectly round. Real holes go oval; see hole_distortion.
        self.show_hole_distortion: bool = True
        self.use_bending_insets: bool | None = None
        self.hole_distortion = None
        #: The etched mark, moved with the plate: ``(points, triangles)``.
        self.plate_marking = None
        self._marking_cache: dict[str, tuple] = {}
        self.plate_fit_warnings: list[str] = []
        self.plate_fit_problems: list[str] = []
        if load_assets():
            self.plate_asset = load_assets()[0]
            self._adopt_asset(self.plate_asset)

        self._undo: list[_Snapshot] = []
        #: What has been written out this session, in order.
        self.exported: list[str] = []

    # -- volume -----------------------------------------------------------

    def load_dicom_folder(self, folder: str, series_uid: str | None = None) -> None:
        volume, geometry, series = load_folder(folder, series_uid)
        self._load_volume(volume, geometry, series, folder)

    def load_sample_scan(self) -> None:
        """Open the bundled public head CBCT (see ``reference_cases``)."""
        volume, geometry, series, _ = load_sample()
        self._load_volume(volume, geometry, series, str(SAMPLE_SCAN))

    def _load_volume(self, volume, geometry, series, folder: str) -> None:
        self.volume = volume
        self._extractor = SurfaceExtractor(volume)
        self.info = VolumeInfo(series=series, geometry=geometry, folder=folder)
        self.threshold = estimate_bone_threshold(volume, thin_bone=geometry.jaw_box)
        self.arch_seeds.clear()
        self.arch_source = ""
        self.arch_curve = None
        self.frames = None
        self.panoramic = None
        self.cross_section = None
        self.planes.clear()
        self.placements.clear()
        self.landmarks.clear()
        self.plate_points.clear()
        self.plate_normals.clear()
        self.plate_plan = None
        self.fitted_plate = None
        self.bent_plate = None
        self.plate_contact = None
        self.cut_applied = False
        self.report = None
        self.symmetry_plane = None
        self.coverage = None
        self._clear_reconstruction()
        self._drop_mandible()
        self._separation_failed.clear()
        self._auto_arch_scan = None
        self._undo.clear()
        self.volume_changed.emit()
        for warning in geometry.warnings:
            self.message.emit(warning)
        self.rebuild_surface()

    def automatic_threshold(self) -> float:
        """The seeded bone threshold for the open scan (see ``threshold``)."""
        if self.volume is None:
            return self.threshold
        jaw_box = bool(self.info is not None and getattr(self.info.geometry, "jaw_box", False))
        return estimate_bone_threshold(self.volume, thin_bone=jaw_box)

    def set_threshold(self, value: float) -> None:
        if self.volume is None:
            return
        self.threshold = float(value)
        self.rebuild_surface()

    def rebuild_surface(self) -> None:
        """The bone surface at the current threshold: all bone, until the
        mandible has been separated again for this threshold."""
        if self.volume is None:
            return
        self._drop_mandible()
        self._set_surface(
            self._extractor.update(self.threshold, keep_fraction=self.SPECK_FRACTION)
        )
        self._schedule_mandible()

    def _set_surface(self, surface) -> None:
        self.surface = surface
        self.projector = (
            SurfaceProjector(self.surface) if self.surface.GetNumberOfPoints() else None
        )
        self.cut_applied = False
        self.fragment_surface = None
        self.retained_surface = None
        self._clear_reconstruction()
        self.surface_changed.emit()
        self.update_plate_plan()

    # -- mandible ----------------------------------------------------------

    @property
    def bone_volume(self) -> Volume | None:
        """The scan as the planning steps should see it: the mandible's bone
        only, once it has been separated from the skull."""
        return self._bone_volume if self._bone_volume is not None else self.volume

    def set_separate_mandible(self, enabled: bool) -> None:
        self.separate_mandible = bool(enabled)
        if self.separate_mandible:
            self.ensure_mandible()
        elif self.volume is not None:
            self.rebuild_surface()

    def _schedule_mandible(self) -> None:
        if self.separate_mandible and self.volume is not None and self.mandible is None:
            self._mandible_timer.start(self.MANDIBLE_DELAY_MS)

    def ensure_mandible(self) -> None:
        """Separate the mandible from the rest of the skull, if it is not yet.

        Runs by itself shortly after a scan is opened or the threshold moves,
        and before any step that needs the mandible alone (cuts, the mirror,
        the plate). The mandible is found in the scan automatically; if it
        cannot be, the arch curve the operator draws is used instead. See
        ``geometry/mandible.py`` for how.
        """
        self._mandible_timer.stop()
        if self.volume is None or not self.separate_mandible or self.mandible is not None:
            return
        scan = (id(self.volume), round(float(self.threshold), 6))
        if self._auto_arch_scan != scan:
            self._auto_arch_scan = scan
            self._auto_arch = find_arch(self.volume, self.threshold)
        if self._auto_arch is not None and (scan, "auto") not in self._separation_failed:
            if self._separate(frames_for(self._auto_arch), "auto"):
                return
        drawn = self.frames is not None and self.arch_source == "drawn"
        if drawn and (scan, self._arch_key()) not in self._separation_failed:
            self._separate(self.frames, "arch")
            return
        if not drawn and not self.mandible_problem:
            self.mandible_problem = (
                "The mandible could not be found in this scan automatically. Draw "
                "the arch curve (step 2) and it will be separated along it."
            )
            self.message.emit(self.mandible_problem)
            self.mandible_changed.emit()

    def separate_along_arch(self) -> None:
        """Separate the mandible again, following the arch curve as drawn."""
        if self.volume is None or self.frames is None or self.arch_source != "drawn":
            self.message.emit("Draw the arch curve first; the separation follows it.")
            return
        self.separate_mandible = True
        self._separate(self.frames, "arch")

    def _arch_key(self):
        return ("arch", tuple(tuple(np.round(seed, 4)) for seed in self.arch_seeds))

    def _separate(self, frames, source: str) -> bool:
        """Run the separation along ``frames``; False if it could not be done."""
        scan = (id(self.volume), round(float(self.threshold), 6))
        self.busy.emit(True)
        try:
            isolation = isolate_mandible(self.volume, self.threshold, frames)
        except ValueError as error:
            self._separation_failed.add((scan, "auto" if source == "auto" else self._arch_key()))
            self.mandible_problem = str(error)
            self.message.emit(str(error))
            self.mandible_changed.emit()
            return False
        finally:
            self.busy.emit(False)
        had_mask = self._bone_volume is not None
        self.mandible = isolation
        self.mandible_source = source
        self.mandible_problem = ""
        if isolation.separated:
            self._bone_volume = masked_bone_volume(self.volume, isolation)
            surface = SurfaceExtractor(self._bone_volume).update(self.threshold)
            self._set_surface(surface)
            self.message.emit(isolation.summary())
        else:
            self._bone_volume = None
            if had_mask:
                self._set_surface(
                    self._extractor.update(self.threshold, keep_fraction=self.SPECK_FRACTION)
                )
        self.mandible_changed.emit()
        if source == "auto" and not self.arch_seeds:
            self._lay_automatic_arch()
        return True

    def _lay_automatic_arch(self) -> bool:
        """Lay the arch curve along the separated mandible, condyle to condyle."""
        if self.mandible is None or self._auto_arch is None or self.mandible_source != "auto":
            return False
        try:
            points = mandible_centreline(self.volume, self.mandible, self._auto_arch)
        except ValueError:
            points = np.asarray(self._auto_arch, dtype=float)
        if len(points) < 2:
            return False
        self.arch_seeds = [np.asarray(p, dtype=float) for p in points]
        self.arch_source = "auto"
        self._rebuild_arch()
        self.arch_laid.emit(float(np.median(np.asarray(self._auto_arch)[:, 2])))
        return self.frames is not None

    def use_automatic_arch(self) -> None:
        """Replace the arch curve with the one found along the mandible."""
        if self.volume is None:
            return
        if self.mandible is None:
            self.ensure_mandible()
        if self.arch_source == "auto" and self.frames is not None:
            return
        if self.mandible is None or self.mandible_source != "auto":
            self.message.emit(
                "The mandible could not be found in this scan automatically; "
                "draw the arch curve on the axial slice instead."
            )
            return
        self._push_undo()
        self._lay_automatic_arch()

    def _drop_mandible(self) -> None:
        self._mandible_timer.stop()
        changed = self.mandible is not None or bool(self.mandible_problem)
        self.mandible = None
        self.mandible_problem = ""
        self._bone_volume = None
        if changed:
            self.mandible_changed.emit()

    @property
    def surface_volume_mm3(self) -> float:
        return mesh_volume_mm3(self.surface) if self.surface is not None else float("nan")

    # -- arch curve --------------------------------------------------------

    #: Two arch seeds closer than this are one click, not two. Nobody laying
    #: 5-10 points along a 100 mm arch means to put two of them a twentieth
    #: of a millimetre apart; a double-click does. A spline through
    #: coincident points is undefined, so the second is dropped.
    SEED_MERGE_TOLERANCE_MM = 0.05

    def add_arch_seed(self, point) -> None:
        point = np.asarray(point, dtype=float).reshape(3)
        if self.arch_source != "drawn":
            # The first click of a hand-drawn curve replaces the automatic one.
            self._push_undo()
            self.arch_seeds = [point]
            self.arch_source = "drawn"
            self._rebuild_arch()
            return
        for existing in self.arch_seeds:
            if np.linalg.norm(existing - point) < self.SEED_MERGE_TOLERANCE_MM:
                self.message.emit(
                    "That point is already on the arch curve — click a little "
                    "further along."
                )
                return
        self._push_undo()
        self.arch_seeds.append(point)
        self._rebuild_arch()

    def remove_last_arch_seed(self) -> None:
        if not self.arch_seeds or self.arch_source != "drawn":
            return
        self._push_undo()
        self.arch_seeds.pop()
        self._rebuild_arch()

    def clear_arch(self) -> None:
        self._push_undo()
        self.arch_seeds.clear()
        self.arch_source = ""
        self._rebuild_arch()

    def _rebuild_arch(self) -> None:
        new_curve = self.frames is None
        self.arch_curve = None
        self.frames = None
        if len(self.arch_seeds) >= 2:
            try:
                self.arch_curve = ArchCurve(np.vstack(self.arch_seeds))
                self.frames = cpr.build_frames(self.arch_curve, self.cpr.step_mm)
            except ValueError as error:
                # Degenerate seeds. Losing the curve is recoverable; raising
                # out of a mouse-click handler is not.
                self.arch_curve = None
                self.frames = None
                self.message.emit(f"The arch curve could not be built: {error}")
        if self.frames is None:
            self.panoramic = None
            self.cross_section = None
        elif new_curve:
            # Open the cross-section on the body, not at a condyle.
            self.cross_section_s = float(np.mean(cpr.body_span(self.frames)))
        self._resolve_all_planes()
        self.arch_changed.emit()
        self.update_reformats()
        self._schedule_mandible()

    def _resolve_all_planes(self) -> None:
        """Rebuild every cut from its placement against the current frames.

        Editing the arch curve moves the frames underneath the cuts. Each cut
        is re-anchored at the point of the new curve nearest where it was
        put, and keeps its angles to the jaw there, so it follows the
        corrected curve instead of being left behind on the old one.
        Re-anchoring by position rather than by distance along the curve is
        what keeps a cut on the same part of the jaw when the curve is
        replaced: the automatic curve starts at a condyle, a drawn one at the
        front of a ramus. The position is where the cut was put
        (``anchor_mm``), not wherever the last curve left it, so a curve
        clicked point by point does not drag the cuts along to its end.
        """
        if self.frames is None:
            return
        for index, placement in enumerate(self.placements):
            if index < len(self.planes):
                if placement.anchor_mm is not None:
                    anchor = np.asarray(placement.anchor_mm, dtype=float)
                else:
                    anchor = np.asarray(self.planes[index].origin, dtype=float) - np.asarray(
                        placement.offset_mm, dtype=float
                    )
                at = self.frames.index_of(self.frames.clamp(placement.s_mm))
                if np.linalg.norm(self.frames.points[at] - anchor) < 1e-6:
                    # The curve has not moved under this cut.
                    moved = placement.replace(s_mm=self.frames.clamp(placement.s_mm))
                else:
                    nearest = int(np.argmin(np.linalg.norm(self.frames.points - anchor, axis=1)))
                    moved = placement.replace(s_mm=float(self.frames.s[nearest]))
                self.placements[index] = moved
                self.planes[index] = self._resolve(moved, self.planes[index].label)

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

    # A cut plane's authoritative state is its :class:`PlanePlacement` — where
    # it sits on the arch and how it is angled relative to the local
    # mandibular frame there. ``self.planes`` is the world-space rendering of
    # that, rebuilt from the placement whenever either changes. Storing the
    # world normal instead is what made a cut slide along the jaw carrying a
    # fixed global angulation.

    def add_plane(self, origin, normal, label: str | None = None) -> None:
        """Add a cut, converting a picked world point into a placement."""
        if len(self.planes) >= constants.MAX_RESECTION_PLANES:
            self.message.emit(
                f"MandiPlan plans up to {constants.MAX_RESECTION_PLANES} cutting planes."
            )
            return
        # A cut must only ever remove mandible: separate it first. That also
        # lays the arch curve the cut is placed along, if there is none yet.
        self.ensure_mandible()
        if self.frames is None:
            # A cut is a position along the mandible plus angles measured
            # against the local jaw frame, so it has nothing to anchor to
            # until the arch curve exists.
            self.message.emit(
                "The mandible could not be found automatically, so there is no "
                "curve to place the cut on: draw the arch curve on the axial "
                "slice (step 2) first."
            )
            return
        self._push_undo()
        index = len(self.planes) + 1
        placement = self._anchored(self._placement_for(origin, normal))
        self.placements.append(placement)
        self.planes.append(self._resolve(placement, label or f"Cut {index}"))
        self.cut_applied = False
        self.update_resection_report()

    #: Where "Add cut" puts each new cut along the mandibular body, as a
    #: fraction of the body's length from its patient-right end. The first
    #: two bound a segment of the right body, the usual first plan to adjust.
    DEFAULT_CUT_FRACTIONS = (0.22, 0.42)

    def add_default_cut(self) -> bool:
        """Add a cut at a sensible place on the body; False if none could be."""
        if self.volume is None:
            self.message.emit("Open a scan first.")
            return False
        if len(self.planes) >= constants.MAX_RESECTION_PLANES:
            self.message.emit(
                f"MandiPlan plans up to {constants.MAX_RESECTION_PLANES} cutting "
                "planes. Move the existing ones, or remove one first."
            )
            return False
        self.ensure_mandible()
        if self.frames is None:
            self.add_plane(np.zeros(3), np.array([1.0, 0.0, 0.0]))  # explains why not
            return False
        start, end = cpr.body_span(self.frames)
        fraction = self.DEFAULT_CUT_FRACTIONS[len(self.planes) % len(self.DEFAULT_CUT_FRACTIONS)]
        s_mm = start + fraction * (end - start)
        if self.placements:
            # Never on top of a cut already placed.
            taken = [p.s_mm for p in self.placements]
            while min(abs(s_mm - t) for t in taken) < 10.0 and s_mm + 15.0 < end:
                s_mm += 15.0
        tangent, _, _ = self.frames.frame_at(s_mm)
        index = self.frames.index_of(s_mm)
        # The second cut faces the other way, so the two bound the segment.
        normal = -tangent if self.planes else tangent
        before = len(self.planes)
        self.add_plane(self.frames.points[index], normal)
        return len(self.planes) > before

    def _placement_for(self, origin, normal) -> PlanePlacement:
        """The placement whose resolved plane best matches a world point.

        Used when a cut is created by picking on the bone rather than by
        entering numbers: the arc position comes from the nearest point on the
        curve, and the side the cut removes from which way the picked normal
        faces along it.
        """
        if self.frames is None:
            return PlanePlacement()
        origin = np.asarray(origin, dtype=float).reshape(3)
        d = np.linalg.norm(self.frames.points - origin, axis=1)
        s_mm = float(self.frames.s[int(np.argmin(d))])
        tangent, _, _ = self.frames.frame_at(s_mm)
        flipped = bool(np.dot(np.asarray(normal, dtype=float).reshape(3), tangent) < 0)
        return PlanePlacement(s_mm=s_mm, flipped=flipped)

    def _resolve(self, placement: PlanePlacement, label: str) -> CutPlane:
        origin, normal = plane_from_placement(self.frames, placement)
        return CutPlane(origin=origin, normal=normal, label=label)

    def set_placement(self, index: int, placement: PlanePlacement) -> None:
        """Replace a cut's placement and rebuild its plane from it."""
        if self.frames is None or index >= len(self.placements):
            return
        placement = self._anchored(placement)
        self.placements[index] = placement
        self.planes[index] = self._resolve(placement, self.planes[index].label)
        self.cut_applied = False
        self.fragment_surface = None
        self.retained_surface = None
        self.update_resection_report()

    def _anchored(self, placement: PlanePlacement) -> PlanePlacement:
        """The placement, remembering the point of the curve it is put at."""
        point = self.frames.point_at(self.frames.clamp(placement.s_mm))
        return placement.replace(anchor_mm=tuple(float(v) for v in point))

    def placement(self, index: int) -> PlanePlacement:
        return self.placements[index]

    def set_plane(self, index: int, origin, normal) -> None:
        plane = self.planes[index]
        self.planes[index] = CutPlane(origin=origin, normal=normal, label=plane.label)
        self.update_resection_report()

    def translate_plane(
        self, index: int, s_mm: float, offset_mm=(0.0, 0.0, 0.0)
    ) -> None:
        """Move a cut along the jaw, keeping its angulation relative to the jaw.

        Only ``s_mm`` and the patient-axis offset change. The base orientation
        is re-derived from the local mandibular frame at the new position and
        the stored yaw, tilt and roll are re-applied to that new frame, so a
        cut dialled to 20 degrees oblique at the body arrives at the angle
        still 20 degrees oblique to the jaw there — not carrying the body's
        world-space orientation with it.
        """
        if self.frames is None or index >= len(self.placements):
            return
        self.set_placement(
            index,
            self.placements[index].replace(
                s_mm=self.frames.clamp(s_mm),
                offset_mm=tuple(np.asarray(offset_mm, dtype=float).reshape(3)),
            ),
        )

    def set_plane_origin(self, index: int, origin) -> None:
        """Move a cut to the arch position nearest a world point.

        Dragging in the 3-D view is a translation: the arc position follows
        the drag, and the angular offsets are untouched and re-applied to the
        frame at wherever the cut lands.
        """
        if self.frames is None or index >= len(self.placements):
            return
        origin = np.asarray(origin, dtype=float).reshape(3)
        d = np.linalg.norm(self.frames.points - origin, axis=1)
        nearest = int(np.argmin(d))
        s_mm = float(self.frames.s[nearest])
        # Whatever of the drag does not lie along the curve is kept as an
        # off-curve offset, so the plane still ends up under the cursor.
        offset = origin - self.frames.points[nearest]
        self.set_placement(
            index,
            self.placements[index].replace(s_mm=s_mm, offset_mm=tuple(offset)),
        )

    def rotate_plane(
        self,
        index: int,
        yaw_deg: float,
        tilt_deg: float,
        roll_deg: float | None = None,
    ) -> None:
        """Re-angle a cut about its own position, relative to the local frame.

        Only the angular offsets change; ``s_mm`` and the patient-axis offset
        are untouched, so the cut does not move.
        """
        if self.frames is None or index >= len(self.placements):
            return
        placement = self.placements[index]
        self.set_placement(
            index,
            placement.replace(
                yaw_deg=float(yaw_deg),
                tilt_deg=float(tilt_deg),
                roll_deg=placement.roll_deg if roll_deg is None else float(roll_deg),
            ),
        )

    def plane_arc_position(self, index: int) -> float:
        """Where a cut plane sits along the arch curve, in mm."""
        if self.frames is None or index >= len(self.placements):
            return float("nan")
        return float(self.placements[index].s_mm)

    def flip_plane(self, index: int) -> None:
        """Swap which side of the cut is removed."""
        if index >= len(self.placements):
            return
        self._push_undo()
        placement = self.placements[index]
        self.set_placement(index, placement.replace(flipped=not placement.flipped))

    def clear_planes(self) -> None:
        self._push_undo()
        self.planes.clear()
        self.placements.clear()
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
        # Every change to the cuts comes through here: bound them along the
        # jaw before anything measures, previews, cuts or mirrors with them.
        bound_resection(self.frames, self.planes)
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
        self.update_plate_fit()
        self._update_fit()
        self.plate_changed.emit()

    def execute_cut(self) -> None:
        self.ensure_mandible()
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
        self.ensure_mandible()
        self.symmetry_plane = estimate_midsagittal_plane(self.bone_volume, self.threshold)
        if self.symmetry_plane.symmetry < SYMMETRY_WARNING:
            self.message.emit(
                f"Only {self.symmetry_plane.symmetry:.0%} of the bone mirrors onto bone "
                f"across the best plane of symmetry (within {SYMMETRY_TOLERANCE_MM:g} mm). "
                "Check the plane in the 3-D view before relying on the mirror."
            )
        self.update_reconstruction()

    def update_reconstruction(self) -> None:
        """Mirror the healthy side into the defect and measure what it misses."""
        if (
            self.reconstruction is not None
            and self._reconstruction_key != self._reconstruction_inputs()
        ):
            self._clear_reconstruction()
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

    def build_reconstruction(self) -> None:
        """Rebuild the resected segment from the mirrored healthy side.

        One flush surface: the mirror is registered to each stump, warped
        smoothly between them, and blended into the retained bone before a
        single surface is contoured. See ``geometry/reconstruction.py``.
        """
        if self.volume is None or self.surface is None or not self.planes:
            self.message.emit("Place the cutting planes before reconstructing.")
            return
        self.ensure_mandible()
        if self.symmetry_plane is None:
            # The computer finds the plane of symmetry itself; the operator can
            # still re-run the estimate from the panel.
            self.estimate_symmetry_plane()
        if self.symmetry_plane is None:
            return
        self.busy.emit(True)
        try:
            self.reconstruction = reconstruct(
                self.bone_volume,
                self.threshold,
                self.surface,
                self.planes,
                self.symmetry_plane,
            )
        finally:
            self.busy.emit(False)
        self._reconstruction_key = self._reconstruction_inputs()
        self._sculptor = None
        self._graft_surface = clip_closed(
            self.reconstruction.surface, self.planes, keep_resected=True
        )
        self._graft_stale = False
        for line in self.reconstruction.report.warnings:
            self.message.emit(line)
        self.reconstruction_changed.emit()

    #: The panel and older callers know this step as building the graft.
    build_graft = build_reconstruction

    @property
    def reconstructed_surface(self):
        return None if self.reconstruction is None else self.reconstruction.surface

    def _reconstruction_inputs(self):
        """What a reconstruction depends on; a change makes it stale."""
        planes = tuple(
            (tuple(np.round(p.origin, 6)), tuple(np.round(p.normal, 9))) for p in self.planes
        )
        plane = self.symmetry_plane
        symmetry = None if plane is None else (
            tuple(np.round(plane.point, 6)), tuple(np.round(plane.normal, 9))
        )
        return (planes, symmetry, round(float(self.threshold), 6), id(self.surface))

    def _clear_reconstruction(self) -> None:
        self.reconstruction = None
        self._reconstruction_key = None
        self._graft_surface = None
        self._graft_stale = False
        self._sculptor = None

    @property
    def graft_surface(self):
        """The mirrored segment alone: the reconstruction inside the cuts."""
        if self._graft_stale and self.reconstruction is not None:
            self._graft_surface = clip_closed(
                self.reconstruction.surface, self.planes, keep_resected=True
            )
            self._graft_stale = False
        return self._graft_surface

    # -- hand edits to the reconstruction ------------------------------------

    @property
    def reconstruction_edited_mm(self) -> float:
        """How far the edits have moved the surface from what was computed."""
        return self._sculptor.max_change_mm() if self._sculptor is not None else 0.0

    @property
    def can_undo_edit(self) -> bool:
        return self._sculptor is not None and self._sculptor.can_undo

    def _ensure_sculptor(self) -> SurfaceSculptor | None:
        if self.reconstruction is None:
            return None
        if self._sculptor is None:
            surface = self.reconstruction.surface
            points = vtk_to_numpy(surface.GetPoints().GetData()).astype(float)
            triangles = vtk_to_numpy(surface.GetPolys().GetConnectivityArray()).reshape(-1, 3)
            self._sculptor = SurfaceSculptor(points, triangles)
        return self._sculptor

    def sculpt(self, point, new_stroke: bool = True) -> None:
        """One dab of the current brush at ``point`` on the reconstruction."""
        sculptor = self._ensure_sculptor()
        if sculptor is None:
            self.message.emit("Reconstruct the jaw first; the brushes refine the result.")
            return
        if new_stroke:
            sculptor.checkpoint()
        if sculptor.stroke(point, self.brush_radius_mm, self.brush_strength, self.brush):
            self._show_edit()

    def smooth_junctions(self) -> None:
        """Round the surface off where each cut meets the mirrored segment."""
        sculptor = self._ensure_sculptor()
        if sculptor is None or not self.planes:
            return
        sculptor.checkpoint()
        moved = sculptor.smooth_near_planes(self.planes)
        self._show_edit()
        self.message.emit(f"Smoothed {moved:,} surface points at the junctions.")

    def undo_edit(self) -> None:
        if self._sculptor is not None and self._sculptor.undo():
            self._show_edit()

    def reset_edits(self) -> None:
        """Back to the reconstruction exactly as computed."""
        if self._sculptor is not None and self._sculptor.edited:
            self._sculptor.reset()
            self._show_edit()

    def _show_edit(self) -> None:
        surface = self.reconstruction.surface
        points = self._sculptor.points
        surface.GetPoints().SetData(numpy_to_vtk(points, deep=True))
        normals = numpy_to_vtk(vertex_normals(points, self._sculptor.triangles), deep=True)
        normals.SetName("Normals")
        surface.GetPointData().SetNormals(normals)
        surface.Modified()
        self._graft_stale = True
        self._plate_projector = None
        self.reconstruction_edited.emit()

    @property
    def plate_projector(self) -> SurfaceProjector | None:
        """Projects plate path clicks onto the bone the plate will sit on:
        the reconstructed jaw once there is one, since the plate spans the
        rebuilt segment, and the patient's own bone before that."""
        surface = self.reconstructed_surface or self.surface
        if surface is None or not surface.GetNumberOfPoints():
            return None
        if self._plate_projector is None or self._plate_projector_source is not surface:
            self._plate_projector = SurfaceProjector(surface)
            self._plate_projector_source = surface
        return self._plate_projector

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
        self.ensure_mandible()
        projector = self.plate_projector
        if projector is None:
            self.message.emit("Extract a bone surface before drawing a plate path.")
            return
        self._push_undo()
        point, _ = projector.project(world_point)
        normal = projector.footprint_normal(point, self.PLATE_FOOTPRINT_MM)
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

    def set_plate_asset(self, asset_id: str | None, length_chosen: bool = True) -> None:
        """Choose which plate model is being planned, by catalogue id.

        ``length_chosen`` False picks only the family: the length then
        follows the drawn path.
        """
        self.plate_length_chosen = bool(length_chosen)
        self.plate_asset = None if asset_id is None else asset_by_id(asset_id)
        if self.plate_asset is not None:
            self._adopt_asset(self.plate_asset)
        self.update_plate_plan()

    def _adopt_asset(self, asset) -> None:
        """Take pitch, section and the length check from the chosen plate.

        The path is resampled at the asset's own screw-hole pitch, so the
        holes that get fitted are the holes the plate really has.
        """
        self.plate.pitch_mm = asset.hole_pitch_mm
        self.plate.width_mm = asset.width_mm
        self.plate.thickness_mm = asset.thickness_mm
        self.plate_system = system_for_asset(asset)

    def use_fitting_length(self) -> None:
        """Switch to the shortest plate of this family that spans the plan."""
        fit, asset = self.fit, self.plate_asset
        if fit is None or fit.option is None or asset is None:
            return
        for candidate in assets_in_family(asset.family):
            if candidate.hole_count == fit.option.holes:
                self.set_plate_asset(candidate.id)
                return

    def set_plate_clearance(self, clearance_mm: float) -> None:
        """How far the plate's inner face stands off the bone, in mm."""
        self.plate_clearance_mm = max(float(clearance_mm), 0.0)
        self.update_plate_fit()

    def update_plate_fit(self) -> None:
        """Place the selected asset on the planned path as a rigid body.

        Rigid means rigid: the mesh is rotated and translated and nothing
        else, so thickness, hole diameter and hole-to-hole spacing come
        through untouched. What the rigid placement cannot reach is reported
        as residual rather than scaled away.
        """
        self.fitted_plate = None
        self.bent_plate = None
        self.plate_contact = None
        self.hole_distortion = None
        self.plate_marking = None
        self.plate_fit_warnings = []
        self.plate_fit_problems = []
        asset = self.plate_asset
        plan = self.plate_plan
        if asset is None or plan is None or len(plan.nodes) < 2:
            return
        try:
            mesh = load_asset_mesh(asset)
        except MeshLoadError as error:
            self.plate_fit_warnings = [str(error)]
            self.message.emit(f"Plate asset could not be loaded: {error}")
            return

        targets = plate_targets(
            plan.nodes,
            plan.normals,
            plan.binormals,
            self.plate_clearance_mm,
            asset.thickness_mm,
        )
        self.fitted_plate = rigid_fit(
            mesh.points,
            mesh.triangles,
            asset.hole_centres_mm,
            asset.hole_axes,
            targets,
            plan.normals,
            plan.binormals,
        )
        mark_points, mark_triangles = self._marking_for(asset)
        self.plate_marking = (
            mark_points @ self.fitted_plate.rotation.T + self.fitted_plate.translation,
            mark_triangles,
        )
        self.plate_fit_warnings = list(self.fitted_plate.warnings)
        if self.fitted_plate.max_residual_mm > 2.0 and not self.plate_bending_enabled:
            self.plate_fit_warnings.append(
                f"Rigid placement leaves the plate up to "
                f"{self.fitted_plate.max_residual_mm:.1f} mm off the planned "
                "path. This plate has to be bent to follow it."
            )
        if self.plate_bending_enabled:
            self._bend_plate(asset, plan, targets)
        self._check_contact(asset, plan)

    def _marking_for(self, asset):
        """The asset's etched mark in its own space, built once per asset."""
        if asset.id not in self._marking_cache:
            from ..render.marking import marking_in_plate_space

            self._marking_cache[asset.id] = marking_in_plate_space(asset)
        return self._marking_cache[asset.id]

    def set_plate_bending(self, enabled: bool) -> None:
        """Bend the plate onto the path, or leave it rigidly placed."""
        self.plate_bending_enabled = bool(enabled)
        self.update_plate_fit()
        self.plate_changed.emit()

    def _bend_plate(self, asset, plan, targets) -> None:
        """Bend the placed plate onto the path, holding the holes rigid."""
        fitted = self.fitted_plate
        if fitted is None or asset.hole_count < 2:
            return
        # The etched mark is bent together with the plate, as one set of
        # points, so it stays on the face it is etched into.
        mark_points, mark_triangles = self.plate_marking or (np.zeros((0, 3)), None)
        plate_count = len(fitted.points)
        try:
            paired, normals, _ = extend_targets(
                asset.hole_count, targets, plan.normals, plan.binormals
            )
            self.bent_plate = bend_to_path(
                np.vstack([fitted.points, mark_points]),
                fitted.triangles,
                fitted.hole_centres,
                fitted.hole_axes,
                paired,
                normals,
                protected_radius_mm=asset.deformation.protected_radius_mm,
                min_bend_radius_mm=asset.deformation.min_bend_radius_mm,
                max_bend_deg_per_node=asset.deformation.max_bend_deg_per_node,
            )
        except ValueError as error:
            self.plate_fit_warnings.append(f"The plate could not be bent: {error}")
            return
        bent_marking = self.bent_plate.points[plate_count:]
        self.bent_plate.points = self.bent_plate.points[:plate_count]
        if mark_triangles is not None:
            self.plate_marking = (bent_marking, mark_triangles)
        self.plate_fit_warnings.extend(self.bent_plate.warnings)
        self.plate_fit_problems.extend(self.bent_plate.problems)
        self._predict_hole_distortion(asset)

    def set_hole_distortion_shown(self, shown: bool) -> None:
        """Show the holes as bending leaves them, or as the catalogue drew them."""
        self.show_hole_distortion = bool(shown)
        self.update_plate_fit()
        self.plate_changed.emit()

    def set_bending_insets(self, use: bool | None) -> None:
        """Whether bending insets are fitted into the threaded holes."""
        self.use_bending_insets = use
        self.update_plate_fit()
        self.plate_changed.emit()

    def _predict_hole_distortion(self, asset) -> None:
        """Work out what this bending plan does to the screw holes.

        Holes do not stay round through contouring unless something holds
        them. Which holes suffer depends on where the bends fall and on what
        the kit does: a tight three-point plier concentrates a bend into a few
        millimetres, a press spreads it and fits insets as well.
        """
        bent = self.bent_plate
        if bent is None or bent.hole_s_mm is None:
            return
        angles = np.nan_to_num(bent.bend_angles_deg)
        report = predict_distortion(
            bent.hole_s_mm,
            asset.hole_diameter_mm,
            asset.thickness_mm,
            angles,
            bent.hole_s_mm,
            self.plate_alloy(),
            self.bending_kit,
            use_insets=self.use_bending_insets,
            locking=bool(getattr(asset, "locking", True)),
        )
        self.hole_distortion = report
        bent.distortion = report
        self.plate_fit_warnings.extend(report.warnings)
        self.plate_fit_problems.extend(report.problems)

        if self.show_hole_distortion and bent.hole_tangents is not None:
            bent.points = ovalise_mesh(
                bent.points,
                bent.hole_centres,
                bent.hole_axes,
                bent.hole_tangents,
                report.holes,
                asset.hole_diameter_mm,
            )

    def note_export(self, what: str) -> None:
        if what not in self.exported:
            self.exported.append(what)
        self.plate_changed.emit()  # the export step reads the list

    def bending_guide(self):
        """The clip-on guide that stops each bend of the selected plate at its
        planned angle (see ``geometry/bending_guide.py``), or ``None`` with a
        message saying what is missing."""
        from ..geometry.bending_guide import plan_bending_guide

        asset, bent = self.plate_asset, self.bent_plate
        if asset is None or bent is None or bent.hole_tangents is None:
            self.message.emit(
                "Draw a plate path with bending switched on before exporting a "
                "bending guide."
            )
            return None
        alloy = self.plate_alloy()
        thickness = asset.thickness_mm
        # Plastic bending happens over the bridge between the rigid rings
        # round two holes; its radius sets how much the plate springs back.
        bridge = max(asset.hole_pitch_mm - 2.0 * asset.deformation.protected_radius_mm, 1.0)

        def overbend(angle_deg: float) -> float:
            if angle_deg <= 0.0:
                return 0.0
            radius = max(bridge / np.radians(angle_deg), alloy.min_bend_radius_mm(thickness))
            return alloy.overbend_deg(angle_deg, radius, thickness)

        return plan_bending_guide(
            asset.hole_centres_mm,
            asset.hole_axes,
            bent.hole_centres,
            bent.hole_axes,
            bent.hole_tangents,
            asset.width_mm,
            thickness,
            asset.seat_diameter_mm or asset.hole_diameter_mm + 1.5,
            overbend=overbend,
            alloy_name=alloy.name,
        )

    def plate_alloy(self) -> Alloy:
        """The alloy the selected plate is made of."""
        asset = self.plate_asset
        material = getattr(asset, "material_id", None) if asset else None
        return alloy_by_id(material or "cp-ti-grade-4")

    def _check_contact(self, asset, plan) -> None:
        """Measure the fitted plate against the bone it has to sit on."""
        placed = self.bent_plate or self.fitted_plate
        if placed is None:
            return
        holes = len(placed.hole_centres)
        nodes, normals = plan.nodes, plan.normals
        if len(nodes) < 1 or holes < 1:
            return
        self.plate_contact = clearance_report(
            placed.hole_centres,
            placed.hole_axes,
            nodes,
            normals,
            asset.thickness_mm,
            target_clearance_mm=self.plate_clearance_mm,
        )
        self.plate_fit_warnings.extend(self.plate_contact.warnings)
        self.plate_fit_problems.extend(self.plate_contact.problems)

    def plate_mesh(self):
        """The plate as it will be displayed and exported: bent if bent."""
        placed = self.bent_plate or self.fitted_plate
        if placed is None:
            return None
        return placed

    def update_plate_plan(self) -> None:
        self.plate_plan = None
        if len(self.plate_points) >= 2:
            path, normals = fair_plate_path(
                np.vstack(self.plate_points), np.vstack(self.plate_normals)
            )
            total = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            if total >= self.plate.pitch_mm:
                self.plate_plan = compute_plate_plan(path, normals, self.plate.pitch_mm)
            else:
                self.message.emit(
                    f"Plate path is {total:.1f} mm, shorter than one "
                    f"{self.plate.pitch_mm:.1f} mm screw-hole pitch."
                )
        self.update_plate_fit()
        self._update_fit()
        self._size_plate_to_path()
        self.plate_changed.emit()

    def _size_plate_to_path(self) -> None:
        """Use the shortest plate of the family that spans the path, unless
        the operator chose a length."""
        fit, asset = self.fit, self.plate_asset
        if self.plate_length_chosen or fit is None or fit.option is None or asset is None:
            return
        if fit.option.holes == asset.hole_count:
            return
        for candidate in assets_in_family(asset.family):
            if candidate.hole_count == fit.option.holes:
                self.plate_asset = candidate
                self._adopt_asset(candidate)
                self.update_plate_fit()
                self._update_fit()
                return

    # -- undo --------------------------------------------------------------

    def _push_undo(self) -> None:
        self._undo.append(
            _Snapshot(
                arch_seeds=copy.deepcopy(self.arch_seeds),
                arch_source=self.arch_source,
                planes=copy.deepcopy(self.planes),
                placements=copy.deepcopy(self.placements),
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
        self.arch_source = state.arch_source
        self.planes = state.planes
        self.placements = state.placements
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
