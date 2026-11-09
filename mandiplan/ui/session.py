"""Application state: the volume, the arch curve, the cuts and the plate.

The widgets read from here and call into here; nothing in this module knows
what a widget looks like.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

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
from ..geometry.mesh_io import Mesh, MeshLoadError
from ..geometry.plate import PlatePlan, compute_plate_plan, fair_plate_path, plan_at_holes
from ..geometry.hole_distortion import ovalise_mesh, predict_distortion
from ..geometry.auto_plate import (
    PREFORM_MIN_DEG,
    advise_holes,
    course_from_neck,
    jaw_route,
    judge_plate,
    place_chain,
    plate_turns,
    recommend_length,
)
from ..geometry.plate_bend import (
    BentPlate,
    ClearanceReport,
    bend_to_path,
    clearance_report,
    seat_plate_chain,
)
from ..geometry.plate_fit import (
    FittedPlate,
    extend_targets,
    plate_targets,
    rigid_fit,
)
from ..materials import Alloy, alloy_by_id
from ..plate_assets import PlateAsset, asset_by_id, assets_in_family, hole_length_mm, load_asset_mesh, load_assets
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
    plate_auto: bool = False


@dataclass
class VolumeInfo:
    series: SeriesInfo
    geometry: SeriesGeometry
    folder: str

    def lines(self, scan: Volume) -> list[str]:
        """What was loaded: ``scan`` is the whole series as shown."""
        nx, ny, nz = (int(v) for v in scan.size_xyz)
        sx, sy, sz = scan.spacing
        lines = [
            f"Series: {self.series.description or '(no description)'}",
            f"Slices: {self.series.n_files or int(scan.size_xyz[2])}",
            f"Matrix: {nx} × {ny} × {nz} voxels",
            f"Voxel size: {sx:.3f} × {sy:.3f} × {sz:.3f} mm",
            f"Field of view: {scan.extent_mm[0]:.1f} × {scan.extent_mm[1]:.1f}"
            f" × {scan.extent_mm[2]:.1f} mm",
        ]
        jaws = getattr(self.geometry, "jaw_volume", None)
        if jaws is not None:
            # A CT's mandible is separated and drawn from a box round its
            # jaws in cubic voxels (see dicom_io).
            lines.append(
                f"Jaw box: {jaws.extent_mm[0]:.0f} × {jaws.extent_mm[1]:.0f} × "
                f"{jaws.extent_mm[2]:.0f} mm in {float(jaws.spacing[0]):.2f} mm cubes"
            )
        return lines


#: A bend past this share of a plate's limit at one hole rules the plate out.
BEND_FATAL_SHARE = 2.0
#: Of what a plate's bridge takes, this share is left to its turning in its own plane
#: when it is laid along the jaw; the rest is for the bends out of it.
IN_PLANE_SHARE = 0.75
#: A hole this far into the bone is within what a rough surface reads, mm.
COLLISION_TOLERANCE_MM = 0.3


class Session(QObject):
    volume_changed = pyqtSignal()
    surface_changed = pyqtSignal()
    arch_changed = pyqtSignal()
    reformat_changed = pyqtSignal()
    resection_changed = pyqtSignal()
    reconstruction_changed = pyqtSignal()
    plate_changed = pyqtSignal()
    guide_view_changed = pyqtSignal()
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

    AUTO_PLATE_DELAY_MS = 350

    def __init__(self, parent=None):
        super().__init__(parent)
        #: The whole series as loaded: what the slices show, and the bone
        #: surface until the mandible has been separated.
        self.scan: Volume | None = None
        #: What the mandible is separated, drawn and planned on: for a
        #: medical CT the box round its jaws in cubic voxels
        #: (``SeriesGeometry.jaw_volume``), otherwise the scan itself. Both
        #: are in the same world millimetres.
        self.volume: Volume | None = None
        self.info: VolumeInfo | None = None
        self.threshold: float = 0.0
        self.surface = None
        self._extractor: SurfaceExtractor | None = None
        self._projector: SurfaceProjector | None = None
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
        #: The plate was placed by ``place_plate_automatically``, not drawn; it
        #: follows the cuts and the jaw until a path is drawn by hand.
        self.plate_auto = False
        self.plate_placement = None
        self.plate_route = None
        #: The automatic route along the plate, ``(points, outward)``, for seating it.
        self._auto_course = None
        #: "neck": from the condylar neck past the far cut. "centred": on the resection.
        self.plate_course = "neck"
        self.plate_course_notes: list[str] = []
        self._retained_by_hand = False
        self._trim_first = 0
        #: Whether this plate will do here, hole by hole. See ``geometry/auto_plate``.
        self.plate_verdict = None
        self._guide = None
        self._guide_mesh = None
        self._guide_steps = []
        self.guide_step = 0
        self._auto_plate_timer = QTimer(self)
        self._auto_plate_timer.setSingleShot(True)
        self._auto_plate_timer.timeout.connect(self._replace_plate)
        self._trimmed_stock = None
        self.fitted_hole_plan = None
        self.fitted_plate: FittedPlate | None = None
        self.bent_plate: BentPlate | None = None
        self.plate_contact: ClearanceReport | None = None
        #: How snugly a key's pins sit in the plate's holes, per side, mm.
        self.guide_pin_play_mm = 0.1
        #: Added to every key's stop, if the fit test shows this plate springs back
        #: more than the estimate allows for, degrees.
        self.guide_extra_overbend_deg = 0.0
        #: The bender's head as measured, in place of what the kit says.
        self.guide_head = None
        #: The printer last chosen, for the keys as for the jaw.
        self.printer_name = "Asiga MAX"
        self.printer_build_mm = None
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
        self.scan = volume
        jaws = getattr(geometry, "jaw_volume", None)
        self.volume = jaws if jaws is not None else volume
        # All bone, until the mandible is separated: the whole scan.
        self._extractor = SurfaceExtractor(self.scan)
        self.info = VolumeInfo(series=series, geometry=geometry, folder=folder)
        self.threshold = self.automatic_threshold()
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
        self._trimmed_stock = None
        self._retained_by_hand = False
        self._trim_first = 0
        self.plate_course_notes = []
        self.fitted_hole_plan = None
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
        # A long, fine-slice CT also gets a jaw box. Cropping alone does not
        # blur thin cortex: only a volume resampled from coarse slices needs
        # the lower seed intended to recover that partially averaged bone.
        thin_bone = self.volume.resampled_from_mm > 0.0
        return estimate_bone_threshold(self.volume, thin_bone=thin_bone)

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
        # Built when first needed: on a whole CT's bone a locator takes
        # seconds, and the first view should not wait for it.
        self._projector = None
        self.cut_applied = False
        self.fragment_surface = None
        self.retained_surface = None
        self._clear_reconstruction()
        self.surface_changed.emit()
        self.update_plate_plan()

    # -- mandible ----------------------------------------------------------

    @property
    def projector(self) -> SurfaceProjector | None:
        """Projects points onto the bone surface; built on first use."""
        if self._projector is None and self.surface is not None and self.surface.GetNumberOfPoints():
            self._projector = SurfaceProjector(self.surface)
        return self._projector

    @projector.setter
    def projector(self, value: SurfaceProjector | None) -> None:
        self._projector = value

    @property
    def bone_volume(self) -> Volume | None:
        """The scan as the planning steps should see it: the mandible's bone
        only, once it has been separated from the skull; until then all bone,
        the volume the bone surface was drawn from."""
        return self._bone_volume if self._bone_volume is not None else self.scan

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
        if self.plate_auto:
            self._auto_plate_timer.start(self.AUTO_PLATE_DELAY_MS)
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
        if self.plate_auto:
            # The plate follows the cuts, once they have stopped moving.
            self._auto_plate_timer.start(self.AUTO_PLATE_DELAY_MS)
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
        self._refit_on_jaw_change()

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
        self._refit_on_jaw_change()

    def _refit_on_jaw_change(self):
        if self.plate_plan is not None:
            self.update_plate_fit()
            self._update_fit()
            self.plate_changed.emit()

    @property
    def plate_projector(self) -> SurfaceProjector | None:
        """Projects plate path clicks onto the bone the plate will sit on:
        the reconstructed jaw once there is one, since the plate spans the
        rebuilt segment, and the patient's own bone before that."""
        surface = self.reconstructed_surface
        if surface is None:
            surface = self.retained_surface if self.cut_applied else self.surface
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
        self.guide_head = None
        self.update_plate_plan()

    def set_guide_head_value(self, name: str, value: float) -> None:
        """One measurement of the bender's head, as taken from the bender in hand."""
        head = self.guide_head or self.bending_kit.head
        self.guide_head = replace(head, source="measured from the bender in hand", **{name: float(value)})
        self._update_fit()
        self.plate_changed.emit()

    def set_guide_setting(self, name: str, value: float) -> None:
        """Pin play or extra overbend, for the keys."""
        if name not in ("guide_pin_play_mm", "guide_extra_overbend_deg"):
            raise ValueError(name)
        setattr(self, name, float(value))
        self._update_fit()
        self.plate_changed.emit()

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
        if self._trimmed_stock and self.fit.option is not None:
            from ..plate_catalog import PlateOption
            active = self.active_plate_asset
            self.fit.option = PlateOption(active.hole_count, active.length_mm)
            self.fit.trim_holes = 0
        if self.plate_length_chosen and self.active_plate_asset is not None:
            uncovered = self.fit.required_mm - self.active_plate_asset.length_mm
            if uncovered > 0.01 and self.fit.option is not None:
                self.fit.problems.append(f"The selected plate leaves {uncovered:.1f} mm of the path uncovered.")
        plan = self.fitted_hole_plan or self.plate_plan
        if span is not None and self.fitted_hole_plan is not None:
            # Count actual retained holes, not virtual stations on a longer path.
            measured = fit_check(plan, self.plate_system, defect_span=span, frames=self.frames)
            self.fit.holes_proximal = measured.holes_proximal
            self.fit.holes_distal = measured.holes_distal
            self.fit.holes_over_defect = measured.holes_over_defect
            self.fit.problems = [v for v in self.fit.problems if "land on retained bone" not in v]
            self.fit.problems.extend(v for v in measured.problems if "land on retained bone" in v)
        self.steps = generate_steps(plan, self.plate_system, self.bending_kit, self.fit)
        if self.bent_plate is not None and not self.plate_system.is_custom:
            from ..bending_steps import steps_from_guide
            try:
                guide = self.bending_guide()
            except ValueError:
                # A kit with nothing for a key to stop: the plain steps stand.
                guide = None
            if guide is not None:
                self.steps = steps_from_guide(
                    guide, stock_holes=self.plate_asset.hole_count, first=self._trim_first if self._trimmed_stock else 0
                )

    def add_plate_point(self, world_point) -> None:
        self.ensure_mandible()
        projector = self.plate_projector
        if projector is None:
            self.message.emit("Extract a bone surface before drawing a plate path.")
            return
        self._push_undo()
        if self.plate_auto:
            # Drawing by hand takes over from the automatic placement.
            self.plate_auto = False
            self.plate_points.clear()
            self.plate_normals.clear()
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
        self.plate_auto = False
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
        self._trimmed_stock = None
        self._retained_by_hand = False
        self._trim_first = 0
        self.plate_length_chosen = bool(length_chosen)
        self.plate_asset = None if asset_id is None else asset_by_id(asset_id)
        if self.plate_asset is not None:
            self._adopt_asset(self.plate_asset)
        # Choosing a plate is all it takes, unless a path was drawn by hand.
        if self.plate_asset is not None and (self.plate_auto or not self.plate_points):
            if self.place_plate_automatically(quiet=True):
                return
        self.update_plate_plan()

    # -- placing the plate without being told where ------------------------

    def _defect_span(self):
        """The resection's arc span on the arch curve, or None."""
        report = self.report
        if self.planes and report is not None and np.isfinite(report.arc_length_mm):
            return (float(report.entry_s_mm), float(report.exit_s_mm))
        return None

    def _route_surface(self):
        return self.reconstructed_surface if self.reconstructed_surface is not None else self.surface

    def place_plate_automatically(self, quiet: bool = False) -> bool:
        """Put the chosen plate where it belongs: no clicks on the bone.

        The length is chosen too, unless the operator picked one. Returns
        False, with the reason shown unless ``quiet``, if there is nothing to
        place it on yet.
        """
        if self.volume is None:
            if not quiet:
                self.message.emit("Open a scan first.")
            return False
        if quiet and self.mandible is None:
            # Not while the scan is still being read: this runs again after.
            return False
        self.ensure_mandible()
        surface = self._route_surface()
        asset = self.plate_asset
        if asset is None or self.frames is None or surface is None or not surface.GetNumberOfPoints():
            if not quiet:
                self.message.emit(
                    "The plate cannot be placed yet: it needs the jaw separated and an arch "
                    "curve along it (steps 1 and 2)."
                )
            return False
        vertices = vtk_to_numpy(surface.GetPoints().GetData()).astype(float)
        route = jaw_route(vertices, self.frames, asset.width_mm, self._in_plane_turn_limit(asset))
        if route is None:
            if not quiet:
                self.message.emit("No route for a plate could be found along this jaw.")
            return False
        defect = self._defect_span()
        system = system_for_asset(asset)
        minimum = max(int(system.min_holes_per_side), 1)
        from_neck = self.plate_course == "neck" and defect is not None

        def describe(item):
            centres, axes = item.hole_centres_mm, item.hole_axes
            return np.linalg.norm(np.diff(centres, axis=0), axis=1), plate_turns(centres, axes)

        if not self.plate_length_chosen and not self._retained_by_hand:
            family = sorted(assets_in_family(asset.family), key=lambda a: a.hole_count)
            choices = [(*describe(item), item.id) for item in family]
            best = recommend_length(route, choices, defect, minimum, from_neck=from_neck)
            if best is not None and best != asset.id:
                self.plate_asset = asset = asset_by_id(best)
                self._adopt_asset(asset)
        notes: list[str] = []
        course = None
        if from_neck:
            if self._retained_by_hand:
                course = course_from_neck(route, *describe(self.active_plate_asset), defect, minimum, keep_all=True)
            else:
                course = course_from_neck(route, *describe(asset), defect, minimum)
        if not self._retained_by_hand:
            if course is not None:
                if not self._cut_stock(asset, course.first, course.count):
                    # Where the plate could not be cut, the whole of it, if that goes.
                    course = course_from_neck(route, *describe(asset), defect, minimum, keep_all=True)
            else:
                self._cut_stock(asset, 0, asset.hole_count)
        active = self.active_plate_asset
        if course is not None:
            placement = course.placement
            neck = route.necks_mm[1 if course.neck_at_route_end else 0]
            start = "the condylar neck" if neck is not None else "the end of the jaw's run"
            notes.append(f"Runs from {start} to {course.holes_past} holes past the cut.")
            if not self._retained_by_hand and course.count < asset.hole_count:
                off = asset.hole_count - course.count
                notes.append(
                    f"Cut off the {off} hole{'' if off == 1 else 's'} this plate does not need: "
                    f"{course.count} of its {asset.hole_count} stay. Change it under Holes to keep."
                )
        else:
            spacings, turns = describe(active)
            placement = place_chain(route, spacings, turns, defect, minimum)
            if from_neck:
                notes.append(
                    "This plate cannot reach from the condylar neck past the far cut, so it is centred on the "
                    "resection instead. A longer plate could run from the neck."
                )
        if placement is None:
            if not quiet:
                self.message.emit("This plate could not be placed along this jaw.")
            return False
        self.plate_route = route
        self.plate_placement = placement
        self.plate_course_notes = notes
        self.plate_auto = True
        self._auto_course = self._course_along(route, placement, active.hole_pitch_mm)
        self.plate_points = [np.asarray(p, dtype=float) for p in placement.points]
        self.plate_normals = [np.asarray(n, dtype=float) for n in placement.outward]
        self.update_plate_plan()
        return True

    @staticmethod
    def _in_plane_turn_limit(asset) -> float:
        """How fast a plate may turn in its own plane along the jaw, degrees per mm:
        a share of what one bridge takes, or what it was made with if more."""
        own = np.abs(plate_turns(asset.hole_centres_mm, asset.hole_axes))
        per_bridge = max(IN_PLANE_SHARE * float(asset.deformation.max_bend_deg_per_node),
                         float(own.max()) if len(own) else 0.0)
        return per_bridge / max(float(asset.hole_pitch_mm), 1e-6)

    @staticmethod
    def _course_along(route, placement, pitch_mm: float):
        """The route itself from the plate's first hole to its last, a pitch past each:
        what the plate is seated along, in its own hole order."""
        first = int(np.argmin(np.linalg.norm(route.points - placement.points[0], axis=1)))
        last = int(np.argmin(np.linalg.norm(route.points - placement.points[-1], axis=1)))
        step = 1 if last >= first else -1
        reach = int(np.ceil(pitch_mm))
        lo = int(np.clip(first - step * reach, 0, len(route.s) - 1))
        hi = int(np.clip(last + step * reach, 0, len(route.s) - 1))
        index = np.arange(lo, hi + step, step)
        if len(index) < 3:
            return None
        return route.points[index].copy(), route.outward[index].copy()

    def _cut_stock(self, stock, first: int, count: int) -> bool:
        """Keep these holes of the stock, unless that is what is already kept. Says
        whether they are now what is kept; if not, the plate is left whole."""
        from ..render.plate_trim import trim_stock

        if count >= stock.hole_count:
            if self._trimmed_stock is not None:
                self._marking_cache.clear()
            self._trimmed_stock = None
            self._trim_first = 0
            return True
        current = self._trimmed_stock
        if current is not None and current[0].hole_count == count and self._trim_first == first and (
            current[0].id == stock.id
        ):
            return True
        self._marking_cache.clear()
        try:
            self._trimmed_stock = trim_stock(stock, int(count), int(first))
            self._trim_first = int(first)
            return True
        except (ValueError, MeshLoadError) as error:
            self.message.emit(str(error))
            self._trimmed_stock = None
            self._trim_first = 0
            return False

    def set_plate_course(self, course: str) -> None:
        """From the condylar neck past the far cut, or centred on the resection."""
        if course not in ("neck", "centred"):
            raise ValueError(course)
        if course == self.plate_course:
            return
        self.plate_course = course
        self._replace_plate()
        self.plate_changed.emit()

    def _replace_plate(self) -> None:
        if self.plate_auto and self.plate_asset is not None:
            self.place_plate_automatically(quiet=True)

    def _judge_plate(self) -> None:
        """Say whether the plate will do where it lies, hole by hole."""
        self.plate_verdict = None
        asset = self.active_plate_asset
        placed = self.bent_plate or self.fitted_plate
        contact = self.plate_contact
        if asset is None or placed is None or contact is None or self.frames is None:
            return
        projector = self.plate_projector
        centres, axes = placed.hole_centres, placed.hole_axes
        if len(contact.clearance_mm) != len(centres):
            return
        stations = self.frames.points[:: max(int(round(1.0 / self.frames.step_mm)), 1)]
        arcs_at = self.frames.s[:: max(int(round(1.0 / self.frames.step_mm)), 1)]
        arcs = arcs_at[np.argmin(np.linalg.norm(centres[:, None, :] - stations[None, :, :], axis=2), axis=1)]
        depths = np.array(
            [projector.through_bone(c, -a) if projector is not None else float("nan") for c, a in zip(centres, axes)]
        )
        defect = self._defect_span()
        holes = advise_holes(arcs, contact.clearance_mm, contact.hole_axis_error_deg, depths, defect)

        limit = float(asset.deformation.max_bend_deg_per_node)
        bends = np.nan_to_num(self.bent_plate.bend_angles_deg) if self.bent_plate is not None else np.zeros(0)
        fatal, cautions = [], []
        worst = float(bends.max()) if len(bends) else 0.0
        if worst > BEND_FATAL_SHARE * limit:
            fatal.append(
                f"The jaw asks for a {worst:.0f}° bend at one hole here, far past the "
                f"{limit:.0f}° this plate is rated for at a hole."
            )
        elif worst > limit:
            cautions.append(
                f"The tightest bend is {worst:.0f}° at a hole, past the {limit:.0f}° this plate is "
                "rated for: bend it in stages and check it for cracks."
            )
        if self.bent_plate is not None and any("fold" in p for p in self.bent_plate.problems):
            fatal.extend(p for p in self.bent_plate.problems if "fold" in p)
        # A rough surface reads a touching plate as a little inside it; only more
        # than that is worth saying.
        if len(contact.clearance_mm) and float(contact.clearance_mm.min()) < -COLLISION_TOLERANCE_MM:
            cautions.extend(contact.problems)

        longer = self._longer_that_would_do(asset, defect)
        own_turns = plate_turns(asset.hole_centres_mm, asset.hole_axes)
        self.plate_verdict = judge_plate(
            holes,
            arcs,
            defect,
            (float(self.frames.s[0]), float(self.frames.s[-1])),
            self.plate_placement if self.plate_auto else None,
            self.plate_route if self.plate_auto else None,
            asset.width_mm,
            fatal,
            cautions,
            is_preformed=bool(np.abs(own_turns).max() > PREFORM_MIN_DEG) if len(own_turns) else False,
            hole_count=asset.hole_count,
            longer_holes=longer,
            notes=self.plate_course_notes if self.plate_auto else (),
        )

    def _longer_that_would_do(self, asset, defect) -> int | None:
        """The hole count of the shortest longer plate of this family that holds."""
        family = sorted(assets_in_family(asset.family), key=lambda a: a.hole_count)
        # Longer than the stock itself: a plate cut short can simply be kept longer.
        longer = [item for item in family if item.hole_count > max(asset.hole_count, self.plate_asset.hole_count)]
        if not longer:
            return None
        if self.plate_route is None or not self.plate_auto:
            return longer[0].hole_count
        system = system_for_asset(asset)
        choices = [
            (
                np.linalg.norm(np.diff(item.hole_centres_mm, axis=0), axis=1),
                plate_turns(item.hole_centres_mm, item.hole_axes),
                item.hole_count,
            )
            for item in longer
        ]
        return recommend_length(
            self.plate_route, choices, defect, max(int(system.min_holes_per_side), 1),
            from_neck=self.plate_course == "neck" and defect is not None,
        )

    def use_suggested_length(self) -> None:
        """Take the length the verdict suggests."""
        verdict, asset = self.plate_verdict, self.plate_asset
        if verdict is None or verdict.suggested_holes is None or asset is None:
            return
        for item in assets_in_family(asset.family):
            if item.hole_count == verdict.suggested_holes:
                self.set_plate_asset(item.id)
                return

    @property
    def active_plate_asset(self):
        """Selected stock after the distal cut, or the intact selected plate."""
        return self._trimmed_stock[0] if self._trimmed_stock else self.plate_asset

    def set_retained_holes(self, count: int) -> None:
        """How many holes of the stock stay, as the operator says: this overrides the
        automatic cut until another plate is chosen."""
        from ..render.plate_trim import trim_stock
        stock = self.plate_asset
        if stock is None:
            return
        first = min(self._trim_first, max(stock.hole_count - int(count), 0))
        try:
            self._trimmed_stock = None if count == stock.hole_count else trim_stock(stock, int(count), first)
        except (ValueError, MeshLoadError) as error:
            self.message.emit(str(error))
            self.plate_changed.emit()
            return
        self._trim_first = first if self._trimmed_stock else 0
        self._retained_by_hand = True
        self.plate_length_chosen = True
        self._marking_cache.clear()
        if self.plate_auto and self.place_plate_automatically(quiet=True):
            return
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
        self._update_fit()
        self.plate_changed.emit()

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
        self.fitted_hole_plan = None
        self.plate_verdict = None
        self.leave_guide_view()
        asset = self.active_plate_asset
        plan = self.plate_plan
        if asset is None or plan is None or len(plan.nodes) < 2:
            return
        try:
            mesh = self._trimmed_stock[1] if self._trimmed_stock else load_asset_mesh(asset)
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
        if self.bent_plate is not None:
            self.fitted_hole_plan = plan_at_holes(self.bent_plate.hole_centres, self.bent_plate.hole_axes, asset.hole_pitch_mm)
        self._judge_plate()

    def _marking_for(self, asset):
        """The asset's etched mark in its own space, built once per asset."""
        key = (asset.id, asset.hole_count, self._trim_first if self._trimmed_stock else 0)
        if key not in self._marking_cache:
            from ..render.marking import marking_in_plate_space

            points, triangles = marking_in_plate_space(self.plate_asset)
            if self._trimmed_stock and len(triangles):
                from ..render.plate_trim import keep_section
                mark = keep_section(Mesh(points, triangles), self.plate_asset.hole_centres_mm,
                                    self._trim_first, asset.hole_count, cap=False)
                points, triangles = mark.points, mark.triangles
            self._marking_cache[key] = points, triangles
        return self._marking_cache[key]

    def set_plate_bending(self, enabled: bool) -> None:
        """Bend the plate onto the path, or leave it rigidly placed."""
        self.plate_bending_enabled = bool(enabled)
        self.update_plate_fit()
        self._update_fit()
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
            if self.plate_projector is not None:
                paired, normals = seat_plate_chain(
                    self.plate_projector,
                    *self._plate_route,
                    np.linalg.norm(np.diff(fitted.hole_centres, axis=0), axis=1),
                    asset.thickness_mm,
                    self.plate_clearance_mm,
                    asset.width_mm,
                    hold_course_across=self.plate_auto and self._auto_course is not None,
                )
            else:
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
        self._update_fit()
        self.plate_changed.emit()

    def set_bending_insets(self, use: bool | None) -> None:
        """Whether bending insets are fitted into the threaded holes."""
        self.use_bending_insets = use
        self.update_plate_fit()
        self._update_fit()
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
        """The bend keys for this plate, as cut, and the bender chosen."""
        from ..geometry.bending_guide import plan_bending_guide

        asset, bent = self.active_plate_asset, self.bent_plate
        if asset is None or bent is None or bent.hole_tangents is None:
            self.message.emit(
                "Choose a plate with bending switched on before exporting a bending guide."
            )
            return None
        alloy = self.plate_alloy()
        thickness = asset.thickness_mm
        extra = self.guide_extra_overbend_deg
        # Plastic bending happens over the bridge between the rigid rings
        # round two holes; its radius sets how much the plate springs back.
        bridge = max(asset.hole_pitch_mm - 2.0 * asset.deformation.protected_radius_mm, 1.0)

        def overbend(angle_deg: float) -> float:
            if angle_deg <= 0.0:
                return 0.0
            radius = max(bridge / np.radians(angle_deg), alloy.min_bend_radius_mm(thickness))
            return alloy.overbend_deg(angle_deg, radius, thickness) + extra

        return plan_bending_guide(
            asset.hole_centres_mm,
            asset.hole_axes,
            bent.hole_centres,
            bent.hole_axes,
            bent.hole_tangents,
            asset.width_mm,
            thickness,
            asset.hole_diameter_mm,
            self.bending_kit,
            overbend=overbend,
            alloy_name=alloy.name,
            play_mm=self.guide_pin_play_mm,
            hole_length_mm=hole_length_mm(self.plate_asset),
            head=self.guide_head,
        )

    # -- rehearsing the bends on the keys ----------------------------------

    def enter_guide_view(self) -> bool:
        """Show the plate as it comes, and step through the keys that bend it."""
        try:
            guide = self.bending_guide()
        except ValueError as error:
            self.message.emit(str(error))
            return False
        if guide is None:
            return False
        self._guide = guide
        self._guide_mesh = {}
        self._guide_steps = guide.steps()
        self.guide_step = 0
        self.guide_view_changed.emit()
        return True

    def leave_guide_view(self) -> None:
        if self._guide is None:
            return
        self._guide = None
        self._guide_mesh = None
        self._guide_steps = []
        self.guide_view_changed.emit()

    @property
    def guide_view_on(self) -> bool:
        return self._guide is not None

    def set_guide_step(self, step: int) -> None:
        if self._guide is None:
            return
        self.guide_step = int(np.clip(step, 0, len(self._guide_steps) - 1))
        self.guide_view_changed.emit()

    @property
    def guide_step_text(self) -> str:
        if self._guide is None:
            return ""
        return self._guide_steps[self.guide_step].text

    @property
    def guide_step_count(self) -> int:
        return len(self._guide_steps)

    def guide_preview(self):
        """``(key mesh or None, head mesh or None, plate points, plate triangles)`` now."""
        if self._guide is None:
            return None
        from ..render.bending_guide_mesh import key_polydata, tool_polydata

        guide = self._guide
        asset = self.active_plate_asset
        mesh = self._trimmed_stock[1] if self._trimmed_stock else load_asset_mesh(asset)
        last = self.guide_step == len(self._guide_steps) - 1
        key = self._guide_steps[self.guide_step].key
        centres, axes = guide.stage(key, done=key is None and last)
        plate = bend_to_path(
            mesh.points,
            mesh.triangles,
            asset.hole_centres_mm,
            asset.hole_axes,
            centres,
            axes,
            protected_radius_mm=asset.deformation.protected_radius_mm,
            min_bend_radius_mm=asset.deformation.min_bend_radius_mm,
            max_bend_deg_per_node=asset.deformation.max_bend_deg_per_node,
        )
        if key is None:
            return None, None, plate.points, plate.triangles
        if key.number not in self._guide_mesh:
            self._guide_mesh[key.number] = (key_polydata(key, preview=True), tool_polydata(guide, key))
        solid, head = self._guide_mesh[key.number]
        return solid, head, plate.points, plate.triangles

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
        projector = self.plate_projector
        if projector is not None:
            # Verify each finished hole against fresh geometry, not the
            # interpolated click path that was used to fit it.
            faces = placed.hole_centres - 0.5 * asset.thickness_mm * placed.hole_axes
            projected = [projector.project(face) for face in faces]
            nodes = np.array([p for p, n in projected])
            # What the plate lies on, not one vertex of a rough surface.
            radius = max(asset.width_mm / 2.0, 2.0)
            normals = np.array([projector.footprint_normal(p, radius) for p, _ in projected])
        self.plate_contact = clearance_report(
            placed.hole_centres,
            placed.hole_axes,
            nodes,
            normals,
            asset.thickness_mm,
            target_clearance_mm=self.plate_clearance_mm,
            matched=projector is not None,
            max_gap_mm=self.plate_clearance_mm + 1.0,
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
            if self.plate_auto and self._auto_course is not None:
                # The jaw's own border, already fair: followed, not smoothed again.
                path, normals = fair_plate_path(*self._auto_course, smoothing_mm=0.5)
            else:
                path, normals = fair_plate_path(
                    np.vstack(self.plate_points), np.vstack(self.plate_normals),
                    smoothing_mm=max(3.0, self.plate.pitch_mm * 0.55),
                )
            self._plate_route = (path, normals)
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
        if self.plate_auto or self.plate_length_chosen or fit is None or fit.option is None or asset is None:
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
                plate_auto=self.plate_auto,
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
        self.plate_auto = state.plate_auto
        # The route along the jaw is the newer placement's: the restored holes stand.
        self._auto_course = None
        if not state.cut_applied:
            self.fragment_surface = None
            self.retained_surface = None
        self.cut_applied = state.cut_applied
        self._rebuild_arch()
        self.update_resection_report()
        self.update_plate_plan()
