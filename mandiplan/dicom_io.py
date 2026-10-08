"""DICOM series discovery, validation and loading.

Two failure modes matter more than anything else this module does:

* a gantry-tilted or sheared acquisition read as if it were a plain stack,
* inconsistent slice spacing collapsed onto a single nominal spacing.

Either silently distorts the volume, and a distorted volume makes every
millimetre the application reports wrong.  Both are detected here and refused
with an explanatory message rather than corrected in place.

One change is made on purpose, and announced: a scan too large to hold in
memory is averaged over whole blocks of voxels (2x2x2, 3x3x3, ...) into a
coarser working grid. A full-head CBCT at 0.25 mm is 300 million voxels, some
8 GB once contoured, and would take the application down on most machines.
Block averaging keeps every world coordinate exact; only the voxel size
changes, and it is reported with the load.

A medical CT is handled in two parts, because it may show the whole body
and is usually reconstructed in slices several times thicker than its
pixels. The whole scan is kept, as any other, and is what the slices and the
first 3-D view show. The jaws are also found in it (``geometry/head_region.py``)
and a box round them is cut from the scan at its full resolution and
resampled to cubic voxels with cubic interpolation: marching cubes on 3 mm
slices draws a bone surface in 3 mm terraces, and on cubic voxels it draws it
smooth. The mandible is separated and drawn from that box
(``SeriesGeometry.jaw_volume``). Both are in the same world millimetres; the
box is announced with the load.

A folder often holds several series of one study: the scout, thin and thick
reconstructions, reformats in other planes, dose reports. The one loaded by
default is the one best suited to the bone (see :func:`choose_series`), not
merely the one with the most files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry.volume import Volume

# A slice-to-slice displacement may deviate from the slice normal by at most
# this angle before the stack is treated as sheared (gantry tilt).
_MAX_SHEAR_DEG = 0.5
# Slice spacing must be consistent to within this fraction of the median.
_MAX_SPACING_VARIATION = 0.01
_MIN_SPACING_TOLERANCE_MM = 0.01
# Direction cosines must be this close to a signed axis permutation.
_MIN_AXIS_ALIGNMENT = 0.999
#: Cubic voxel size a CT's jaw box is resampled to, mm: its own pixel size,
#: but no coarser than the upper bound (a CBCT's) and no finer than the lower.
CT_VOXEL_MM = (0.5, 0.75)
#: A CT longer than this shows more than a head, mm, and is cut down to the
#: jaws. A CBCT never is: its field of view is the jaws.
HEAD_LENGTH_MM = 260.0
#: Voxels coarser than this along any axis, mm, step the bone surface
#: visibly: the jaw box is resampled to cubic voxels.
SMOOTH_VOXEL_MM = 1.0
#: Largest working volume, in voxels, before it is block-averaged. 80 M voxels
#: is 320 MB as float32, which leaves room for the surface, the reformats and
#: the reconstruction on an 8 GB machine.
WORKING_VOXEL_BUDGET = 80_000_000


class DicomLoadError(RuntimeError):
    """Raised when a series cannot be loaded as an undistorted volume."""


@dataclass
class SeriesInfo:
    series_uid: str
    description: str
    modality: str
    files: list[str]
    #: DICOM ImageType values, e.g. ("ORIGINAL", "PRIMARY", "AXIAL").
    image_type: tuple[str, ...] = ()

    @property
    def n_files(self) -> int:
        return len(self.files)

    def __str__(self) -> str:
        desc = self.description or "(no description)"
        return f"{desc} [{self.modality}, {self.n_files} slices]"


@dataclass
class SeriesGeometry:
    """Header-derived geometry of a series, before pixel data is read."""

    slice_spacing_mm: float
    in_plane_spacing_mm: tuple[float, float]
    orientation: np.ndarray  # (2, 3) row and column direction cosines
    normal: np.ndarray  # (3,) slice normal
    gantry_tilt_deg: float
    max_shear_deg: float
    spacing_variation: float
    warnings: list[str] = field(default_factory=list)
    #: Voxels averaged per axis to make the working grid (1 = native).
    working_factor: int = 1
    #: True when this is a medical CT whose jaws were found (see ``_ct_jaws``).
    jaw_box: bool = False
    #: For such a CT: the box round the jaws, cut from the scan at its full
    #: resolution and made of cubic voxels, in the scan's world millimetres.
    #: The mandible is separated and drawn from it; the whole scan is what
    #: the slices show. None for any other scan.
    jaw_volume: Volume | None = None


def list_series(folder: str | Path) -> list[SeriesInfo]:
    """Every DICOM series found under ``folder`` (recursively)."""
    import SimpleITK as sitk

    folder = str(folder)
    reader = sitk.ImageSeriesReader()
    uids = reader.GetGDCMSeriesIDs(folder)
    if not uids:
        uids = tuple()

    found: list[SeriesInfo] = []
    for uid in uids:
        files = list(reader.GetGDCMSeriesFileNames(folder, uid))
        if not files:
            continue
        desc, modality, image_type = _series_labels(files[0])
        found.append(
            SeriesInfo(
                series_uid=uid,
                description=desc,
                modality=modality,
                files=files,
                image_type=image_type,
            )
        )

    if not found:
        # GetGDCMSeriesIDs only looks at one directory level; walk sub-folders.
        for sub in sorted(p for p in Path(folder).rglob("*") if p.is_dir()):
            found.extend(list_series(sub))
    return found


def _series_labels(path: str) -> tuple[str, str, tuple[str, ...]]:
    import pydicom

    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    image_type = getattr(ds, "ImageType", ())
    if isinstance(image_type, str):
        image_type = image_type.split("\\")
    return (
        str(getattr(ds, "SeriesDescription", "")),
        str(getattr(ds, "Modality", "")),
        tuple(str(v).strip().upper() for v in image_type),
    )


#: Image types that are pictures of the patient rather than a volume.
_NOT_A_VOLUME = ("LOCALIZER", "SCOUT", "TOPOGRAM", "MIP", "MINIP", "VR", "SSD", "DRR")
#: Slice spacings within this of each other are the same, mm.
_SPACING_TIE_MM = 0.05


def _series_rank(series: SeriesInfo):
    """Sort key for :func:`choose_series`; None if the series is not a volume.

    Read from the first and last slice headers only, so ranking a study of
    many series stays quick.
    """
    import pydicom

    if series.n_files < 2 or any(t in series.image_type for t in _NOT_A_VOLUME):
        return None
    try:
        first, last = (
            pydicom.dcmread(path, stop_before_pixels=True, force=True)
            for path in (series.files[0], series.files[-1])
        )
        orientation = np.asarray([float(v) for v in first.ImageOrientationPatient])
        spacing = [float(v) for v in first.PixelSpacing]
        steps = np.asarray([float(v) for v in last.ImagePositionPatient]) - np.asarray(
            [float(v) for v in first.ImagePositionPatient]
        )
    except (AttributeError, TypeError, ValueError):
        return None
    normal = np.cross(orientation[:3], orientation[3:])
    if np.linalg.norm(normal) < 0.5:
        return None
    normal /= np.linalg.norm(normal)
    length = abs(float(steps @ normal))
    slice_mm = length / (series.n_files - 1)
    if slice_mm <= 0:
        return None
    return (
        series.modality.upper() == "CT",
        # Every scanner acquires axially; a coronal or sagittal series is a
        # reformat of that, resampled once already.
        abs(float(normal[2])) > 0.9,
        # The finest slices resolve the bite and the joint spaces best.
        -round(slice_mm / _SPACING_TIE_MM),
        series.image_type[:1] == ("ORIGINAL",),
        -round(min(spacing), 3),
        length,
        series.n_files,
    )


def choose_series(series: list[SeriesInfo]) -> SeriesInfo | None:
    """The series to plan on, of several in one folder.

    A CT study usually holds a scout, a dose report, thin and thick axial
    reconstructions, and coronal and sagittal reformats of them; the reformats
    can hold the most files. The one chosen is an axial volume (the plane the
    scanner acquired in), with the finest slices, an original reconstruction
    before a derived one, then the finest pixels and the longest coverage.
    None when no series is a volume.
    """
    ranked = [(rank, s) for s in series if (rank := _series_rank(s)) is not None]
    if not ranked:
        return None
    return max(ranked, key=lambda item: item[0])[1]


def inspect_geometry(files: list[str]) -> SeriesGeometry:
    """Read slice headers and check the stack is a true rectilinear volume.

    Raises :class:`DicomLoadError` with a plain-language explanation when the
    series cannot be represented without distortion.
    """
    import pydicom

    if len(files) < 2:
        raise DicomLoadError(
            "This series has fewer than 2 slices; MandiPlan needs a volume."
        )

    positions = []
    orientations = []
    tilts = []
    pixel_spacings = []
    for path in files:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        if not hasattr(ds, "ImagePositionPatient") or not hasattr(
            ds, "ImageOrientationPatient"
        ):
            raise DicomLoadError(
                f"{Path(path).name} has no ImagePositionPatient/ImageOrientationPatient; "
                "the slice geometry is unknown and cannot be used for measurement."
            )
        positions.append([float(v) for v in ds.ImagePositionPatient])
        orientations.append([float(v) for v in ds.ImageOrientationPatient])
        if hasattr(ds, "GantryDetectorTilt"):
            tilts.append(abs(float(ds.GantryDetectorTilt)))
        if hasattr(ds, "PixelSpacing"):
            pixel_spacings.append([float(v) for v in ds.PixelSpacing])

    positions = np.asarray(positions)
    orientations = np.asarray(orientations)

    if np.max(np.abs(orientations - orientations[0])) > 1e-4:
        raise DicomLoadError(
            "Slice orientation is not constant across this series. "
            "MandiPlan will not resample a variable-orientation stack, because "
            "doing so silently would distort every measurement."
        )

    row = orientations[0, :3]
    col = orientations[0, 3:]
    normal = np.cross(row, col)
    normal /= np.linalg.norm(normal)

    gantry_tilt = max(tilts) if tilts else 0.0
    if gantry_tilt > 0.1:
        raise DicomLoadError(
            f"Gantry/detector tilt is {gantry_tilt:.2f}°. A tilted acquisition read "
            "as a plain stack is sheared, so MandiPlan refuses it. Re-export the "
            "series with tilt correction applied."
        )

    proj = positions @ normal
    order = np.argsort(proj)
    positions = positions[order]
    proj = proj[order]

    steps = np.diff(proj)
    if np.any(steps <= 0):
        raise DicomLoadError(
            "Two or more slices share the same position along the slice normal; "
            "the series appears to contain duplicates or interleaved acquisitions."
        )
    median_step = float(np.median(steps))
    variation = float(np.max(np.abs(steps - median_step)))
    tolerance = max(_MIN_SPACING_TOLERANCE_MM, _MAX_SPACING_VARIATION * median_step)
    if variation > tolerance:
        raise DicomLoadError(
            f"Slice spacing is not uniform: it varies by {variation:.3f} mm about a "
            f"median of {median_step:.3f} mm (tolerance {tolerance:.3f} mm). "
            "A non-uniform stack cannot be represented as a regular volume without "
            "resampling, which MandiPlan will not do silently."
        )

    displacement = np.diff(positions, axis=0)
    lengths = np.linalg.norm(displacement, axis=1)
    cosines = np.clip((displacement @ normal) / lengths, -1.0, 1.0)
    shear = float(np.max(np.degrees(np.arccos(cosines))))
    if shear > _MAX_SHEAR_DEG:
        raise DicomLoadError(
            f"Slice positions step {shear:.2f}° away from the slice normal, i.e. the "
            "stack is sheared (gantry tilt or a tilted reconstruction). MandiPlan "
            "refuses it rather than produce a distorted volume."
        )

    warnings: list[str] = []
    if pixel_spacings:
        ps = np.asarray(pixel_spacings)
        if np.max(np.abs(ps - ps[0])) > 1e-4:
            raise DicomLoadError("In-plane pixel spacing is not constant across slices.")
        in_plane = (float(ps[0][1]), float(ps[0][0]))  # (column, row) -> (x, y)
    else:
        raise DicomLoadError("PixelSpacing is missing; the volume has no scale.")

    if abs(median_step - in_plane[0]) > 1e-3 or abs(median_step - in_plane[1]) > 1e-3:
        warnings.append(
            f"Anisotropic voxels: {in_plane[0]:.3f} × {in_plane[1]:.3f} × "
            f"{median_step:.3f} mm."
        )

    return SeriesGeometry(
        slice_spacing_mm=median_step,
        in_plane_spacing_mm=in_plane,
        orientation=np.vstack([row, col]),
        normal=normal,
        gantry_tilt_deg=gantry_tilt,
        max_shear_deg=shear,
        spacing_variation=variation,
        warnings=warnings,
    )


def _reorient_to_lps(
    array: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    factor: int = 1,
) -> Volume:
    """Transpose/flip an axis-aligned array so index axes follow world +x, +y, +z.

    ``array`` is indexed ``[axis2, axis1, axis0]`` (SimpleITK convention) and
    ``direction`` is the 3x3 matrix whose column ``a`` is the world direction
    of image axis ``a``. With ``factor > 1`` the result is averaged over
    ``factor``³ blocks on the way, without a full-resolution float copy.
    """
    out, new_spacing, new_origin = _lps_view(array, spacing, origin, direction)
    if factor > 1:
        data = block_average(out, factor)
        # A block's value belongs at the centre of the voxels it averaged.
        new_origin = new_origin + 0.5 * (factor - 1) * new_spacing
        new_spacing = new_spacing * factor
    else:
        data = np.ascontiguousarray(out, dtype=np.float32)
    return Volume(array=data, spacing=new_spacing, origin=new_origin)


def _lps_view(array, spacing, origin, direction):
    """``(view, spacing, origin)``: the array as indexed ``[z, y, x]`` in
    patient axes, as a view (no copy), after refusing oblique series."""
    world_axis = np.argmax(np.abs(direction), axis=0)  # per image axis
    if sorted(world_axis.tolist()) != [0, 1, 2]:
        raise DicomLoadError(
            "The series axes do not map one-to-one onto the patient axes; "
            "the acquisition is oblique and MandiPlan will not resample it."
        )
    signs = np.sign(direction[world_axis, np.arange(3)])
    alignment = np.abs(direction[world_axis, np.arange(3)])
    if np.any(alignment < _MIN_AXIS_ALIGNMENT):
        worst = float(np.degrees(np.arccos(alignment.min())))
        raise DicomLoadError(
            f"The volume axes are {worst:.2f}° off the patient axes (oblique "
            "acquisition). MandiPlan refuses it rather than measure a skewed volume."
        )

    image_axis_for_world = {int(world_axis[a]): a for a in range(3)}
    # numpy axis for image axis a is (2 - a)
    perm = [2 - image_axis_for_world[w] for w in (2, 1, 0)]  # -> (z, y, x)
    out = np.transpose(array, perm)

    new_spacing = np.empty(3)
    new_origin = np.empty(3)
    flips = []
    for w in range(3):
        a = image_axis_for_world[w]
        n = array.shape[2 - a]
        sp = float(spacing[a])
        new_spacing[w] = sp
        if signs[a] >= 0:
            new_origin[w] = origin[w]
        else:
            new_origin[w] = origin[w] - (n - 1) * sp
            flips.append(w)
    for w in flips:
        out = np.flip(out, axis=2 - w)  # world x,y,z -> numpy axis 2,1,0
    return out, new_spacing, new_origin


def block_average(array: np.ndarray, factor: int) -> np.ndarray:
    """Mean over ``factor``³ blocks, as float32; trailing partial blocks are dropped.

    Works one output slice at a time so the full-resolution data is never
    converted to float in one piece.
    """
    nz, ny, nx = (int(v) // factor for v in array.shape)
    if min(nz, ny, nx) < 2:
        raise DicomLoadError("The series is too small to reduce to a working grid.")
    out = np.empty((nz, ny, nx), dtype=np.float32)
    for k in range(nz):
        slab = np.asarray(
            array[k * factor : (k + 1) * factor, : ny * factor, : nx * factor],
            dtype=np.float32,
        )
        out[k] = slab.reshape(factor, ny, factor, nx, factor).mean(axis=(0, 2, 4))
    return out


def working_factor(shape, budget: int = WORKING_VOXEL_BUDGET) -> int:
    """Smallest whole-voxel block size that brings ``shape`` within ``budget``."""
    voxels = float(np.prod([int(v) for v in shape]))
    factor = 1
    while voxels / factor**3 > budget:
        factor += 1
    return factor


def load_series(
    files: list[str], voxel_budget: int = WORKING_VOXEL_BUDGET, modality: str = ""
) -> tuple[Volume, SeriesGeometry]:
    """Validate and load a DICOM series into an LPS-aligned :class:`Volume`.

    The volume is always the whole series. For a medical CT
    (``modality == "CT"``) the box round its jaws is made too, in cubic
    voxels, as ``geometry.jaw_volume``; see the module docstring.
    """
    import SimpleITK as sitk

    geometry = inspect_geometry(files)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(list(files))
    image = reader.Execute()

    # A view in the stored type: no full-resolution float copy is made.
    array = sitk.GetArrayViewFromImage(image)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    # Not every "CT" is a medical CT: many CBCT scanners write that modality
    # too. What matters is whether the scan shows more than the head, or has
    # voxels coarse enough to step the bone surface; a CBCT does neither.
    length = _length_mm(array.shape, spacing, direction)
    if modality.upper() == "CT" and (length > HEAD_LENGTH_MM or spacing.max() > SMOOTH_VOXEL_MM):
        geometry.jaw_volume = _ct_jaws(array, spacing, origin, direction, geometry, voxel_budget)
        geometry.jaw_box = geometry.jaw_volume is not None
    factor = working_factor(array.shape, voxel_budget)
    volume = _reorient_to_lps(array, spacing, origin, direction, factor)
    del array, image
    if factor > 1:
        native = float(np.min(spacing))
        geometry.working_factor = factor
        geometry.warnings.append(
            f"This scan is {np.prod(volume.size_xyz) * factor**3 / 1e6:.0f} million "
            f"voxels at {native:.2f} mm, more than fits in memory for planning. "
            f"MandiPlan works on it at {float(volume.spacing.min()):.2f} mm "
            f"({np.prod(volume.size_xyz) / 1e6:.0f} million voxels), each the average "
            f"of {factor}×{factor}×{factor} scan voxels. Distances are still in "
            "millimetres from the DICOM spacing."
        )
    return volume, geometry


def _length_mm(shape, spacing, direction) -> float:
    """Head-to-foot extent of a series, mm."""
    direction = np.asarray(direction, dtype=float).reshape(3, 3)
    axis = int(np.argmax(np.abs(direction[2])))  # image axis along world z
    return float((shape[2 - axis] - 1) * spacing[axis])


def _ct_jaws(array, spacing, origin, direction, geometry, voxel_budget) -> Volume | None:
    """The box round the jaws of a CT, in cubic voxels; None if the jaws are
    not found (the mandible is then separated from the whole scan)."""
    from .geometry.head_region import coarse_view, find_jaw_region

    view, spacing, origin = _lps_view(array, spacing, origin, direction)
    region = find_jaw_region(coarse_view(view, spacing, origin))
    if region is None:
        geometry.warnings.append(
            "The jaws could not be found in this CT on their own. If the mandible "
            "is not found automatically either, draw the arch curve on the axial "
            "slice (step 2)."
        )
        return None
    shape = np.array(view.shape[::-1])  # x, y, z
    lo = np.clip(np.floor((region.lo - origin) / spacing).astype(int), 0, shape - 1)
    hi = np.clip(np.ceil((region.hi - origin) / spacing).astype(int) + 1, lo + 2, shape)
    crop = np.ascontiguousarray(view[lo[2] : hi[2], lo[1] : hi[1], lo[0] : hi[0]], dtype=np.float32)
    crop_origin = origin + lo * spacing
    extent = (hi - lo - 1) * spacing

    found = (
        f"The jaws are a {extent[0]:.0f} × {extent[1]:.0f} × {extent[2]:.0f} mm box "
        "of this CT; the mandible is separated and drawn from that box, at the "
        "scan's full resolution"
    )
    if spacing.max() <= SMOOTH_VOXEL_MM:
        geometry.warnings.append(found + ". Distances are in millimetres from the DICOM spacing.")
        return Volume(array=crop, spacing=spacing.copy(), origin=crop_origin)

    voxel = float(np.clip(spacing[:2].min(), *CT_VOXEL_MM))
    while np.prod(np.floor(extent / voxel) + 1) > voxel_budget:
        voxel *= 1.1
    volume = _resample_cubic(crop, spacing, crop_origin, voxel)
    geometry.warnings.append(
        found + f", resampled from {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm "
        f"voxels to {voxel:.2f} mm cubes so its surface is smooth rather than stepped. "
        "Distances are in millimetres from the DICOM spacing."
    )
    return volume


def _resample_cubic(array: np.ndarray, spacing, origin, voxel: float) -> Volume:
    """``array`` resampled to cubic ``voxel``-sized voxels, cubic B-spline."""
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(array)
    image.SetSpacing([float(v) for v in spacing])
    image.SetOrigin([float(v) for v in origin])
    extent = (np.array(array.shape[::-1]) - 1) * np.asarray(spacing, dtype=float)
    size = [int(n) for n in np.floor(extent / voxel) + 1]
    reference = sitk.Image(size, sitk.sitkFloat32)
    reference.SetSpacing([voxel] * 3)
    reference.SetOrigin([float(v) for v in origin])
    out = sitk.Resample(
        image, reference, sitk.Transform(), sitk.sitkBSpline, float(array.min()), sitk.sitkFloat32
    )
    # The cubic overshoots a little beside metal; keep within the scan's range.
    data = np.clip(sitk.GetArrayFromImage(out), float(array.min()), float(array.max()))
    return Volume(
        array=data,
        spacing=np.full(3, voxel),
        origin=np.asarray(origin, dtype=float),
        resampled_from_mm=float(np.max(spacing)),
    )


def load_folder(
    folder: str | Path,
    series_uid: str | None = None,
    voxel_budget: int = WORKING_VOXEL_BUDGET,
) -> tuple[Volume, SeriesGeometry, SeriesInfo]:
    """Load one series from ``folder``: ``series_uid``, or else the one
    :func:`choose_series` picks."""
    series = list_series(folder)
    if not series:
        raise DicomLoadError(f"No DICOM series found in {folder}.")
    if series_uid is not None:
        matches = [s for s in series if s.series_uid == series_uid]
        if not matches:
            raise DicomLoadError(f"Series {series_uid} is not in {folder}.")
        chosen = matches[0]
    else:
        # A folder of nothing but single images still gets inspect_geometry's
        # plain explanation of why it cannot be used.
        chosen = choose_series(series) or max(series, key=lambda s: s.n_files)
    volume, geometry = load_series(chosen.files, voxel_budget, chosen.modality)
    volumes = [s for s in series if s is not chosen and _series_rank(s) is not None]
    if series_uid is None and volumes:
        geometry.warnings.insert(
            0,
            f"This folder holds {len(volumes) + 1} scan series; MandiPlan opened "
            f"\"{chosen.description or chosen.series_uid}\", whose "
            f"{geometry.slice_spacing_mm:.1f} mm slices resolve the bone best.",
        )
    return volume, geometry, chosen
