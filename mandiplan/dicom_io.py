"""DICOM series discovery, validation and loading.

Two failure modes matter more than anything else this module does:

* a gantry-tilted or sheared acquisition read as if it were a plain stack,
* inconsistent slice spacing collapsed onto a single nominal spacing.

Either silently distorts the volume, and a distorted volume makes every
millimetre the application reports wrong.  Both are detected here and refused
with an explanatory message rather than corrected in place.
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


class DicomLoadError(RuntimeError):
    """Raised when a series cannot be loaded as an undistorted volume."""


@dataclass
class SeriesInfo:
    series_uid: str
    description: str
    modality: str
    files: list[str]

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
        desc, modality = _series_labels(files[0])
        found.append(
            SeriesInfo(series_uid=uid, description=desc, modality=modality, files=files)
        )

    if not found:
        # GetGDCMSeriesIDs only looks at one directory level; walk sub-folders.
        for sub in sorted(p for p in Path(folder).rglob("*") if p.is_dir()):
            found.extend(list_series(sub))
    return found


def _series_labels(path: str) -> tuple[str, str]:
    import pydicom

    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    return str(getattr(ds, "SeriesDescription", "")), str(getattr(ds, "Modality", ""))


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
    array: np.ndarray, spacing: np.ndarray, origin: np.ndarray, direction: np.ndarray
) -> Volume:
    """Transpose/flip an axis-aligned array so index axes follow world +x, +y, +z.

    ``array`` is indexed ``[axis2, axis1, axis0]`` (SimpleITK convention) and
    ``direction`` is the 3x3 matrix whose column ``a`` is the world direction
    of image axis ``a``.
    """
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

    return Volume(
        array=np.ascontiguousarray(out, dtype=np.float32),
        spacing=new_spacing,
        origin=new_origin,
    )


def load_series(files: list[str]) -> tuple[Volume, SeriesGeometry]:
    """Validate and load a DICOM series into an LPS-aligned :class:`Volume`."""
    import SimpleITK as sitk

    geometry = inspect_geometry(files)

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(list(files))
    image = reader.Execute()

    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    return _reorient_to_lps(array, spacing, origin, direction), geometry


def load_folder(
    folder: str | Path, series_uid: str | None = None
) -> tuple[Volume, SeriesGeometry, SeriesInfo]:
    """Load one series from ``folder`` (the largest one unless ``series_uid`` is given)."""
    series = list_series(folder)
    if not series:
        raise DicomLoadError(f"No DICOM series found in {folder}.")
    if series_uid is not None:
        matches = [s for s in series if s.series_uid == series_uid]
        if not matches:
            raise DicomLoadError(f"Series {series_uid} is not in {folder}.")
        chosen = matches[0]
    else:
        chosen = max(series, key=lambda s: s.n_files)
    volume, geometry = load_series(chosen.files)
    return volume, geometry, chosen
