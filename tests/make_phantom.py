"""Synthetic CBCT-like phantom with analytically known geometry.

There is no patient CBCT in this repository, so every accuracy claim is
checked against a volume whose geometry is generated from a closed-form
curve.  A "mandible" is an elliptical cross-section (semi-axes ``a``
buccolingual and ``b`` superior-inferior) swept along a circular arc of
radius ``R`` lying in an axial plane.  That gives exactly known

    arc length              L      = R * dtheta
    cross-section extents   2a, 2b
    turn angle per node     delta  = pitch / R
    solid volume            V      = pi * a * b * R * dtheta   (Pappus)

The intensity profile is a linear ramp across the surface, so the half-maximum
isosurface coincides with the analytic surface to sub-voxel accuracy and
surface-derived measurements are not biased by the choice of threshold.

Run directly to write a DICOM series you can open in the app::

    python tests/make_phantom.py [output_dir]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mandiplan.geometry.volume import Volume  # noqa: E402


@dataclass(frozen=True)
class PhantomSpec:
    """Parameters of the analytic phantom.  All lengths in millimetres."""

    radius_mm: float = 32.0
    theta_start_deg: float = -70.0
    theta_end_deg: float = 70.0
    semi_axis_bl_mm: float = 6.0  # a, buccolingual
    semi_axis_si_mm: float = 9.0  # b, superior-inferior
    centre_xy: tuple[float, float] = (0.0, 0.0)
    centre_z: float = 0.0
    spacing: tuple[float, float, float] = (0.3, 0.3, 0.6)
    margin_mm: float = 8.0
    air_value: float = 100.0
    bone_value: float = 1600.0
    edge_mm: float | None = None  # default: two voxels of the coarsest axis
    noise_sigma: float = 25.0
    seed: int = 0

    @property
    def theta_span_rad(self) -> float:
        return np.radians(self.theta_end_deg - self.theta_start_deg)

    @property
    def arc_length_mm(self) -> float:
        """Analytic arc length of the swept centre-line."""
        return self.radius_mm * self.theta_span_rad

    @property
    def volume_mm3(self) -> float:
        """Analytic volume of the swept solid (Pappus's centroid theorem)."""
        return (
            np.pi
            * self.semi_axis_bl_mm
            * self.semi_axis_si_mm
            * self.radius_mm
            * self.theta_span_rad
        )

    @property
    def half_max_value(self) -> float:
        return 0.5 * (self.air_value + self.bone_value)

    @property
    def ramp_mm(self) -> float:
        return self.edge_mm if self.edge_mm is not None else 2.0 * max(self.spacing)

    def centre_line(self, n: int = 512, extend_deg: float = 0.0) -> np.ndarray:
        """Points on the analytic centre-line of the sweep.

        ``extend_deg`` continues the same circle past both ends of the bone,
        which is what a user does when they draw the arch curve across the
        whole field of view rather than stopping at the region of interest.
        """
        theta = np.linspace(
            np.radians(self.theta_start_deg - extend_deg),
            np.radians(self.theta_end_deg + extend_deg),
            n,
        )
        cx, cy = self.centre_xy
        return np.column_stack(
            [
                cx + self.radius_mm * np.cos(theta),
                cy + self.radius_mm * np.sin(theta),
                np.full_like(theta, self.centre_z),
            ]
        )

    def point_at_arc(self, s_mm: float) -> np.ndarray:
        theta = np.radians(self.theta_start_deg) + s_mm / self.radius_mm
        cx, cy = self.centre_xy
        return np.array(
            [
                cx + self.radius_mm * np.cos(theta),
                cy + self.radius_mm * np.sin(theta),
                self.centre_z,
            ]
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        pts = self.centre_line(1024)
        r_out = self.radius_mm + self.semi_axis_bl_mm
        r_in = self.radius_mm - self.semi_axis_bl_mm
        cx, cy = self.centre_xy
        theta = np.linspace(
            np.radians(self.theta_start_deg), np.radians(self.theta_end_deg), 1024
        )
        xs = np.concatenate([cx + r_out * np.cos(theta), cx + r_in * np.cos(theta)])
        ys = np.concatenate([cy + r_out * np.sin(theta), cy + r_in * np.sin(theta)])
        lo = np.array(
            [
                xs.min() - self.margin_mm,
                ys.min() - self.margin_mm,
                self.centre_z - self.semi_axis_si_mm - self.margin_mm,
            ]
        )
        hi = np.array(
            [
                xs.max() + self.margin_mm,
                ys.max() + self.margin_mm,
                self.centre_z + self.semi_axis_si_mm + self.margin_mm,
            ]
        )
        _ = pts
        return lo, hi


def signed_distance_field(spec: PhantomSpec, x, y, z) -> np.ndarray:
    """First-order signed distance to the phantom surface (negative inside)."""
    cx, cy = spec.centre_xy
    dx, dy = x - cx, y - cy
    rho = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)

    dr = rho - spec.radius_mm
    dz = z - spec.centre_z
    a, b = spec.semi_axis_bl_mm, spec.semi_axis_si_mm
    q = (dr / a) ** 2 + (dz / b) ** 2
    grad = np.sqrt((2 * dr / a**2) ** 2 + (2 * dz / b**2) ** 2)
    tube = np.where(grad > 1e-12, (q - 1.0) / np.maximum(grad, 1e-12), -min(a, b))

    t0 = np.radians(spec.theta_start_deg)
    t1 = np.radians(spec.theta_end_deg)
    # Distance to the nearer end cap, as arc length at radius rho: negative
    # inside the angular wedge, positive beyond either end.
    wedge = np.maximum(t0 - theta, theta - t1) * rho

    return np.maximum(tube, wedge)


def make_phantom(spec: PhantomSpec = PhantomSpec()) -> Volume:
    """Build the phantom volume for ``spec``."""
    lo, hi = spec.bounds()
    spacing = np.asarray(spec.spacing, dtype=float)
    size = np.maximum(np.ceil((hi - lo) / spacing).astype(int) + 1, 2)

    x = lo[0] + np.arange(size[0]) * spacing[0]
    y = lo[1] + np.arange(size[1]) * spacing[1]
    z = lo[2] + np.arange(size[2]) * spacing[2]
    xx = x[None, None, :]
    yy = y[None, :, None]
    zz = z[:, None, None]

    sdf = signed_distance_field(spec, xx, yy, zz)
    ramp = spec.ramp_mm
    frac = np.clip(0.5 - sdf / ramp, 0.0, 1.0)
    array = spec.air_value + (spec.bone_value - spec.air_value) * frac

    if spec.noise_sigma > 0:
        rng = np.random.default_rng(spec.seed)
        array = array + rng.normal(0.0, spec.noise_sigma, size=array.shape)

    return Volume(array=array.astype(np.float32), spacing=spacing, origin=lo)


# --------------------------------------------------------------------------
# DICOM export, so the phantom can be opened through the real loading path.
# --------------------------------------------------------------------------


def write_dicom_series(
    volume: Volume, out_dir: str | Path, patient_id: str = "PHANTOM"
) -> Path:
    """Write ``volume`` as a CT DICOM series (one file per axial slice)."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.dcm"):
        old.unlink()

    study_uid = generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()

    data = np.asarray(volume.array, dtype=np.float64)
    intercept = float(np.floor(data.min()))
    slope = max((float(data.max()) - intercept) / 32000.0, 1e-3)
    stored = np.rint((data - intercept) / slope).astype(np.int16)

    sx, sy, sz = (float(v) for v in volume.spacing)
    ox, oy, oz = (float(v) for v in volume.origin)
    nz, ny, nx = volume.array.shape

    for k in range(nz):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()

        ds = Dataset()
        ds.file_meta = meta

        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.PatientName = "MandiPlan^Phantom"
        ds.PatientID = patient_id
        ds.Modality = "CT"
        ds.SeriesDescription = "MandiPlan analytic phantom"
        ds.StudyDate = "20200101"
        ds.SeriesNumber = 1
        ds.InstanceNumber = k + 1

        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [ox, oy, oz + k * sz]
        ds.PixelSpacing = [sy, sx]  # [row spacing, column spacing]
        ds.SliceThickness = sz
        ds.SpacingBetweenSlices = sz
        ds.SliceLocation = oz + k * sz

        ds.Rows = ny
        ds.Columns = nx
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1
        ds.RescaleIntercept = intercept
        ds.RescaleSlope = slope
        ds.RescaleType = "US"
        ds.PixelData = stored[k].tobytes()

        pydicom.dcmwrite(out_dir / f"slice_{k:04d}.dcm", ds, enforce_file_format=True)

    return out_dir


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else Path("build/phantom_dicom")
    spec = PhantomSpec()
    volume = make_phantom(spec)
    write_dicom_series(volume, out)
    print(f"phantom written to {out.resolve()}")
    print(f"  size          {tuple(int(v) for v in volume.size_xyz)} voxels (x, y, z)")
    print(f"  spacing       {tuple(volume.spacing)} mm")
    print(f"  arc length    {spec.arc_length_mm:.3f} mm")
    print(f"  cross-section {2 * spec.semi_axis_bl_mm:.1f} x {2 * spec.semi_axis_si_mm:.1f} mm")
    print(f"  solid volume  {spec.volume_mm3:.1f} mm^3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
