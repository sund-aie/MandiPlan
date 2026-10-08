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
    arch_centre_deg: float = -90.0  # chin direction: -90 deg is anterior in LPS
    half_span_deg: float = 70.0
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
    def theta_start_deg(self) -> float:
        return self.arch_centre_deg - self.half_span_deg

    @property
    def theta_end_deg(self) -> float:
        return self.arch_centre_deg + self.half_span_deg

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


# --------------------------------------------------------------------------
# A CT of the whole body, to test finding the jaws in one.
# --------------------------------------------------------------------------

#: Hounsfield-like values of the body CT.
CT_AIR, CT_SOFT, CT_BONE = -1000.0, 40.0, 1200.0


def make_body_ct(
    spec: PhantomSpec = PhantomSpec(),
    spacing: tuple[float, float, float] = (1.2, 1.2, 4.0),
    foot_mm: float = -600.0,
) -> Volume:
    """A head-to-pelvis CT in thick slices, with the phantom mandible in it.

    Soft tissue in the shape of a head, neck and torso; a skull vault round
    a brain above the mandible; a spine down the back; ribs, which never
    close a ring in an axial slice; and a pelvic ring, which does, for a few
    centimetres. The mandible is ``spec``'s, at ``spec.centre_z``.
    """
    sx, sy, sz = spacing
    x = np.arange(-110.0, 110.0 + 1e-6, sx)
    y = np.arange(-110.0, 110.0 + 1e-6, sy)
    z = np.arange(foot_mm, spec.centre_z + 200.0 + 1e-6, sz)
    X, Y = np.meshgrid(x, y)
    array = np.full((len(z), len(y), len(x)), CT_AIR, dtype=np.float32)
    head_centre = np.array([spec.centre_xy[0], spec.centre_xy[1] + 25.0, spec.centre_z + 90.0])
    for k, zk in enumerate(z):
        section = array[k]
        dz = zk - head_centre[2]
        if zk > spec.centre_z - 40.0:  # head
            r = 1.0 - (dz / 115.0) ** 2
            if r > 0:
                inside = ((X - head_centre[0]) / (75.0 * np.sqrt(r))) ** 2 + (
                    (Y - head_centre[1]) / (95.0 * np.sqrt(r))
                ) ** 2 <= 1.0
                section[inside] = CT_SOFT
                if dz > -30.0:  # the vault: a bone shell round the brain
                    rb = 1.0 - (dz / 105.0) ** 2
                    if rb > 0:
                        e = ((X - head_centre[0]) / (66.0 * np.sqrt(rb))) ** 2 + (
                            (Y - head_centre[1]) / (86.0 * np.sqrt(rb))
                        ) ** 2
                        section[(e <= 1.0) & (e > 0.82)] = CT_BONE
        if spec.centre_z - 120.0 < zk <= spec.centre_z - 20.0:  # neck
            section[((X / 45.0) ** 2 + ((Y - 30.0) / 45.0) ** 2) <= 1.0] = CT_SOFT
        if zk <= spec.centre_z - 120.0:  # torso
            section[((X / 105.0) ** 2 + ((Y - 10.0) / 85.0) ** 2) <= 1.0] = CT_SOFT
            if zk > foot_mm + 120.0 and int((zk - foot_mm) // 25.0) % 2 == 0:
                # a rib on each side: an arc, open at the front and the back
                e = (X / 95.0) ** 2 + ((Y - 10.0) / 75.0) ** 2
                rib = (e <= 1.0) & (e > 0.85) & (np.abs(X) > 25.0) & (Y > -40.0)
                section[rib] = CT_BONE
            if foot_mm + 20.0 < zk < foot_mm + 60.0:
                # the pelvic ring, closed for a few centimetres
                e = (X / 80.0) ** 2 + ((Y - 10.0) / 60.0) ** 2
                section[(e <= 1.0) & (e > 0.8)] = CT_BONE
        if zk <= spec.centre_z - 10.0:  # the spine
            vertebra = ((X / 14.0) ** 2 + ((Y - 55.0) / 12.0) ** 2) <= 1.0
            section[vertebra] = CT_BONE
    # The mandible, in the soft tissue.
    zz = z[:, None, None]
    sdf = signed_distance_field(spec, X[None], Y[None], zz)
    frac = np.clip(0.5 - sdf / (2.0 * max(spacing)), 0.0, 1.0)
    jaw = frac > 0.0
    array[jaw] = np.maximum(array[jaw], CT_SOFT + (CT_BONE - CT_SOFT) * frac[jaw])
    rng = np.random.default_rng(spec.seed)
    array += rng.normal(0.0, 15.0, size=array.shape).astype(np.float32)
    return Volume(array=array, spacing=np.asarray(spacing, dtype=float),
                  origin=np.array([x[0], y[0], z[0]]))
