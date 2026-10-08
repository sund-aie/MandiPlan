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
    volume: Volume,
    out_dir: str | Path,
    patient_id: str = "PHANTOM",
    description: str = "MandiPlan analytic phantom",
    image_type: tuple[str, ...] = ("ORIGINAL", "PRIMARY", "AXIAL"),
    plane: str = "axial",
    prefix: str = "slice",
    series_number: int = 1,
    study_uid: str | None = None,
) -> Path:
    """Write ``volume`` as a CT DICOM series, one file per slice.

    ``plane="coronal"`` writes it as coronal slices instead, as a scanner
    writes a coronal reformat. Several series can share a folder when each
    has its own ``prefix``; only files with that prefix are replaced.
    """
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{prefix}_*.dcm"):
        old.unlink()

    study_uid = study_uid or generate_uid()
    series_uid = generate_uid()
    frame_uid = generate_uid()

    data = np.asarray(volume.array, dtype=np.float64)
    intercept = float(np.floor(data.min()))
    slope = max((float(data.max()) - intercept) / 32000.0, 1e-3)
    stored = np.rint((data - intercept) / slope).astype(np.int16)

    sx, sy, sz = (float(v) for v in volume.spacing)
    ox, oy, oz = (float(v) for v in volume.origin)
    nz, ny, nx = volume.array.shape
    if plane == "coronal":
        # Rows run down the patient (-z), columns along +x; one slice per y.
        stored = np.ascontiguousarray(np.transpose(stored, (1, 0, 2))[:, ::-1, :])
        slices = [
            ([1.0, 0.0, 0.0, 0.0, 0.0, -1.0], [ox, oy + j * sy, oz + (nz - 1) * sz], [sz, sx], sy)
            for j in range(ny)
        ]
    else:
        slices = [
            ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [ox, oy, oz + k * sz], [sy, sx], sz)
            for k in range(nz)
        ]

    for k, (orientation, position, pixel_spacing, thickness) in enumerate(slices):
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
        ds.SeriesDescription = description
        ds.ImageType = list(image_type)
        ds.StudyDate = "20200101"
        ds.SeriesNumber = series_number
        ds.InstanceNumber = k + 1

        ds.ImageOrientationPatient = orientation
        ds.ImagePositionPatient = position
        ds.PixelSpacing = pixel_spacing  # [row spacing, column spacing]
        ds.SliceThickness = thickness
        ds.SpacingBetweenSlices = thickness
        ds.SliceLocation = position[2]

        ds.Rows, ds.Columns = stored[k].shape
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

        pydicom.dcmwrite(out_dir / f"{prefix}_{k:04d}.dcm", ds, enforce_file_format=True)

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
    frac = np.clip(0.5 - sdf / (2.0 * min(spacing)), 0.0, 1.0)
    jaw = frac > 0.0
    array[jaw] = np.maximum(array[jaw], CT_SOFT + (CT_BONE - CT_SOFT) * frac[jaw])
    rng = np.random.default_rng(spec.seed)
    array += rng.normal(0.0, 15.0, size=array.shape).astype(np.float32)
    return Volume(array=array, spacing=np.asarray(spacing, dtype=float),
                  origin=np.array([x[0], y[0], z[0]]))


# --------------------------------------------------------------------------
# A head-and-neck CT with the teeth in occlusion, to test separating the
# mandible where it touches the rest of the skull.
# --------------------------------------------------------------------------

#: Hounsfield-like values of the dentate CT's tissues.
CT_CROWN, CT_METAL, CT_MARROW = 2100.0, 3071.0, 300.0


@dataclass(frozen=True)
class DentateSpec:
    """A head-and-neck CT of a dentate jaw in occlusion. Lengths in mm.

    Everything the separation has to undo on a real CT is here, at a CT's
    resolution: the upper and lower crowns touch along the whole arch, each
    condyle sits in its fossa a millimetre below the skull base, each
    coronoid a millimetre below its zygomatic arch, and the maxilla, the
    palate, the skull base and the spine are one piece of bone. The scan is
    reconstructed in slices thicker than its pixels, so the joint spaces and
    the bite blur across, and it starts below the vault, as a neck CT does.
    ``roll_deg`` turns the head about its front-to-back axis, which tilts the
    bite and puts one condyle higher than the other; ``metal`` puts a
    filling in each lower first molar, with the streaks a CT draws from it
    across the slices it lies in.
    """

    spacing: tuple[float, float, float] = (0.75, 0.75, 2.0)
    roll_deg: float = 0.0
    metal: bool = False
    noise_sigma: float = 20.0
    seed: int = 0
    #: Radius of the front of the arch curve, through the middle of the body.
    arch_radius_mm: float = 30.0
    #: The body runs on straight behind the curve this far, to the angle.
    arch_back_mm: float = 22.0
    #: Height of the bite above the middle of the body.
    bite_mm: float = 18.0
    #: Gap between upper and lower crowns; 0 is teeth in contact.
    bite_gap_mm: float = 0.0
    condyle_centre: tuple[float, float, float] = (46.0, 30.0, 57.0)
    field_mm: tuple[float, float] = (220.0, 220.0)
    z_range_mm: tuple[float, float] = (-90.0, 100.0)

    @property
    def molar_s(self) -> float:
        """Arc length from the right end of the tooth row to the first molar."""
        return 14.0


def _arch_distance(x, y, radius, back):
    """Distance in plan from the arch curve: a front semicircle and a
    straight run behind each end."""
    front = np.abs(np.hypot(x, y) - radius)
    side = np.where(
        y > back, np.hypot(np.abs(x) - radius, y - back), np.abs(np.abs(x) - radius)
    )
    return np.where(y <= 0.0, front, side)


def _arch_point(s, radius, back, extra=10.0):
    """Point and tangent at arc length ``s`` along the tooth row, from the
    right end (``extra`` mm up the right straight run) to the left."""
    half = np.pi * radius
    if s < extra:  # right straight run, coming forward
        return np.array([-radius, extra - s]), np.array([0.0, -1.0])
    if s > extra + half:  # left straight run, going back
        return np.array([radius, s - extra - half]), np.array([0.0, 1.0])
    theta = np.pi + (s - extra) / radius
    return (
        np.array([radius * np.cos(theta), radius * np.sin(theta)]),
        np.array([-np.sin(theta), np.cos(theta)]),
    )


def _teeth(x, y, z, radius, back, centre_z, half_along, half_across, half_up, count=15, extra=10.0):
    """A row of crowns (ellipsoids) along the arch."""
    out = np.zeros(np.broadcast(x, y, z).shape, dtype=bool)
    length = np.pi * radius + 2.0 * extra
    for s in np.linspace(half_along, length - half_along, count):
        (cx, cy), (tx, ty) = _arch_point(s, radius, back, extra)
        dx, dy = x - cx, y - cy
        along = dx * tx + dy * ty
        across = -dx * ty + dy * tx
        out |= (along / half_along) ** 2 + (across / half_across) ** 2 + ((z - centre_z) / half_up) ** 2 <= 1.0
    return out


def _ramus(x, y, z, side, spec):
    """One ramus: a plate rising and turning outwards from the angle to the
    condylar neck and the coronoid, in the (y, z) outline below."""
    from mandiplan.geometry.plate_profile import point_in_polygon

    cx, _, cz = spec.condyle_centre
    r = spec.arch_radius_mm
    plate_x = r + (z + 11.0) * (cx - r) / (cz + 11.0)
    near = np.abs(side * x - plate_x) <= 2.5
    outline = np.array(
        [(8, 4), (14, -10), (32, -12), (36, 0), (33, 48), (31, 54), (27, 50), (22, 43),
         (17, 50), (13, 56), (10, 50), (9, 30)], dtype=float
    )
    out = np.zeros(np.broadcast(x, y, z).shape, dtype=bool)
    if near.any():
        yy = np.broadcast_to(y, out.shape)[near]
        zz = np.broadcast_to(z, out.shape)[near]
        out[near] = point_in_polygon(np.column_stack([yy, zz]), outline)
    return out


def _bones(x, y, z, spec):
    """Occupancy of each tissue at anatomical points: (mandible, crowns of
    the lower teeth, upper teeth, maxilla and skull, hyoid and spine, metal,
    the marrow of the mandibular body)."""
    r, back, bite = spec.arch_radius_mm, spec.arch_back_mm, spec.bite_mm
    cx, cy, cz = spec.condyle_centre
    along = _arch_distance(x, y, r, back)
    body = (along / 5.5) ** 2 + (z / 11.0) ** 2 <= 1.0
    # A cortex 2 mm thick round marrow far below any bone threshold.
    marrow = (along / 3.5) ** 2 + (z / 9.0) ** 2 <= 1.0
    lower = _teeth(x, y, z, r, back, bite - 3.5, 3.4, 4.0, 3.5)
    upper = _teeth(x, y, z, r + 1.0, back, bite + spec.bite_gap_mm + 3.5, 3.5, 4.3, 3.5)
    rami = _ramus(x, y, z, 1.0, spec) | _ramus(x, y, z, -1.0, spec)
    condyles = np.zeros_like(body)
    fossae = np.zeros_like(body)
    for side in (-1.0, 1.0):
        e = ((x - side * cx) / 9.0) ** 2 + ((y - cy) / 4.5) ** 2 + ((z - cz) / 4.0) ** 2
        condyles |= e <= 1.0
        # The neck, widening from the ramus into the head.
        widen = np.clip((z - (cz - 14.0)) / 14.0, 0.0, 1.0)
        neck = ((x - side * (cx - 4.0 * (1.0 - widen))) / (2.5 + 2.5 * widen)) ** 2 + (
            (y - (cy - 1.0)) / 3.0
        ) ** 2 <= 1.0
        condyles |= neck & (z >= cz - 14.0) & (z <= cz)
        # The fossa: a dome over the condyle, as clear of it as a joint
        # space is (some 2.5 mm above, 2 mm in front and behind), which
        # 2 mm slices blur across.
        inner = ((x - side * cx) / 12.0) ** 2 + ((y - cy) / 6.5) ** 2 + ((z - cz) / 6.5) ** 2
        outer = ((x - side * cx) / 16.0) ** 2 + ((y - cy) / 10.5) ** 2 + ((z - cz) / 10.0) ** 2
        fossae |= (inner > 1.0) & (outer <= 1.0) & (z >= cz + 1.0)
    mandible = body | rami | condyles | lower

    upper_curve = _arch_distance(x, np.minimum(y, 12.0) + np.maximum(y - 12.0, 0.0) * 1e3, r + 1.0, 12.0)
    maxilla = (upper_curve / 6.0) ** 2 + ((z - bite - 11.0) / 7.0) ** 2 <= 1.0
    inside = np.where(y <= 0.0, np.hypot(x, y) < r - 3.0, (np.abs(x) < r - 3.0) & (y <= 12.0))
    palate = inside & (z >= bite + 15.0) & (z <= bite + 18.0)
    pterygoid = (np.abs(x) >= 12.0) & (np.abs(x) <= 16.0) & (y >= 8.0) & (y <= 26.0) & (z >= bite + 12.0) & (z <= cz + 9.0)
    base = (z >= cz + 6.5) & (z <= cz + 10.5) & (np.abs(x) <= 62.0) & (y >= -5.0) & (y <= 70.0)
    # The zygomatic arch, over the coronoid, back to the eminence in front of
    # the condyle, where it joins the skull base.
    arch_bar = (np.abs(x) >= 42.0) & (np.abs(x) <= 50.0) & (y >= -8.0) & (y <= 22.0) & (z >= cz + 0.5) & (z <= cz + 7.0)
    zygoma = (np.abs(x) >= 34.0) & (np.abs(x) <= 50.0) & (y >= -12.0) & (y <= -2.0) & (z >= bite + 12.0) & (z <= cz + 4.5)
    skull = maxilla | palate | pterygoid | base | arch_bar | zygoma | fossae

    spine = (np.hypot(x, y - 58.0) <= 8.0) & (z <= cz + 7.0)
    hyoid = (np.abs(np.hypot(x, y - 10.0) - 16.0) <= 2.5) & (y < 10.0) & (np.abs(z + 28.0) <= 2.5)
    metal = np.zeros_like(body)
    if spec.metal:
        for s in (spec.molar_s, np.pi * r + 20.0 - spec.molar_s):
            (mx, my), _ = _arch_point(s, r, back)
            metal |= ((x - mx) / 2.5) ** 2 + ((y - my) / 2.5) ** 2 + ((z - bite + 1.0) / 1.5) ** 2 <= 1.0
    return mandible, lower, upper, skull & ~mandible, spine | hyoid, metal, marrow & ~lower


def make_dentate_ct(spec: DentateSpec = DentateSpec()):
    """The CT of ``spec``, and the share of each voxel that is each bone.

    Returns ``(volume, truth)``: ``truth`` maps "mandible", "upper" (the
    upper teeth) and "skull" (every other bone of the head) to volumes on
    the CT's grid holding the share of each voxel that bone fills.
    Each slice averages four planes through its thickness, and each pixel
    two by two points across it, as a CT detector integrates.
    """
    sx, sy, sz = spec.spacing
    x = np.arange(-spec.field_mm[0] / 2, spec.field_mm[0] / 2 + 1e-6, sx)
    y = np.arange(-spec.field_mm[1] / 2 + 20.0, spec.field_mm[1] / 2 + 20.0 + 1e-6, sy)
    z = np.arange(spec.z_range_mm[0], spec.z_range_mm[1] + 1e-6, sz)
    # The bone lies within this box; only it is sampled finely.
    bx = (np.abs(x) <= 66.0)
    by = (y >= -30.0) & (y <= 75.0)
    fx = (x[bx][:, None] + np.array([-0.25, 0.25]) * sx).ravel()
    fy = (y[by][:, None] + np.array([-0.25, 0.25]) * sy).ravel()
    FX, FY = np.meshgrid(fx, fy)
    roll = np.radians(spec.roll_deg)
    ca, sa = np.cos(roll), np.sin(roll)
    pivot = 20.0
    shape = (len(z), len(y), len(x))
    values = np.full(shape, CT_AIR, dtype=np.float32)
    truth = {name: np.zeros(shape, dtype=np.float32) for name in ("mandible", "upper", "skull")}
    X2, Y2 = np.meshgrid(x, y)
    for k, zk in enumerate(z):
        # Soft tissue: the head above the jaw, the neck below.
        head = ((X2 / 75.0) ** 2 + ((Y2 - 20.0) / 95.0) ** 2 <= 1.0) & (zk > -40.0)
        neck = (np.hypot(X2, Y2 - 30.0) <= 55.0) & (zk <= -40.0)
        values[k][head | neck] = CT_SOFT
        shares = np.zeros((7,) + FX.shape, dtype=np.float32)
        for dz in (-0.375, -0.125, 0.125, 0.375):
            # Anatomical coordinates of the plane, the head rolled about y.
            za = zk + dz * sz
            ax = FX * ca + (za - pivot) * sa
            az = -FX * sa + (za - pivot) * ca + pivot
            for i, part in enumerate(_bones(ax, FY, az, spec)):
                shares[i] += part
        shares /= 4.0
        # Two by two fine points to a pixel.
        ny_b, nx_b = int(by.sum()), int(bx.sum())
        shares = shares.reshape(7, ny_b, 2, nx_b, 2).mean(axis=(2, 4))
        mandible, lower, upper, skull, other, metal, marrow = shares
        crown = np.clip(lower + upper, 0.0, 1.0)
        bone = np.clip(mandible - lower - marrow + skull + other, 0.0, 1.0 - crown)
        metal = np.minimum(metal, 1.0)
        block = values[k][np.ix_(by, bx)]
        soft = np.clip(1.0 - bone - crown - marrow, 0.0, 1.0)
        block = block * soft + CT_BONE * bone + CT_CROWN * crown + CT_MARROW * marrow
        block = block * (1.0 - metal) + CT_METAL * metal
        values[k][np.ix_(by, bx)] = block
        truth["mandible"][k][np.ix_(by, bx)] = mandible
        truth["upper"][k][np.ix_(by, bx)] = upper
        truth["skull"][k][np.ix_(by, bx)] = skull
    origin = np.array([x[0], y[0], z[0]])
    if spec.metal:
        _add_streaks(values, x, y, z, spec, ca, sa, pivot)
    rng = np.random.default_rng(spec.seed)
    values += rng.normal(0.0, spec.noise_sigma, size=shape).astype(np.float32)
    np.clip(values, CT_AIR - 24.0, CT_METAL, out=values)
    spacing = np.asarray(spec.spacing, dtype=float)
    return Volume(values, spacing, origin), {
        name: Volume(share, spacing, origin) for name, share in truth.items()
    }


def _add_streaks(values, x, y, z, spec, ca, sa, pivot):
    """Bright and dark rays from each filling across the slices it lies in,
    as a CT draws them, and a bright band between the two fillings."""
    r, back = spec.arch_radius_mm, spec.arch_back_mm
    X2, Y2 = np.meshgrid(x, y)
    centres = []
    for s in (spec.molar_s, np.pi * r + 20.0 - spec.molar_s):
        (mx, my), _ = _arch_point(s, r, back)
        mz = spec.bite_mm - 1.0
        # Back to scanner coordinates.
        wx = (mx * ca - (mz - pivot) * sa)
        wz = (mx * sa + (mz - pivot) * ca) + pivot
        centres.append((wx, my, wz))
    for k, zk in enumerate(z):
        for wx, wy, wz in centres:
            weight = np.clip(1.0 - abs(zk - wz) / 2.0, 0.0, 1.0)
            if weight <= 0:
                continue
            dist = np.hypot(X2 - wx, Y2 - wy)
            angle = np.arctan2(Y2 - wy, X2 - wx)
            rays = np.cos(9.0 * angle) * np.exp(-dist / 25.0) * (dist > 3.0)
            values[k] += (600.0 * weight * rays).astype(np.float32)
        (ax_, ay_, az_), (bx_, by_, bz_) = centres
        weight = np.clip(1.0 - abs(zk - 0.5 * (az_ + bz_)) / 2.5, 0.0, 1.0)
        if weight > 0:
            # Across the mouth from one filling to the other.
            d = np.array([bx_ - ax_, by_ - ay_])
            length = float(np.linalg.norm(d))
            u = d / length
            along = (X2 - ax_) * u[0] + (Y2 - ay_) * u[1]
            off = np.abs(-(X2 - ax_) * u[1] + (Y2 - ay_) * u[0])
            band = (along > 3.0) & (along < length - 3.0) & (off < 0.8)
            values[k][band] += 500.0 * weight
