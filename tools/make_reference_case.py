"""Build the bundled reference CBCT from the public 3D Slicer sample.

Source: 3D Slicer SampleData "CBCT-MR Head", file DZ-CBCT.nrrd,
SHA256 4ce7aa75278b5a7b757ed0c8d7a6b3caccfc3e2973b020532456dbc8f3def7db.
The 3D Slicer project states that this data set was donated by the person
visible in the images, to be used without any restrictions.

This script does not download anything. Fetch the file yourself and run::

    python tools/make_reference_case.py path/to/DZ-CBCT.nrrd

It resamples the scan onto the patient axes (the original grid is tilted by a
few degrees), crops it to the jaws with enough skull above them to exercise
the mandible separation, averages it to 0.75 mm and stores it as int16 with
the arch curve points used by the tests.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mandiplan.dicom_io import block_average  # noqa: E402
from mandiplan.geometry import cpr  # noqa: E402
from mandiplan.geometry.mandible import isolate_mandible  # noqa: E402
from mandiplan.geometry.spline import ArchCurve  # noqa: E402
from mandiplan.geometry.threshold import estimate_bone_threshold  # noqa: E402
from mandiplan.geometry.volume import Volume  # noqa: E402

SHA256 = "4ce7aa75278b5a7b757ed0c8d7a6b3caccfc3e2973b020532456dbc8f3def7db"
OUT = ROOT / "mandiplan" / "data" / "reference" / "dz_cbct_jaws.npz"
FACTOR = 3  # 0.25 mm -> 0.75 mm


def axis_aligned(path: Path) -> Volume:
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    size = np.array(image.GetSize())
    corners = np.array(
        [
            image.TransformIndexToPhysicalPoint([int(c * (n - 1)) for c, n in zip(corner, size)])
            for corner in np.ndindex(2, 2, 2)
        ]
    )
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    spacing = np.array(image.GetSpacing())
    shape = np.ceil((hi - lo) / spacing).astype(int) + 1
    reference = sitk.Image([int(v) for v in shape], image.GetPixelID())
    reference.SetSpacing(image.GetSpacing())
    reference.SetOrigin(lo.tolist())
    reference.SetDirection(np.eye(3).ravel().tolist())
    low = float(sitk.GetArrayViewFromImage(image).min())
    resampled = sitk.Resample(image, reference, sitk.Transform(), sitk.sitkLinear, low)
    return Volume(sitk.GetArrayFromImage(resampled).astype(np.float32), spacing, lo)


def arch_points(volume: Volume, threshold: float, z_mm: float, count: int = 11) -> np.ndarray:
    """Points along the middle of the mandibular body on one axial slice."""
    import SimpleITK as sitk

    k = int(round((z_mm - volume.origin[2]) / volume.spacing[2]))
    section = (volume.array[k] >= threshold).astype(np.uint8)
    labels = sitk.GetArrayFromImage(
        sitk.RelabelComponent(sitk.ConnectedComponent(sitk.GetImageFromArray(section), True))
    )
    rows, cols = np.nonzero(labels == 1)
    centre_col, centre_row = cols.mean(), rows.max() + 5
    angle = np.arctan2(rows - centre_row, cols - centre_col)
    edges = np.linspace(angle.min(), angle.max(), count + 1)
    points = []
    for a, b in zip(edges[:-1], edges[1:]):
        inside = (angle >= a) & (angle < b)
        if inside.sum() > 5:
            points.append(volume.index_to_world([cols[inside].mean(), rows[inside].mean(), k]))
    return np.array(points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("nrrd", type=Path, help="DZ-CBCT.nrrd from the 3D Slicer sample data")
    parser.add_argument("--body-z", type=float, default=-28.5,
                        help="world z (mm) of an axial slice through the mandibular body")
    args = parser.parse_args()

    digest = hashlib.sha256(args.nrrd.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"{args.nrrd} is not the expected file (sha256 {digest}).")

    full = axis_aligned(args.nrrd)
    threshold = estimate_bone_threshold(full)
    seeds = arch_points(full, threshold, args.body_z)
    frames = cpr.build_frames(ArchCurve(seeds), 0.25)
    isolation = isolate_mandible(full, threshold, frames)

    kk, jj, ii = np.nonzero(isolation.mask)
    lo = full.origin + np.array([ii.min(), jj.min(), kk.min()]) * full.spacing - 8.0
    hi = full.origin + np.array([ii.max(), jj.max(), kk.max()]) * full.spacing + 8.0
    bite = isolation.bite.heights_mm[isolation.bite.has_teeth].max()
    hi[2] = max(hi[2], bite + 62.0)
    i0 = np.maximum(np.floor((lo - full.origin) / full.spacing).astype(int), 0)
    i1 = np.minimum(np.ceil((hi - full.origin) / full.spacing).astype(int), full.size_xyz)
    crop = full.array[i0[2] : i1[2], i0[1] : i1[1], i0[0] : i1[0]]
    reduced = block_average(crop, FACTOR)
    origin = full.origin + i0 * full.spacing + 0.5 * (FACTOR - 1) * full.spacing
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        array=np.round(reduced).astype(np.int16),
        spacing=full.spacing * FACTOR,
        origin=origin,
        arch_points=seeds,
        source_sha256=np.array(SHA256),
    )
    print(f"{OUT.relative_to(ROOT)}: {reduced.shape} voxels at {full.spacing[0] * FACTOR:.2f} mm, "
          f"{OUT.stat().st_size / 1e6:.1f} MB; {isolation.summary()}")


if __name__ == "__main__":
    main()
