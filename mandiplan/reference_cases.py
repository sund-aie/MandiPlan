"""The real scan bundled with MandiPlan.

``data/reference/dz_cbct_jaws.npz`` is a head CBCT from the 3D Slicer sample
data ("CBCT-MR Head", DZ-CBCT.nrrd), which the 3D Slicer project states was
donated by the person in the images to be used without restriction. It is
cropped to the jaws, with enough skull above them to show the mandible being
separated, and averaged to 0.75 mm; ``tools/make_reference_case.py``
rebuilds it from the public file.

It is here for two reasons: every test of the planning chain on real anatomy
(teeth in occlusion, fillings, joints) runs on it, and File › Open sample
scan lets the whole workflow be tried without patient data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dicom_io import SeriesGeometry, SeriesInfo
from .geometry.volume import Volume

REFERENCE_DIR = Path(__file__).resolve().parent / "data" / "reference"
SAMPLE_SCAN = REFERENCE_DIR / "dz_cbct_jaws.npz"
SAMPLE_DESCRIPTION = "Sample head CBCT (3D Slicer sample data, donated without restriction)"


def load_sample() -> tuple[Volume, SeriesGeometry, SeriesInfo, np.ndarray]:
    """The sample scan, its geometry, and arch curve points on its mandible."""
    with np.load(SAMPLE_SCAN) as data:
        volume = Volume(
            array=data["array"].astype(np.float32),
            spacing=data["spacing"],
            origin=data["origin"],
        )
        arch_points = np.asarray(data["arch_points"], dtype=float)
    sx, sy, sz = (float(v) for v in volume.spacing)
    geometry = SeriesGeometry(
        slice_spacing_mm=sz,
        in_plane_spacing_mm=(sx, sy),
        orientation=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        normal=np.array([0.0, 0.0, 1.0]),
        gantry_tilt_deg=0.0,
        max_shear_deg=0.0,
        spacing_variation=0.0,
        warnings=[
            "This is the bundled sample scan, cropped to the jaws and averaged to "
            f"{sx:.2f} mm. It is for trying the workflow, not a patient."
        ],
    )
    series = SeriesInfo(
        series_uid="mandiplan-sample-dz-cbct",
        description=SAMPLE_DESCRIPTION,
        modality="CT",
        files=[],
    )
    return volume, geometry, series, arch_points
