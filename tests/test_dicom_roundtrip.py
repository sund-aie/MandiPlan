"""DICOM writing, validation and loading."""

from __future__ import annotations

import numpy as np
import pydicom
import pytest

from make_phantom import PhantomSpec, make_phantom, write_dicom_series
from mandiplan.dicom_io import DicomLoadError, inspect_geometry, list_series, load_folder


@pytest.fixture(scope="module")
def written_series(tmp_path_factory):
    spec = PhantomSpec(spacing=(0.25, 0.5, 0.8), seed=3)
    volume = make_phantom(spec)
    folder = tmp_path_factory.mktemp("phantom_dicom")
    write_dicom_series(volume, folder)
    return spec, volume, folder


def test_series_is_discovered(written_series):
    _, _, folder = written_series
    series = list_series(folder)
    assert len(series) == 1
    assert series[0].modality == "CT"
    assert series[0].n_files > 10


def test_spacing_and_orientation_survive_the_round_trip(written_series):
    spec, volume, folder = written_series
    loaded, geometry, _ = load_folder(folder)

    assert np.allclose(loaded.spacing, volume.spacing, atol=1e-9)
    assert np.allclose(loaded.origin, volume.origin, atol=1e-6)
    assert loaded.array.shape == volume.array.shape
    assert geometry.slice_spacing_mm == pytest.approx(spec.spacing[2], abs=1e-6)
    assert geometry.in_plane_spacing_mm == pytest.approx(spec.spacing[:2], abs=1e-9)
    assert np.allclose(geometry.normal, [0, 0, 1], atol=1e-9)


def test_voxel_values_and_world_positions_survive(written_series):
    _, volume, folder = written_series
    loaded, _, _ = load_folder(folder)

    quantisation = (volume.array.max() - volume.array.min()) / 32000.0
    assert np.max(np.abs(loaded.array - volume.array)) < quantisation

    rng = np.random.default_rng(1)
    lo, hi = volume.bounds_mm
    pts = lo + rng.random((300, 3)) * (hi - lo)
    assert np.allclose(loaded.sample(pts), volume.sample(pts), atol=quantisation * 2)


def test_anisotropy_is_reported_as_a_warning(written_series):
    _, _, folder = written_series
    _, geometry, _ = load_folder(folder)
    assert any("Anisotropic" in w for w in geometry.warnings)


def test_gantry_tilt_is_refused(written_series, tmp_path):
    _, _, folder = written_series
    tilted = tmp_path / "tilted"
    tilted.mkdir()
    for path in sorted(folder.glob("*.dcm")):
        ds = pydicom.dcmread(path)
        ds.GantryDetectorTilt = 12.0
        pydicom.dcmwrite(tilted / path.name, ds, enforce_file_format=True)

    with pytest.raises(DicomLoadError, match="tilt"):
        load_folder(tilted)


def test_sheared_slice_positions_are_refused(written_series, tmp_path):
    _, _, folder = written_series
    sheared = tmp_path / "sheared"
    sheared.mkdir()
    files = sorted(folder.glob("*.dcm"))
    for k, path in enumerate(files):
        ds = pydicom.dcmread(path)
        pos = [float(v) for v in ds.ImagePositionPatient]
        pos[1] += 0.4 * k  # slide every slice posteriorly: a sheared stack
        ds.ImagePositionPatient = pos
        pydicom.dcmwrite(sheared / path.name, ds, enforce_file_format=True)

    with pytest.raises(DicomLoadError, match="sheared"):
        inspect_geometry([str(p) for p in sorted(sheared.glob("*.dcm"))])


def test_irregular_slice_spacing_is_refused(written_series, tmp_path):
    _, _, folder = written_series
    irregular = tmp_path / "irregular"
    irregular.mkdir()
    files = sorted(folder.glob("*.dcm"))
    for k, path in enumerate(files):
        ds = pydicom.dcmread(path)
        pos = [float(v) for v in ds.ImagePositionPatient]
        if k > len(files) // 2:
            pos[2] += 0.35  # a gap in the middle of the stack
        ds.ImagePositionPatient = pos
        pydicom.dcmwrite(irregular / path.name, ds, enforce_file_format=True)

    with pytest.raises(DicomLoadError, match="not uniform"):
        inspect_geometry([str(p) for p in sorted(irregular.glob("*.dcm"))])
