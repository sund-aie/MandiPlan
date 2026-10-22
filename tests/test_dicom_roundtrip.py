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


def test_a_scan_too_large_for_memory_is_block_averaged_with_exact_geometry(written_series):
    """World positions survive the working grid; only the voxel size changes."""
    spec, volume, folder = written_series
    budget = volume.array.size // 5
    loaded, geometry, _ = load_folder(folder, voxel_budget=budget)

    assert geometry.working_factor == 2
    assert loaded.array.size <= budget
    assert np.allclose(loaded.spacing, 2 * volume.spacing)
    # Each working voxel is the mean of the 2x2x2 block it covers, and sits
    # at that block's centre.
    assert np.allclose(loaded.origin, volume.origin + 0.5 * volume.spacing, atol=1e-6)
    block = volume.array[:2, :2, :2].astype(float)
    assert loaded.array[0, 0, 0] == pytest.approx(block.mean(), abs=1.0)
    centre = loaded.index_to_world(np.array(loaded.size_xyz) // 2)
    assert loaded.sample(centre[None])[0] == pytest.approx(volume.sample(centre[None])[0], abs=60.0)
    assert any("millimetres" in w for w in geometry.warnings)


def _write_multiframe(volume, path, descending=True):
    """``volume`` as one enhanced CT file, its geometry in functional groups,
    frames top-down as cone-beam scanners write them."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    data = np.asarray(volume.array, dtype=np.float64)
    stored = np.clip(np.rint(data + 1000.0), 0, 65535).astype(np.uint16)
    order = np.arange(stored.shape[0])[::-1] if descending else np.arange(stored.shape[0])
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
    ds.Modality, ds.ImageType, ds.SeriesDescription = "CT", ["ORIGINAL", "PRIMARY"], "CBCT multi-frame"
    ds.Rows, ds.Columns = stored.shape[1:]
    ds.NumberOfFrames = len(order)
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    sx, sy, sz = (float(v) for v in volume.spacing)
    measures = Dataset()
    measures.PixelSpacing, measures.SliceThickness = [sy, sx], sz
    shared = Dataset()
    shared.PixelMeasuresSequence = Sequence([measures])
    ds.SharedFunctionalGroupsSequence = Sequence([shared])
    frames = []
    for k in order:
        item, position, orientation, transform = Dataset(), Dataset(), Dataset(), Dataset()
        position.ImagePositionPatient = [float(volume.origin[0]), float(volume.origin[1]), float(volume.origin[2] + k * sz)]
        orientation.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        transform.RescaleSlope, transform.RescaleIntercept, transform.RescaleType = 1.0, -1000.0, "HU"
        item.PlanePositionSequence = Sequence([position])
        item.PlaneOrientationSequence = Sequence([orientation])
        item.PixelValueTransformationSequence = Sequence([transform])
        frames.append(item)
    ds.PerFrameFunctionalGroupsSequence = Sequence(frames)
    ds.PixelData = stored[order].tobytes()
    pydicom.dcmwrite(path, ds, enforce_file_format=True)


def test_a_multiframe_cbct_loads_like_the_same_scan_in_slices(tmp_path, phantom, spec):
    """A CBCT stored as one enhanced multi-frame file, frames top-down."""
    folder = tmp_path / "multiframe"
    folder.mkdir()
    _write_multiframe(phantom, folder / "scan.dcm")
    loaded, geometry, series = load_folder(folder)
    assert series.n_files == 1
    assert np.allclose(loaded.spacing, spec.spacing)
    assert np.allclose(loaded.origin, phantom.origin, atol=1e-4)
    assert loaded.array.shape == phantom.array.shape
    assert np.abs(loaded.array - np.rint(phantom.array)).max() <= 1.0


def test_a_multiframe_scan_too_large_is_averaged_from_disk(tmp_path, phantom):
    folder = tmp_path / "multiframe_big"
    folder.mkdir()
    _write_multiframe(phantom, folder / "scan.dcm", descending=False)
    budget = phantom.array.size // 6
    loaded, geometry, _ = load_folder(folder, voxel_budget=budget)
    assert geometry.working_factor == 2
    assert np.allclose(loaded.spacing, phantom.spacing * 2)
    assert np.allclose(loaded.origin, phantom.origin + 0.5 * phantom.spacing, atol=1e-4)
