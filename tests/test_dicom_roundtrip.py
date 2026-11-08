"""DICOM writing, validation and loading."""

from __future__ import annotations

from pathlib import Path

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


def _head_up_series(folder, spacing_tag=None, fill=None):
    """A little CT, bright at its top slice, optionally as odd scanners write it."""
    import pydicom

    from make_phantom import write_dicom_series
    from mandiplan.geometry.volume import Volume

    array = np.full((12, 32, 32), -1000.0, dtype=np.float32)
    array[:, 8:24, 8:24] = 40.0
    array[-1, 12:20, 12:20] = 2000.0  # the top of the head
    array[0, 12:16, 12:16] = 700.0  # and a mark at the foot end only
    if fill is not None:
        yy, xx = np.mgrid[:32, :32]
        array[:, (yy - 15.5) ** 2 + (xx - 15.5) ** 2 > 15.5**2] = fill
    volume = Volume(array=array, spacing=np.array([1.0, 1.0, 2.5]), origin=np.array([-16.0, -16.0, -100.0]))
    write_dicom_series(volume, folder)
    files = sorted(Path(folder).rglob("*.dcm"))
    assert files
    if spacing_tag is not None:
        for path in files:
            ds = pydicom.dcmread(path)
            ds.SpacingBetweenSlices = spacing_tag
            ds.save_as(path)
    return array


def test_a_negative_slice_spacing_does_not_turn_the_scan_upside_down(tmp_path):
    """Some CT scanners write SpacingBetweenSlices as -2.5. The head stays up."""
    from mandiplan.dicom_io import load_folder

    array = _head_up_series(tmp_path, spacing_tag="-2.5")
    volume, _, _ = load_folder(tmp_path)
    assert volume.origin[2] == pytest.approx(-100.0)
    assert volume.spacing[2] == pytest.approx(2.5)
    np.testing.assert_allclose(volume.array, array, atol=1.0)


def test_the_fill_outside_a_ct_circle_counts_as_air_not_tissue(tmp_path):
    from mandiplan.dicom_io import load_folder

    _head_up_series(tmp_path, fill=-3024.0)
    volume, _, _ = load_folder(tmp_path)
    assert float(volume.array.min()) >= -1100.0


def test_a_real_low_value_is_not_mistaken_for_padding():
    from mandiplan.dicom_io import _without_padding

    # A CBCT's darkest values run on smoothly from its air: no gap, no padding.
    rng = np.random.default_rng(1)
    array = rng.integers(0, 3000, size=(30, 40, 40)).astype(np.int16)
    array[array < 200] = 0
    assert _without_padding(array) is array


def test_a_series_in_nested_folders_is_listed_once(tmp_path):
    from mandiplan.dicom_io import list_series

    nested = tmp_path / "DICOM" / "ST000000" / "SE000000"
    nested.mkdir(parents=True)
    _head_up_series(nested)
    assert len(list_series(tmp_path)) == 1

