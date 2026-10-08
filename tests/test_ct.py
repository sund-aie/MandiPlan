"""A medical CT, of the head and neck or the whole body, in thick slices.

The whole scan is kept and shown; the jaws are found in it, and the mandible
is separated and drawn from a box round them resampled to cubic voxels, so
its surface is as smooth as a CBCT's. A CBCT, or any scan of the head alone
in fine voxels, is loaded as it always was.

The dentate CT (``make_phantom.make_dentate_ct``) is what a real one throws
at the separation: the teeth in occlusion along the whole arch, condyles a
millimetre under their fossae and coronoids under the zygomatic arches,
2 mm slices that blur all of those across, and no skull vault in the scan.
"""

from __future__ import annotations

import numpy as np
import pytest

from make_phantom import (
    CT_BONE,
    CT_SOFT,
    DentateSpec,
    PhantomSpec,
    make_body_ct,
    make_dentate_ct,
    make_phantom,
    write_dicom_series,
)
from mandiplan.dicom_io import CT_VOXEL_MM, list_series, load_folder
from mandiplan.geometry.head_region import coarse_view, find_jaw_region, tissue_thresholds, vault_top
from mandiplan.geometry.mandible import arch_axes, find_arch, frames_for, isolate_mandible
from mandiplan.geometry.threshold import estimate_bone_threshold
from mandiplan.geometry.volume import Volume


@pytest.fixture(scope="module")
def body_ct():
    spec = PhantomSpec()
    return spec, make_body_ct(spec)


@pytest.fixture(scope="module")
def body_ct_folder(tmp_path_factory, body_ct):
    spec, volume = body_ct
    folder = tmp_path_factory.mktemp("body_ct")
    write_dicom_series(volume, folder, patient_id="BODY-CT")
    return spec, folder


def test_the_head_is_found_by_its_skull_vault_not_the_pelvis(body_ct):
    spec, volume = body_ct
    coarse = coarse_view(volume.array, volume.spacing, volume.origin)
    body, bone = tissue_thresholds(coarse)
    assert CT_SOFT - 600.0 < body < CT_SOFT
    assert CT_SOFT < bone < CT_BONE
    top = vault_top(coarse, body, bone)
    # The vault's top is near the top of the head, nowhere near the pelvis.
    assert top is not None and spec.centre_z + 150.0 < top < spec.centre_z + 210.0
    region = find_jaw_region(coarse)
    assert region is not None
    assert region.arch_level_mm == pytest.approx(spec.centre_z, abs=6.0)
    assert np.all(region.lo[:2] < np.array(spec.centre_xy) - spec.radius_mm)
    assert region.size_mm[2] < 220.0


def test_a_body_ct_is_kept_whole_and_its_jaws_made_cubic(body_ct_folder, body_ct):
    """The whole scan is what is shown; the jaws are a cubic box of it."""
    spec, folder = body_ct_folder
    _, original = body_ct
    volume, geometry, series = load_folder(folder)
    # All 800 mm of it, at the scan's own voxels.
    assert np.allclose(volume.spacing, original.spacing)
    assert volume.extent_mm[2] == pytest.approx(original.extent_mm[2])
    assert geometry.jaw_box
    jaws = geometry.jaw_volume
    # Cubic voxels, as fine as the scan's pixels allow, whatever its slices.
    assert np.allclose(jaws.spacing, jaws.spacing[0])
    assert CT_VOXEL_MM[0] <= jaws.spacing[0] <= CT_VOXEL_MM[1]
    assert jaws.resampled_from_mm == pytest.approx(original.spacing[2])
    # A box round the jaws, in the scan's own millimetres.
    assert jaws.extent_mm[2] < 220.0
    lo, hi = jaws.bounds_mm
    assert lo[2] < spec.centre_z - 20.0 < spec.centre_z + 20.0 < hi[2]
    scan_lo, scan_hi = volume.bounds_mm
    assert np.all(lo >= scan_lo - 1e-6) and np.all(hi <= scan_hi + 1e-6)
    assert any("box of this CT" in w for w in geometry.warnings)

    # The mandible is in the box, where it was, at its own size.
    threshold = estimate_bone_threshold(jaws, thin_bone=True)
    arch = find_arch(jaws, threshold)
    assert arch is not None
    assert np.mean(arch[:, 2]) == pytest.approx(spec.centre_z, abs=6.0)
    radius = np.linalg.norm(arch[:, :2] - np.array(spec.centre_xy), axis=1)
    assert np.allclose(radius, spec.radius_mm, atol=3.0)
    # And the box's voxels sample the scan's, millimetre for millimetre.
    point = np.array([spec.radius_mm, 0.0, spec.centre_z])
    assert jaws.sample(point[None])[0] == pytest.approx(volume.sample(point[None])[0], abs=150.0)


def test_cubic_voxels_smooth_the_bone_surface(body_ct_folder):
    """4 mm slices draw a surface in 4 mm terraces; cubic voxels do not."""
    from mandiplan.render.surface import SurfaceExtractor
    from vtkmodules.util.numpy_support import vtk_to_numpy

    _, folder = body_ct_folder
    _, geometry, _ = load_folder(folder)
    jaws = geometry.jaw_volume
    surface = SurfaceExtractor(jaws).update(estimate_bone_threshold(jaws, thin_bone=True))
    z = vtk_to_numpy(surface.GetPoints().GetData())[:, 2]
    # On the 4 mm slices every vertex height would be one of a few levels per
    # slice; resampled, the heights spread evenly.
    levels = np.unique(np.round(z / 0.25))
    assert len(levels) > (z.max() - z.min()) / 1.0


def test_a_scan_of_the_jaws_in_fine_voxels_is_not_touched(tmp_path):
    spec = PhantomSpec()
    folder = tmp_path / "phantom"
    write_dicom_series(make_phantom(spec), folder)
    volume, geometry, _ = load_folder(folder)
    assert not geometry.jaw_box and geometry.jaw_volume is None
    assert np.allclose(volume.spacing, spec.spacing)


def test_thick_slice_ct_seeds_a_lower_threshold(body_ct):
    """Thin bone averaged with soft tissue reads low on thick slices."""
    _, volume = body_ct
    usual = estimate_bone_threshold(volume)
    thin = estimate_bone_threshold(volume, thin_bone=True)
    assert CT_SOFT < thin < usual


def test_a_head_lying_turned_is_still_found():
    spec = PhantomSpec(arch_centre_deg=-90.0 + 35.0, centre_xy=(4.0, -3.0))
    volume = make_phantom(spec)
    arch = find_arch(volume, spec.half_max_value)
    assert arch is not None
    radius = np.linalg.norm(arch[:, :2] - np.array(spec.centre_xy), axis=1)
    assert np.allclose(radius, spec.radius_mm, atol=2.0)
    # The arch's own axes turn with it: "back" points away from the chin.
    _, back, lateral = arch_axes(arch)
    chin = np.array([np.cos(np.radians(spec.arch_centre_deg)), np.sin(np.radians(spec.arch_centre_deg))])
    assert float(back @ chin) < -0.9
    assert abs(float(lateral @ chin)) < 0.3


def test_a_straight_head_keeps_the_patient_axes():
    spec = PhantomSpec()
    arch = find_arch(make_phantom(spec), spec.half_max_value)
    _, back, lateral = arch_axes(arch)
    assert np.allclose(back, [0.0, 1.0]) and np.allclose(lateral, [1.0, 0.0])


def test_a_body_ct_plans_like_a_cbct(qt_app, body_ct_folder, body_ct):
    """Open the CT: the whole scan is there, then the jaws, the curve and the cuts."""
    from mandiplan.ui.session import Session

    spec, folder = body_ct_folder
    _, original = body_ct
    session = Session()
    session.load_dicom_folder(str(folder))
    assert session.info.geometry.jaw_box
    # What is shown is the whole scan, and before the mandible is separated
    # the bone surface is all of its bone, from the skull to the pelvis.
    assert session.scan.extent_mm[2] == pytest.approx(original.extent_mm[2])
    assert session.volume is session.info.geometry.jaw_volume
    bounds = session.surface.GetBounds()
    assert bounds[5] - bounds[4] > 500.0
    # The thick-slice seed, and the reset button gives the same.
    assert session.threshold == pytest.approx(session.automatic_threshold())
    assert session.add_default_cut() and session.add_default_cut()
    assert session.mandible is not None and session.arch_source == "auto"
    # The phantom's jaw is the front of an arch only, some 73 mm long.
    assert 10.0 < session.report.arc_length_mm < 50.0
    # Once separated, the surface is the mandible's, from the cubic box.
    lo, hi = session.volume.bounds_mm
    bounds = session.surface.GetBounds()
    assert lo[2] - 1.0 <= bounds[4] and bounds[5] <= hi[2] + 1.0
    text = "\n".join(session.info.lines(session.scan))
    assert f"{original.extent_mm[2]:.1f} mm" in text and "Jaw box" in text


# -- a dentate head-and-neck CT, its teeth in occlusion -----------------------


def _write_study(volume: Volume, folder) -> None:
    """The series a scanner writes for a neck CT: the axial images, a
    coronal reformat of them (more files than the axial series), a scout."""
    from pydicom.uid import generate_uid

    study = generate_uid()
    write_dicom_series(volume, folder, description="Neck 2.0 axial", prefix="axial", study_uid=study)
    write_dicom_series(
        volume, folder, description="Neck MPR cor", image_type=("DERIVED", "PRIMARY", "MPR"),
        plane="coronal", prefix="coronal", series_number=2, study_uid=study,
    )
    scout = Volume(volume.array[:, :, ::2][:2], volume.spacing * [2.0, 1.0, 1.0], volume.origin)
    write_dicom_series(
        scout, folder, description="Topogram", image_type=("ORIGINAL", "PRIMARY", "LOCALIZER"),
        prefix="scout", series_number=3, study_uid=study,
    )


@pytest.fixture(scope="module")
def dentate(tmp_path_factory):
    spec = DentateSpec()
    volume, truth = make_dentate_ct(spec)
    folder = tmp_path_factory.mktemp("dentate_ct")
    _write_study(volume, folder)
    return spec, volume, truth, folder


@pytest.fixture(scope="module")
def tilted_dentate(tmp_path_factory):
    """A head rolled 6 degrees, so one condyle stands higher than the other
    and the bite is tilted, with a filling and its streaks in each first molar."""
    spec = DentateSpec(roll_deg=6.0, metal=True, seed=3)
    volume, truth = make_dentate_ct(spec)
    folder = tmp_path_factory.mktemp("tilted_dentate_ct")
    write_dicom_series(volume, folder)
    return spec, volume, truth, folder


def test_the_axial_series_is_opened_not_the_reformat_with_more_files(dentate):
    spec, volume, _, folder = dentate
    found = {s.description: s for s in list_series(folder)}
    assert found["Neck MPR cor"].n_files > found["Neck 2.0 axial"].n_files
    scan, geometry, series = load_folder(folder)
    assert series.description == "Neck 2.0 axial"
    assert np.allclose(scan.spacing, spec.spacing)
    assert any("Neck 2.0 axial" in w for w in geometry.warnings)


def _jaw_level(volume: Volume) -> float:
    return find_jaw_region(coarse_view(volume.array, volume.spacing, volume.origin)).arch_level_mm


def test_the_jaws_are_found_in_a_neck_ct_with_no_vault(dentate):
    """No vault to measure down from: the mandible is still the lowest wide
    U, not the maxilla's above it, even with a sharp kernel's noise."""
    spec, volume, _, _ = dentate
    assert vault_top(coarse_view(volume.array, volume.spacing, volume.origin), *tissue_thresholds(
        coarse_view(volume.array, volume.spacing, volume.origin))) is None
    assert -11.0 < _jaw_level(volume) < spec.bite_mm
    noisy = volume.array + np.random.default_rng(1).normal(0.0, 150.0, volume.array.shape).astype(np.float32)
    assert -11.0 < _jaw_level(Volume(noisy, volume.spacing, volume.origin)) < spec.bite_mm


def _score(volume: Volume, threshold: float, isolation, truth) -> dict:
    """How much of each true bone, at the threshold, the mandible holds."""
    points = volume.origin + np.indices(volume.array.shape)[::-1].reshape(3, -1).T * volume.spacing
    bone = volume.array >= threshold
    share = {name: t.sample(points).reshape(volume.array.shape) > 0.5 for name, t in truth.items()}
    mask = isolation.mask
    return {name: float((mask & part & bone).sum() / max((part & bone).sum(), 1)) for name, part in share.items()}


def _separate(folder):
    _, geometry, _ = load_folder(folder)
    jaws = geometry.jaw_volume
    threshold = estimate_bone_threshold(jaws, thin_bone=True)
    arch = find_arch(jaws, threshold)
    assert arch is not None
    return jaws, threshold, isolate_mandible(jaws, threshold, frames_for(arch))


def _condyle_held(volume: Volume, isolation, centre) -> bool:
    i = np.round((np.asarray(centre) - volume.origin) / volume.spacing).astype(int)
    r = np.ceil(2.0 / volume.spacing).astype(int)
    return bool(isolation.mask[i[2] - r[2] : i[2] + r[2] + 1, i[1] - r[1] : i[1] + r[1] + 1,
                               i[0] - r[0] : i[0] + r[0] + 1].any())


def test_a_closed_bite_ct_is_separated_condyle_to_condyle(dentate):
    spec, _, truth, folder = dentate
    jaws, threshold, isolation = _separate(folder)
    assert "at the bite" in isolation.summary() and not isolation.warnings
    held = _score(jaws, threshold, isolation, truth)
    assert held["mandible"] > 0.9
    assert held["upper"] < 0.01  # no upper tooth
    assert held["skull"] < 0.002  # no maxilla, palate, skull base or spine
    cx, cy, cz = spec.condyle_centre
    for side in (-1.0, 1.0):
        assert _condyle_held(jaws, isolation, (side * cx, cy, cz))


def test_a_tilted_bite_with_fillings_keeps_both_condyles(tilted_dentate):
    """The rami's seeds end at different heights when the bite is tilted;
    each condyle is traced from its own. A filling's streaks are not kept."""
    import SimpleITK as sitk

    spec, _, truth, folder = tilted_dentate
    jaws, threshold, isolation = _separate(folder)
    right, left = isolation.ramus_heights_mm
    assert abs(right - left) < 5.0 and not isolation.warnings
    held = _score(jaws, threshold, isolation, truth)
    assert held["mandible"] > 0.9 and held["upper"] < 0.01 and held["skull"] < 0.002
    # Nothing of the mandible stands more than 2 mm off the true bone.
    points = jaws.origin + np.indices(jaws.array.shape)[::-1].reshape(3, -1).T * jaws.spacing
    outside = ~(truth["mandible"].sample(points).reshape(jaws.array.shape) > 0.5)
    image = sitk.GetImageFromArray(outside.astype(np.uint8))
    image.SetSpacing([float(v) for v in jaws.spacing])
    depth = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(
        image, insideIsPositive=True, squaredDistance=False, useImageSpacing=True))
    assert (isolation.mask & (depth > 2.0)).sum() < 0.005 * isolation.mask.sum()


def test_the_whole_ct_is_shown_from_the_start(qt_app, dentate):
    """The reported fault: on opening, the slices showed only a section of
    the CT. They show the whole scan, from the first frame and after the
    mandible is separated; the mandible is drawn smooth from the jaw box."""
    from mandiplan.render.surface import slice_smoothing
    from mandiplan.ui.main_window import MainWindow

    spec, volume, _, folder = dentate
    window = MainWindow()
    window.show()
    window.start()
    try:
        window.load_folder(str(folder))
        qt_app.processEvents()
        session = window.session
        assert session.scan.extent_mm == pytest.approx(volume.extent_mm)
        for name, axis in (("axial", 2), ("coronal", 1), ("sagittal", 0)):
            assert window.slice_sliders[name].maximum() == volume.size_xyz[axis] - 1
        assert window.slice_views["axial"]._image.shape == volume.array.shape[1:]
        # The first 3-D view is all the scan's bone, not a box of it.
        bounds = session.surface.GetBounds()
        assert bounds[4] < session.volume.bounds_mm[0][2] - 10.0  # the spine, below the jaws
        session.ensure_mandible()
        qt_app.processEvents()
        assert session.mandible is not None and not session.mandible.warnings
        assert window.slice_views["axial"]._image.shape == volume.array.shape[1:]
        assert window.slice_sliders["axial"].maximum() == volume.size_xyz[2] - 1
        # The axial slice went to the mandibular body.
        z = window._slice_world_coord("axial")
        assert -12.0 < z < spec.bite_mm
        assert slice_smoothing(session.bone_volume) is not None
        assert "Jaw box" in window.volume_panel.info.text()
        assert f"{volume.extent_mm[2]:.1f} mm" in window.volume_panel.info.text()
    finally:
        window.close()


def test_the_occlusal_plane_ignores_a_stretch_the_bite_search_lost():
    """Where a filling's streaks lead the first bite search astray, the plane
    the rest of the arch agrees on is kept."""
    from mandiplan.geometry.mandible import robust_plane

    rng = np.random.default_rng(0)
    points = np.column_stack([rng.uniform(-30, 30, 200), rng.uniform(-30, 10, 200), np.zeros(200)])
    z = 0.1 * points[:, 0] - 0.05 * points[:, 1] + 20.0 + rng.normal(0.0, 0.3, 200)
    z[:50] -= 8.0  # a quarter of the stations followed a streak 8 mm low
    plane = robust_plane(points, z)
    assert plane is not None
    assert np.allclose(plane(points[50:, 0], points[50:, 1]), z[50:], atol=1.5)
    assert robust_plane(points, 3.0 * points[:, 0]) is None  # implausibly steep


def test_only_a_volume_made_from_thick_slices_is_drawn_smoothed():
    from mandiplan.render.surface import SurfaceExtractor, mesh_volume_mm3, slice_smoothing

    spec = PhantomSpec(spacing=(0.5, 0.5, 0.5))
    native = make_phantom(spec)
    resampled = Volume(native.array, native.spacing, native.origin, resampled_from_mm=2.0)
    assert slice_smoothing(native) is None
    assert slice_smoothing(resampled) is not None
    plain = SurfaceExtractor(native).update(spec.half_max_value)
    smooth = SurfaceExtractor(resampled).update(spec.half_max_value)
    # The filter does not shrink the bone.
    assert mesh_volume_mm3(smooth) == pytest.approx(mesh_volume_mm3(plain), rel=0.02)
    assert mesh_volume_mm3(smooth) == pytest.approx(spec.volume_mm3, rel=0.03)
