"""A medical CT, of the whole body and in thick slices, handled like a CBCT.

The jaws are found in it, only a box round them is kept, and that box is
resampled to cubic voxels so the bone surface is smooth. A CBCT, or any scan
of the head alone in fine voxels, is loaded as it always was.
"""

from __future__ import annotations

import numpy as np
import pytest

from make_phantom import CT_BONE, CT_SOFT, PhantomSpec, make_body_ct, make_phantom, write_dicom_series
from mandiplan.dicom_io import CT_VOXEL_MM, load_folder
from mandiplan.geometry.head_region import coarse_view, find_jaw_region, tissue_thresholds, vault_top
from mandiplan.geometry.mandible import arch_axes, find_arch
from mandiplan.geometry.threshold import estimate_bone_threshold


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


def test_a_body_ct_is_cut_down_to_its_jaws_in_cubic_voxels(body_ct_folder):
    spec, folder = body_ct_folder
    volume, geometry, series = load_folder(folder)
    assert geometry.jaw_box
    # Cubic voxels, as fine as the scan's pixels allow, whatever its slices.
    assert np.allclose(volume.spacing, volume.spacing[0])
    assert CT_VOXEL_MM[0] <= volume.spacing[0] <= CT_VOXEL_MM[1]
    # A box round the jaws, not the 800 mm of the scan.
    assert volume.extent_mm[2] < 220.0
    lo, hi = volume.bounds_mm
    assert lo[2] < spec.centre_z - 20.0 < spec.centre_z + 20.0 < hi[2]
    assert any("keeps the jaws" in w for w in geometry.warnings)
    assert not any(w.startswith("Anisotropic") for w in geometry.warnings)

    # The mandible is in it, where it was, at its own size.
    threshold = estimate_bone_threshold(volume, thin_bone=True)
    arch = find_arch(volume, threshold)
    assert arch is not None
    assert np.mean(arch[:, 2]) == pytest.approx(spec.centre_z, abs=6.0)
    radius = np.linalg.norm(arch[:, :2] - np.array(spec.centre_xy), axis=1)
    assert np.allclose(radius, spec.radius_mm, atol=3.0)


def test_cubic_voxels_smooth_the_bone_surface(body_ct_folder):
    """3 mm slices draw a surface in 3 mm terraces; cubic voxels do not."""
    from mandiplan.render.surface import SurfaceExtractor
    from vtkmodules.util.numpy_support import vtk_to_numpy

    _, folder = body_ct_folder
    volume, _, _ = load_folder(folder)
    surface = SurfaceExtractor(volume).update(estimate_bone_threshold(volume, thin_bone=True))
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
    assert not geometry.jaw_box
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


def test_a_body_ct_plans_like_a_cbct(qt_app, body_ct_folder):
    """Open the CT, and the jaws, the curve and the cuts are there."""
    from mandiplan.ui.session import Session

    spec, folder = body_ct_folder
    session = Session()
    session.load_dicom_folder(str(folder))
    assert session.info.geometry.jaw_box
    # The thick-slice seed, and the reset button gives the same.
    assert session.threshold == pytest.approx(session.automatic_threshold())
    assert session.add_default_cut() and session.add_default_cut()
    assert session.mandible is not None and session.arch_source == "auto"
    assert 15.0 < session.report.arc_length_mm < 50.0
    assert "Jaw box" in "\n".join(session.info.lines(session.volume))
