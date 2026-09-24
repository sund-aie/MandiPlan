"""Bone threshold seeding.  CBCT gray values are not Hounsfield units."""

from __future__ import annotations

import numpy as np

from mandiplan.geometry.threshold import estimate_bone_threshold, otsu_threshold


def test_otsu_separates_two_gaussian_populations():
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(100, 20, 50_000), rng.normal(1600, 40, 20_000)])
    counts, edges = np.histogram(data, bins=256)
    assert 400 < otsu_threshold(counts, edges) < 1300


def test_seed_threshold_lies_between_air_and_bone(phantom, spec):
    threshold = estimate_bone_threshold(phantom)
    assert spec.air_value < threshold < spec.bone_value


def test_seed_threshold_follows_the_gray_value_scale(spec):
    """The same geometry on a shifted gray-value scale gets a shifted threshold."""
    from make_phantom import PhantomSpec, make_phantom

    shifted = make_phantom(
        PhantomSpec(
            spacing=spec.spacing,
            air_value=-500.0,
            bone_value=2500.0,
            noise_sigma=spec.noise_sigma,
        )
    )
    threshold = estimate_bone_threshold(shifted)
    assert -500.0 < threshold < 2500.0
    assert threshold > estimate_bone_threshold(make_phantom(PhantomSpec()))


def test_a_head_scan_with_soft_tissue_gets_a_bone_threshold():
    """Air, soft tissue, bone and a few metal voxels, like a real CBCT.

    A two-class split falls between air and skin here and would draw the face;
    the seed has to fall between soft tissue and bone.
    """
    from mandiplan.geometry.volume import Volume

    rng = np.random.default_rng(3)
    air = rng.normal(-1000, 40, 600_000)
    soft = rng.normal(0, 60, 300_000)
    bone = rng.normal(900, 120, 59_700)
    metal = rng.normal(12000, 500, 300)
    data = np.concatenate([air, soft, bone, metal]).astype(np.float32)
    rng.shuffle(data)
    volume = Volume(data.reshape(60, 80, 200), np.array([0.3, 0.3, 0.3]))
    threshold = estimate_bone_threshold(volume)
    assert 250 < threshold < 700


def test_otsu_three_class_finds_both_gaps():
    from mandiplan.geometry.threshold import otsu_three_class

    rng = np.random.default_rng(1)
    data = np.concatenate(
        [rng.normal(0, 30, 40_000), rng.normal(500, 30, 40_000), rng.normal(1000, 30, 40_000)]
    )
    counts, edges = np.histogram(data, bins=200)
    t1, t2 = otsu_three_class(counts, edges)
    assert 150 < t1 < 350
    assert 650 < t2 < 850
