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
