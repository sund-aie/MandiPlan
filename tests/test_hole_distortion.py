"""What contouring does to the screw holes.

An earlier version of this application held every hole perfectly circular
through bending and called that a feature. It is not what happens. AO Surgery
Reference is explicit that without bending insets the holes deform during
contouring and precise seating of locking screws cannot be guaranteed, and
that preformed plates keep their fatigue life precisely because they need less
bending. These tests pin that behaviour down.
"""

from __future__ import annotations

import numpy as np
import pytest

from mandiplan.geometry.hole_distortion import (
    LOCKING_MARGINAL_MM,
    LOCKING_TOLERANCE_MM,
    ovalise_mesh,
    predict_distortion,
)
from mandiplan.materials import ALLOYS, alloy_by_id
from mandiplan.plate_catalog import kit_by_id, load_kits

HOLES = np.arange(6) * 9.0
DIAMETER = 2.9
THICKNESS = 2.4


def run(kit_id, bend_positions, angles=(12.0, 12.0), alloy_id="cp-ti-grade-4", **kw):
    return predict_distortion(
        HOLES,
        DIAMETER,
        THICKNESS,
        np.asarray(angles, dtype=float),
        np.asarray(bend_positions, dtype=float),
        alloy_by_id(alloy_id),
        kit_by_id(kit_id),
        **kw,
    )


# -- the correction: holes do go out of round ----------------------------


def test_bending_through_a_hole_takes_it_out_of_round():
    report = run("bending-pliers", [9.0, 27.0])
    worst = report.worst
    assert worst is not None
    assert worst.out_of_round_mm > LOCKING_MARGINAL_MM
    assert worst.major_mm > DIAMETER > worst.minor_mm
    assert "out of round" in worst.verdict
    assert report.problems


def test_bending_between_holes_with_a_local_instrument_spares_them():
    """Which is exactly why the technique is taught that way."""
    report = run("bending-pliers", [13.5, 22.5])
    assert report.worst.out_of_round_mm == pytest.approx(0.0, abs=1e-12)
    assert report.locking_capable == len(report.holes)
    assert not report.problems


def test_a_wide_grip_catches_holes_even_when_bending_between_them():
    """Bending irons spread the bend over most of a pitch, so it reaches."""
    between = run("bending-irons", [13.5, 22.5])
    assert between.worst.out_of_round_mm > LOCKING_TOLERANCE_MM
    # ...but still less than driving the bend straight through a hole.
    through = run("bending-irons", [9.0, 27.0])
    assert through.worst.out_of_round_mm > between.worst.out_of_round_mm


def test_bending_insets_protect_the_holes():
    """AO: without insets the holes deform and locking seating is not assured."""
    without = run("bending-pliers", [9.0, 27.0], use_insets=False)
    with_insets = run("bending-pliers", [9.0, 27.0], use_insets=True)

    # An order of magnitude less distortion, which moves the holes out of the
    # "will not take a locking screw" class and into "marginal".
    assert with_insets.worst.out_of_round_mm < without.worst.out_of_round_mm / 5.0
    assert without.worst.out_of_round_mm > LOCKING_MARGINAL_MM
    assert with_insets.worst.out_of_round_mm < LOCKING_MARGINAL_MM
    assert without.problems and not with_insets.problems

    # Insets reduce, they do not abolish: a residual gets through, so the
    # model does not claim a bent hole is as good as an unbent one.
    assert with_insets.worst.out_of_round_mm > 0.0


def test_a_press_with_insets_is_the_kindest_kit():
    through = {kit: run(kit, [9.0, 27.0]).worst.out_of_round_mm
               for kit in ("bending-pliers", "bending-irons", "bar-bending-press")}
    assert through["bar-bending-press"] < through["bending-irons"]
    assert through["bar-bending-press"] < through["bending-pliers"]
    # The tightest instrument concentrates the bend and does the most damage.
    assert through["bending-pliers"] > through["bending-irons"]


def test_a_larger_bend_angle_distorts_more():
    small = run("bending-irons", [9.0], angles=[5.0])
    large = run("bending-irons", [9.0], angles=[25.0])
    assert large.worst.out_of_round_mm > small.worst.out_of_round_mm


def test_no_bend_leaves_every_hole_nominal():
    report = run("bending-irons", [9.0], angles=[0.0])
    for hole in report.holes:
        assert hole.major_mm == pytest.approx(DIAMETER)
        assert hole.minor_mm == pytest.approx(DIAMETER)
        assert hole.takes_locking_screw
    assert report.fatigue_fraction == pytest.approx(1.0)


# -- material behaviour ---------------------------------------------------


def test_a_less_ductile_alloy_suffers_more():
    """Ti-6Al-4V has a third the elongation of 316L and shows it."""
    titanium = run("bending-irons", [9.0], angles=[18.0], alloy_id="ti-6al-4v-eli")
    steel = run("bending-irons", [9.0], angles=[18.0], alloy_id="316lvm")
    assert titanium.fatigue_fraction < steel.fatigue_fraction


def test_springback_ranks_the_alloys_as_metallurgy_predicts():
    """FCC steel springs back least; alpha-beta titanium most."""
    ratios = {a.id: a.springback_ratio(30.0, 2.4) for a in ALLOYS}
    assert ratios["316lvm"] > ratios["cp-ti-grade-2"]
    assert ratios["cp-ti-grade-2"] > ratios["cp-ti-grade-4"]
    assert ratios["cp-ti-grade-4"] > ratios["ti-6al-4v-eli"]
    for ratio in ratios.values():
        assert 0.0 < ratio <= 1.0


def test_overbend_exceeds_the_target_angle():
    for alloy in ALLOYS:
        assert alloy.overbend_deg(20.0, 30.0, 2.4) > 20.0


def test_a_tight_bend_exceeds_the_rated_elongation():
    titanium = alloy_by_id("ti-6al-4v-eli")
    assert titanium.strain_headroom(60.0, 2.4) < 1.0
    assert titanium.strain_headroom(8.0, 2.4) > 1.0


def test_every_alloy_records_its_lattice():
    lattices = {a.id: a.lattice for a in ALLOYS}
    assert lattices["316lvm"] == "FCC"
    assert lattices["ti-6al-4v-eli"] == "HCP + BCC"
    assert lattices["cp-ti-grade-4"] == "HCP"
    for alloy in ALLOYS:
        assert alloy.standard.startswith("ASTM")
        assert alloy.elongation_pct > 0
        assert alloy.yield_mpa < alloy.ultimate_mpa


# -- fatigue --------------------------------------------------------------


def test_heavy_bending_cuts_the_predicted_fatigue_life():
    gentle = run("bar-bending-press", [13.5], angles=[5.0])
    brutal = run("bending-pliers", [9.0, 18.0, 27.0], angles=[25.0, 25.0, 25.0])
    assert brutal.fatigue_fraction < gentle.fatigue_fraction
    assert brutal.fatigue_fraction < 0.5
    assert any("fatigue" in p for p in brutal.problems)


# -- twisting shears rather than stretches -------------------------------


def test_twisting_forceps_skew_the_holes():
    report = run("twisting-forceps", [9.0], angles=[15.0])
    assert report.worst.skew_deg > 0.0
    plier = run("bending-pliers", [9.0], angles=[15.0])
    assert plier.worst.skew_deg == 0.0


# -- the distortion is applied to the mesh, not just reported ------------


def test_the_mesh_actually_changes_shape():
    from mandiplan.plate_assets import asset_by_id, load_asset_mesh

    asset = asset_by_id("generic-recon-2.4-12h")
    mesh = load_asset_mesh(asset)
    holes = np.linalg.norm(asset.hole_centres_mm - asset.hole_centres_mm[0], axis=1)
    report = predict_distortion(
        holes,
        asset.hole_diameter_mm,
        asset.thickness_mm,
        np.full(len(holes), 18.0),
        holes,
        asset.alloy,
        kit_by_id("bending-pliers"),
    )
    tangents = np.tile([1.0, 0.0, 0.0], (asset.hole_count, 1))
    moved = ovalise_mesh(
        mesh.points,
        asset.hole_centres_mm,
        asset.hole_axes,
        tangents,
        report.holes,
        asset.hole_diameter_mm,
    )
    assert moved.shape == mesh.points.shape
    assert not np.allclose(moved, mesh.points)

    # The hole really is oval afterwards: wider along the plate, narrower across.
    centre, axis = asset.hole_centres_mm[0], asset.hole_axes[0]
    offset = mesh.points - centre
    radial = offset - np.outer(offset @ axis, axis)
    wall = np.abs(np.linalg.norm(radial, axis=1) - asset.hole_diameter_mm / 2) < 0.03
    after = moved[wall] - centre
    along = np.abs(after @ np.array([1.0, 0.0, 0.0])).max()
    across = np.abs(after @ np.array([0.0, 1.0, 0.0])).max()
    assert along > asset.hole_diameter_mm / 2
    assert across < asset.hole_diameter_mm / 2


def test_a_round_prediction_leaves_the_mesh_alone():
    from mandiplan.plate_assets import asset_by_id, load_asset_mesh

    asset = asset_by_id("generic-recon-2.4-12h")
    mesh = load_asset_mesh(asset)
    holes = np.linalg.norm(asset.hole_centres_mm - asset.hole_centres_mm[0], axis=1)
    report = predict_distortion(
        holes,
        asset.hole_diameter_mm,
        asset.thickness_mm,
        np.zeros(len(holes)),
        holes,
        asset.alloy,
        kit_by_id("bar-bending-press"),
    )
    tangents = np.tile([1.0, 0.0, 0.0], (asset.hole_count, 1))
    moved = ovalise_mesh(
        mesh.points, asset.hole_centres_mm, asset.hole_axes, tangents,
        report.holes, asset.hole_diameter_mm,
    )
    assert np.allclose(moved, mesh.points)


# -- catalogue integrity --------------------------------------------------


def test_every_kit_declares_its_hole_behaviour():
    for kit in load_kits():
        assert kit.bend_localisation_mm > 0
        assert 0.0 <= kit.hole_protection <= 1.0
        assert kit.signature, kit.id
        if kit.bending_insets:
            assert kit.hole_protection > 0.5, kit.id


def test_mismatched_bend_inputs_are_refused():
    with pytest.raises(ValueError):
        predict_distortion(
            HOLES, DIAMETER, THICKNESS, [1.0, 2.0], [3.0],
            alloy_by_id("cp-ti-grade-4"),
        )
    with pytest.raises(ValueError):
        predict_distortion(
            HOLES, 0.0, THICKNESS, [1.0], [3.0], alloy_by_id("cp-ti-grade-4"),
        )
