"""Finding the fibula in a CT of the legs, checked on the leg phantom."""

from __future__ import annotations

import numpy as np
import pytest

from make_leg_phantom import LegSpec, make_leg_phantom
from mandiplan.geometry.fibula import find_fibulae


def _off_axis_mm(spec: LegSpec, fibula) -> float:
    """How far the found line strays from the phantom's true axis."""
    worst = 0.0
    for p in fibula.points:
        c = spec.fibula_centre(fibula.side, p[2])
        worst = max(worst, float(np.hypot(p[0] - c[0], p[1] - c[1])))
    return worst


@pytest.mark.parametrize("side", ["right", "left"])
def test_the_fibula_of_one_leg_is_found_on_its_own_side(side):
    spec = LegSpec(side=side, noise_hu=30.0)
    (fibula,) = find_fibulae(make_leg_phantom(spec))
    assert fibula.side == side
    assert _off_axis_mm(spec, fibula) < 0.5
    # The line runs distal to proximal: up the leg.
    assert fibula.proximal_end[2] > fibula.distal_end[2] + 300.0
    # It is found to within a few millimetres of each tip, never past one:
    # where the tip thins out or meets the tibia it is lost a little early.
    low, high = spec.fibula_tip_z()
    assert low - 0.5 <= fibula.distal_end[2] <= low + 8.0
    assert high - 10.0 <= fibula.proximal_end[2] <= high + 0.5
    # Lateral is away from the tibia, towards the leg's own side.
    assert fibula.lateral @ spec.lateral(side) > 0.7


def test_both_legs_in_one_scan_give_two_fibulae():
    spec = LegSpec(side="both", noise_hu=30.0)
    found = find_fibulae(make_leg_phantom(spec))
    assert sorted(f.side for f in found) == ["left", "right"]
    for fibula in found:
        assert _off_axis_mm(spec, fibula) < 0.5
        assert fibula.length_mm > spec.fibula_length_mm - 15.0


def test_a_leaning_leg_is_followed_along_its_lean():
    spec = LegSpec(side="right", tilt_deg=6.0, spacing=(0.8, 0.8, 1.25))
    (fibula,) = find_fibulae(make_leg_phantom(spec))
    axis = fibula.proximal_end - fibula.distal_end
    axis /= np.linalg.norm(axis)
    assert np.degrees(np.arccos(np.clip(axis @ spec.fibula_axis(), -1.0, 1.0))) < 0.5
    assert _off_axis_mm(spec, fibula) < 0.5
    # Its length is measured along the bone, not up the scan.
    assert spec.fibula_length_mm - 15.0 < fibula.length_mm <= spec.fibula_length_mm + 0.5
    assert fibula.axis_at(150.0) @ spec.fibula_axis() > 0.999


def test_the_threshold_decides_what_is_bone():
    # Above the cortex nothing is bone, and nothing is found.
    volume = make_leg_phantom(LegSpec(side="right"))
    assert find_fibulae(volume, threshold=2000.0) == []
    assert len(find_fibulae(volume, threshold=400.0)) == 1


def test_a_scan_without_legs_gives_no_fibula():
    volume = make_leg_phantom(LegSpec(side="right"))
    volume.array[:] = -1000.0
    assert find_fibulae(volume) == []


def test_points_along_the_line_are_measured_from_the_distal_end():
    spec = LegSpec(side="right")
    (fibula,) = find_fibulae(make_leg_phantom(spec))
    middle = fibula.point_at(100.0)
    assert abs(middle[2] - (fibula.distal_end[2] + 100.0)) < 0.5
    # Past either end the line carries straight on.
    below = fibula.point_at(-20.0)
    assert abs(below[2] - (fibula.distal_end[2] - 20.0)) < 0.5
    lateral = fibula.lateral_at(100.0)
    assert abs(lateral @ fibula.axis_at(100.0)) < 1e-6


def test_a_leg_ct_loads_whole_without_hunting_for_jaws(tmp_path):
    from make_phantom import write_dicom_series

    from mandiplan.dicom_io import load_folder

    spec = LegSpec(side="left")
    write_dicom_series(make_leg_phantom(spec), tmp_path, patient_id="LEG", description="leg")
    volume, geometry, _ = load_folder(tmp_path, find_jaws=False)
    assert getattr(geometry, "jaw_volume", None) is None
    (fibula,) = find_fibulae(volume)
    assert fibula.side == "left"
    assert _off_axis_mm(spec, fibula) < 0.6


def test_the_fibula_surface_leaves_the_tibia_out():
    import vtk

    from mandiplan.render.fibula_mesh import fibula_surface

    spec = LegSpec(side="right")
    volume = make_leg_phantom(spec)
    (fibula,) = find_fibulae(volume)
    surface = fibula_surface(volume, fibula, 200.0)
    x0, x1, y0, y1, _, _ = surface.GetBounds()
    centre = spec.fibula_centre("right", 150.0)
    reach = spec.head_radius_mm * 1.2
    assert centre[0] - reach < x0 and x1 < centre[0] + reach
    assert centre[1] - reach < y0 and y1 < centre[1] + reach
    mass = vtk.vtkMassProperties()
    mass.SetInputData(surface)
    mass.Update()
    # A shaft of about 7 mm equivalent radius, wider at its ends: some 60 to 80 cm3.
    assert 50_000.0 < mass.GetVolume() < 90_000.0
