"""Placing a plate with no clicks: where it goes, its screws, and whether it will do."""

from __future__ import annotations

import numpy as np
import pytest
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from mandiplan.geometry import cpr
from mandiplan.geometry.auto_plate import (
    BAD,
    CAUTION,
    GOOD,
    NO_BONE,
    SCREW,
    SKIP,
    SPARE,
    advise_holes,
    jaw_route,
    judge_plate,
    place_chain,
    plate_turns,
    recommend_length,
)
from mandiplan.geometry.spline import ArchCurve

PITCH = 8.0


def _jaw():
    """A U of a jaw as a tube of bone, condyle to condyle, with its arch curve."""
    theta = np.radians(np.linspace(-90.0, 90.0, 13))
    body = np.column_stack([50 * np.sin(theta), -50 * np.cos(theta), np.zeros(13)])
    seeds = np.vstack(
        [[[-50.0, 25.0, 45.0], [-50.0, 15.0, 15.0]], body, [[50.0, 15.0, 15.0], [50.0, 25.0, 45.0]]]
    )
    frames = cpr.build_frames(ArchCurve(seeds), 0.5)
    line = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    for p in frames.points[::4]:
        points.InsertNextPoint(*p)
    line.SetPoints(points)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(points.GetNumberOfPoints(), list(range(points.GetNumberOfPoints())))
    line.SetLines(cells)
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(line)
    tube.SetRadius(9.0)
    tube.SetNumberOfSides(48)
    tube.CappingOn()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(tube.GetOutputPort())
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(triangles.GetOutputPort())
    clean.Update()
    return frames, vtk_to_numpy(clean.GetOutput().GetPoints().GetData()).astype(float)


@pytest.fixture(scope="module")
def jaw():
    return _jaw()


@pytest.fixture(scope="module")
def route(jaw):
    frames, vertices = jaw
    return jaw_route(vertices, frames, 8.0)


def _straight(holes):
    spacings = np.full(holes - 1, PITCH)
    return spacings, np.zeros(max(holes - 2, 0))


def test_the_route_runs_on_the_outer_face_above_the_lower_border(jaw, route):
    frames, vertices = jaw
    assert route is not None and route.valid.mean() > 0.9
    # Halfway round the chin: the middle of the jaw.
    i = int(np.argmin(np.abs(route.s - 0.5 * frames.length_mm)))
    point = route.points[i]
    # A tube of radius 9 about z = 0: a plate 8 wide, its edge 2 above the border,
    # has its middle 6 above the border, so at z = -3.
    assert point[2] == pytest.approx(-3.0, abs=1.5)
    # On the outside of the arch, on the bone, not on its far side.
    assert point[1] < -55.0
    assert route.outward[i] @ np.array([0.0, -1.0, 0.0]) > 0.9
    nearest = np.min(np.linalg.norm(vertices[::3] - point, axis=1))
    assert nearest < 2.5


def test_a_plate_is_centred_on_the_resection_with_the_same_holes_each_side(jaw, route):
    frames, _ = jaw
    middle = 0.4 * frames.length_mm
    defect = (middle - 12.0, middle + 12.0)
    placed = place_chain(route, *_straight(12), defect)
    assert placed is not None and not placed.off_route
    assert min(placed.holes_before, placed.holes_after) >= 3
    assert abs(placed.holes_before - placed.holes_after) <= 1
    centre = 0.5 * (placed.arcs_mm[0] + placed.arcs_mm[-1])
    assert centre == pytest.approx(middle, abs=PITCH)
    np.testing.assert_allclose(np.linalg.norm(np.diff(placed.points, axis=0), axis=1), PITCH, atol=0.05)


def test_with_no_resection_a_straight_plate_goes_round_the_chin(jaw, route):
    frames, _ = jaw
    placed = place_chain(route, *_straight(10), None)
    centre = 0.5 * (placed.arcs_mm[0] + placed.arcs_mm[-1])
    assert centre == pytest.approx(0.5 * frames.length_mm, abs=12.0)


def test_a_preformed_plate_goes_to_where_the_jaw_has_its_shape():
    """A plate made with a corner is put at the jaw's corner, the right way round."""
    # The route: straight, a 50-degree corner in its plane, straight.
    s = np.arange(0.0, 200.0, 1.0)
    heading = np.radians(np.interp(s, [0.0, 120.0, 130.0, 200.0], [0.0, 0.0, 50.0, 50.0]))
    points = np.column_stack([np.cumsum(np.cos(heading)), np.cumsum(np.sin(heading)), np.zeros(len(s))])
    outward = np.tile([0.0, 0.0, 1.0], (len(s), 1))
    from mandiplan.geometry.auto_plate import JawRoute

    route = JawRoute(s, points, outward, np.full(len(s), 20.0), np.ones(len(s), dtype=bool))
    # A plate with the same corner between its fifth and seventh holes.
    angles = np.radians([0, 0, 0, 0, 0, 25, 50, 50, 50, 50])
    steps = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(10)]) * PITCH
    centres = np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])
    axes = np.tile([0.0, 0.0, 1.0], (11, 1))
    turns = plate_turns(centres, axes)
    placed = place_chain(route, np.linalg.norm(np.diff(centres, axis=0), axis=1), turns)
    assert placed.turn_needed_deg < 12.0
    assert not placed.reversed
    # Its corner holes lie at the jaw's corner.
    assert placed.arcs_mm[5] == pytest.approx(125.0, abs=10.0)
    # Bent the other way, it could only be made to fit by turning it end for end.
    mirrored = place_chain(route, np.linalg.norm(np.diff(centres, axis=0), axis=1), -turns)
    assert mirrored.reversed


def test_the_shortest_length_that_holds_four_screws_each_side_is_recommended(jaw, route):
    frames, _ = jaw
    middle = 0.4 * frames.length_mm
    defect = (middle - 12.0, middle + 12.0)
    plates = [(*_straight(n), n) for n in (6, 8, 10, 12, 14, 20)]
    chosen = recommend_length(route, plates, defect)
    placed = place_chain(route, *_straight(chosen), defect)
    assert min(placed.holes_before, placed.holes_after) >= 4
    shorter = place_chain(route, *_straight(plates[[p[2] for p in plates].index(chosen) - 1][2]), defect)
    assert min(shorter.holes_before, shorter.holes_after) < 4
    # With nothing long enough, the longest there is.
    assert recommend_length(route, plates[:2], defect) == 8
    assert recommend_length(route, plates, None) == 10


def _advice(defect=(40.0, 60.0)):
    arcs = np.arange(10.0, 96.0, 8.0)
    gaps = np.full(len(arcs), 0.6)
    angles = np.full(len(arcs), 5.0)
    depths = np.full(len(arcs), 11.0)
    return arcs, gaps, angles, depths, defect


def test_each_hole_gets_the_right_advice():
    arcs, gaps, angles, depths, defect = _advice()
    gaps[0] = 4.0
    angles[1] = 60.0
    holes = advise_holes(arcs, gaps, angles, depths, defect)
    states = [h.state for h in holes]
    assert states[0] == NO_BONE and "off the bone" in holes[0].why
    assert states[1] == SKIP and "off the bone" in holes[1].why
    for hole, arc in zip(holes, arcs):
        if defect[0] <= arc <= defect[1]:
            assert hole.state == NO_BONE and "resection" in hole.why
        elif min(abs(arc - defect[0]), abs(arc - defect[1])) < 5.0:
            assert hole.state == SKIP
    # The nearest sound holes outward from each cut carry the screws.
    screws = [a for a, h in zip(arcs, holes) if h.state == SCREW]
    assert sum(a < defect[0] for a in screws) == 2  # all that are sound before it
    assert sum(a > defect[1] for a in screws) == 4
    spare = [a for a, h in zip(arcs, holes) if h.state == SPARE]
    assert all(a > max(screws) for a in spare)


def test_a_hole_with_too_little_bone_under_it_gets_no_screw():
    arcs, gaps, angles, depths, defect = _advice(None)
    depths[2] = 2.0
    depths[3] = float("nan")
    holes = advise_holes(arcs, gaps, angles, depths, None)
    assert holes[2].state == SKIP and holes[3].state == SKIP


def _verdict(defect=(44.0, 60.0), arcs=None, **kwargs):
    arcs = np.arange(0.0, 104.0, 8.0) if arcs is None else arcs
    n = len(arcs)
    holes = advise_holes(arcs, np.full(n, 0.6), np.full(n, 5.0), np.full(n, 11.0), defect)
    return judge_plate(holes, arcs, defect, (-30.0, 240.0), **kwargs)


def test_a_plate_that_holds_well_is_applicable():
    verdict = _verdict(arcs=np.arange(4.0, 104.0, 8.0))
    assert verdict.level == GOOD and verdict.headline == "Applicable" and not verdict.reasons


def test_three_screws_a_side_will_do_but_is_said():
    verdict = _verdict(arcs=np.arange(4.0, 92.0, 8.0), longer_holes=14)
    assert verdict.level == CAUTION
    assert any("4 are better" in r for r in verdict.reasons)
    assert verdict.suggested_holes == 14


def test_a_plate_too_short_for_the_resection_is_not_applicable():
    verdict = _verdict(defect=(30.0, 60.0), arcs=np.arange(10.0, 80.0, 8.0), longer_holes=16)
    assert verdict.level == BAD and verdict.headline == "Not applicable"
    assert any("at least 3" in r and "one side" in r for r in verdict.reasons)
    assert verdict.suggested_holes == 16


def test_a_resection_through_to_the_condyle_has_nothing_to_hold_on_to():
    verdict = _verdict(defect=(100.0, 240.0), arcs=np.arange(4.0, 96.0, 8.0))
    assert verdict.level == BAD
    assert any("condyle" in r for r in verdict.reasons)


def test_bends_past_the_plates_limit_rule_it_out():
    verdict = _verdict(arcs=np.arange(4.0, 104.0, 8.0), bend_problems=["The jaw asks for too much."])
    assert verdict.level == BAD and "The jaw asks for too much." in verdict.reasons


def test_a_preformed_plate_that_does_not_suit_the_site_is_not_applicable(route):
    arcs = np.arange(4.0, 104.0, 8.0)
    placed = place_chain(route, *_straight(len(arcs)), None)
    placed.turn_needed_deg = 40.0
    verdict = _verdict(arcs=arcs, placement=placed, is_preformed=True)
    assert verdict.level == BAD and any("does not suit" in r for r in verdict.reasons)
    # A straight plate bent that way is a matter of bending, not of the wrong plate.
    assert _verdict(arcs=arcs, placement=placed, is_preformed=False).level == GOOD


def test_a_plate_wider_than_the_bone_it_lies_on_is_not_applicable(route):
    arcs = np.arange(4.0, 104.0, 8.0)
    placed = place_chain(route, *_straight(len(arcs)), None)
    verdict = _verdict(arcs=arcs, placement=placed, route=route, plate_width_mm=40.0)
    assert verdict.level == BAD and any("hang over" in r for r in verdict.reasons)
