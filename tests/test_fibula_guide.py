"""The fibula cutting guide, made for a plan on the leg phantom and followed in use."""

from __future__ import annotations

import numpy as np
import pytest
from vtkmodules.util.numpy_support import vtk_to_numpy

from make_leg_phantom import LegSpec, make_leg_phantom
from mandiplan.geometry.fibula import find_fibulae
from mandiplan.geometry.fibula_guide import (
    GAP_MM,
    REACH_MM,
    SLOT_MM,
    SLOT_REACH_MM,
    check_guide,
    guide_field_on,
    plan_guide,
)
from mandiplan.geometry.fibula_plan import choose_joints, defect_course, plan_fibula, section_rim, with_guide
from mandiplan.render.bending_guide_mesh import _is_closed
from mandiplan.render.fibula_guide_mesh import clearance_to_bone, guide_grid, guide_polydata
from mandiplan.render.fibula_mesh import fibula_surface
from test_fibula_plan import _cuts, _horseshoe


@pytest.fixture(scope="module")
def leg():
    volume = make_leg_phantom(LegSpec(side="right", spacing=(0.8, 0.8, 1.0)))
    (fibula,) = find_fibulae(volume)
    surface = fibula_surface(volume, fibula, 200.0)
    points = vtk_to_numpy(surface.GetPoints().GetData()).astype(float)
    return volume, fibula, surface, points


def _plan(leg, face: str | None):
    """Three pieces round the horseshoe's arch; spaced for a guide on ``face``,
    or, with None, for the bone alone."""
    _, fibula, _, points = leg
    frames, vertices = _horseshoe()
    course = defect_course(vertices, frames, _cuts(frames, 40.0, 120.0), 7.0)
    joints, deviation = choose_joints(course.points)

    def rim_at(s, frame):
        rim = section_rim(points, fibula, s, frame)
        if face is None:
            return rim
        return with_guide(rim, fibula.side_at(s, face), frame[:, 0], REACH_MM, SLOT_REACH_MM + 0.5)

    return plan_fibula(course, fibula, joints, deviation, rim_at=rim_at)


def _made(leg, plan, face: str):
    volume, fibula, surface, points = leg
    guide = plan_guide(plan, fibula, points, face)
    grid = guide_grid(guide, volume, fibula, 200.0)
    solid = guide_field_on(guide, grid)
    mesh = guide_polydata(guide, grid, solid)
    check = check_guide(guide, plan, grid, solid, clearance_to_bone(mesh, surface))
    return guide, grid, mesh, check


@pytest.fixture(scope="module")
def lateral(leg):
    plan = _plan(leg, "lateral")
    return (plan, *_made(leg, plan, "lateral"))


@pytest.fixture(scope="module")
def anterior(leg):
    plan = _plan(leg, "anterior")
    return (plan, *_made(leg, plan, "anterior"))


def test_a_guide_on_the_lateral_face_works_in_use(lateral):
    plan, guide, _, mesh, check = lateral
    assert check.ok, check.lines()
    # Seated, it stands off the bone by the gap it was made with, at its closest.
    assert GAP_MM - 0.05 <= check.bone_clear_mm <= GAP_MM + 0.2
    assert check.lift_off_mm == 0.0
    assert check.kerf_into_piece_mm == 0.0 and check.uncut_share == 0.0
    # One closed solid, every piece screwed down and numbered.
    assert _is_closed(mesh)
    assert {s.piece for s in guide.screws} == {p.number for p in plan.pieces}
    assert len(guide.labels) == len(plan.pieces) + 1  # and the P
    # The two ends of the graft are slots; each joint a window or two slots.
    assert not guide.openings[0].is_window and not guide.openings[-1].is_window
    assert sum(len(o.cuts) for o in guide.openings) == 2 * len(plan.pieces)


def test_a_guide_on_the_anterior_face_works_and_takes_less_fibula(lateral, anterior):
    plan, _, _, mesh, check = anterior
    assert check.ok, check.lines()
    assert _is_closed(mesh)
    # Each wedge narrows towards the lateral side, the side that faces out of the
    # jaw: a guide there needs its cuts further apart than one in front.
    assert plan.harvest_length_mm < lateral[0].harvest_length_mm - 10.0


def test_every_slot_has_its_kept_wall_on_the_planned_cut(lateral):
    plan, guide, grid, _, _ = lateral
    faces = {id(f) for p in plan.pieces for f in (p.distal_face, p.proximal_face)}
    assert len(guide.cuts) == len(faces)
    for cut in guide.cuts:
        if not any(o.cuts == (cut,) for o in guide.openings):
            continue
        # Just on the kept piece's side of the cut, no slot; just past it, the slot.
        across = cut.point + (cut.centre * cut.across) + 0.5 * cut.top * cut.travel
        kept = across - 0.1 * cut.normal
        thrown = across + 0.5 * SLOT_MM * cut.normal
        assert cut.slot(kept[None, :])[0] > 0.0
        assert cut.slot(thrown[None, :])[0] < 0.0


def test_the_check_finds_cuts_that_would_cross_in_the_guide(leg):
    # Spaced for the bone alone, the wedges narrow into a lateral guide and the
    # two cuts of a joint come together inside it: a blade no longer fits.
    plan = _plan(leg, None)
    *_, check = _made(leg, plan, "lateral")
    assert not check.ok
    assert check.blade_clear_mm < 0.0


def test_the_bone_distance_agrees_with_the_bone_surface(lateral, leg):
    _, _, grid, _, _ = lateral
    points = leg[3]
    lo = grid.origin + 1.0
    hi = grid.origin + np.array(grid.shape_xyz) * grid.step - 1.0
    inside = points[np.all((points > lo) & (points < hi), axis=1)]
    distance = grid.sample(inside)
    assert abs(float(np.median(distance))) < 0.05
    assert float(np.percentile(np.abs(distance), 95)) < 0.15


def _oblique(leg, a: float, b: float, degrees: float, face: str = "lateral"):
    """A plan for a defect whose cuts are turned ``degrees`` from square, as on a real jaw."""
    from mandiplan.geometry.resection import CutPlane

    _, fibula, _, points = leg
    frames, vertices = _horseshoe()

    def turned(s, into, angle):
        tangent, up, _ = frames.frame_at(s)
        r = np.radians(angle)
        return CutPlane(frames.point_at(s), into * (tangent * np.cos(r) + np.cross(up, tangent) * np.sin(r)))

    course = defect_course(vertices, frames, [turned(a, 1.0, degrees), turned(b, -1.0, -degrees)], 7.0)
    joints, deviation = choose_joints(course.points, max_pieces=2)

    def rim_at(s, frame):
        rim = section_rim(points, fibula, s, frame)
        return with_guide(rim, fibula.side_at(s, face), frame[:, 0], REACH_MM, SLOT_REACH_MM + 0.5)

    return plan_fibula(course, fibula, joints, deviation, rim_at=rim_at)


def test_oblique_cuts_still_give_a_guide_that_works(leg):
    # The resection's cuts turned 45 degrees from square: the graft's ends lean
    # by some 33 degrees, the pieces are short, and a joint stands between them.
    plan = _oblique(leg, 45.0, 88.0, 45.0)
    assert len(plan.pieces) == 2
    assert min(f.lean_deg for f in (plan.pieces[0].distal_face, plan.pieces[-1].proximal_face)) > 30.0
    *_, check = _made(leg, plan, "lateral")
    assert check.ok, check.lines()


def test_a_screw_stays_in_its_piece_however_steep_the_ends(leg):
    # Turned the other way, the cuts leave one piece whose ends lean 77 degrees:
    # the plan says so, and each screw's whole path is still inside the piece,
    # though placed only by where the guide's surface is, it would graze a cut.
    plan = _oblique(leg, 50.0, 92.0, -50.0)
    assert len(plan.pieces) == 1 and plan.pieces[0].distal_face.lean_deg > 70.0
    assert any("leans" in w for w in plan.warnings)
    *_, check = _made(leg, plan, "lateral")
    assert check.ok, check.lines()
    assert check.screw_margin_mm >= 2.0
