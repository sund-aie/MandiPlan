"""Every export carries the attribution, in the file itself."""

from __future__ import annotations

import numpy as np
import pytest
import vtk

from helpers import circular_arc_path, rel_error
from mandiplan.constants import ATTRIBUTION, DISCLAIMER
from mandiplan.exporting import (
    read_stl_attribution,
    write_bend_csv,
    write_plan_summary_csv,
    write_steps_csv,
    write_surface_stl,
    write_template_stl,
)
from mandiplan.geometry.plate import compute_plate_plan
from mandiplan.plate_catalog import fit_check, kit_by_id, system_by_id


@pytest.fixture(scope="module")
def plan():
    path = circular_arc_path(30.0, 2.0)
    normals = np.tile([0.0, 0.0, 1.0], (len(path), 1))
    return compute_plate_plan(path, normals, 9.0)


def _attribute_bytes(path) -> bytes:
    raw = path.read_bytes()
    count = int.from_bytes(raw[80:84], "little")
    return b"".join(raw[84 + i * 50 + 48 : 84 + i * 50 + 50] for i in range(count))


def test_the_template_stl_header_carries_the_attribution(plan, tmp_path):
    out = write_template_stl(tmp_path / "template.stl", plan, 12.0, 2.0)
    assert read_stl_attribution(out) == ATTRIBUTION
    assert "Ahmed Alsunaidi" in read_stl_attribution(out)
    assert "vesper" in read_stl_attribution(out)


def test_the_attribution_is_also_in_the_triangle_attribute_bytes(plan, tmp_path):
    """A second copy inside the geometry block, not only in the header."""
    out = write_template_stl(tmp_path / "template.stl", plan, 12.0, 2.0)
    recovered = _attribute_bytes(out).decode("utf-8", "replace")
    assert ATTRIBUTION[:20] in recovered


def test_stamping_does_not_damage_the_mesh(plan, tmp_path):
    out = write_template_stl(tmp_path / "template.stl", plan, 12.0, 2.0)
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(out))
    reader.Update()
    mesh = reader.GetOutput()
    assert mesh.GetNumberOfPolys() > 0

    mass = vtk.vtkMassProperties()
    mass.SetInputData(mesh)
    mass.Update()
    expected = 12.0 * 2.0 * plan.total_length_mm
    assert rel_error(mass.GetVolume(), expected) < 0.05


def test_a_surface_stl_is_stamped_too(tmp_path):
    source = vtk.vtkSphereSource()
    source.Update()
    out = write_surface_stl(tmp_path / "surface.stl", source.GetOutput())
    assert read_stl_attribution(out) == ATTRIBUTION


@pytest.mark.parametrize("writer", ["bends", "steps", "summary"])
def test_every_csv_carries_the_attribution_and_the_disclaimer(writer, plan, tmp_path):
    system = system_by_id("recon-2.4-bar")
    if writer == "bends":
        out = write_bend_csv(tmp_path / "a.csv", plan, 12.0, 2.0)
    elif writer == "steps":
        fit = fit_check(plan, system)
        from mandiplan.bending_steps import generate_steps

        steps = generate_steps(plan, system, kit_by_id("bending-irons"), fit)
        out = write_steps_csv(tmp_path / "b.csv", steps, system, kit_by_id("bending-irons"), fit)
    else:
        out = write_plan_summary_csv(tmp_path / "c.csv", [("a", "1")], "title")

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"# {DISCLAIMER}"
    assert lines[1] == f"# {ATTRIBUTION}"
