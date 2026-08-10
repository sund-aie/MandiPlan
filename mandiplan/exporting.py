"""CSV and STL export.

Every exported CSV carries the same non-clinical-use header as the status bar,
because a table of bend angles is exactly the artefact that gets printed and
carried into a workshop.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import vtk

from .constants import DISCLAIMER
from .geometry.plate import PlatePlan, ribbon_mesh
from .render.convert import triangles_to_polydata

BEND_COLUMNS = [
    "node",
    "cumulative_mm",
    "in_plane_bend_deg",
    "out_of_plane_bend_deg",
    "twist_deg",
    "segment_length_mm",
]

_SIGN_CONVENTIONS = [
    "Sign conventions (right-hand rule, degrees):",
    "  in-plane bend      rotation about the outward surface normal n; positive is",
    "                     counter-clockwise seen from the plate's outer face.",
    "  out-of-plane bend  rotation about the binormal b = t x n; positive lifts the",
    "                     plate away from the bone, negative bends it into the bone.",
    "  twist              rotation of the plate's outer face about the direction of",
    "                     travel t; positive is right-handed about t.",
]


def _write_header(handle, lines: list[str]) -> None:
    handle.write(f"# {DISCLAIMER}\n")
    for line in lines:
        handle.write(f"# {line}\n")


def write_bend_csv(path: str | Path, plan: PlatePlan, width_mm: float, thickness_mm: float) -> Path:
    """Write the plate bend table, units in the column names."""
    path = Path(path)
    lines = [
        "MandiPlan reconstruction-plate bend instructions",
        f"screw-hole pitch: {plan.pitch_mm:.2f} mm",
        f"plate ribbon: {width_mm:.2f} mm wide x {thickness_mm:.2f} mm thick",
        f"nodes: {len(plan.nodes)}",
        f"total plate length along the path: {plan.total_length_mm:.2f} mm",
        "",
        *_SIGN_CONVENTIONS,
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        _write_header(handle, lines)
        writer = csv.writer(handle)
        writer.writerow(BEND_COLUMNS)
        for node in plan.bends:
            writer.writerow(
                [
                    node.index,
                    f"{node.cumulative_mm:.3f}",
                    _num(node.in_plane_deg),
                    _num(node.out_of_plane_deg),
                    _num(node.twist_deg),
                    _num(node.segment_length_mm),
                ]
            )
    return path


def write_plan_summary_csv(path: str | Path, rows: list[tuple[str, str]], title: str) -> Path:
    """Write a two-column quantity/value summary (resection readout, measurements)."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        _write_header(handle, [title])
        writer = csv.writer(handle)
        writer.writerow(["quantity", "value"])
        writer.writerows(rows)
    return path


def write_template_stl(
    path: str | Path, plan: PlatePlan, width_mm: float, thickness_mm: float
) -> Path:
    """Write the bending template as a binary STL."""
    path = Path(path)
    points, triangles = ribbon_mesh(plan, width_mm, thickness_mm)
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(triangles_to_polydata(points, triangles))
    writer.Write()
    return path


def write_surface_stl(path: str | Path, polydata) -> Path:
    """Write any surface (e.g. the resected fragment) as a binary STL."""
    path = Path(path)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(polydata)
    triangles.Update()
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(triangles.GetOutput())
    writer.Write()
    return path


def _num(value: float) -> str:
    return "" if not np.isfinite(value) else f"{value:.3f}"
