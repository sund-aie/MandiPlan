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

from .constants import ATTRIBUTION, DISCLAIMER
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


#: Binary STL keeps an 80-byte header ahead of the triangle count, and two
#: spare bytes on every triangle. The attribution goes in both, so stripping
#: the header alone does not remove it.
_STL_HEADER_BYTES = 80


def stamp_stl(path: Path, attribution: str = ATTRIBUTION) -> Path:
    """Write the attribution into a binary STL's header and attribute bytes.

    The header is the conventional place and survives most round-trips. The
    per-triangle attribute field is a second copy in the geometry block, so a
    tool that rewrites the header still carries the credit. Neither is
    tamper-proof — any file can be edited — but both travel with the mesh
    through ordinary use.
    """
    data = bytearray(path.read_bytes())
    if len(data) < _STL_HEADER_BYTES + 4:
        return path

    header = attribution.encode("utf-8", "replace")[:_STL_HEADER_BYTES]
    data[: len(header)] = header
    data[len(header) : _STL_HEADER_BYTES] = b" " * (_STL_HEADER_BYTES - len(header))

    count = int.from_bytes(data[_STL_HEADER_BYTES : _STL_HEADER_BYTES + 4], "little")
    mark = attribution.encode("utf-8", "replace")
    start = _STL_HEADER_BYTES + 4
    for triangle in range(count):
        offset = start + triangle * 50 + 48
        if offset + 2 > len(data):
            break
        data[offset] = mark[(triangle * 2) % len(mark)]
        data[offset + 1] = mark[(triangle * 2 + 1) % len(mark)]
    path.write_bytes(bytes(data))
    return path


def read_stl_attribution(path: Path) -> str:
    """The attribution recovered from a binary STL header, for checking."""
    with Path(path).open("rb") as handle:
        return handle.read(_STL_HEADER_BYTES).decode("utf-8", "replace").rstrip()


def _write_header(handle, lines: list[str]) -> None:
    handle.write(f"# {DISCLAIMER}\n")
    handle.write(f"# {ATTRIBUTION}\n")
    for line in lines:
        handle.write(f"# {line}\n")


def asset_provenance_lines(asset=None, fitted=None) -> list[str]:
    """Which plate this plan was made with, and what that plate actually is.

    A plan that names a plate has to say whether that plate is a licensed
    exact asset or a generic approximation, and how it was placed. Anything
    less invites the file being read as a device selection.
    """
    if asset is None:
        return ["plate asset: none selected"]
    lines = [
        f"plate asset id: {asset.id}",
        f"plate asset: {asset.name}",
        f"plate asset status: {asset.status_label}",
        f"plate asset exact geometry: {str(asset.exact).lower()}",
        f"plate asset source: {asset.provenance}",
        f"plate asset licence: {asset.licence}",
        f"plate thickness: {asset.thickness_mm:.2f} mm",
        f"plate holes: {asset.hole_count} at {asset.hole_pitch_mm:.2f} mm pitch",
    ]
    if fitted is not None:
        lines.append(
            "plate placement: rigid (rotation and translation only; no scaling)"
        )
        lines.append(
            f"plate placement residual: max {fitted.max_residual_mm:.2f} mm, "
            f"rms {fitted.rms_residual_mm:.2f} mm"
        )
        for warning in fitted.warnings:
            lines.append(f"plate fit warning: {warning}")
    if not asset.exact:
        lines.append(
            "NOTE: this plate is a generic parametric approximation. It is not "
            "manufacturer-specific and must not be read as a device selection."
        )
    return lines


def write_bend_csv(
    path: str | Path,
    plan: PlatePlan,
    width_mm: float,
    thickness_mm: float,
    asset=None,
    fitted=None,
) -> Path:
    """Write the plate bend table, units in the column names."""
    path = Path(path)
    lines = [
        "MandiPlan reconstruction-plate bend instructions",
        f"screw-hole pitch: {plan.pitch_mm:.2f} mm",
        f"bending template ribbon: {width_mm:.2f} mm wide x {thickness_mm:.2f} mm thick",
        f"nodes: {len(plan.nodes)}",
        f"total plate length along the path: {plan.total_length_mm:.2f} mm",
        "",
        *asset_provenance_lines(asset, fitted),
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


def write_steps_csv(
    path: str | Path, steps, system, kit, fit=None
) -> Path:
    """Write the bench bending steps for one plate system and one kit."""
    path = Path(path)
    lines = [
        "MandiPlan plate bending steps",
        f"plate system: {system.name}",
        f"bending kit: {kit.name}",
        "distances are measured from the proximal cut end of the plate",
    ]
    if fit is not None:
        lines.append(f"fit: {fit.verdict}")
        lines.extend(f"problem: {p}" for p in fit.problems)
    lines.append(
        "plate dimensions here are generic profiles by size class; check them "
        "against the specification sheet of the system you are holding"
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        _write_header(handle, lines)
        writer = csv.writer(handle)
        writer.writerow(["step", "kind", "distance_from_cut_end_mm", "angle_deg",
                         "instrument", "instruction"])
        for step in steps:
            writer.writerow([
                step.order,
                step.kind,
                _num(step.distance_mm),
                _num(step.angle_deg),
                step.instrument,
                step.text,
            ])
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


def write_plate_stl(path: str | Path, fitted) -> Path:
    """Write the fitted plate — the real asset mesh under its placement.

    This is the same geometry the viewport draws. There is no separate proxy
    to fall out of step with it.
    """
    path = Path(path)
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(triangles_to_polydata(fitted.points, fitted.triangles))
    writer.Write()
    return stamp_stl(path)


def write_template_stl(
    path: str | Path, plan: PlatePlan, width_mm: float, thickness_mm: float
) -> Path:
    """Write a plain swept bending template as a binary STL.

    This is a bending aid, not the implant: a rectangular ribbon along the
    planned path, for marking a blank. It is never what the plate looks like —
    use :func:`write_plate_stl` for the plate itself.
    """
    path = Path(path)
    points, triangles = ribbon_mesh(plan, width_mm, thickness_mm)
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(triangles_to_polydata(points, triangles))
    writer.Write()
    return stamp_stl(path)


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
    return stamp_stl(path)


def _num(value: float) -> str:
    return "" if not np.isfinite(value) else f"{value:.3f}"
