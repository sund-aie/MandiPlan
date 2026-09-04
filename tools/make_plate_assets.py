"""Generate the generic reconstruction-plate mesh assets.

These are **generic parametric approximations**, not manufacturer implants.
No vendor CAD is used, referenced or implied. Every asset this writes is
labelled as generic in its catalogue entry, and the application shows that
label wherever the plate is selected.

Each plate is built as a real solid, not a swept rectangle:

* the silhouette is the union of circular lobes centred on the screw holes,
  which overlap to give the scalloped waist of a reconstruction bar;
* the screw holes are genuine through-holes cut in the cap before extrusion,
  so the solid has one topological handle per hole;
* thickness is a real extrusion with both faces capped, and the result is
  merged and checked watertight.

This is a build tool, not part of the application: it may use VTK, which
``mandiplan/geometry`` may not. It makes no network calls. Run it to
regenerate everything after changing a profile:

    python3 tools/make_plate_assets.py

Assets land in ``mandiplan/data/plates/`` with a ``plates.json`` catalogue.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import vtk

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "mandiplan" / "data" / "plates"

#: Resolution of the generated boundaries. High enough that a hole is round
#: to well under the tolerance any bending check uses, low enough that the
#: whole catalogue stays a couple of megabytes.
LOBE_SEGMENTS = 26
HOLE_SEGMENTS = 36

GENERIC_NOTICE = (
    "Generic parametric approximation - not manufacturer-specific and not for "
    "clinical device selection."
)
PROVENANCE = (
    "Generated procedurally by tools/make_plate_assets.py in this repository "
    "from the published dimensional classes of reconstruction plates. No "
    "manufacturer CAD, drawing or measurement was used."
)
LICENCE = "CC0-1.0 (generated in-repo; no third-party rights)"


@dataclass
class PlateProfile:
    """The parameters a generic plate family is generated from."""

    family: str
    name: str
    category: str
    hole_pitch_mm: float
    width_mm: float
    thickness_mm: float
    hole_diameter_mm: float
    hole_counts: tuple[int, ...]
    #: In-plane preformed curvature radius, mm. None for a straight bar.
    preform_radius_mm: float | None = None
    #: Preformed angle at the middle of the plate, degrees, for angle plates.
    preform_angle_deg: float | None = None
    min_bend_radius_mm: float = 15.0
    max_bend_deg_per_node: float = 15.0
    notes: str = ""
    system_id: str = ""

    @property
    def lobe_radius_mm(self) -> float:
        return self.width_mm / 2.0

    @property
    def hole_radius_mm(self) -> float:
        return self.hole_diameter_mm / 2.0


PROFILES: tuple[PlateProfile, ...] = (
    PlateProfile(
        family="generic-recon-2.4",
        name="Generic 2.4 mm reconstruction bar, straight",
        category="reconstruction bar",
        hole_pitch_mm=9.0,
        width_mm=12.0,
        thickness_mm=2.4,
        hole_diameter_mm=2.9,
        hole_counts=(8, 12, 16, 20),
        system_id="recon-2.4-bar",
        notes="Load-bearing bar for a segmental defect. Bicortical screws.",
    ),
    PlateProfile(
        family="generic-recon-2.7",
        name="Generic 2.7 mm reconstruction bar, straight",
        category="reconstruction bar",
        hole_pitch_mm=10.0,
        width_mm=13.0,
        thickness_mm=2.7,
        hole_diameter_mm=3.2,
        hole_counts=(10, 16),
        system_id="recon-2.7-bar",
        notes="Heavier bar for a long defect or a poor-quality bone bed.",
    ),
    PlateProfile(
        family="generic-body-curved",
        name="Generic 2.4 mm body plate, preformed curve",
        category="mandibular body",
        hole_pitch_mm=9.0,
        width_mm=12.0,
        thickness_mm=2.4,
        hole_diameter_mm=2.9,
        hole_counts=(10, 14),
        preform_radius_mm=55.0,
        notes=(
            "Preformed to a 55 mm in-plane radius, roughly the lateral "
            "curvature of an adult mandibular body."
        ),
    ),
    PlateProfile(
        family="generic-angle",
        name="Generic 2.4 mm angle plate, preformed",
        category="mandibular angle and ramus",
        hole_pitch_mm=9.0,
        width_mm=12.0,
        thickness_mm=2.4,
        hole_diameter_mm=2.9,
        hole_counts=(10, 14),
        preform_angle_deg=115.0,
        notes=(
            "Preformed with a 115 degree turn at the middle, for a defect "
            "crossing the angle into the ramus."
        ),
    ),
)


def lobed_outline(
    centres: np.ndarray, lobe_radius: float, segments: int = LOBE_SEGMENTS
) -> np.ndarray:
    """The boundary of a chain of overlapping circles, as a closed loop.

    Returns (N, 2) points counter-clockwise. The lobes must overlap — that is
    what produces the scalloped waist between screw holes; a chain of
    separated circles is not a plate.
    """
    centres = np.asarray(centres, dtype=float).reshape(-1)
    if len(centres) == 1:
        angles = np.linspace(0.0, 2 * np.pi, segments * 4, endpoint=False)
        return np.column_stack(
            [centres[0] + lobe_radius * np.cos(angles), lobe_radius * np.sin(angles)]
        )

    pitch = float(np.diff(centres).min())
    if 2.0 * lobe_radius <= pitch:
        raise ValueError(
            f"lobes of radius {lobe_radius} mm do not overlap at a {pitch} mm "
            "pitch; a plate outline needs a continuous silhouette"
        )
    # Where neighbouring lobes cross, measured from a lobe's own centre.
    theta = math.atan2(
        math.sqrt(lobe_radius**2 - (pitch / 2.0) ** 2), pitch / 2.0
    )

    upper: list[tuple[float, float]] = []
    for i, centre in enumerate(centres):
        start = math.pi if i == 0 else math.pi - theta
        end = 0.0 if i == len(centres) - 1 else theta
        angles = np.linspace(start, end, segments)
        arc = [
            (centre + lobe_radius * math.cos(a), lobe_radius * math.sin(a))
            for a in angles
        ]
        # Consecutive arcs meet exactly at the crossing point; keep it once.
        upper.extend(arc if i == 0 else arc[1:])

    lower = [(x, -y) for x, y in reversed(upper[1:-1])]
    return np.array(upper + lower, dtype=float)


def _circle(cx: float, cy: float, radius: float, segments: int) -> np.ndarray:
    angles = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])


def _loops_to_polydata(loops: list[np.ndarray]) -> vtk.vtkPolyData:
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for loop in loops:
        start = points.GetNumberOfPoints()
        for x, y in loop:
            points.InsertNextPoint(float(x), float(y), 0.0)
        lines.InsertNextCell(len(loop) + 1)
        for i in range(len(loop)):
            lines.InsertCellPoint(start + i)
        lines.InsertCellPoint(start)
    poly.SetPoints(points)
    poly.SetLines(lines)
    return poly


def build_plate(profile: PlateProfile, holes: int) -> tuple[vtk.vtkPolyData, np.ndarray]:
    """A watertight plate solid and its hole centres, in local plate space.

    Local space: +x runs along the plate, +y across its width, +z is the
    outward face — the side that faces the surgeon, away from bone. The
    midplane is z = 0, so the solid spans -t/2 to +t/2.
    """
    pitch = profile.hole_pitch_mm
    centres = np.arange(holes, dtype=float) * pitch
    centres -= centres.mean()  # centre the plate on its own origin

    outline = lobed_outline(centres, profile.lobe_radius_mm)
    # Hole loops wind the other way so the triangulator reads them as holes.
    hole_loops = [
        _circle(c, 0.0, profile.hole_radius_mm, HOLE_SEGMENTS)[::-1] for c in centres
    ]

    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputData(_loops_to_polydata([outline, *hole_loops]))

    extrude = vtk.vtkLinearExtrusionFilter()
    extrude.SetInputConnection(triangulator.GetOutputPort())
    extrude.SetExtrusionTypeToNormalExtrusion()
    extrude.SetVector(0.0, 0.0, 1.0)
    extrude.SetScaleFactor(profile.thickness_mm)
    extrude.CappingOn()

    # Extrusion leaves the cap and wall points unmerged, which is why the
    # result is only watertight after cleaning.
    clean = vtk.vtkCleanPolyData()
    clean.SetInputConnection(extrude.GetOutputPort())
    clean.SetTolerance(1e-6)
    clean.PointMergingOn()

    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(clean.GetOutputPort())

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(triangles.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    solid = normals.GetOutput()
    # Extrusion runs from z = 0 to z = t; shift so the midplane is z = 0.
    solid = _translate(solid, (0.0, 0.0, -profile.thickness_mm / 2.0))

    hole_centres = np.column_stack(
        [centres, np.zeros(holes), np.zeros(holes)]
    )
    if profile.preform_radius_mm or profile.preform_angle_deg:
        solid, hole_centres = _preform(solid, hole_centres, profile)
    return solid, hole_centres


def _translate(polydata: vtk.vtkPolyData, offset) -> vtk.vtkPolyData:
    transform = vtk.vtkTransform()
    transform.Translate(*offset)
    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetTransform(transform)
    filt.SetInputData(polydata)
    filt.Update()
    return filt.GetOutput()


def _preform(
    solid: vtk.vtkPolyData, hole_centres: np.ndarray, profile: PlateProfile
) -> tuple[vtk.vtkPolyData, np.ndarray]:
    """Bend a straight blank into its preformed shape.

    In-plane only, about the +z axis, so the plate's faces stay faces and the
    screw-hole axes stay perpendicular to them. Arc length along +x is
    preserved exactly, which is what keeps the hole pitch correct.
    """
    if profile.preform_angle_deg is not None:
        half_span = float(np.abs(hole_centres[:, 0]).max())
        radius = half_span / math.radians(180.0 - profile.preform_angle_deg) * 2.0
    else:
        radius = float(profile.preform_radius_mm)

    def bend(points: np.ndarray) -> np.ndarray:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        angle = x / radius
        # The neutral axis sits on the plate midline, so y offsets ride in and
        # out of the bend radius rather than being stretched along it.
        r = radius - y
        return np.column_stack([r * np.sin(angle), radius - r * np.cos(angle), z])

    array = _points_of(solid)
    bent = bend(array)
    out = vtk.vtkPolyData()
    out.DeepCopy(solid)
    for i, point in enumerate(bent):
        out.GetPoints().SetPoint(i, *point)
    return out, bend(hole_centres)


def _points_of(polydata: vtk.vtkPolyData) -> np.ndarray:
    from vtkmodules.util.numpy_support import vtk_to_numpy

    return vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)


def audit(solid: vtk.vtkPolyData) -> dict:
    """Watertightness, volume and genus — the evidence the holes are real."""
    edges = vtk.vtkFeatureEdges()
    edges.SetInputData(solid)
    edges.BoundaryEdgesOn()
    edges.NonManifoldEdgesOn()
    edges.FeatureEdgesOff()
    edges.ManifoldEdgesOff()
    edges.Update()

    mass = vtk.vtkMassProperties()
    mass.SetInputData(solid)
    mass.Update()

    vertices = solid.GetNumberOfPoints()
    faces = solid.GetNumberOfCells()
    euler = vertices - (faces * 3 // 2) + faces
    return {
        "open_edges": int(edges.GetOutput().GetNumberOfCells()),
        "volume_mm3": round(float(mass.GetVolume()), 4),
        "surface_area_mm2": round(float(mass.GetSurfaceArea()), 4),
        "points": vertices,
        "triangles": faces,
        "euler_characteristic": int(euler),
        "genus": int((2 - euler) // 2),
    }


def write_stl(solid: vtk.vtkPolyData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(path))
    writer.SetFileTypeToBinary()
    writer.SetInputData(solid)
    writer.Write()


def catalogue_entry(
    profile: PlateProfile, holes: int, solid, hole_centres, mesh_name: str
) -> dict:
    bounds = solid.GetBounds()
    report = audit(solid)
    centreline = hole_centres.tolist()
    return {
        "id": f"{profile.family}-{holes}h",
        "name": f"{profile.name}, {holes} holes",
        "family": profile.family,
        "category": profile.category,
        "exact": False,
        "generic": True,
        "status_label": "Generic parametric approximation",
        "notice": GENERIC_NOTICE,
        "mesh": mesh_name,
        "mesh_format": "stl",
        "units": "mm",
        "unit_scale": 1.0,
        "coordinate_convention": (
            "+x along the plate, +y across its width, +z out of the outer "
            "face (away from bone); origin at the plate's centre, midplane z=0"
        ),
        "thickness_mm": profile.thickness_mm,
        "width_mm": profile.width_mm,
        "hole_count": holes,
        "hole_pitch_mm": profile.hole_pitch_mm,
        "hole_diameter_mm": profile.hole_diameter_mm,
        "length_mm": round(float(bounds[1] - bounds[0]), 4),
        "hole_centres_mm": [[round(v, 6) for v in row] for row in centreline],
        "hole_axes": [[0.0, 0.0, 1.0] for _ in range(holes)],
        "centreline_mm": [[round(v, 6) for v in row] for row in centreline],
        "side": "universal",
        "mirrorable": True,
        "preform_radius_mm": profile.preform_radius_mm,
        "preform_angle_deg": profile.preform_angle_deg,
        "deformation": {
            "min_bend_radius_mm": profile.min_bend_radius_mm,
            "max_bend_deg_per_node": profile.max_bend_deg_per_node,
            "protected_radius_mm": round(profile.hole_diameter_mm * 1.15, 4),
            "bendable": "inter-hole bridges only",
        },
        "provenance": PROVENANCE,
        "licence": LICENCE,
        "system_id": profile.system_id or None,
        "audit": report,
        "notes": profile.notes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--family", help="regenerate only this family", default=None
    )
    args = parser.parse_args(argv)

    entries = []
    failures = []
    for profile in PROFILES:
        if args.family and profile.family != args.family:
            continue
        for holes in profile.hole_counts:
            solid, centres = build_plate(profile, holes)
            report = audit(solid)
            mesh_name = f"{profile.family}-{holes}h.stl"
            write_stl(solid, args.out / mesh_name)
            entries.append(
                catalogue_entry(profile, holes, solid, centres, mesh_name)
            )
            flag = "ok "
            if report["open_edges"] or report["genus"] != holes:
                flag = "BAD"
                failures.append(mesh_name)
            print(
                f"  {flag} {mesh_name:<32} {report['triangles']:>6} tris  "
                f"genus {report['genus']:>2} (holes {holes:>2})  "
                f"open edges {report['open_edges']}"
            )

    catalogue = {
        "note": [
            "Generic reconstruction-plate assets generated in this repository.",
            GENERIC_NOTICE,
            "No manufacturer CAD, part number or trade dress is used or implied.",
            "Drop a licensed exact asset and its metadata here to add one; set",
            "exact to true only when the provenance genuinely supports it.",
        ],
        "generated_by": "tools/make_plate_assets.py",
        "plates": entries,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plates.json").write_text(
        json.dumps(catalogue, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n{len(entries)} assets -> {args.out}")
    if failures:
        print(f"FAILED: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
