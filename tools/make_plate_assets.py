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
    #: Alloy id from mandiplan/materials.py.
    material_id: str = "cp-ti-grade-4"
    #: True for plates that carry the mandible across a defect. A 1.0-1.5 mm
    #: adaptation plate is not one, and saying so matters.
    load_bearing: bool = True
    #: Dimensional class this profile is drawn from, for the record.
    dimensional_class: str = ""
    #: Half-width of the waist between lobes, mm. Only needed when the lobes
    #: are too far apart to overlap, as on a narrow low-profile plate: there
    #: the silhouette is separated lobes joined by a straight waisted bridge.
    waist_half_width_mm: float | None = None

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
        material_id="cp-ti-grade-4",
        dimensional_class="2.4 mm load-bearing reconstruction class",
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
        material_id="cp-ti-grade-4",
        dimensional_class="2.7 mm load-bearing reconstruction class",
        notes="Heavier bar for a long defect or a poor-quality bone bed.",
    ),
    PlateProfile(
        family="generic-recon-2.0",
        name="Generic 2.0 mm reconstruction plate, straight",
        category="reconstruction bar",
        hole_pitch_mm=8.0,
        width_mm=10.0,
        thickness_mm=2.0,
        hole_diameter_mm=2.5,
        hole_counts=(10, 14, 20),
        material_id="ti-6al-4v-eli",
        dimensional_class=(
            "2.0 mm primary-reconstruction class; published technique guides "
            "treat 2.0 mm as the thinnest load-bearing reconstruction profile"
        ),
        notes=(
            "Thinner load-bearing profile in the stronger alloy. Less bulk "
            "under the soft tissue; springs back harder and cracks sooner "
            "than a CP-titanium bar of the same shape."
        ),
    ),
    PlateProfile(
        family="generic-recon-2.5",
        name="Generic 2.5 mm reconstruction plate, straight",
        category="reconstruction bar",
        hole_pitch_mm=9.0,
        width_mm=13.5,
        thickness_mm=2.5,
        hole_diameter_mm=2.9,
        hole_counts=(12, 18, 24),
        material_id="cp-ti-grade-4",
        dimensional_class=(
            "2.5 mm heavy reconstruction class; published systems in this "
            "class run from 6 to 29 screw holes at uniform 2.5 mm thickness"
        ),
        notes="Heavy bar for a long continuity defect. Stiff to contour.",
    ),
    PlateProfile(
        family="generic-lowprofile-1.8",
        name="Generic 1.8 mm low-profile reconstruction plate",
        category="reconstruction bar",
        hole_pitch_mm=7.5,
        width_mm=6.0,
        thickness_mm=1.8,
        hole_diameter_mm=2.2,
        hole_counts=(8, 12, 16, 24),
        material_id="ti-6al-4v-eli",
        dimensional_class="1.8 mm x 6 mm low-profile class, 4 to 24 holes",
        notes=(
            "Narrow, low-profile bar for a thin soft-tissue envelope. The "
            "6 mm width leaves little material either side of a hole, so "
            "bending through one is especially unforgiving."
        ),
        waist_half_width_mm=1.8,
    ),
    PlateProfile(
        family="generic-adaptation-1.0",
        name="Generic 1.0 mm adaptation plate",
        category="adaptation and trauma",
        hole_pitch_mm=6.0,
        width_mm=5.0,
        thickness_mm=1.0,
        hole_diameter_mm=2.0,
        hole_counts=(4, 6, 8),
        material_id="cp-ti-grade-2",
        load_bearing=False,
        dimensional_class="1.0 mm trauma and adaptation class",
        min_bend_radius_mm=4.0,
        max_bend_deg_per_node=30.0,
        notes=(
            "NOT load-bearing. Trauma and orthognathic fixation only; it "
            "will not carry a mandible across a continuity defect."
        ),
        waist_half_width_mm=1.5,
    ),
    PlateProfile(
        family="generic-adaptation-1.5",
        name="Generic 1.5 mm adaptation plate",
        category="adaptation and trauma",
        hole_pitch_mm=7.0,
        width_mm=6.0,
        thickness_mm=1.5,
        hole_diameter_mm=2.2,
        hole_counts=(6, 8, 12),
        material_id="cp-ti-grade-4",
        load_bearing=False,
        dimensional_class="1.5 mm intermediate trauma class",
        min_bend_radius_mm=6.0,
        max_bend_deg_per_node=25.0,
        notes=(
            "Intermediate trauma profile. Used with a vascularised bone "
            "graft, not as a bridging device on its own."
        ),
        waist_half_width_mm=1.8,
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
        material_id="cp-ti-grade-4",
        dimensional_class="preformed body class",
        notes=(
            "Preformed to a 55 mm in-plane radius, roughly the lateral "
            "curvature of an adult mandibular body. Preforming is the point: "
            "less bending on the table means the threaded holes keep their "
            "shape and the plate keeps its fatigue life."
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
        material_id="cp-ti-grade-4",
        dimensional_class="preformed angle class",
        notes=(
            "Preformed with a 115 degree turn at the middle, for a defect "
            "crossing the angle into the ramus."
        ),
    ),
    PlateProfile(
        family="generic-hemimandibular",
        name="Generic hemimandibular reconstruction plate, preformed",
        category="hemimandibular",
        hole_pitch_mm=9.0,
        width_mm=12.5,
        thickness_mm=2.8,
        hole_diameter_mm=2.9,
        hole_counts=(22,),
        preform_angle_deg=120.0,
        material_id="cp-ti-grade-4",
        dimensional_class=(
            "hemimandibular class; published contoured plates in this class "
            "use a 5 + 17 hole split about the angle at 2.8 mm profile height"
        ),
        notes=(
            "One half of the mandible, ramus to symphysis, preformed about "
            "the angle. Very stiff; shape by press rather than by hand."
        ),
    ),
)


def lobed_outline(
    centres: np.ndarray,
    lobe_radius: float,
    segments: int = LOBE_SEGMENTS,
    waist_half_width: float | None = None,
) -> np.ndarray:
    """The silhouette of a chain of lobes centred on the screw holes.

    Returns (N, 2) points counter-clockwise. Two cases, both real:

    * **Overlapping lobes** — a wide bar whose lobes are more than half a
      pitch in radius. The boundary is the arcs between the points where
      neighbouring lobes cross, which gives the scalloped waist of a
      reconstruction bar.
    * **Separated lobes** — a narrow low-profile plate whose lobes do not
      reach each other. They are joined by a straight waisted bridge of
      ``waist_half_width``, which is the ladder silhouette those plates
      actually have. The waist has to be narrower than the lobe, or the plate
      is simply a constant-width bar with no waist at all.
    """
    centres = np.asarray(centres, dtype=float).reshape(-1)
    if len(centres) == 1:
        angles = np.linspace(0.0, 2 * np.pi, segments * 4, endpoint=False)
        return np.column_stack(
            [centres[0] + lobe_radius * np.cos(angles), lobe_radius * np.sin(angles)]
        )

    pitch = float(np.diff(centres).min())
    overlapping = 2.0 * lobe_radius > pitch

    if overlapping:
        # Where neighbouring lobes cross, measured from a lobe's own centre.
        theta = math.atan2(
            math.sqrt(lobe_radius**2 - (pitch / 2.0) ** 2), pitch / 2.0
        )
    else:
        if waist_half_width is None:
            raise ValueError(
                f"lobes of radius {lobe_radius} mm do not reach each other at a "
                f"{pitch} mm pitch, so this profile needs a waist_half_width_mm "
                "to join them"
            )
        if not 0.0 < waist_half_width < lobe_radius:
            raise ValueError(
                f"waist half-width {waist_half_width} mm must be between 0 and "
                f"the lobe radius {lobe_radius} mm"
            )
        # Where the straight waist meets the lobe.
        theta = math.asin(waist_half_width / lobe_radius)
        if 2.0 * lobe_radius * math.cos(theta) >= pitch:
            raise ValueError(
                "the waist is too wide to fit between these lobes; widen the "
                "pitch or narrow the waist"
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
        # Overlapping lobes meet exactly at the crossing point, so it is kept
        # once; separated lobes are joined by the straight waist instead.
        upper.extend(arc if (i == 0 or not overlapping) else arc[1:])

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

    outline = lobed_outline(
        centres, profile.lobe_radius_mm,
        waist_half_width=profile.waist_half_width_mm,
    )
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
        "material_id": profile.material_id,
        "load_bearing": profile.load_bearing,
        "dimensional_class": profile.dimensional_class,
        "waist_half_width_mm": profile.waist_half_width_mm,
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
