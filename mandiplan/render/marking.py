"""Laser marking on the plate face.

Real reconstruction plates carry an etched mark — material designation, size,
hole count, a lot or reference number — on the outer face. It is laser-etched,
and on CMF implants the etch is roughly **50 to 200 micrometres deep**. That
is the real scale, and it is what this module renders.

It is worth being plain about the scale, because it is easy to over-imagine:
100 um is about the thickness of a sheet of paper. Marking is not a
nanometre-scale feature and nothing on a titanium implant is; a nanometre is
ten thousand times finer than the etch, well below the surface roughness of
the plate itself and far below anything an optical instrument resolves. What
you can see on a real plate, and what you can see here, is the etch.

The marking is built as its own polydata and drawn as a separate dark actor
sitting in the etched recess, rather than being boolean-subtracted from the
plate solid. That keeps the plate watertight — which the exports and the
genus checks depend on — while showing the mark at its true depth and size.
"""

from __future__ import annotations

import numpy as np
import vtk

#: Depth of a laser etch on a CMF implant face, in millimetres.
ETCH_DEPTH_MM = 0.10
#: Cap height of the etched characters, in millimetres.
ETCH_TEXT_HEIGHT_MM = 1.6


def marking_text(asset) -> str:
    """The line etched on a plate: what it is made of and what size it is."""
    alloy = asset.alloy
    designation = {
        "ti-6al-4v-eli": "Ti6Al4V ELI",
        "cp-ti-grade-4": "CP Ti GR4",
        "cp-ti-grade-2": "CP Ti GR2",
        "316lvm": "316LVM",
    }.get(alloy.id, alloy.name)
    return (
        f"{designation}  {asset.thickness_mm:.1f}x{asset.width_mm:.1f}  "
        f"{asset.hole_count}H  {asset.id.upper()}"
    )


def build_marking(
    text: str,
    origin,
    along,
    across,
    outward,
    height_mm: float = ETCH_TEXT_HEIGHT_MM,
    depth_mm: float = ETCH_DEPTH_MM,
) -> vtk.vtkPolyData:
    """Extruded characters lying in the plate's outer face.

    ``origin`` is where the text starts, ``along`` runs down the plate,
    ``across`` runs over its width, and ``outward`` points out of the face
    being marked. The glyphs are sunk ``depth_mm`` into that face.
    """
    source = vtk.vtkVectorText()
    source.SetText(text)
    source.Update()
    flat = source.GetOutput()
    if flat.GetNumberOfPoints() == 0:
        return vtk.vtkPolyData()

    extrude = vtk.vtkLinearExtrusionFilter()
    extrude.SetInputData(flat)
    extrude.SetExtrusionTypeToNormalExtrusion()
    extrude.SetVector(0.0, 0.0, 1.0)
    extrude.SetScaleFactor(depth_mm)
    extrude.CappingOn()
    extrude.Update()
    glyphs = extrude.GetOutput()

    # vtkVectorText draws at a cap height of about 1 unit; scale to millimetres.
    bounds = flat.GetBounds()
    natural = max(bounds[3] - bounds[2], 1e-6)
    scale = height_mm / natural

    along = _unit(along)
    across = _unit(across)
    outward = _unit(outward)

    basis = vtk.vtkMatrix4x4()
    basis.Identity()
    for row in range(3):
        basis.SetElement(row, 0, along[row] * scale)
        basis.SetElement(row, 1, across[row] * scale)
        # Extrusion runs along +z in glyph space; send it into the face.
        basis.SetElement(row, 2, -outward[row])
        basis.SetElement(row, 3, origin[row])

    transform = vtk.vtkTransform()
    transform.SetMatrix(basis)
    placed = vtk.vtkTransformPolyDataFilter()
    placed.SetTransform(transform)
    placed.SetInputData(glyphs)
    placed.Update()
    return placed.GetOutput()


def marking_for_plate(asset, hole_centres, hole_axes) -> vtk.vtkPolyData:
    """Place the etched mark on a fitted plate's outer face.

    Sits alongside the middle screw holes, running down the plate, offset
    across the width so it does not cross a hole.
    """
    centres = np.asarray(hole_centres, dtype=float).reshape(-1, 3)
    axes = np.asarray(hole_axes, dtype=float).reshape(-1, 3)
    if len(centres) < 2:
        return vtk.vtkPolyData()

    middle = len(centres) // 2
    outward = _unit(axes[middle])
    along = _unit(centres[min(middle + 1, len(centres) - 1)] - centres[max(middle - 1, 0)])
    across = np.cross(outward, along)

    text = marking_text(asset)
    # Start it back along the plate so the line is roughly centred, and push
    # it to the edge of the width so it clears the screw holes.
    span = len(text) * ETCH_TEXT_HEIGHT_MM * 0.62
    origin = (
        centres[middle]
        - along * (span / 2.0)
        + across * (asset.width_mm * 0.28)
        + outward * (asset.thickness_mm / 2.0)
    )
    return build_marking(text, origin, along, across, outward)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v
