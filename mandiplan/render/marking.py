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
#: Default cap height of the etched characters, in millimetres. Each plate's
#: own layout usually asks for less; a reconstruction bar has about 1.5 mm of
#: solid face beside its screw seats.
ETCH_TEXT_HEIGHT_MM = 0.9


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
) -> vtk.vtkPolyData:  # noqa: ARG001 - ``across`` is derived, see below
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
    outward = _unit(outward)
    # Glyph x runs along the plate, glyph y across its face, and the
    # extrusion sinks into it. The basis must be right-handed
    # (x cross y = -outward) or the lettering comes out mirrored, so ``across``
    # is derived here rather than trusted from the caller.
    across = _unit(np.cross(outward, along))

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
    placed = vtk.vtkTransformFilter()
    placed.SetTransform(transform)
    placed.SetInputData(glyphs)
    placed.Update()
    return placed.GetPolyDataOutput()


def marking_in_plate_space(asset) -> tuple[np.ndarray, np.ndarray]:
    """The etched mark as ``(points, triangles)`` in the asset's own space.

    Laid out by the asset itself: each catalogue entry says where its
    lettering fits — short codes in the solid runs of face between the screw
    seats and the edge, which is where real plates carry them. A token that
    would overrun its run is scaled down to fit rather than spilling across a
    notch or a seat. Plates too narrow to mark (miniplates) carry none.

    Built on the unplaced plate so the caller can move it through exactly the
    same placement and bend as the plate itself.
    """
    from vtkmodules.util.numpy_support import vtk_to_numpy

    anchors = getattr(asset, "marking", ()) or ()
    pieces_points, pieces_tris = [], []
    offset = 0
    for anchor in anchors:
        text = str(anchor.get("text", "")).strip()
        if not text:
            continue
        height = float(anchor.get("height_mm", ETCH_TEXT_HEIGHT_MM))
        origin = np.asarray(anchor["origin"], dtype=float)
        along = np.asarray(anchor["along"], dtype=float)
        outward = np.asarray(anchor["outward"], dtype=float)
        glyphs = build_marking(text, origin, along, None, outward, height_mm=height)
        if glyphs.GetNumberOfPoints() == 0:
            continue
        points = vtk_to_numpy(glyphs.GetPoints().GetData()).astype(float)
        # Shrink about the origin, within the face, if the run is too short.
        limit = float(anchor.get("max_length_mm", 0.0))
        length = float(np.ptp((points - origin) @ _unit(along)))
        if limit > 0 and length > limit:
            factor = limit / length
            rel = points - origin
            depth = (rel @ _unit(outward))[:, None] * _unit(outward)
            points = origin + (rel - depth) * factor + depth
        triangulate = vtk.vtkTriangleFilter()
        triangulate.SetInputData(glyphs)
        triangulate.Update()
        tris = vtk_to_numpy(
            triangulate.GetOutput().GetPolys().GetConnectivityArray()
        ).astype(np.int64).reshape(-1, 3)
        pieces_points.append(points)
        pieces_tris.append(tris + offset)
        offset += len(points)
    if not pieces_points:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    return np.vstack(pieces_points), np.vstack(pieces_tris)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v
