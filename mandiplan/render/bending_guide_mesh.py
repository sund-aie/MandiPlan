"""The bending guide as a printable mesh.

Each convex block of each saddle (``geometry/bending_guide.py``) is cut out
of a box by its own planes, so every face — the stop faces above all — is
exactly flat and exactly where the plan puts it. The blocks of a saddle
overlap where they join, as the parts of a printed model usually do; slicers
merge overlapping solids of one object. Each saddle also carries its hole
number, raised on the outer face of one wall, to match it to the guide table.
"""

from __future__ import annotations

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ..geometry.bending_guide import (
    Block,
    BendingGuide,
    GuideSaddle,
    saddle_blocks,
    saddle_dimensions,
)

#: Hole numbers on the saddle walls.
LABEL_HEIGHT_MM = 3.0
LABEL_RAISE_MM = 0.4


def block_polydata(block: Block) -> vtk.vtkPolyData:
    """One convex block as a closed, outward-facing triangle mesh."""
    # A box comfortably larger than the block, cut down by every plane.
    extent = 200.0
    cube = vtk.vtkCubeSource()
    cube.SetBounds(-extent, extent, -extent, extent, -extent, extent)
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputConnection(cube.GetOutputPort())
    triangles.Update()
    planes = vtk.vtkPlaneCollection()
    for normal, offset in zip(block.normals, block.offsets):
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            continue
        plane = vtk.vtkPlane()
        plane.SetOrigin(*(normal * offset / length**2))
        # vtkClipClosedSurface keeps the side the plane normal points into.
        plane.SetNormal(*(-normal / length))
        planes.AddItem(plane)
    clip = vtk.vtkClipClosedSurface()
    clip.SetInputConnection(triangles.GetOutputPort())
    clip.SetClippingPlanes(planes)
    clip.GenerateFacesOn()
    clip.TriangulationErrorDisplayOff()
    clean = vtk.vtkTriangleFilter()
    clean.SetInputConnection(clip.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


def saddle_polydata(guide: BendingGuide, saddle: GuideSaddle) -> vtk.vtkPolyData:
    """One saddle, in the plate's supplied frame."""
    parts = vtk.vtkAppendPolyData()
    for block in saddle_blocks(guide, saddle):
        mesh = block_polydata(block)
        if mesh.GetNumberOfCells():
            parts.AddInputData(mesh)
    parts.AddInputData(_label(str(saddle.hole + 1), saddle_dimensions(guide)))
    parts.Update()
    return _to_plate_frame(parts.GetOutput(), saddle)


def _label(text: str, d: dict[str, float]) -> vtk.vtkPolyData:
    """Raised digits on the outer face of the +y wall, read from outside."""
    source = vtk.vtkVectorText()
    source.SetText(text)
    source.Update()
    extrude = vtk.vtkLinearExtrusionFilter()
    extrude.SetInputData(source.GetOutput())
    extrude.SetExtrusionTypeToNormalExtrusion()
    extrude.SetVector(0.0, 0.0, 1.0)
    extrude.SetScaleFactor(LABEL_RAISE_MM + 0.1)
    extrude.CappingOn()
    extrude.Update()
    glyphs = extrude.GetOutput()
    b = glyphs.GetBounds()
    scale = LABEL_HEIGHT_MM / max(b[3] - b[2], 1e-6)
    centre = np.array([(b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0])

    # Glyph x reads left to right seen from +y, which is -x; glyph y is up
    # (+z); the extrusion stands out of the wall (+y). Right-handed, so the
    # digits are not mirrored. It starts 0.1 mm inside the wall so the two
    # print as one.
    matrix = vtk.vtkMatrix4x4()
    matrix.Identity()
    columns = (np.array([-scale, 0.0, 0.0]), np.array([0.0, 0.0, scale]), np.array([0.0, 1.0, 0.0]))
    offset = np.array([centre[0] * scale, d["outer_y"] - 0.1, -centre[1] * scale])
    for row in range(3):
        for col in range(3):
            matrix.SetElement(row, col, float(columns[col][row]))
        matrix.SetElement(row, 3, float(offset[row]))
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    placed = vtk.vtkTransformFilter()
    placed.SetTransform(transform)
    placed.SetInputData(glyphs)
    placed.Update()
    triangles = vtk.vtkTriangleFilter()
    triangles.SetInputData(placed.GetPolyDataOutput())
    triangles.Update()
    return triangles.GetOutput()


def _to_plate_frame(polydata: vtk.vtkPolyData, saddle: GuideSaddle) -> vtk.vtkPolyData:
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)
    placed = saddle.centre + points @ saddle.frame.T
    out = vtk.vtkPolyData()
    out.DeepCopy(polydata)
    out.GetPoints().SetData(numpy_to_vtk(placed, deep=True))
    return out


def guide_polydata(guide: BendingGuide) -> vtk.vtkPolyData:
    """The whole guide, lying as it prints best: saddle tops down on z = 0.

    Turned half a turn about the plate's length, which is a rotation and
    not a mirror, so the stops and the digits keep their handedness. Printed
    this way the only overhangs are the 0.6 mm snap lips.
    """
    parts = vtk.vtkAppendPolyData()
    for saddle in guide.saddles:
        parts.AddInputData(saddle_polydata(guide, saddle))
    parts.Update()
    merged = vtk.vtkPolyData()
    merged.DeepCopy(parts.GetOutput())
    points = vtk_to_numpy(merged.GetPoints().GetData()).astype(float)
    points[:, 1:] *= -1.0
    points[:, 2] -= points[:, 2].min()
    merged.GetPoints().SetData(numpy_to_vtk(points, deep=True))
    return merged
