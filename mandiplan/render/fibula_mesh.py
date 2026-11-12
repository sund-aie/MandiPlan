"""The fibula's own surface, out of a leg CT that holds the tibia beside it.

The bone above the threshold is kept only within a tube round the fibula's line
(``geometry/fibula.py``), between its two ends, and on the fibula's side of the
tibia: nearer the fibula's rim than the tibia's, each bone's rim taken as a
circle of its own section's size. A split half-way between their centres would
run through the tibia, which is twice the fibula's width. Contoured, that is
the fibula alone; where the two touch, at the head, it ends in a small facet.
"""

from __future__ import annotations

import numpy as np
import vtk

from ..geometry.fibula import fibula_volume
from .surface import SurfaceExtractor


def fibula_surface(volume, fibula, threshold: float):
    """The fibula's surface, at ``threshold``."""
    return SurfaceExtractor(fibula_volume(volume, fibula)).update(threshold)


def _keep_side(face, into_piece: bool = True) -> vtk.vtkPlane:
    """A VTK plane for one end of a piece. ``vtkClipClosedSurface`` keeps the side
    a plane's normal points into, and a face's normal points out of its piece."""
    plane = vtk.vtkPlane()
    plane.SetOrigin(*np.asarray(face.point, dtype=float))
    normal = -np.asarray(face.normal, dtype=float) if into_piece else np.asarray(face.normal, dtype=float)
    plane.SetNormal(*normal)
    return plane


def cut_piece(surface: vtk.vtkPolyData, piece) -> vtk.vtkPolyData:
    """The piece of the fibula between its two ends, capped, where it lies in the leg."""
    planes = vtk.vtkPlaneCollection()
    planes.AddItem(_keep_side(piece.distal_face))
    planes.AddItem(_keep_side(piece.proximal_face))
    clip = vtk.vtkClipClosedSurface()
    clip.SetInputData(surface)
    clip.SetClippingPlanes(planes)
    clip.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(clip.GetOutput())
    return out


def moved(polydata: vtk.vtkPolyData, transform) -> vtk.vtkPolyData:
    """``polydata`` carried by a rigid ``transform`` (``x -> R x + t``)."""
    matrix = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            matrix.SetElement(i, j, float(transform.rotation[i, j]))
        matrix.SetElement(i, 3, float(transform.translation[i]))
    carry = vtk.vtkTransform()
    carry.SetMatrix(matrix)
    move = vtk.vtkTransformPolyDataFilter()
    move.SetInputData(polydata)
    move.SetTransform(carry)
    move.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(move.GetOutput())
    return out


#: One colour per piece, in the order they lie along the jaw.
PIECE_COLOURS = (
    (231, 138, 95),
    (232, 190, 92),
    (138, 186, 110),
    (104, 166, 205),
    (176, 132, 200),
    (214, 120, 150),
    (150, 150, 150),
)


def coloured(pieces: list[vtk.vtkPolyData]) -> vtk.vtkPolyData:
    """The pieces as one mesh, each point carrying its piece's colour."""
    from vtkmodules.util.numpy_support import numpy_to_vtk

    append = vtk.vtkAppendPolyData()
    for index, piece in enumerate(pieces):
        if piece.GetNumberOfPoints() == 0:
            continue
        tagged = vtk.vtkPolyData()
        tagged.ShallowCopy(piece)
        colour = np.tile(np.array(PIECE_COLOURS[index % len(PIECE_COLOURS)], dtype=np.uint8), (piece.GetNumberOfPoints(), 1))
        array = numpy_to_vtk(colour, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        array.SetName("Colors")
        tagged.GetPointData().SetScalars(array)
        append.AddInputData(tagged)
    if append.GetTotalNumberOfInputConnections() == 0:
        return vtk.vtkPolyData()
    append.Update()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(append.GetOutputPort())
    normals.SplittingOn()
    normals.SetFeatureAngle(50.0)
    normals.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(normals.GetOutput())
    return out


def cut_faces(plan, half_mm: float = 14.0) -> vtk.vtkPolyData:
    """Every cut on the leg as a thin square, to show where the saw goes."""
    append = vtk.vtkAppendPolyData()
    for piece in plan.pieces:
        for face in (piece.distal_face, piece.proximal_face):
            normal = np.asarray(face.normal, dtype=float)
            helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            u = np.cross(normal, helper)
            u /= np.linalg.norm(u)
            v = np.cross(normal, u)
            source = vtk.vtkPlaneSource()
            centre = np.asarray(face.point, dtype=float)
            source.SetOrigin(*(centre - half_mm * u - half_mm * v))
            source.SetPoint1(*(centre + half_mm * u - half_mm * v))
            source.SetPoint2(*(centre - half_mm * u + half_mm * v))
            source.Update()
            append.AddInputData(source.GetOutput())
    if append.GetTotalNumberOfInputConnections() == 0:
        return vtk.vtkPolyData()
    append.Update()
    out = vtk.vtkPolyData()
    out.DeepCopy(append.GetOutput())
    return out
