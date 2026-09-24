"""Build the reconstructed mandible as one surface.

The geometry — registration, the donor warp, the blend — is in
``mandiplan.geometry.reconstruction`` and is numpy only. This module feeds it
the bone surface and the scan, and contours the result with VTK.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from ..geometry.reconstruction import (
    ReconstructionReport,
    RigidTransform,
    compose_field,
    depth_into_resection,
    icp,
    junction_step_mm,
    stump_bands,
)
from ..geometry.volume import Volume
from .surface import SurfaceExtractor

#: Surface scalar marking which part of the reconstruction is which.
REGION_RETAINED = 0.0
REGION_DONOR = 1.0
REGION_NO_DONOR = 2.0


@dataclass
class Reconstruction:
    surface: vtk.vtkPolyData  # the whole reconstructed mandible, one mesh
    report: ReconstructionReport
    volume: Volume  # the reconstructed field, for measurement and slicing


def _surface_points(polydata: vtk.vtkPolyData, limit: int = 60000) -> np.ndarray:
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)
    if len(points) > limit:
        points = points[:: int(np.ceil(len(points) / limit))]
    return points


def register_junctions(surface, planes, symmetry, near_mm=1.0, far_mm=12.0):
    """One rigid correction per cut, bringing the mirror onto that stump."""
    points = _surface_points(surface)
    reflected = symmetry.reflect(points)
    stumps = stump_bands(planes, points, near_mm, far_mm)
    donors = stump_bands(planes, reflected, near_mm, far_mm)
    registrations = []
    for i in range(len(planes)):
        registrations.append(icp(reflected[donors[i]], points[stumps[i]]))
    return registrations


def reconstruct(
    volume: Volume,
    threshold: float,
    surface: vtk.vtkPolyData,
    planes,
    symmetry,
    band_mm: float = 3.0,
    register: bool = True,
    fallback_field=None,
    fallback_name: str = "the patient's pre-operative contour",
) -> Reconstruction:
    """Mirror the healthy side into the defect and return one flush surface.

    ``register=False`` is the plain mirror, kept for comparison: the donor is
    the reflection alone, with no correction at either stump.
    """
    registrations = register_junctions(surface, planes, symmetry) if register else []
    if registrations and all(reg.points_used for reg in registrations):
        pairs = [(i, reg.transform) for i, reg in enumerate(registrations)]
    else:
        # A stump with too little bone to register keeps the plain mirror.
        pairs = [(i, RigidTransform.identity()) for i in range(len(planes))]

    # Work only where the result can differ from the scan: inside the
    # resection and its blend band, around the bone that is there.
    points = _surface_points(surface)
    near = depth_into_resection(planes, points) > -(band_mm + 2.0)
    if not np.any(near):
        near = np.ones(len(points), dtype=bool)
    margin = 8.0
    box_min = points[near].min(axis=0) - margin
    box_max = points[near].max(axis=0) + margin

    field, lo, donorless, report = compose_field(
        volume,
        planes,
        symmetry,
        pairs,
        box_min,
        box_max,
        band_mm=band_mm,
        threshold=threshold,
        fallback_field=fallback_field,
        fallback_name=fallback_name,
    )
    report.registrations = registrations
    if report.donorless_voxels:
        report.warnings.append(
            "Check the no-donor span before relying on it: it is not mirrored "
            "anatomy."
        )

    array = volume.array.astype(np.float32, copy=True)
    k0, j0, i0 = lo[2], lo[1], lo[0]
    nz, ny, nx = field.shape
    array[k0 : k0 + nz, j0 : j0 + ny, i0 : i0 + nx] = field
    rebuilt = Volume(array=array, spacing=volume.spacing, origin=volume.origin)
    result = SurfaceExtractor(rebuilt).update(threshold, largest_component=True)
    _label_regions(result, planes, donorless, lo, volume, band_mm)
    return Reconstruction(result, report, rebuilt)


def _label_regions(polydata, planes, donorless, lo, volume, band_mm) -> None:
    """Point scalars: retained bone, mirrored donor, or no-donor fill."""
    points = vtk_to_numpy(polydata.GetPoints().GetData()).astype(float)
    region = np.full(len(points), REGION_RETAINED, dtype=np.float32)
    inside = depth_into_resection(planes, points) >= 0.0
    region[inside] = REGION_DONOR
    if donorless.any():
        index = np.round((points - volume.origin) / volume.spacing).astype(int) - np.asarray(lo)
        nz, ny, nx = donorless.shape
        valid = (
            (index[:, 0] >= 0) & (index[:, 0] < nx)
            & (index[:, 1] >= 0) & (index[:, 1] < ny)
            & (index[:, 2] >= 0) & (index[:, 2] < nz)
        )
        flagged = np.zeros(len(points), dtype=bool)
        flagged[valid] = donorless[index[valid, 2], index[valid, 1], index[valid, 0]]
        region[flagged & inside] = REGION_NO_DONOR
    scalars = numpy_to_vtk(region, deep=True)
    scalars.SetName("region")
    polydata.GetPointData().SetScalars(scalars)


def junction_steps(rebuilt: Volume, threshold: float, planes) -> list[float]:
    """Mean surface step across each junction of a reconstruction, mm."""
    return [junction_step_mm(rebuilt, threshold, plane) for plane in planes]
