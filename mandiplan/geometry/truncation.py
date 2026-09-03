"""Reject cases whose mandible runs out of the field of view.

A mandible whose ramus or condyle is cut off by the edge of the scan is not a
whole shape, and averaging it into a shape library quietly shortens every
reference it contributes to. The test is blunt and does not need a model: if
the mandible label touches the boundary of the volume, the bone continues
outside the scan and the case is incomplete.

Pure numpy — no Qt, no VTK.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Voxels this close to a face still count as touching it, so that a mandible
#: separated from the edge by a single partial-volume voxel is not accepted.
_EDGE_VOXELS = 1


@dataclass
class TruncationReport:
    """Whether a labelled case is whole enough to enter the shape library."""

    truncated: bool
    faces: list[str] = field(default_factory=list)
    voxels_on_boundary: int = 0

    @property
    def usable(self) -> bool:
        return not self.truncated

    def summary(self) -> str:
        if not self.truncated:
            return "Mandible is fully inside the field of view."
        where = ", ".join(self.faces)
        return (
            f"Mandible reaches the edge of the scan at: {where}. "
            f"{self.voxels_on_boundary} label voxels lie on the boundary, so the "
            "ramus or condyle is cut off and the case is incomplete."
        )


def check_field_of_view(labels: np.ndarray, mandible_label: int = 2) -> TruncationReport:
    """Check a label volume for a mandible that leaves the scan.

    ``labels`` is indexed ``[k, j, i]`` -> world ``(z, y, x)``, matching
    :class:`~mandiplan.geometry.volume.Volume`. ``mandible_label`` is 2 in the
    DentalSegmentator label scheme.

    The superior and posterior faces are where the ramus and condyle leave a
    short field of view, but every face is reported: a mandible touching any
    boundary is not a complete shape.
    """
    mask = np.asarray(labels) == mandible_label
    if not mask.any():
        return TruncationReport(truncated=True, faces=["no mandible label found"])

    edge = _EDGE_VOXELS
    faces: list[str] = []
    total = 0
    checks = (
        ("inferior", mask[:edge, :, :]),
        ("superior", mask[-edge:, :, :]),
        ("anterior", mask[:, :edge, :]),
        ("posterior", mask[:, -edge:, :]),
        ("patient right", mask[:, :, :edge]),
        ("patient left", mask[:, :, -edge:]),
    )
    for name, slab in checks:
        count = int(slab.sum())
        if count:
            faces.append(name)
            total += count

    return TruncationReport(truncated=bool(faces), faces=faces, voxels_on_boundary=total)


def mask_from_volume(volume, threshold: float) -> np.ndarray:
    """A pseudo-label array from a thresholded grayscale volume.

    Lets the same field-of-view check run on a dataset that ships raw volumes
    rather than labels, before any segmentation model has been installed.
    """
    return np.where(np.asarray(volume.array) >= threshold, 2, 0).astype(np.uint8)
