"""What a mouse click means right now."""

from __future__ import annotations

from enum import Enum


class Mode(Enum):
    NAVIGATE = "Navigate"
    MEASURE = "Measure"
    ANGLE = "Angle"
    ARCH = "Arch curve"
    PLATE = "Plate path"
    LANDMARK = "Margin point"
    SCULPT = "Refine jaw"

    @property
    def hint(self) -> str:
        return {
            Mode.NAVIGATE: "Drag to rotate the 3-D view; scroll to zoom.",
            Mode.MEASURE: "Click two points to measure between them.",
            Mode.ANGLE: (
                "Click three points; the angle is measured at the second one."
            ),
            Mode.ARCH: (
                "Click 5–10 points in the axial view along the buccal cortex, "
                "where the plate will sit — not through the dental arch."
            ),
            Mode.PLATE: "Click along the bone in the 3-D view to draw the plate path.",
            Mode.LANDMARK: "Click on the bone to mark a tumour margin point.",
            Mode.SCULPT: (
                "Click or drag on the reconstructed jaw to smooth, fill or carve it; "
                "pick the brush in the reconstruction step. Right-drag still rotates."
            ),
        }[self]
