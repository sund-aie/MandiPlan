"""Shared constants. Defaults live here; none of them are clinical facts."""

APP_NAME = "MandiPlan"

DISCLAIMER = (
    "RESEARCH AND EDUCATION USE ONLY — NOT A MEDICAL DEVICE — "
    "NOT FOR CLINICAL DECISION-MAKING"
)

# Curved planar reformation defaults (all millimetres).
CPR_STEP_MM = 0.2
CPR_SLAB_MM = 10.0
CPR_MODE = "max"  # "max" (crisper cortex) or "mean" (ray-sum, OPG-like)
CROSS_SECTION_WIDTH_MM = 40.0

# Reconstruction-plate defaults. Screw-hole pitch and ribbon size vary by
# plate system; these are starting values, editable in the UI.
PLATE_PITCH_MM = 9.0
PLATE_WIDTH_MM = 12.0
PLATE_THICKNESS_MM = 2.0

MAX_RESECTION_PLANES = 2
