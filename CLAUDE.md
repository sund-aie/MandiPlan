# MandiPlan

## Commands
- Run app: `python -m mandiplan`
- Tests: `pytest` (must pass with no skips)
- Regenerate phantom: `python tests/make_phantom.py`

## Non-negotiable
- All measurements in millimetres, derived from DICOM voxel spacing. Never measure in index or pixel space.
- CBCT gray values are NOT Hounsfield units. Bone threshold is always user-adjustable, never hardcoded.
- No status-bar disclaimer banner. The project owner removed it deliberately; do not reinstate it. The disclaimer stays in the About dialog and in the header of every exported CSV.
- No network calls, ever.

## Style
- VTK pipelines built once and updated via SetInputData, not rebuilt per frame.
- Geometry math in `mandiplan/geometry/`, kept free of Qt and VTK imports so it stays unit-testable.
