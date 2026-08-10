# Decisions

Points where the specification left room, or where the obvious implementation
would have been wrong. One line each.

## Geometry and measurement

- **CPR sampling is numpy, not `vtkImageReslice`.** Trilinear sampling of the
  volume at arbitrary world points keeps `mandiplan/geometry/` free of VTK, so
  every millimetre is unit-testable without a render context.
- **Panoramic pixel size equals the arch sampling step.** Both output axes then
  share one millimetre scale and the view is 1:1 by construction rather than by
  a conversion factor that could drift.
- **Cross-sections are a single plane**, as specified — no slab averaging along
  the tangent. Noise is visible; a quieter image would be a different image.
- **Catmull-Rom end conditions continue the local curvature** (ghost point on
  the circle through the first three seeds) instead of reflecting the chord.
  With chord reflection the tangent at the ends of the arch curve was off by
  half a seed spacing — 11° for 7 seeds — which tilts the end cross-sections
  and inflates their measured buccolingual width.
- **Otsu returns the middle of the near-optimal plateau.** When two populations
  are cleanly separated the between-class variance is flat across the whole
  empty gap, and a plain `argmax` returns the very edge of the noise floor.
- **No surface smoothing.** A smoothed isosurface no longer coincides with the
  gray-value boundary that was thresholded, and every distance is taken off
  that surface.

## DICOM

- **Volumes are re-oriented to axis-aligned LPS on load; anything oblique is
  refused.** Gantry tilt, sheared slice positions, non-uniform slice spacing,
  variable orientation and oblique direction cosines all raise `DicomLoadError`
  with an explanation. Resampling them silently is the failure this application
  can least afford.
- **The series with the most slices is loaded** when a folder holds several.

## Resection

- **"Margin distances" are signed perpendicular distances from each cut plane
  to user-placed tumour-margin points.** Lesion segmentation is out of scope,
  so there is nothing else to measure from; positive means the point lies on
  the resected side.
- **`CutPlane.normal` points into the fragment being removed.** With two planes
  the resected fragment is the intersection of the half-spaces; the retained
  bone is the union of the two outer pieces, so it is produced by clipping with
  each plane separately and appending.
- **Segment length is reported twice** — arc length along the arch curve and
  the straight line between the two cuts — because the two differ by several
  millimetres across a curved mandible and only the first is the plate length.

## Plate

- **The screw-hole pitch is arc length along the drawn path.** Consecutive
  nodes are therefore separated by a chord that is slightly shorter than the
  pitch; the table shows both the cumulative distance and the segment length.
- **Bend signs follow the right-hand rule about the named frame axis**, and the
  convention is spelled out in the `mandiplan/geometry/plate.py` docstring and
  repeated in the CSV header.
- **A buccal plate's arch curvature appears as out-of-plane bend.** That is what
  the definitions in the specification give: the plate's own plane is
  perpendicular to the surface normal, so wrapping it around the arch bends it
  about an axis lying in that plane.

## Interface

- **2-D views are QPainter widgets, not VTK image viewers.** They own the
  millimetre-to-pixel mapping outright, which is what keeps a pixel-space
  measurement path from appearing, and it avoids a second render context per
  pane.
- **Arch seeds are placed in the axial view** at the currently displayed slice's
  z, which is where the specification puts them.
- **Clicks in the 3-D view are projected onto the surface** with
  `vtkCellLocator`, and the normal is interpolated barycentrically across the
  picked triangle.

## Tests

- **The suite starts Xvfb when it runs on headless Linux.** VTK's 3-D widgets
  abort the interpreter under Qt's `offscreen` platform plugin, and skipping
  the widget tests would leave the interactive half of the application
  unexercised. macOS and desktop Linux need nothing.
