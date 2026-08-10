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

## Mirror reconstruction

- **The mid-sagittal plane is searched for, not assumed.** Coordinate descent
  over one lateral offset and two tilt angles, maximising the fraction of bone
  voxels whose reflection is also bone. A full grid over the three costs twenty
  times as much and lands in the same place, because the parameters are close
  to independent once the volume is in patient axes.
- **The symmetry score is reported, not just used.** Mirroring is only as good
  as the patient's symmetry, and a case where the best plane maps 60% of the
  bone onto bone should be visible as such before anyone trusts the graft.
- **A defect crossing the midline is reported as partly un-mirrorable** rather
  than silently returning the piece that could be mirrored. For a symmetric
  arch the un-mirrorable span is exactly twice the distance from the midline to
  the nearer cut, which is what the tests check.
- **The un-mirrorable span is interpolated from the patient, not predicted from
  a population.** Cross-sections at the two ends of the gap are blended and
  swept along the arch curve. A statistical shape model would need a curated
  library of real mandibles, which cannot be assembled inside an application
  that makes no network calls and ships no patient data.
- **Profile extents are measured at the interpolated threshold crossing.**
  Counting whole samples above the threshold loses up to half a sample at each
  end, which was a 4.8% underestimate of the reconstructed volume.
- **Reflection reverses handedness**, so the mirrored surface is re-wound and
  its normals recomputed; otherwise the graft renders inside-out and its
  enclosed volume comes out negative.

## Plate catalogue and bench steps

- **The catalogue is data, not code, and is generic by size class.** No vendor
  names, no part numbers, no dimensions presented as manufacturer
  specifications. Publishing invented specifications as fact would be worse
  than publishing none.
- **`bend_warning_deg` and `min_bend_radius_mm` are working limits the user
  sets**, and are named and documented that way rather than as ratings.
- **Fit is judged on screw purchase as well as length.** A plate long enough to
  span the defect but with two holes on retained bone is refused, because the
  length alone was never the question.
- **Bench distances are measured from the plate's proximal cut end**, not from
  the first screw hole, because that is where a ruler starts.
- **Bend directions are named in patient anatomy** — anterior, superior,
  toward the patient's left — rather than in the frame conventions, so a step
  can be followed without first reading the geometry documentation.
- **A kit that cannot make a bend says so** instead of omitting the step; the
  step still states the angle and names what is needed.

## Packaging

- **`run_mandiplan.command` is the supported way in.** macOS has no `python`,
  only `python3`, and the double-click launcher removes the question entirely
  by building its own virtual environment on first run.
- **PyInstaller excludes the TLS and plotting stacks.** The application makes
  no network calls and reads no encrypted DICOM, so `cryptography`, `requests`
  and `urllib3` are dead weight — and collecting them broke the build.
- **The bundle was built and run on Linux** to prove the spec collects what it
  needs. The macOS-only steps (`sips`, `iconutil`, `BUNDLE`, `hdiutil`) are
  written but unrun here.

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
