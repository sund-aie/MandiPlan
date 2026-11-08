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
- **No surface smoothing,** with one exception. A smoothed isosurface no
  longer coincides with the gray-value boundary that was thresholded, and every
  distance is taken off that surface. The exception is a medical CT's jaw box,
  interpolated from slices thicker than its voxels: its isosurface steps in
  ledges one slice high whatever the interpolation, and those are the
  scanner's, not the bone's. It is low-pass filtered with a windowed-sinc
  filter (no shrinkage), whose passband falls with the ratio of slice to voxel;
  on a 2 mm CT the surface moved 0.3 mm on average, under 0.7 mm for 95 % of
  it.

## DICOM

- **Volumes are re-oriented to axis-aligned LPS on load; anything oblique is
  refused.** Gantry tilt, sheared slice positions, non-uniform slice spacing,
  variable orientation and oblique direction cosines all raise `DicomLoadError`
  with an explanation. Resampling them silently is the failure this application
  can least afford.
- **The series that resolves the bone best is loaded** when a folder holds
  several, not the one with the most files: a CT study's coronal reformat often
  has more files than its axial series. Axial first (the acquisition plane),
  then the finest slices, an original reconstruction before a derived one, the
  finest pixels, the longest coverage; scouts and MIPs never.
- **A medical CT is kept whole; its jaws are a second, cubic volume.** The
  slices and the first 3-D view show the scan as acquired. The mandible is
  separated, drawn and planned on a box round the jaws cut from the scan at
  full resolution and resampled to cubic voxels, in the same world
  millimetres. Cropping the scan to that box instead hid most of the CT, and
  when the jaws were found in the wrong place it hid the mandible as well.

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

## Ideas taken from the commercial planners

Two of these were looked at for structure: NemoFAB's numbered wizard and IPS
CaseDesigner's ribbon plus context panel. Almost all of their content is
orthognathic — cephalometrics, occlusion, splints, soft-tissue simulation, Le
Fort I, the Arnett protocol — which the specification excludes and which a
segmental resection tool has no use for. Three things did carry over, none of
them by copying a name or a layout:

- **The work is a numbered sequence that knows where you are.** Both products
  present planning as ordered steps rather than a pile of tools. MandiPlan's
  steps are its own, and each one reports whether it is finished, what it is
  waiting for, or what to do next.
- **A cut can be set by numbers, not only by dragging.** CaseDesigner gives
  segment movement explicit millimetre and degree entry with a finish action.
  The same argument applies to an osteotomy plane: dragging is fast but not
  repeatable or communicable.
- **Angle measurement beside distance.** Both list it as a diagnostic tool and
  it was missing.

Deliberately not taken: anything orthognathic, the vendors' button names
(Spanish wizard labels for a different operation), and nerve auto-tracing,
which the specification excludes.

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

## One cut and two: resolved along the jaw, not by infinite planes

A plane through one side of the body also crosses the other side, and a
half-space takes or leaves the far ramus by accident. A saw cuts one place. So
near a cut the plane decides, and elsewhere the position along the arch curve
does: one cut takes the jaw from there to the far condyle, two take what lies
between. The mesh is parted by each plane everywhere, each piece assigned along
the jaw, and the walls left at the far crossings removed, so the result is closed.

## Faint condylar heads

The back of a condylar head is a thin plate over marrow and can fall below a
threshold that suits the rest of the jaw. The head alone is read at a level
between the soft tissue beside it and the threshold, both measured in the scan
(CBCT gray values are not Hounsfield units), never entering the skull's side of
the joint. The threshold stays the user's; the level moves with it.

## A plate's course is faired as a whole

Reading the bone hole by hole copies its surface texture into the plate, each
lobe at its own angle. The route is faired along its length first (course across
the bone most, height off it less, the face's turn in between), then lifted onto
any prominence it would pass through, then the holes are walked along it at the
plate's exact spacing.

## The bending guide is a key for each bend

Replaces the comb with pins, which the owner found odd and hard to use. A comb only checks
a bend after the fact: the hand decides when to stop. A key stops the bend. It lies
between the plate and the bender's head with the plate's finished shape on its faces, so the
plate goes no further once flat on it, however hard the handles are pressed, and neither
the angle nor the bender's exact kinematics come into it. Only two lobes of the plate touch
its faces, which lift away beyond them, so earlier and later bends do not matter. Pins in the
two holes locate it and are conical, so they stay engaged while the hole turns. It is
printed on its side so those faces are upright.

Order and force are from the maker: in-plane first, per the DePuy Synthes sternal guide for
the 329.142. The bridge's plastic moment is about 4.5 N·m on a 2.4 plate, so about 1000 N
on the key at a 9 mm reach, about 40-60 MPa in the key: a calculation, to be checked on the
fit test. A printed bending form is not new: Song et al. (J Korean Assoc Oral Maxillofac
Surg 2021;47:233) printed upper and lower supports for a press, with a rod to locate the
plate and rounded edges against stress. That is one case report.

The plan's small turns are gathered, since a key per bridge per direction came to 34 keys
for an 18-hole plate. In-plane error only moves a hole along the bone, so it gets 0.8 mm;
out-of-plane error lifts the plate off it, so it gets 0.25 mm. The plate in the app is the
plan; the plate the keys make is within what the table states, and the rehearsal draws the latter.

## A plate runs from the condylar neck

The owner's practice: a long plate goes up the neck, along the inferior border on the facial
surface, to a little past the cut, its excess cut off. Asked to check, not to take this on
trust. The AO Surgery Reference supports the inferior and posterior borders, the lateral
surface, and three or preferably four screws per side, all for fractures. Nothing found says
a plate must start at the neck. So it is the default because the owner asked, the option
to centre it stays, and the README says what is and is not supported.

The neck is read rather than assumed: a section's run of bone from its lower border,
ignoring what lies past a 3 mm gap (the coronoid above the notch), is followed up the ramus
to where it narrows before the head. A height-times-breadth product finds it where height
alone would not, since the head is wider across the jaw than along it.

## The print kit is built from a solid, on a grid

A real jaw's isosurface is hollow, so a connector has nothing to stand in, and the
earlier approach of keying the mesh failed on every real scan. The surface is
filled and contoured again at 0.3 mm, then every vertex near the scan's surface is
put back on it, so the working surface loses nothing to the grid.

## Parts connect by the jaw's own section, plug in socket

Pegs of 4 and 3 mm were too small: the owner feared the jaw would break under pressure, and a
4 mm peg's section modulus of 6.3 mm3 gives about 0.4 N·m in a 60 MPa resin, about what
pressing a plate 40 mm from the joint asks for. The plug is the section itself, drawn in by a
wall, and strength is taken as the weaker of plug and wall for bending each way, not as the
plug's alone: a thin wall would only move the failure. On the slim jaws tried it comes to 27 to
50 mm3, against 150 to 190 for the whole section there: a useful step, not parity, and the
kit reports it rather than claim otherwise. An interlocking jigsaw joint was tried earlier and
dropped at the owner's request. A half-lap was considered and is weaker than this for any
section. Flat faces keep the jaw's length exact.

## Support rods are in the part's own file

The owner asked for files ready for the printer. A resin printer needs a rod under each
lowest point before anything else, or that layer starts from nothing, so those are placed
first, by reading the part along vertical rays. Rods are simple closed shells overlapping the
part, which printers' software accepts, not one watertight union: slicing them is not a problem
and a boolean of hundreds of rods on a scan's mesh would be slow and fragile. The model alone
is exported too, at the same coordinates.

## The plate places itself

Choosing a plate is the whole input. The route runs just above the lower border
because that keeps screws clear of tooth roots and the nerve on any jaw, and
needs no segmentation of either, which the scans could not support. Where a course
from the neck does not apply, position is chosen by scoring, not by rules per plate:
holes balanced round the resection and least bending in the plate's own plane, which
puts a preformed plate at the angle without knowing it is an angle plate. The verdict's red is kept for what cannot
be bent or trimmed away, so it means something when it shows.

## The rehearsal starts from a fresh plate

A plate out of the packet is straight, and each key's bend is made on a plate holding all the
bends before it, released. So the rehearsal runs the bends forward as hinges at the bridges, in the
key's own frame at each stop, and draws the plate, the key and the bender's head at that step.

## Negative slice spacing and CT padding

Both came from one real scan. A negative SpacingBetweenSlices made the reader
report a reversed stacking axis over slices in file order: an upside-down,
mirrored volume, with no error. Placement now comes from ImagePositionPatient.
The constant outside a CT's reconstructed circle counted as air and moved the
bone threshold into soft tissue; a constant below -1100, covering over 1% of the
scan, is now air. The rule needs Hounsfield-like values, and a CBCT without them
keeps its gray values untouched.

