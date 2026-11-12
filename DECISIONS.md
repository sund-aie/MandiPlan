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
a bend after the fact: the hand decides when to stop. A key stops the bend. It is a die,
the bender's post the punch: the plate is pushed into the key's shallow V until its two
lobes lie on it, and then each lobe is held between the post and the key and the bridge
carries no more moment however hard the handles are pressed. Order is from the maker:
in-plane first, per the DePuy Synthes sternal guide for the 329.142. A printed bending form
is not new: Song et al. (J Korean Assoc Oral Maxillofac Surg 2021;47:233) printed upper and
lower supports for a press, with a rod to locate the plate. That is one case report.

The owner asked for the keys to be tested, and following each press step by step changed
the first design three ways. It bore on the head's lobes through thin ends, about 3.6 mm
under them; at a hard squeeze, 2000 N on the post, that is about 160 MPa, more than printing
resin takes, so a key now sits up in the arch and bears on it, the lobes clear of the
plate. Its ends rose only 8 degrees past its faces, so a bend over about 16 degrees ran the
fresh plate into them; they now rise at 60. And an in-plane key cannot stop a bend at all:
a lobe is round about its hole, so against a flat face it turns about the hole and still
touches it. In-plane keys are therefore trays to check on, with pins in four holes, not
press keys. The second pin of every key is flattened along the plate, as fixtures for two
holes do, so the holes can close or spread as the bridge bends without binding.

The plan's small turns are gathered, since a key per bridge per direction came to 34 keys
for an 18-hole plate. In-plane error only moves a hole along the bone, so it gets 0.8 mm;
out-of-plane error lifts the plate off it, so it gets 0.25 mm. The plate in the app is the
plan; the plate the keys make is within what the table states, and the rehearsal draws the
latter.

## The plate lies flush with the border

The owner found the plate high on the jaw. The route was right, 2 mm above the lower border,
but seating the plate faired its course across the bone over about 13 mm, which is right for
clicks and wrong for a route already on the border: it lifted the plate 2 to 5 mm in the body
and up to 11 mm across the angle of the jaw. Now the route's edge is 0.5 mm above the border,
only the wrap round the arch is faired, and the angle is rounded just enough for the plate's
in-plane bending, a quarter under its bridge's rating.

## A cut plane is a square sized to the bone

VTK draws an unclipped plane the length of its box's diagonal, about 3.5 times the box's
half-size, so a plane meant to cover the bone came out nearly twice too big. The square is
now drawn directly, upright with the jaw, centred on the outline of the bone the plane cuts
and just covering it, and redrawn as it is dragged. The widget's origin is moved to that
centre, on the same plane, and what it reports is taken back to the cut. Once a cut is
executed its planes leave the view, and nothing can move them until Undo cut.

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


## The fibula graft follows the open planner, with the commercial defaults

The owner asked for fibula planning after a search of how others do it (the research
report, `reports/Fibula flap mandible planning.md`, is kept out of the repository with its
notes). The method is BoneReconstructionPlanner's, read from its code: straight pieces whose
ends are the jaw's planes, each joint the plane half-way between its two pieces, each piece a
stretch of fibula carried to the jaw by one rigid transform, the pieces laid up the fibula from
the ankle, each pushed up until its cuts clear the last plus 1.5 mm, and a guide of a shell 0.3
mm off the bone over 120° with slotted towers. Four things are added:

- The joints are chosen, not clicked: the chain of chords with its joints on the course that
  strays least from it, the fewest pieces within 2 mm, none under 20 mm. Raith and Modabber
  found deviation from the target line tracks surgeons' ratings; no threshold is published, so
  2 mm is the top of what a guided cut lands within. BoneReconstructionPlanner has no shortest
  piece; Fry's 1 cm pieces lacked a periosteal vessel a third of the time.
- No piece is twisted on the next. The frame is carried from piece to piece by the least
  rotation, as Leclerc's planner does, so every joint is one plain wedge and the vessels stay
  along one side. BoneReconstructionPlanner turns all its planes together instead.
- The defaults are the vendors': 70 mm kept above the lateral malleolus (Materialise; KLS Martin
  80; atlases 50 to 60), a height choice from the lower border up (KLS Martin's form), 1 to 5
  pieces automatically (their forms offer 1 to 5 and 1 to 6). The knee's 60 mm has no source;
  none publishes one.
- The guide is followed in use before it is trusted, as the bend keys are: seated, lifted off,
  a blade down every slot, every screw driven. On synthetic legs that found three faults: a
  joint's cuts kept apart across the bone but not along the slot's length past it; a window
  bounded by each cut's own measure across, shut at its ends; the two cuts' measures across
  adding to nothing, which left the guide in three parts. On a real jaw with a synthetic leg
  it found four more: the cuts kept apart round the bone's outline grown outwards, not to
  the corners of the box a tower is; screws placed by the guide's surface alone, one
  entering the wedge where a cut leant 37 degrees; the check itself following a screw from
  the wrong side of the bone; and, for an end cut leaning 77 degrees, a tower leaning over
  the bone so the guide could not be lifted off. Each has a test now. The towers' lower parts
  that caused the last also rounded the guide's inner face towards the bone where they met
  it; with nothing left under the bone seen from the guide's face, it lands at the 0.3 mm
  asked for, at its closest.

Where a joint's two cuts are too close for two slots with plastic between, the joint gets one
window whose two walls guide the blade for each cut: a flange, which Lohn's 540 test cuts found
more accurate than a closed slot. Each wedge narrows towards the outside of its bend, so a guide
on that face needs its cuts kept apart through its own depth, and costs bone. Which face that is
depends on the bends (round the arch, the lateral; up the angle, the front or back), so the face
is a setting, as DePuy offers, and the panel says what the other would take.

The bone the guide is shaped to is the scan's own surface at the threshold, as a signed distance
on a 0.25 mm grid; at the bone mesh's vertices it reads 0.01 mm on average.
