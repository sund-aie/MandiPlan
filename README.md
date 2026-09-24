# MandiPlan

**RESEARCH AND EDUCATION USE ONLY — NOT A MEDICAL DEVICE — NOT FOR CLINICAL
DECISION-MAKING**

A local desktop tool for planning a segmental mandibular resection and the
reconstruction plate that follows it, from a CBCT series.

A plate length taken off a panoramic radiograph is an estimate: OPG
magnification is non-uniform and site-dependent, published figures ranging from
roughly 5% to 35%. A CBCT-derived curved reformat brings linear error down to
roughly 1.3–8%. MandiPlan does the measurement-and-planning step from the CBCT,
on your own machine, in the time it takes to draw a curve — rather than through
a vendor engineering turnaround.

Rendering here is plain on purpose. Measurement is not: every number the
application shows is derived from DICOM voxel spacing in millimetres, never
from screen pixels.

## Get it

```sh
cd ~/Desktop
git clone https://github.com/sund-aie/MandiPlan.git
cd MandiPlan
```

`git clone` makes a folder named `MandiPlan` **inside** the folder you are
standing in, so after cloning you have to `cd MandiPlan` before anything else
works. Run `ls` — if you do not see `requirements.txt` and `run_mandiplan.command`
listed, you are not in the repository, and every command below will fail with
"no such file or directory".

## Run it on macOS

**The simplest way: double-click `run_mandiplan.command` in Finder.** On first
run it builds a private virtual environment beside itself, installs the
dependencies into it, and starts the application. After that it just starts.
(macOS will refuse to run a downloaded script until you right-click it and
choose Open, once.)

From a terminal, standing in the repository, use `python3` — macOS has no
command called `python`, which is what `zsh: command not found: python` means:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m mandiplan
```

To open a folder of DICOM files straight away, pass it as an argument:

```sh
python3 -m mandiplan ~/Desktop/my_cbct_folder
```

Python 3.11 or newer is required; get it from python.org if `python3 --version`
says otherwise. Everything runs on Linux and Windows too, with `python3`
or `python` as that system names it.

### If it starts but no window appears

Run the display check and note which of its three windows you actually see:

```sh
python3 tools/check_display.py
```

The 3-D view is a VTK widget inside Qt, and VTK offers two ways to embed it.
The default suits X11; macOS needs the other one, which is why MandiPlan picks
`QOpenGLWidget` there. If your machine wants the opposite, force it:

```sh
MANDIPLAN_VTK_WIDGET=QWidget python3 -m mandiplan
```

## Build a double-clickable app

Standing in the repository:

```sh
./packaging/build_macos.sh
```

This produces `dist/MandiPlan.app` and `dist/MandiPlan.dmg` (drag onto
Applications to install), with the icon in `packaging/icon.png`. The bundle is
about a gigabyte, most of it VTK, and it is **not code signed or notarised**, so
the first launch needs a right-click → Open.

The bundle was built and run on Linux to check that it collects everything it
needs; the macOS-only steps in that script — `sips`/`iconutil` for the icon,
the `.app` bundle and the disk image — have not been run here, so treat the
first macOS build as the one that proves them.

## Other commands

Run the test suite (it must pass with no skips):

```sh
pytest
```

Generate the phantom, which writes `build/phantom_dicom` for you to open in the
application:

```sh
python3 tests/make_phantom.py
```

With no patient CBCT to hand, run `tests/make_phantom.py` and open the folder
it writes. The phantom is a synthetic mandible-like arc whose geometry is known
in closed form — 78.19 mm of arc, a 12 × 18 mm elliptical cross-section — so you
can check what the application reports against what it should report.

## The plate library

Choosing a plate changes the geometry, not just a number. Every entry in
`mandiplan/data/plates/` is a real watertight solid, loaded and placed as
itself; there is no proxy shape anywhere in the viewport or in any export.

**Everything shipped here is generic** — built by `tools/make_plate_assets.py`
from published dimensional classes, not from any manufacturer's CAD, and
labelled "Generic parametric approximation" in the interface and in every
export. Manufacturers do not publish their CAD; what they and the literature
do publish is dimensions, and those are what these plates follow:

| family | section | pitch | built from |
| --- | --- | --- | --- |
| Low-profile reconstruction 2.4 | 2.4 × 8.0 mm | 8 mm | FE study: conventional low-profile 2.4 plate, 135 × 8 × 2.4 mm |
| Locking reconstruction 2.4 | 2.5 × 8.0 mm, threaded holes | 8 mm | locking reconstruction plate 2.4: 2.5 mm thick, 2.4/3.0 locking screws |
| Heavy reconstruction | 2.8 × 10 mm | 9 mm | 2.8 mm profile-height plates; ~10 mm catalogue widths |
| Primary reconstruction 1.5 | 1.5 × 6.0 mm | 7 mm | 1.5 mm plates for primary reconstruction, not load-bearing |
| Miniplate 2.0 | 1.0 × 4.3 mm | 6 mm | FE studies: 2.0 4-hole miniplate, 26 × 4.3 × 1.0 mm |
| Preformed body, angle, hemimandibular (5 + 17), full mandibular (6 + 13 + 6) | as their straight family | | published preformed configurations |

Each is built the way real plates are: straight parallel edges with small
notches between holes (where the plate is meant to bend), rounded ends,
conical screw seats, a conical thread in the seat of locking holes (drawn as
rings at the thread pitch, not a helix), and a rounded edge break. Miniplates
are eyelets on a narrow bar. Preformed plates are curved the way they sit on
the jaw: body curvature through the plate's thickness, the gonial-angle turn
in the plate's own plane.

The faces are meshed finely (no edge along the plate longer than about a
millimetre) because a coarse face bends as flat chords cutting across the
curved plate. Every asset is checked on build and in the tests: watertight,
consistently wound outward, and genus equal to its hole count — a watertight
solid's genus counts its handles, and a through-hole is a handle.

To add a licensed exact asset, drop the mesh (STL, PLY or OBJ) and an entry
into `mandiplan/data/plates/`. An entry may only set `"exact": true` if it also
carries provenance and a licence, and a mesh whose size does not match its
metadata is refused, which catches a metre- or inch-authored file.

    python3 tools/make_plate_assets.py     # regenerate the library
    python3 tools/screenshot.py --plate    # fit one and capture it

## What bending does to the screw holes

An earlier version of this application held every screw hole perfectly
circular through bending and presented that as a feature. That was wrong. AO
Surgery Reference is explicit:

> Without bending insets, the holes become deformed during contouring of the
> plate, and precise seating of the locking screws cannot be guaranteed.

and, on preformed plates:

> The minimal intraoperative bending required in preformed plates preserves
> the optimal threaded-hole shape, resulting in a plate with increased fatigue
> life compared to standard reconstruction plates.

So MandiPlan predicts the distortion, applies it to the plate's geometry so
you can see it, and says what it means for the screw. The prediction depends
on the alloy, on where the bends fall relative to the holes, and on the kit —
which is the point. Two 12° bends in a CP-Ti Grade 4 bar:

| kit | bends **between** holes | bends **through** holes |
| --- | --- | --- |
| Three-point pliers | 0 µm, fatigue 100% | **409 µm, fatigue 7%** |
| Bending irons | 115 µm, fatigue 47% | 232 µm, fatigue 22% |
| Bar press with insets | 11 µm, fatigue 93% | 14 µm, fatigue 91% |

The pliers concentrate a bend into about five millimetres, so they are the
most precise instrument *between* holes and the most destructive *through*
one. The irons spread the bend over most of a pitch, which is gentler through
a hole but catches the neighbours even when you aim between them. The press
fits insets, and insets are what the technique is really about.

Each hole is reported as its major and minor axis, how far out of round it is
in micrometres, and a verdict: round enough for a locking screw, marginal, or
use a non-locking screw. Turn `Show holes as bending leaves them` off in the
inspector to see the catalogue's nominal circles instead.

### Materials

Each plate family declares an alloy, and the alloy decides how it behaves
under an iron. The space lattice is recorded because it is the reason:

| alloy | standard | lattice | yield | elongation | springback (overbend to hold 20° at R30) |
| --- | --- | --- | --- | --- | --- |
| Ti-6Al-4V ELI | ASTM F136 | HCP + BCC | 795 MPa | 10% | 27.0° |
| CP-Ti Grade 4 | ASTM F67 | HCP | 483 MPa | 15% | 24.1° |
| CP-Ti Grade 2 | ASTM F67 | HCP | 275 MPa | 20% | 22.2° |
| 316LVM steel | ASTM F138 | FCC | 190 MPa | 40% | 20.8° |

FCC austenite has twelve slip systems, which is why 316LVM is by far the most
forgiving to bend. Alpha titanium is HCP with few slip systems, so it work-
hardens fast and springs back. Ti-6Al-4V is alpha+beta: strongest, least
ductile, springs back hardest, cracks soonest.

These are **published standard minima and nominal handbook values, not
measurements of the plate in your hand.** A real lot is usually stronger.

### The etched marking

Real plates carry a laser mark on the outer face. MandiPlan renders it at its
true depth — **100 µm**, about the thickness of a sheet of paper. Nothing on a
titanium implant is marked at nanometre scale; a nanometre is ten thousand
times finer than the etch, below the surface roughness of the plate and far
below what any optical instrument resolves. What you can see on a real plate
is the etch, and that is what is drawn.

## Where a resection plane gets its orientation

A cut is stored as a `PlanePlacement` — an arc position along the mandible
plus three angles — and never as a world-space normal. The world plane is
re-derived from that placement every time either changes.

At arc position `s` the arch curve gives a right-handed local mandibular
frame:

| axis | meaning | how it is built |
| --- | --- | --- |
| `T` | direction of travel along the arch | unit tangent of the arch spline |
| `U` | anatomical superior reference | patient superior, parallel-transported along the curve from `s = 0` |
| `B` | buccolingual | `T x U` |

`U` is *transported*, not taken from a global axis. At each step the frame is
rotated by exactly the rotation carrying the previous tangent onto the current
one — the minimal rotation, which adds no twist. Building the up-axis from a
fixed reference instead (`T x z` and similar) degenerates wherever the curve
runs parallel to that reference, which on a mandible is the ramus; the
previous implementation refused such a curve outright.

**The default cut normal is `T`** — perpendicular to the local arch. That is
the osteotomy a saw makes held square to the bone at that point.

The three offsets are then applied to that frame, each about an axis of the
frame as already rotated: `roll` about `T`, then `yaw` about the rolled `U`,
then `tilt` about the twice-rotated `B`. Roll does not move the cut by itself
— a plane is invariant under rotation about its own normal — it chooses the
axes yaw and tilt subsequently act about.

Because the angles are stored relative to this frame, **moving a cut along the
jaw carries its obliquity with it**. Dial 20 degrees at the body, drag to the
angle, and the plane arrives still 20 degrees oblique to the arch there, with
a different world-space normal. Translation writes only `s_mm` and the
patient-axis offset; rotation writes only the angles. Which side the cut
removes is a flag on the placement, so it survives translation too.

To see it:

    python3 tools/screenshot.py --cut-sweep

That prints the normal and the obliquity at three positions and writes
`artifacts/screenshots/cut-{body,angle,ramus}-viewport.png`.

## The interface

A three-zone workspace. A compact command bar across the top carries the
identity, the workflow tools and, on the right, the view controls, undo, fit,
help and the inspector toggle. The 3-D planning viewport is the centre and
keeps most of the window. A context inspector sits on the right; it is
width-limited, resizeable, and collapses entirely with the panel button or
Ctrl+B when you want the whole window for the anatomy.

The palette is deliberately quiet. One accent blue marks the active tool and
primary actions and appears nowhere else; the viewport is a flat cool
near-white so the ivory bone reads against it without a border around the
view. Planning state is carried by hue rather than saturation — muted warm red
for the resected fragment, pale teal for the mirrored segment, sand where there
was no mirror donor, titanium grey
for the plate, amber only while a landmark is being edited.

Icons are drawn in code (`mandiplan/ui/icons.py`), so there are no image files
to ship and nothing to fetch. Design tokens live in `mandiplan/ui/theme.py`;
change a colour there and the whole application follows.

To capture the interface:

    python3 tools/screenshot.py --phantom

That writes `artifacts/screenshots/workspace.png` and, separately,
`workspace-viewport.png`. The viewport is captured through VTK because Qt
cannot read back an OpenGL surface — in the window grab the 3-D view appears
as noise, which is a screenshot artifact and not what you see on screen.


## The workflow

The planning panel is a numbered sequence, and the strip above it says where
you are: a dot per step, green once it is finished, and a line naming either
what has been achieved or the next thing to do. Steps that cannot be started
yet say what they are waiting for.

**1 · Load and threshold.** *File → Open DICOM folder*, or *File → Open
sample scan* to try everything on the real head CBCT bundled with the
application. MandiPlan reads the voxel spacing from the header, re-orients the
volume to patient axes, and shows the matrix, voxel size and field of view in
millimetres. A series with gantry tilt, sheared slice positions, non-uniform
slice spacing or an oblique orientation is refused with an explanation rather
than loaded distorted. A scan too large to plan on in memory (a 0.25 mm head
scan is 300 million voxels) is averaged over whole blocks of voxels into a
coarser working grid, and the load says so; world coordinates stay exact.

CBCT gray values are not Hounsfield units — they depend on the scanner, the
field of view and the exposure — so there is no fixed bone threshold. The
slider is seeded from the scan's own histogram: a real head scan holds air,
soft tissue and bone, and the seed is the split between the last two (a
two-class split draws the skin), over a range that ignores metal fillings.
Move it by eye; the isosurface follows as you drag. No smoothing is applied,
because a smoothed surface no longer coincides with the gray values you
thresholded.

**2 · Arch curve and reformat.** Switch to the *Slices* tab, press *Place arch
points*, and click 5–10 points along the jaw in the axial view. A cubic spline
is fitted through them and resampled at uniform **arc length**.

Once the curve is drawn the mandible is **separated from the rest of the
skull** by itself. With the teeth in occlusion a real scan is one connected
mass of bone — mandible, maxilla and skull — and a cutting plane, being
infinite, would take a slice of all three. MandiPlan finds the bite along the
arch (the dark line between bright upper and lower crowns), cuts only the tooth
contacts along it, and splits any remaining contact at the jaw joints at its
thinnest, darkest point. The bone panel reports the separated mandible's
volume; the checkbox there shows all bone again if you need it.

Draw the curve along the buccal cortex, where the plate will actually sit — not
through the dental arch. Arc length is the right quantity for a plate, but only
if the curve follows the path the plate will take.

The *Panoramic (CPR)* tab then shows the jaw flattened along that curve. Its
x-axis is arc length in millimetres and its y-axis is superior–inferior
position in millimetres, so the view is 1:1 and measurement in it is direct.
Each column aggregates a slab (10 mm by default) across the buccolingual
direction, either as maximum intensity (default, crisper cortical outline) or
as a mean ray-sum that looks like an OPG. Below it is a buccolingual
cross-section at any point along the curve; ← and → step it along, Shift for
5 mm steps.

**Measuring, at any step.** Choose *Measure* and click two points, or *Angle* and click
three — the angle is reported at the middle point, which is how you record a
gonial angle or check a bend against the plan. In the 3-D view you get
a straight-line distance. In the panoramic view you get three numbers, labelled
for what they are: arc length along the arch curve, superior–inferior
distance, and the flattened distance between the points. The x-axis of that
view is arc length, not a straight line, and the readout says so — across a
curved mandible the two differ by several millimetres.

**3 · Resection.** *Add cutting plane* drops a plane perpendicular to the arch
curve; drag its handles in the 3-D view, or set it by numbers in the panel:
position along the arch curve in millimetres, obliquity (yaw about the superior
axis), inclination (tilt about the buccolingual direction), and offsets in
patient axes. Dragging is quick; an osteotomy you have to describe, check or
hand over needs numbers. Bone that the current planes would
remove is tinted red as you move them. Up to two planes, which covers a
segmental resection. *Execute cut* separates the fragment and reports the
resected segment as both arc length along the curve and straight-line distance
between the cuts, plus the fragment volume. *Mark tumour margin point* places
landmarks, and the readout gives the signed distance from each cut plane to
each of them — positive means the point is on the resected side. *Undo cut*
puts it back; Ctrl+Z undoes editing steps.

**4 · Reconstruction.** *Reconstruct from the healthy side* finds the patient's
plane of symmetry (and reports how much bone mirrors onto bone, so a patient
too asymmetric for mirroring shows it), mirrors the healthy side into the
defect, and makes it **flush**: the mirror is registered to each cut stump
separately, the two corrections are blended smoothly across the defect, and
the mirror is blended into the retained bone over 3 mm before a single surface
is built. The result is one closed piece that replaces the bone in the view —
ivory where it is the patient's own bone, pale teal where it is mirrored. The
panel reports each junction's mismatch before and after registration.

Where the defect crosses the midline there is no healthy counterpart to
mirror; that part is filled from the pre-operative contour, coloured sand, and
reported, never silently merged.

If a junction still wants rounding, *Smooth the junctions* does it for you, and
*Brush on the jaw* (or key 7) gives Smooth, Fill and Carve brushes with a size
and a strength: left-drag on the jaw, right-drag to turn it. Every stroke can
be undone, *Back to computed* restores the computed surface, and the panel says
how far your edits have moved it.

**5 · Plate.** *Draw plate path on the bone* and click along the outer face of
the jaw — after reconstructing, across the rebuilt segment too, since the plate
follows the reconstruction. Then choose the plate and its length from the
library; its pitch, section and the lengths it comes in drive everything
below. Pick the bending kit you will use.

Clicks are projected onto the bone, the normal under each is fitted over the
plate's footprint rather than taken from one point of a rough surface, and the
path is a smooth curve through the clicks (a plate cannot follow a wiggle
shorter than its hole pitch). It is resampled at the screw-hole pitch, and each
interior node gets three signed angles in the plate's own frame:

| angle | what it is |
|---|---|
| in-plane bend | the contour bend in the plate's own plane, about the surface normal |
| out-of-plane bend | rotation about the binormal; positive lifts the plate off the bone |
| twist | rotation of the plate's face about the direction of travel |

The full convention is in the docstring of `mandiplan/geometry/plate.py` and is
repeated in the CSV header. Note that for a plate on the buccal surface most of
the arch curvature shows up as out-of-plane bend, since the plate's own plane
is the tangent plane of the bone.

The panel then answers the question that matters at the bench in one line:
**does this plate fit?** If a different length of the same plate spans the
plan better it offers it. The reasons sit under *Why*: how many screw holes land
on retained bone either side of the defect (three for a load-bearing plate),
whether a bridge is bent past what the alloy takes over that length (about 26°
for a 2.4 mm grade 4 plate on an 8 mm pitch), how far the bone contact is from
the planned standoff, and what the bends do to the screw holes.

The **Bench steps** tab turns all of that into instructions for the kit you
picked, measured from the proximal cut end of the plate because that is what a
ruler measures, and pointed in patient anatomy rather than in frame
conventions:

> 1. Take the 10-hole 2.4 mm reconstruction bar (90.0 mm). Trim 1 hole from the
>    distal end to 81.0 mm.
> 2. Mark at 13.5 mm from the proximal end (hole 1). Bend 12.4° across the
>    plate's face, so the distal end moves toward the patient's left, in 2
>    passes of about 6.2° each. Grip 10 mm either side of the mark.
> 4. At 31.4 mm (hole 3): 1.0° of twist clockwise is needed; bending irons
>    cannot twist. Use twisting forceps here.

**6 · Export.** Every file in one place: the reconstructed jaw, the mirrored
segment alone, the resected segment, the bent plate, the bending guide, the bend
table, the bench steps and the resection summary. Every STL carries the
attribution; every table carries the disclaimer.

### The bending guide

A plate is bent by eye against a model, over and over. The bending guide
removes the guessing. It is a chain of saddles, one per screw hole, printed in
one piece (TPU 95A, or PETG) and clipped onto the plate as it comes out of the
packet, straight or preformed. A window over each hole leaves it free for the
bending irons; a thin strap links each saddle to the next.

Where two saddles meet, over the bridge between two holes, their ends are cut
on the mitre of the planned bend at that bridge, overbent by the alloy's
springback. On the side the bend closes they stand apart by a wedge; bend the
bridge that way until the two saddles touch, let go, and the plate relaxes onto
the plan. On the other side the ends are square and simply move apart, so the
guide also shows which way to bend. Each saddle carries its hole number, and
the table written next to the STL says, bridge by bridge, what the bend is and
when to stop. Twist closes no gap and has no stop; it is listed for checking by
eye.

### The plate catalogue is yours to correct

The plate library (`mandiplan/data/plates`) is generic, dimensioned from
published size classes, and every plate says so. Its working limits — how far a
bridge may be bent, how tight a radius the alloy takes — are derived from the
alloy's rated ductility and the bridge between two screw seats, not from any
manufacturer's rating.

`mandiplan/data/bending_kits.json` describes the instruments: bending irons,
three-point pliers, a bar press, twisting forceps, and a template wire. Add your
department's set with its own instrument names and the steps will read the way
your kit is labelled.

## Privacy

MandiPlan runs entirely on your machine. It makes no network calls, has no
analytics, no accounts and no cloud sync; `tests/test_no_network.py` fails if
any module so much as imports something that could open a connection.

**De-identify DICOM before use.** Nothing here removes patient identifiers, and
exported files carry no PHI only because they carry no header data at all.

## How it is checked

Accuracy is checked two ways.

**Against a phantom** generated from a closed-form curve — an elliptical
cross-section swept along a circular arc, with configurable anisotropic voxels,
a bone-like intensity plateau and Gaussian noise — where arc length,
cross-section axes, turn angle per node and solid volume are all known exactly.

| what is checked | tolerance | worst measured |
|---|---|---|
| panoramic arc length of the bone vs. analytic | 2% | 0.01% |
| cross-section width and height vs. known ellipse axes | 2% | 0.9% |
| measurement on 0.25 × 0.5 × 0.8 mm anisotropic voxels | 1% | 0.7% |
| in-plane bend on a constant-radius arc vs. analytic turn | 1° | 0.00001° |
| twist on a planar path | ≈0 | 0° |
| total plate length vs. analytic arc length | 2% | 0.38% |
| resected fragment volume vs. analytic wedge | 3% | 0.03% |
| mid-sagittal plane offset on a symmetric phantom | 0.6 mm | 0.06 mm |
| mirrored volume vs. the fragment it replaces | 5% | 0.02% |
| un-mirrorable span across the midline vs. analytic | 1.5 mm | 0.0 mm |
| DICOM round trip: spacing and orientation preserved | exact | exact |

**Against a real scan.** `mandiplan/data/reference/dz_cbct_jaws.npz` is a head
CBCT from the 3D Slicer sample data ("CBCT-MR Head"), which the 3D Slicer
project states was donated by the person in the images to be used without
restriction; `tools/make_reference_case.py` rebuilds it from the public file
and checks its checksum. It has everything a phantom lacks: soft tissue in the
histogram, teeth in occlusion, fillings, condyles in their fossae, asymmetry.
The tests check that the threshold lands on bone, that the separated mandible
keeps both condyles and none of the maxilla or palate, that a cut removes
mandible only, and that the reconstruction of a lateral defect is one piece
with junction steps under 0.15 mm and smaller than a plain mirror's (0.05 mm
against 0.47 mm on this scan). The same checks were run by hand on the two
dental-surgery CBCTs in the 3D Slicer sample data.

The bending guide is checked geometrically: two neighbouring saddles do not
touch up to the stop angle and do just past it, for a curl and for an in-plane
bend, and bending the wrong way opens the gap.

`pytest` runs all of it, including end-to-end passes through the real
application: load, separate, cut, reconstruct, refine, draw a plate, bend it,
export.

## Known limitations

- **One real reference scan, not a population.** The reconstruction is the
  patient's own mirrored anatomy; where the defect crosses the midline and
  there is nothing to mirror, the gap is filled from the pre-operative contour,
  which is wrong wherever the lesion has already changed the bone. A
  statistical shape model built from many de-identified mandibles would do
  better there; it needs that data set, which this application does not ship.
- Mirroring assumes the healthy side is normal. A patient whose contralateral
  side is also diseased, previously operated, or simply asymmetric will get a
  target that is wrong in exactly the way the symmetry score warns about.
- The mandible separation needs teeth or joint contacts it can find. Heavy
  metal artefact across the bite, or a curve drawn far from the mandible, can
  leave a piece on the wrong side; the bone panel's checkbox shows all bone,
  and the separated volume is reported so a wrong split is visible.
- The plate library is generic. Its dimensions come from published size
  classes, not from any manufacturer, and nothing in the application knows the
  real bending characteristics of the plate in your hand.
- The bending guide's stops are as good as the print: 0.1 mm of printing error
  on a saddle wall is about 1.3° of bend. Twist has no stop. A flexible filament
  gives a little under load; bend slowly to contact.
- Two cutting planes. That covers a segmental resection but not a
  hemimandibulectomy with a condylar cut.
- Margin distances are measured to points you place by eye, not to a segmented
  lesion.
- Multi-frame enhanced-CT DICOM is not handled; one file per slice is expected.
- Nothing is saved between runs except what you export. There is no case file.
