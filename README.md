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

## The workflow

**1 · Load and threshold.** *File → Open DICOM folder*. MandiPlan reads the
voxel spacing from the header, re-orients the volume to patient axes, and shows
the matrix, voxel size and field of view in millimetres. A series with gantry
tilt, sheared slice positions, non-uniform slice spacing or an oblique
orientation is refused with an explanation rather than loaded distorted.

CBCT gray values are not Hounsfield units — they depend on the scanner, the
field of view and the exposure — so there is no fixed bone threshold. The
histogram panel seeds one from Otsu's method over this volume's own histogram
and leaves it on a live slider; the isosurface follows as you drag. "Keep
largest connected component" is the only morphology applied. No smoothing is
applied at all, because a smoothed surface no longer coincides with the
gray values you thresholded.

**2 · Arch curve and reformat.** Switch to the *Slices* tab, press *Place arch
points*, and click 5–10 points along the jaw in the axial view. A cubic spline
is fitted through them and resampled at uniform **arc length**.

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

**3 · Measure.** Choose *Measure* and click two points. In the 3-D view you get
a straight-line distance. In the panoramic view you get three numbers, labelled
for what they are: arc length along the arch curve, superior–inferior
distance, and the flattened distance between the points. The x-axis of that
view is arc length, not a straight line, and the readout says so — across a
curved mandible the two differ by several millimetres.

**4 · Resection.** *Add cutting plane* drops a plane perpendicular to the arch
curve; drag its handles in the 3-D view. Bone that the current planes would
remove is tinted red as you move them. Up to two planes, which covers a
segmental resection. *Execute cut* separates the fragment and reports the
resected segment as both arc length along the curve and straight-line distance
between the cuts, plus the fragment volume. *Mark tumour margin point* places
landmarks, and the readout gives the signed distance from each cut plane to
each of them — positive means the point is on the resected side. *Undo cut*
puts it back; Ctrl+Z undoes editing steps.

**5 · Mirror reconstruction.** *Estimate mid-sagittal plane* finds the
patient's plane of symmetry by maximising the overlap of the bone with its own
reflection, and reports the score it reached — a low score means this patient
is not symmetric enough for mirroring to be trusted, and you can see that
before you rely on it. *Mirror healthy side into the defect* reflects the
retained bone and clips it to the resection: that mirrored piece is the
reconstruction target, and it exports as an STL.

Where the defect crosses the midline, mirroring runs out of donor: the bone
that would be mirrored into the crossing part is inside the resection too. The
panel measures that span and says so rather than quietly returning a partial
graft. For that span the missing bone is estimated by blending this patient's
own cross-sections at the two ends of the gap and sweeping them along the arch
curve — an interpolation of their anatomy, shown in a different colour from the
mirrored part so the two are never confused.

**6 · Plate.** Pick a plate system and a bending kit at the top of the panel.
The system sets the screw-hole pitch and the plate's width and thickness; the
kit decides which instrument each instruction names and how many passes a bend
is split into.

*Draw plate path on the bone* and click along the bone in the 3-D view; clicks
are projected onto the surface. The path is resampled at the screw-hole pitch,
and each interior node gets three signed angles in the plate's own frame:

| angle | what it is |
|---|---|
| in-plane bend | the contour bend in the plate's own plane, about the surface normal |
| out-of-plane bend | rotation about the binormal; positive lifts the plate off the bone |
| twist | rotation of the plate's face about the direction of travel |

The full convention is in the docstring of `mandiplan/geometry/plate.py` and is
repeated in the CSV header. Note that for a plate on the buccal surface most of
the arch curvature shows up as out-of-plane bend, since the plate's own plane
is the tangent plane of the bone.

The panel then answers the question that matters at the bench: **does this
plate fit?** It picks the shortest length in the chosen system that covers the
path, says how many holes to trim, counts how many screw holes land on retained
bone either side of the defect, and refuses the plan in red when either side
has too few for purchase or when no length in the system is long enough — in
which case it says a custom plate is needed rather than leaving you to work it
out. Bends steeper than the working limit set for that system are listed as
notes.

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

*Export bend table (CSV)* writes the angle table, *Export bench steps (CSV)*
the instructions above, and *Export bending template (STL)* sweeps the plate's
own cross-section along the path for printing and bending against.

### The plate catalogue is yours to correct

`mandiplan/data/plate_systems.json` holds generic profiles grouped by size
class — 2.0, 2.4 and 2.7 mm bars, a pre-bent angle bar, and a custom plate. It
is **not a manufacturer's catalogue and carries no part numbers**: the pitches,
widths, thicknesses and hole counts are plausible defaults, not specifications.
Check them against the sheet for the system you actually hold and edit the file
to match; `bend_warning_deg` and `min_bend_radius_mm` are working limits you set
for yourself, not manufacturer ratings.

`mandiplan/data/bending_kits.json` is the same idea for instruments: bending
irons, three-point pliers, a bar press, twisting forceps, and a template wire.
Add your department's set with its own instrument names and the steps will read
the way your kit is labelled.

## Privacy

MandiPlan runs entirely on your machine. It makes no network calls, has no
analytics, no accounts and no cloud sync; `tests/test_no_network.py` fails if
any module so much as imports something that could open a connection.

**De-identify DICOM before use.** Nothing here removes patient identifiers, and
exported files carry no PHI only because they carry no header data at all.

## How it is checked

There is no patient CBCT in this repository, so accuracy is checked against a
phantom generated from a closed-form curve: an elliptical cross-section swept
along a circular arc, with configurable anisotropic voxels, a bone-like
intensity plateau and Gaussian noise. Arc length, cross-section axes, turn
angle per node and solid volume are all known exactly.

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
| mirrored graft volume vs. the fragment it replaces | 5% | 0.02% |
| un-mirrorable span across the midline vs. analytic | 1.5 mm | 0.0 mm |
| estimated segment across the gap vs. analytic | 3% | 0.4% |
| DICOM round trip: spacing and orientation preserved | exact | exact |

The cross-section figures are dominated by where the bone edge level is put,
not by the reformat: sampling the same profile at the exact half-maximum gives
0.04%. That is the same judgement a human makes placing a caliper on a cortical
outline, and it is the practical floor on any of these measurements.

`pytest` runs all of it, including an end-to-end pass through the real
application: load, flatten, measure, resect, draw a plate, export.

## Known limitations

- **There is no library of mandibles behind the reconstruction.** Where
  mirroring has no donor, the missing bone is interpolated from this patient's
  own cross-sections at the two ends of the gap. It is not predicted from a
  population of normal mandibles, because building that needs a curated set of
  real CBCT scans, and this application makes no network calls and ships no
  patient data. If you assemble a de-identified set yourself, a statistical
  shape model fitted to it would replace the interpolation — that is a
  substantial piece of work, not a setting.
- Mirroring assumes the healthy side is normal. A patient whose contralateral
  side is also diseased, previously operated, or simply asymmetric will get a
  target that is wrong in exactly the way the symmetry score warns about.
- The plate catalogue is generic. Dimensions come from an editable file, not
  from any manufacturer, and nothing in the application knows the real bending
  characteristics of the alloy in your hand.
- The bending steps assume the plate starts straight (except for the pre-bent
  angle profile, which is only told to you, not modelled bend by bend).
- Segmentation is threshold plus largest-connected-component. Scatter from
  restorations, or a mandible touching the maxilla at the threshold you pick,
  will need the threshold moved by hand; there is no editing brush.
- Two cutting planes. That covers a segmental resection but not a
  hemimandibulectomy with a condylar cut.
- The bending template is a ribbon swept along the path. At a very tight bend
  its inner corners can self-intersect; check the STL before printing.
- Margin distances are measured to points you place by eye, not to a segmented
  lesion.
- Multi-frame enhanced-CT DICOM is not handled; one file per slice is expected.
- The arch curve is planar — seeds are placed in one axial slice. A jaw with
  significant vertical curvature across the region of interest is flattened
  against a curve that does not follow it in z.
- Rebuilding the panoramic reformat resamples the whole volume, so on a large
  field of view it takes a few seconds after each change to the curve or to the
  slab settings.
- Nothing is saved between runs except what you export. There is no case file.
