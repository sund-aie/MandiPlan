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

## Install

Python 3.11 or newer.

```sh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Primary platform is macOS on Apple Silicon; it runs on Linux and Windows too.
On a headless Linux machine the test suite needs `Xvfb` installed (it starts
one itself); an ordinary desktop session needs nothing.

## Run

```sh
python -m mandiplan                 # or: python -m mandiplan /path/to/dicom
pytest                              # must pass with no skips
python tests/make_phantom.py        # writes build/phantom_dicom to open in the app
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

**5 · Plate.** *Draw plate path on the bone* and click along the bone in the
3-D view; clicks are projected onto the surface. The path is resampled at the
screw-hole pitch of your plate system (9 mm by default, editable — it is not a
constant), and each interior node gets three signed angles in the plate's own
frame:

| angle | what it is |
|---|---|
| in-plane bend | the contour bend in the plate's own plane, about the surface normal |
| out-of-plane bend | rotation about the binormal; positive lifts the plate off the bone |
| twist | rotation of the plate's face about the direction of travel |

The full convention is in the docstring of `mandiplan/geometry/plate.py` and is
repeated in the CSV header. Note that for a plate on the buccal surface most of
the arch curvature shows up as out-of-plane bend, since the plate's own plane
is the tangent plane of the bone.

*Export bend table (CSV)* writes the table; *Export bending template (STL)*
sweeps a rectangular ribbon (12 × 2 mm by default) along the path for printing
and bending against.

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
| DICOM round trip: spacing and orientation preserved | exact | exact |

The cross-section figures are dominated by where the bone edge level is put,
not by the reformat: sampling the same profile at the exact half-maximum gives
0.04%. That is the same judgement a human makes placing a caliper on a cortical
outline, and it is the practical floor on any of these measurements.

`pytest` runs all of it, including an end-to-end pass through the real
application: load, flatten, measure, resect, draw a plate, export.

## Known limitations

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
- Nothing is saved between runs except what you export. There is no case file.
