# Audit — UI, resection planes, plate geometry

State of the code before the September 2026 rework. Written so the reasoning
behind the three following commits is recoverable later.

## 1. UI and theme

`mandiplan/ui/theme.py` holds a semantic palette (background, surface, border,
text, accent, danger) and one ~200-line `stylesheet()` returning a single QSS
blob. Colours are already restrained; the problems are structural.

- The planning dock is added to `LeftDockWidgetArea`, so the inspector sits on
  the left and the 3-D viewport is pushed right.
- The toolbar is text-only. There are no icons, no wordmark, no Undo/Redo/Fit
  View, and no tooltips, so an icon-only control could not be introduced
  without leaving it unlabelled.
- Panel content is a `QToolBox` accordion of five fixed pages. It is not
  context-sensitive: selecting a tool in the toolbar does not change what the
  panel offers.
- Several sizes are hard-coded in pixels (`setMinimumWidth(400)`, font sizes in
  `px`), which breaks at large font scaling.
- Empty panels render blank rather than explaining what to do first.
- There is no screenshot or visual-regression infrastructure anywhere in
  `tests/` or `tools/`.

## 2. Resection planes

`geometry/spline.py` is sound. `ArchCurve` is a genuine 3-D centripetal
Catmull-Rom with a proper arc-length table, `sample_at(s) -> (point, tangent)`,
`length_mm` and `arc_position_of`. Nothing needs a second curve representation.

The failures are downstream of it.

**`build_frames` refuses non-axial curves.** `geometry/cpr.py` computes
`normals = t x z_hat` and raises when the curve runs parallel to the superior
axis: *"arch curve is locally parallel to the superior axis; seed points must
lie in an axial plane"*. There is no stored up-axis and no parallel transport,
so a curve climbing the ramus cannot be represented at all.

**Plane state is world-space only.** `SessionState` stores
`CutPlane(origin, normal)` and nothing else. `plane_arc_position()`
reverse-engineers the arc position by nearest-point search over
`frames.points` every time it is needed.

**Translation freezes the normal.** `SessionState.translate_plane` calls
`plane_from_frame(...)` but discards the returned normal and re-uses
`plane.normal`. The plane therefore slides along the jaw carrying whatever
world-space angulation it had, which is the reported "slides left-right with a
fixed arbitrary angle" behaviour. `plane_from_frame` itself does derive the
normal from the local tangent; it is simply never consulted on a move.

`yaw` also rotates about the global `SUPERIOR = (0, 0, 1)` rather than about a
transported local up axis.

## 3. Plate geometry

**The rectangle is `ribbon_mesh()`, `geometry/plate.py`.** It sweeps a
four-corner rectangular cross-section along the resampled path — `p +/- half*b`
for the width and `+ thickness*n` for the depth, four vertices per station.
It has no outline, no lobes, no holes, and no rounded edges. It is reached from
exactly two places:

- `ui/view3d.py`, `refresh_plate()` — what is drawn in the viewport;
- `exporting.py`, `write_template_stl()` — what is written to STL.

Display and export therefore agree with each other and are both wrong in the
same way.

**Plate selection carries no geometry.** `plate_catalog.PlateSystem` holds
`hole_pitch_mm`, `end_margin_mm`, `width_mm`, `thickness_mm`, `hole_counts` and
working limits. There is no mesh path, no hole coordinate, no centreline, no
provenance field. `mandiplan/data/` contains three JSON files and no mesh
assets of any kind. Choosing a different plate system changes the screw pitch
and the fit verdict; it changes nothing about the rendered shape.

**Screw positions are decorative.** `refresh_plate` feeds `plan.nodes` into a
sphere glyph. The markers are not derived from hole geometry because no hole
geometry exists; they are the resampled path stations.

The orange in the viewport is `MARK_COLOUR = (1.0, 0.75, 0.2)` on the landmark
glyphs. The plate actor itself is currently pale blue,
`PLATE_COLOUR = (0.55, 0.70, 0.95)`.

## Consequences for the rework

1. Keep `ArchCurve`. Add a transported frame to `ArchFrames` rather than a
   second curve. Parallel transport of `z` along a curve lying in the axial
   plane returns `z` exactly, so the panoramic reformat is unaffected.
2. Make `(s_mm, yaw, tilt, roll)` the authoritative plane state and derive
   `CutPlane` from it, so translation re-derives the normal from the local
   frame while the operator's angular offsets survive the move.
3. Replace `ribbon_mesh` at both call sites with a real asset mesh. Removing it
   from `view3d` alone would leave the export writing a rectangle.
