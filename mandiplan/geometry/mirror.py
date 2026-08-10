"""Mirror reconstruction of a resected mandibular segment.

The healthy side is the best available template for the missing side, so the
reconstruction target is the retained bone reflected in the patient's
mid-sagittal plane. Two things stop that from being the whole answer, and both
are reported rather than hidden:

* **A defect that crosses the midline has no donor.** The tissue that would be
  mirrored into the crossing part is itself inside the resection, so mirroring
  returns nothing there. That span is measured and reported separately.
* **The mid-sagittal plane is an estimate.** It is found by maximising the
  overlap of the bone mask with its own reflection, and the score it achieved
  is reported so a poorly symmetric case is visible rather than assumed.

For the span mirroring cannot cover, the missing bone is estimated
geometrically: the cross-section is blended between the two ends of the gap and
swept along the arch curve. That is an interpolation of this patient's own
anatomy, not a prediction from a population of mandibles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SUPERIOR = np.array([0.0, 0.0, 1.0])


@dataclass
class MidSagittalPlane:
    """The patient's estimated plane of symmetry."""

    point: np.ndarray  # a point on the plane, world mm
    normal: np.ndarray  # unit normal, close to the patient's left-right axis
    symmetry: float  # fraction of bone voxels whose reflection is also bone

    def __post_init__(self) -> None:
        self.point = np.asarray(self.point, dtype=float).reshape(3)
        n = np.asarray(self.normal, dtype=float).reshape(3)
        self.normal = n / np.linalg.norm(n)

    def signed_distance(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        return (pts - self.point) @ self.normal

    def reflect(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        flat = pts.reshape(-1, 3)
        offset = (flat - self.point) @ self.normal
        return (flat - 2.0 * offset[:, None] * self.normal).reshape(pts.shape)

    @property
    def tilt_deg(self) -> float:
        """How far the plane is tilted from the patient's left-right axis."""
        return float(np.degrees(np.arccos(min(1.0, abs(self.normal[0])))))


def _normal_from_angles(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """A near-lateral normal, tilted by two small angles."""
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    n = np.array([np.cos(yaw) * np.cos(pitch), np.sin(yaw), np.sin(pitch)])
    return n / np.linalg.norm(n)


def _bone_points(volume, threshold: float, max_points: int = 40000) -> np.ndarray:
    """World coordinates of bone voxels, subsampled for the plane search."""
    mask = volume.array >= threshold
    k, j, i = np.nonzero(mask)
    if len(i) == 0:
        raise ValueError("no voxel is above the bone threshold")
    rng = np.random.default_rng(0)
    # Shuffle before trimming: the search subsamples this array with a stride,
    # and a stride through index order would sample a regular lattice.
    order = rng.permutation(len(i))[:max_points]
    index = np.column_stack([i[order], j[order], k[order]]).astype(float)
    return volume.origin + index * volume.spacing


def _symmetry_score(points: np.ndarray, volume, threshold: float, plane) -> float:
    """Fraction of bone points whose mirror image also lands on bone."""
    reflected = plane.reflect(points)
    values = volume.sample(reflected, fill=float(volume.array.min()))
    return float(np.mean(values >= threshold))


#: (points used, lateral half-range mm, lateral step mm, tilt half-range deg, tilt step deg)
_SEARCH_STAGES = (
    (4000, 20.0, 2.0, 8.0, 2.0),
    (12000, 3.0, 0.5, 2.0, 0.5),
    (40000, 0.6, 0.15, 0.6, 0.15),
)


def estimate_midsagittal_plane(
    volume, threshold: float, max_points: int = 40000
) -> MidSagittalPlane:
    """Find the plane that best maps the bone onto itself.

    Coordinate descent over the plane's lateral offset and two small tilt
    angles, coarse to fine, using more of the bone at each refinement. A full
    grid search over the three parameters costs twenty times as much and lands
    in the same place: the parameters are close to independent for a plane that
    starts nearly lateral, which it does, because the volume is in patient
    (LPS) axes.
    """
    points = _bone_points(volume, threshold, max_points)
    centre = points.mean(axis=0)
    parameters = [0.0, 0.0, 0.0]  # lateral offset mm, yaw deg, pitch deg

    def evaluate(sample: np.ndarray, values: list[float]) -> float:
        normal = _normal_from_angles(values[1], values[2])
        candidate = MidSagittalPlane(centre + normal * values[0], normal, 0.0)
        return _symmetry_score(sample, volume, threshold, candidate)

    def scan(sample: np.ndarray, index: int, half: float, step: float) -> None:
        centre_value = parameters[index]
        best_value, best_score = centre_value, -1.0
        for value in np.arange(centre_value - half, centre_value + half + 1e-9, step):
            trial = list(parameters)
            trial[index] = float(value)
            found = evaluate(sample, trial)
            if found > best_score:
                best_score, best_value = found, float(value)
        parameters[index] = best_value

    for n_points, d_half, d_step, a_half, a_step in _SEARCH_STAGES:
        sample = points[:: max(1, len(points) // n_points)]
        for _ in range(2):
            scan(sample, 0, d_half, d_step)
            scan(sample, 1, a_half, a_step)
            scan(sample, 2, a_half, a_step)

    normal = _normal_from_angles(parameters[1], parameters[2])
    return MidSagittalPlane(
        centre + normal * parameters[0], normal, evaluate(points, parameters)
    )


@dataclass
class MirrorCoverage:
    """How much of a defect the healthy side can actually be mirrored into."""

    entry_mm: float
    exit_mm: float
    covered_mm: float
    uncovered_mm: float
    crosses_midline: bool
    midline_s_mm: float
    uncovered_spans: list[tuple[float, float]] = field(default_factory=list)

    @property
    def defect_mm(self) -> float:
        return abs(self.exit_mm - self.entry_mm)

    @property
    def covered_fraction(self) -> float:
        return self.covered_mm / self.defect_mm if self.defect_mm else 0.0

    def summary(self) -> str:
        if not self.crosses_midline:
            return (
                f"The defect is {self.defect_mm:.1f} mm and lies on one side of the "
                "midline, so the healthy side mirrors onto all of it."
            )
        return (
            f"The defect is {self.defect_mm:.1f} mm and crosses the midline at "
            f"{self.midline_s_mm:.1f} mm along the arch curve. Mirroring covers "
            f"{self.covered_mm:.1f} mm; the remaining {self.uncovered_mm:.1f} mm has "
            "no healthy counterpart to mirror, because the bone that would be "
            "mirrored into it is inside the resection as well."
        )


def mirror_coverage(frames, plane: MidSagittalPlane, entry_mm: float, exit_mm: float,
                    step_mm: float = 0.5) -> MirrorCoverage:
    """Which parts of the defect the mirrored healthy side reaches."""
    entry_mm, exit_mm = sorted((float(entry_mm), float(exit_mm)))
    samples = np.arange(entry_mm, exit_mm + 1e-9, step_mm)
    if len(samples) < 2:
        samples = np.array([entry_mm, exit_mm])

    indices = np.clip(
        np.round(samples / frames.step_mm).astype(int), 0, len(frames.s) - 1
    )
    points = frames.points[indices]
    mirrored = plane.reflect(points)
    donor_s = _nearest_arc_positions(frames, mirrored)
    uncovered = (donor_s >= entry_mm) & (donor_s <= exit_mm)

    distances = plane.signed_distance(points)
    crosses = bool(np.any(distances > 0) and np.any(distances < 0))
    midline_s = float("nan")
    if crosses:
        sign_change = np.where(np.sign(distances[:-1]) * np.sign(distances[1:]) < 0)[0]
        if len(sign_change):
            i = int(sign_change[0])
            w = distances[i] / (distances[i] - distances[i + 1])
            midline_s = float(samples[i] + w * (samples[i + 1] - samples[i]))

    return MirrorCoverage(
        entry_mm=entry_mm,
        exit_mm=exit_mm,
        covered_mm=float(np.sum(~uncovered) * step_mm),
        uncovered_mm=float(np.sum(uncovered) * step_mm),
        crosses_midline=crosses,
        midline_s_mm=midline_s,
        uncovered_spans=_contiguous_spans(samples, uncovered),
    )


def _nearest_arc_positions(frames, points: np.ndarray) -> np.ndarray:
    """Arc position of the curve point closest to each of ``points``."""
    diff = points[:, None, :] - frames.points[None, :, :]
    return frames.s[np.argmin(np.einsum("ijk,ijk->ij", diff, diff), axis=1)]


def _contiguous_spans(samples: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = float(samples[i])
        elif not flag and start is not None:
            spans.append((start, float(samples[i - 1])))
            start = None
    if start is not None:
        spans.append((start, float(samples[-1])))
    return spans


@dataclass
class CrossSectionProfile:
    """The bone outline at one arc position, approximated as an ellipse."""

    s_mm: float
    centre_u_mm: float
    centre_z_mm: float
    semi_bl_mm: float
    semi_si_mm: float


def _crossing_span(values: np.ndarray, level: float) -> tuple[float, float] | None:
    """First and last crossings of ``level``, in fractional sample units.

    Interpolating the crossing rather than counting samples above the level
    matters here: whole-sample counting loses up to half a sample at each end,
    which on a 12 mm cortex is a two to three percent underestimate.
    """
    above = np.nonzero(values >= level)[0]
    if len(above) == 0:
        return None
    first, last = int(above[0]), int(above[-1])

    def interp(inside: int, outside: int) -> float:
        a, b = values[outside], values[inside]
        if a == b:
            return float(inside)
        return outside + (level - a) / (b - a) * (inside - outside)

    start = interp(first, first - 1) if first > 0 else float(first)
    end = interp(last, last + 1) if last < len(values) - 1 else float(last)
    return start, end


def estimate_profile(
    cross_section, threshold: float, s_mm: float
) -> CrossSectionProfile | None:
    """Fit an ellipse to the bone in a buccolingual cross-section."""
    mask = cross_section.image >= threshold
    if not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    centre_u = cross_section.col_to_mm(float(cols.mean()))
    centre_z = cross_section.row_to_mm(float(rows.mean()))

    # One refinement pass: measure across the middle, then re-measure through
    # the centre that gave.
    for _ in range(2):
        row = int(np.clip(round(cross_section.mm_to_row(centre_z)), 0, mask.shape[0] - 1))
        col = int(np.clip(round(cross_section.mm_to_col(centre_u)), 0, mask.shape[1] - 1))
        span_u = _crossing_span(cross_section.image[row, :], threshold)
        span_z = _crossing_span(cross_section.image[:, col], threshold)
        if span_u is None or span_z is None:
            return None
        centre_u = cross_section.col_to_mm(0.5 * (span_u[0] + span_u[1]))
        centre_z = cross_section.row_to_mm(0.5 * (span_z[0] + span_z[1]))

    return CrossSectionProfile(
        s_mm=float(s_mm),
        centre_u_mm=centre_u,
        centre_z_mm=centre_z,
        semi_bl_mm=0.5 * (span_u[1] - span_u[0]) * cross_section.pixel_mm,
        semi_si_mm=0.5 * (span_z[1] - span_z[0]) * cross_section.pixel_mm,
    )


def bridge_mesh(
    frames,
    start: CrossSectionProfile,
    end: CrossSectionProfile,
    station_step_mm: float = 1.0,
    ring_points: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep a blended elliptical cross-section along the arch between two ends.

    Returns ``(points, triangles)`` for a closed surface. This is the estimate
    of the missing bone where mirroring has no donor: an interpolation between
    the patient's own cross-sections at the two ends of the gap.
    """
    s_a, s_b = sorted((start.s_mm, end.s_mm))
    if s_b - s_a < station_step_mm:
        raise ValueError("bridge span is shorter than one station")
    if s_a == start.s_mm:
        first, second = start, end
    else:
        first, second = end, start

    stations = np.arange(s_a, s_b + 1e-9, station_step_mm)
    theta = np.linspace(0.0, 2 * np.pi, ring_points, endpoint=False)
    rings = np.empty((len(stations), ring_points, 3))

    for k, s in enumerate(stations):
        w = (s - s_a) / (s_b - s_a)
        semi_bl = (1 - w) * first.semi_bl_mm + w * second.semi_bl_mm
        semi_si = (1 - w) * first.semi_si_mm + w * second.semi_si_mm
        centre_u = (1 - w) * first.centre_u_mm + w * second.centre_u_mm
        centre_z = (1 - w) * first.centre_z_mm + w * second.centre_z_mm

        index = int(np.clip(round(s / frames.step_mm), 0, len(frames.s) - 1))
        p = frames.points[index]
        n = frames.normals[index]
        u = centre_u + semi_bl * np.cos(theta)
        z = centre_z + semi_si * np.sin(theta)
        rings[k, :, 0] = p[0] + u * n[0]
        rings[k, :, 1] = p[1] + u * n[1]
        rings[k, :, 2] = z

    return _loft(rings)


def _loft(rings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a stack of closed rings into a capped tube."""
    n_rings, ring_points, _ = rings.shape
    points = rings.reshape(-1, 3)
    centre_start = points[:ring_points].mean(axis=0)
    centre_end = points[-ring_points:].mean(axis=0)
    start_index = len(points)
    points = np.vstack([points, centre_start, centre_end])
    end_index = start_index + 1

    tris: list[tuple[int, int, int]] = []
    for k in range(n_rings - 1):
        a = k * ring_points
        b = (k + 1) * ring_points
        for e in range(ring_points):
            e2 = (e + 1) % ring_points
            tris.append((a + e, b + e, b + e2))
            tris.append((a + e, b + e2, a + e2))
    for e in range(ring_points):
        e2 = (e + 1) % ring_points
        tris.append((start_index, e2, e))
        last = (n_rings - 1) * ring_points
        tris.append((end_index, last + e, last + e2))

    triangles = np.array(tris, dtype=np.int64)
    if _signed_volume(points, triangles) < 0:
        triangles = triangles[:, ::-1].copy()
    return points, triangles


def _signed_volume(points: np.ndarray, triangles: np.ndarray) -> float:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
