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

Turning the mirror into a flush reconstruction (registration to each stump,
blending into the retained bone, one surface) happens in
``geometry/reconstruction.py``; this module supplies the plane and coverage.
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
