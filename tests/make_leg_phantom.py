"""Synthetic lower-leg CT with a tibia and a fibula of known geometry.

There is no leg CT in this repository, so fibula detection and fibula planning
are checked against a volume whose bones are built from closed-form shapes, in
Hounsfield units as a CT angiogram of the leg would give them:

- a straight fibula along ``z``, its distal tip (the lateral malleolus) at
  ``z = 0`` and its proximal tip (the head) at ``z = fibula_length_mm``,
  a rounded triangle in section, wider at the head and the malleolus;
- a thicker tibia beside it, medial and a little anterior, running past both
  ends of the fibula as a tibia does, flared at the knee and the ankle;
- the soft tissue of the leg round both, and air outside.

The fibula lies lateral to the tibia: towards the patient's right (-x in LPS)
for a right leg, left (+x) for a left one. ``side="both"`` puts both legs in
the scan, as a CT angiogram of the legs usually has them, ``leg_separation_mm``
apart. ``tilt_deg`` leans the leg sideways, its knee towards patient left, so
the bones do not run straight along the slices.

Run directly to write a DICOM series you can open in the app::

    python tests/make_leg_phantom.py [output_dir] [right|left|both]
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mandiplan.geometry.volume import Volume  # noqa: E402

AIR_HU = -1000.0
SOFT_HU = 40.0
CORTEX_HU = 1400.0
MARROW_HU = 250.0


@dataclass(frozen=True)
class LegSpec:
    """A lower leg. All lengths in millimetres, positions in patient LPS."""

    side: str = "right"
    fibula_length_mm: float = 340.0
    #: The fibula's shaft, head and malleolus, as the radius of a circle of the
    #: same area as its section.
    fibula_radius_mm: float = 7.0
    head_radius_mm: float = 12.5
    malleolus_radius_mm: float = 10.0
    tibia_radius_mm: float = 11.5
    cortex_mm: float = 2.5
    #: Where the fibula's axis is, and the tibia's beside it.
    #: A right leg's fibula and tibia, relative to its leg's middle; a left
    #: leg's are mirrored.
    fibula_xy: tuple[float, float] = (-22.0, 8.0)
    tibia_xy: tuple[float, float] = (0.0, -5.0)
    #: With both legs in the scan, their tibiae are this far apart, mm.
    leg_separation_mm: float = 180.0
    #: The leg leans this much sideways, its knee towards patient left (+x).
    tilt_deg: float = 0.0
    spacing: tuple[float, float, float] = (1.0, 1.0, 2.0)
    ramp_mm: float = 1.0
    noise_hu: float = 0.0
    seed: int = 3

    def sides(self) -> tuple[str, ...]:
        return ("right", "left") if self.side == "both" else (self.side,)

    def _leg_x(self, side: str) -> float:
        if self.side != "both":
            return 0.0
        return -0.5 * self.leg_separation_mm if side == "right" else 0.5 * self.leg_separation_mm

    def fibula_centre(self, side: str | None = None, z: float = 0.0) -> np.ndarray:
        """Where the fibula's axis crosses height ``z``, (x, y) mm."""
        side = side or self.sides()[0]
        x, y = self.fibula_xy
        mirror = 1.0 if side == "right" else -1.0
        return np.array([self._leg_x(side) + x * mirror + self._lean(z), y])

    def tibia_centre(self, side: str | None = None, z: float = 0.0) -> np.ndarray:
        side = side or self.sides()[0]
        x, y = self.tibia_xy
        mirror = 1.0 if side == "right" else -1.0
        return np.array([self._leg_x(side) + x * mirror + self._lean(z), y])

    def _lean(self, z: float) -> float:
        return float(np.tan(np.radians(self.tilt_deg)) * z)

    def fibula_axis(self) -> np.ndarray:
        """The fibula's direction, distal to proximal."""
        a = np.array([np.tan(np.radians(self.tilt_deg)), 0.0, 1.0])
        return a / np.linalg.norm(a)

    def fibula_tip_z(self) -> tuple[float, float]:
        """Heights of the fibula's distal and proximal tips: it is
        ``fibula_length_mm`` long along its leaning axis."""
        return 0.0, self.fibula_length_mm * float(np.cos(np.radians(self.tilt_deg)))

    def lateral(self, side: str | None = None) -> np.ndarray:
        """The patient's lateral direction for a leg, LPS."""
        side = side or self.sides()[0]
        return np.array([-1.0, 0.0, 0.0]) if side == "right" else np.array([1.0, 0.0, 0.0])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        top = self.fibula_length_mm + 45.0
        points = [
            centre
            for side in self.sides()
            for z in (-40.0, top)
            for centre in (self.fibula_centre(side, z), self.tibia_centre(side, z))
        ]
        lo = np.min(points, axis=0) - 55.0
        hi = np.max(points, axis=0) + 55.0
        return (np.array([lo[0], lo[1], -40.0]), np.array([hi[0], hi[1], top]))


def fibula_radius(spec: LegSpec, z: np.ndarray) -> np.ndarray:
    """Equivalent radius of the fibula's section at height ``z``; 0 past its tips."""
    length = spec.fibula_length_mm
    r = np.full_like(z, spec.fibula_radius_mm, dtype=float)
    # The malleolus: widening over its lowest 45 mm, rounded to a tip at z = 0.
    low = np.clip(1.0 - z / 45.0, 0.0, 1.0)
    r = r + (spec.malleolus_radius_mm - spec.fibula_radius_mm) * np.sin(np.pi * 0.5 * low) ** 2
    tip = np.clip(z / 8.0, 0.0, 1.0)
    r = r * np.sqrt(np.clip(1.0 - (1.0 - tip) ** 2, 0.0, 1.0))
    # The head: widening over its top 40 mm, rounded to a tip at z = length.
    high = np.clip(1.0 - (length - z) / 40.0, 0.0, 1.0)
    r = r + (spec.head_radius_mm - spec.fibula_radius_mm) * np.sin(np.pi * 0.5 * high) ** 2
    cap = np.clip((length - z) / 10.0, 0.0, 1.0)
    r = r * np.sqrt(np.clip(1.0 - (1.0 - cap) ** 2, 0.0, 1.0))
    return np.where((z < 0.0) | (z > length), 0.0, r)


def tibia_radius(spec: LegSpec, z: np.ndarray) -> np.ndarray:
    """The tibia runs past both ends of the fibula: from the ankle to the knee."""
    length = spec.fibula_length_mm
    r = np.full_like(z, spec.tibia_radius_mm, dtype=float)
    r = r + 8.0 * np.clip(1.0 - (z + 25.0) / 60.0, 0.0, 1.0) ** 2  # the plafond
    r = r + 14.0 * np.clip((z - (length - 60.0)) / 90.0, 0.0, 1.0) ** 2  # the plateau
    return np.where((z < -25.0) | (z > length + 35.0), 0.0, r)


def _section_distance(dx, dy, radius, twist):
    """Signed distance, roughly, to a rounded triangle of equivalent ``radius``."""
    angle = np.arctan2(dy, dx)
    rim = radius * (1.0 + 0.14 * np.cos(3.0 * (angle - twist)))
    return np.hypot(dx, dy) - rim


def make_leg_phantom(spec: LegSpec = LegSpec()) -> Volume:
    """Build the lower-leg phantom for ``spec``."""
    lo, hi = spec.bounds()
    spacing = np.asarray(spec.spacing, dtype=float)
    size = np.maximum(np.ceil((hi - lo) / spacing).astype(int) + 1, 2)
    x = lo[0] + np.arange(size[0]) * spacing[0]
    y = lo[1] + np.arange(size[1]) * spacing[1]
    z = lo[2] + np.arange(size[2]) * spacing[2]
    xx, yy = np.meshgrid(x, y, indexing="xy")  # (ny, nx)
    array = np.empty((len(z), len(y), len(x)), dtype=np.float32)
    ramp = spec.ramp_mm
    # Along the leaning axis the bones' sections are a little longer in z.
    along = float(np.cos(np.radians(spec.tilt_deg)))
    for k, zk in enumerate(z):
        value = np.full(xx.shape, AIR_HU, dtype=np.float64)
        for side in spec.sides():
            f, t = spec.fibula_centre(side, zk), spec.tibia_centre(side, zk)
            middle = 0.5 * (f + t)
            # The leg's soft tissue, an ellipse round both bones.
            soft = np.hypot((xx - middle[0]) / 45.0, (yy - middle[1]) / 40.0) < 1.0
            value[soft] = SOFT_HU
            mirror = 1.0 if side == "right" else -1.0
            for centre, radius, twist in (
                (f, float(fibula_radius(spec, np.array([zk / along]))[0]), 0.6),
                (t, float(tibia_radius(spec, np.array([zk / along]))[0]), -0.4),
            ):
                if radius <= 0.0:
                    continue
                d = _section_distance(mirror * (xx - centre[0]), yy - centre[1], radius, twist)
                outer = np.clip(0.5 - d / ramp, 0.0, 1.0)
                inner = np.clip(0.5 - (d + spec.cortex_mm) / ramp, 0.0, 1.0)
                bone = MARROW_HU * inner + CORTEX_HU * (outer - inner)
                value = value * (1.0 - outer) + bone
        array[k] = value
    if spec.noise_hu > 0:
        rng = np.random.default_rng(spec.seed)
        array = array + rng.normal(0.0, spec.noise_hu, size=array.shape).astype(np.float32)
    return Volume(array=array, spacing=spacing, origin=lo)


if __name__ == "__main__":
    from make_phantom import write_dicom_series

    out = Path(sys.argv[1] if len(sys.argv) > 1 else "leg_phantom_dicom")
    side = sys.argv[2] if len(sys.argv) > 2 else "right"
    volume = make_leg_phantom(LegSpec(side=side))
    write_dicom_series(volume, out, patient_id="LEGPHANTOM", description=f"MandiPlan {side} leg phantom")
    print(f"Wrote {side} leg phantom to {out}")
