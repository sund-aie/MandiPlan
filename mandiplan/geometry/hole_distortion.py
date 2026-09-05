"""What bending actually does to a screw hole.

An earlier version of this application held every screw hole perfectly
circular through bending and presented that as a feature. It is not what
happens. AO Surgery Reference is explicit:

    "Without bending insets, the holes become deformed during contouring of
    the plate, and precise seating of the locking screws cannot be
    guaranteed."

and, on preformed plates:

    "The minimal intraoperative bending required in preformed plates preserves
    the optimal threaded-hole shape, resulting in a plate with increased
    fatigue life compared to standard reconstruction plates."

So a hole near a bend goes out of round, a locking screw may then not seat,
and fatigue life falls. This module predicts that, and :func:`ovalise_mesh`
applies it to the plate's geometry so the operator can see it rather than read
about it.

The model
---------
For a bend of radius ``R`` in a plate of thickness ``t``, the outer fibre
carries strain ``e = t / (2R + t)``. A hole is a stress raiser: the strain in
the material immediately around a circular hole is amplified by a
concentration factor, taken here as ``Kt = 2.2`` — between the classical 3.0
for a hole in a wide plate under uniaxial tension and the lower value that
applies in bending, where the strain gradient through the thickness relieves
the peak.

A hole only sees the part of the bend that lands on it, so the strain is
scaled by how much of the instrument's bend arc overlaps the hole. That is
where the kit matters: three-point pliers concentrate a bend into about five
millimetres and put nearly all of it through whatever is there, while a press
spreads it over twelve and fits insets as well.

The hole then stretches along the plate and draws in across it by Poisson's
ratio, giving

    major = d (1 + e_eff)          minor = d (1 - nu e_eff)

Twisting instruments shear rather than stretch, so their holes come out as a
skewed rhomboid rather than an oval; that is reported as a skew angle.

Limits of the model
-------------------
This is a closed-form estimate, not a finite-element result. It does not model
thread geometry, work hardening from previous passes, the Bauschinger effect
on reverse bends, or the true three-dimensional stress field around a
countersink. It predicts the right direction and the right order of magnitude
so a plan can flag a hole worth checking. **Measure the hole before trusting a
locking screw to it.**
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Strain concentration at a circular hole in a plate under bending.
STRESS_CONCENTRATION = 2.2

#: How far out of round a threaded hole may go and still take a locking screw.
#: Locking threads engage over a small radial tolerance; past the second
#: figure the thread no longer indexes the screw head reliably.
LOCKING_TOLERANCE_MM = 0.03
LOCKING_MARGINAL_MM = 0.10

#: Even with insets fitted, some deformation gets through.
INSET_RESIDUAL = 0.08


@dataclass
class HoleDistortion:
    """One screw hole after bending."""

    index: int
    nominal_diameter_mm: float
    major_mm: float
    minor_mm: float
    skew_deg: float
    strain: float
    absorbed_bend_deg: float
    protected: bool

    @property
    def out_of_round_mm(self) -> float:
        return float(self.major_mm - self.minor_mm)

    @property
    def ovality_pct(self) -> float:
        return float(100.0 * self.out_of_round_mm / self.nominal_diameter_mm)

    @property
    def takes_locking_screw(self) -> bool:
        return self.out_of_round_mm <= LOCKING_TOLERANCE_MM

    @property
    def verdict(self) -> str:
        if self.out_of_round_mm <= LOCKING_TOLERANCE_MM:
            return "round; locking screw seats"
        if self.out_of_round_mm <= LOCKING_MARGINAL_MM:
            return "slightly oval; locking screw marginal, non-locking fine"
        return "out of round; use a non-locking screw or re-drill"

    def describe(self) -> str:
        return (
            f"Hole {self.index + 1}: {self.major_mm:.3f} x {self.minor_mm:.3f} mm "
            f"({self.ovality_pct:.1f}% oval, {self.out_of_round_mm * 1000:.0f} um "
            f"out of round) — {self.verdict}"
        )


@dataclass
class DistortionReport:
    """Every hole after a bending plan, and what it means for the plate."""

    holes: list[HoleDistortion] = field(default_factory=list)
    #: Remaining fatigue life as a fraction of an unbent plate's.
    fatigue_fraction: float = 1.0
    kit_signature: str = ""
    warnings: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def worst(self) -> HoleDistortion | None:
        return max(self.holes, key=lambda h: h.out_of_round_mm, default=None)

    @property
    def locking_capable(self) -> int:
        return sum(1 for h in self.holes if h.takes_locking_screw)

    def summary_lines(self) -> list[str]:
        if not self.holes:
            return ["No holes analysed."]
        worst = self.worst
        lines = [
            f"{self.locking_capable} of {len(self.holes)} holes stay round "
            f"enough for a locking screw.",
            f"Worst: {worst.describe()}" if worst else "",
            f"Estimated fatigue life after bending: "
            f"{self.fatigue_fraction * 100:.0f}% of an unbent plate.",
        ]
        if self.kit_signature:
            lines.append(self.kit_signature)
        return [line for line in lines if line]


def predict_distortion(
    hole_positions_mm: np.ndarray,
    hole_diameter_mm: float,
    thickness_mm: float,
    bend_angles_deg: np.ndarray,
    bend_positions_mm: np.ndarray,
    alloy,
    kit=None,
    use_insets: bool | None = None,
) -> DistortionReport:
    """Predict what a bending plan does to every screw hole.

    ``hole_positions_mm`` and ``bend_positions_mm`` are arc positions along the
    plate. ``bend_angles_deg`` is the turn taken at each bend position.
    """
    hole_positions = np.asarray(hole_positions_mm, dtype=float).reshape(-1)
    bend_positions = np.asarray(bend_positions_mm, dtype=float).reshape(-1)
    bend_angles = np.abs(np.nan_to_num(np.asarray(bend_angles_deg, dtype=float)))
    if len(bend_angles) != len(bend_positions):
        raise ValueError("each bend angle needs a position along the plate")
    if thickness_mm <= 0 or hole_diameter_mm <= 0:
        raise ValueError("thickness and hole diameter must be positive")

    localisation = float(getattr(kit, "bend_localisation_mm", 10.0)) or 10.0
    protection = float(getattr(kit, "hole_protection", 0.0))
    insets = (
        bool(getattr(kit, "bending_insets", False))
        if use_insets is None
        else bool(use_insets)
    )
    if insets and protection <= 0.0:
        # The operator says they are fitting insets with a kit that does not
        # list them; credit the protection AO describes, less a residual.
        protection = 1.0 - INSET_RESIDUAL
    if not insets:
        protection = 0.0
    shears = bool(getattr(kit, "can_twist", False))

    report = DistortionReport(kit_signature=getattr(kit, "signature", "") or "")
    cumulative_strain = 0.0

    for index, position in enumerate(hole_positions):
        # How much of each bend lands on this hole: a triangular overlap
        # between the instrument's bend arc and the hole itself.
        reach = (localisation + hole_diameter_mm) / 2.0
        overlap = np.clip(1.0 - np.abs(bend_positions - position) / reach, 0.0, 1.0)
        absorbed = float(np.sum(bend_angles * overlap))

        if absorbed <= 0.0:
            report.holes.append(
                HoleDistortion(
                    index=index,
                    nominal_diameter_mm=hole_diameter_mm,
                    major_mm=hole_diameter_mm,
                    minor_mm=hole_diameter_mm,
                    skew_deg=0.0,
                    strain=0.0,
                    absorbed_bend_deg=0.0,
                    protected=insets,
                )
            )
            continue

        # The radius this much turn implies over the arc it is taken across:
        # arc length over angle in radians.
        radius = localisation / max(np.radians(absorbed), 1e-9)
        strain = alloy.strain_at_bend(radius, thickness_mm) * STRESS_CONCENTRATION
        strain *= 1.0 - protection
        cumulative_strain += strain

        major = hole_diameter_mm * (1.0 + strain)
        minor = hole_diameter_mm * (1.0 - alloy.poisson * strain)
        skew = float(np.degrees(strain)) if shears else 0.0

        report.holes.append(
            HoleDistortion(
                index=index,
                nominal_diameter_mm=hole_diameter_mm,
                major_mm=float(major),
                minor_mm=float(minor),
                skew_deg=skew,
                strain=float(strain),
                absorbed_bend_deg=absorbed,
                protected=insets,
            )
        )

    report.fatigue_fraction = _fatigue_fraction(report, alloy, cumulative_strain)
    _judge(report, alloy, insets, kit)
    return report


def _fatigue_fraction(report: DistortionReport, alloy, cumulative_strain: float) -> float:
    """Rough remaining fatigue life after cold work and hole ovalisation.

    Cold work at a bend and an out-of-round hole both cut fatigue life, and AO
    warns plainly that excessive or repeated bending ends in plate fracture.
    This is an index for comparing plans, not a cycle count.
    """
    worst = report.worst
    ovality = worst.ovality_pct / 100.0 if worst else 0.0
    work = cumulative_strain / max(alloy.elongation_pct / 100.0, 1e-6)
    fraction = float(np.exp(-1.6 * work) * np.exp(-3.0 * ovality))
    return float(np.clip(fraction, 0.02, 1.0))


def _judge(report: DistortionReport, alloy, insets: bool, kit) -> None:
    compromised = [h for h in report.holes if not h.takes_locking_screw]
    severe = [h for h in report.holes if h.out_of_round_mm > LOCKING_MARGINAL_MM]

    if severe:
        report.problems.append(
            f"{len(severe)} screw hole(s) are more than "
            f"{LOCKING_MARGINAL_MM * 1000:.0f} um out of round after bending. "
            "A locking screw will not index in these; plan non-locking screws "
            "there, move the bend off the hole, or fit bending insets."
        )
    elif compromised:
        report.warnings.append(
            f"{len(compromised)} screw hole(s) go slightly oval. Non-locking "
            "screws are unaffected; check any locking screw before relying on it."
        )

    if not insets and getattr(kit, "insets_available", False) and compromised:
        report.warnings.append(
            "This kit can take bending insets. Fitting them into the holes that "
            "will carry locking screws keeps those holes round while the plate "
            "is contoured."
        )
    if report.fatigue_fraction < 0.5:
        report.problems.append(
            f"Estimated fatigue life is down to "
            f"{report.fatigue_fraction * 100:.0f}% of an unbent plate. "
            f"{alloy.name} tolerates repeated bending poorly; consider a "
            "preformed plate or a fresh one shaped in fewer passes."
        )
    elif report.fatigue_fraction < 0.75:
        report.warnings.append(
            f"Cold work has taken estimated fatigue life to "
            f"{report.fatigue_fraction * 100:.0f}% of an unbent plate."
        )


def ovalise_mesh(
    points: np.ndarray,
    hole_centres: np.ndarray,
    hole_axes: np.ndarray,
    plate_tangents: np.ndarray,
    distortions: list[HoleDistortion],
    hole_diameter_mm: float,
) -> np.ndarray:
    """Apply predicted hole distortion to the plate mesh, so it is visible.

    Vertices on and immediately around each hole wall are stretched along the
    plate and drawn in across it by that hole's predicted major and minor
    axes. The effect fades out over a couple of hole diameters so the plate
    body is not disturbed.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3).copy()
    hole_centres = np.asarray(hole_centres, dtype=float).reshape(-1, 3)
    hole_axes = np.asarray(hole_axes, dtype=float).reshape(-1, 3)
    plate_tangents = np.asarray(plate_tangents, dtype=float).reshape(-1, 3)
    radius = hole_diameter_mm / 2.0
    falloff = hole_diameter_mm * 2.0

    for distortion in distortions:
        if distortion.out_of_round_mm < 1e-9:
            continue
        i = distortion.index
        centre = hole_centres[i]
        axis = hole_axes[i] / np.linalg.norm(hole_axes[i])
        along = plate_tangents[i] - float(np.dot(plate_tangents[i], axis)) * axis
        norm = np.linalg.norm(along)
        if norm < 1e-9:
            continue
        along = along / norm
        across = np.cross(axis, along)

        offset = points - centre
        # Distance from the hole's own axis, in the plane of the plate.
        radial = offset - np.outer(offset @ axis, axis)
        distance = np.linalg.norm(radial, axis=1)
        touched = distance < radius + falloff
        if not np.any(touched):
            continue

        # 1 on the hole wall, fading to 0 two diameters out.
        weight = np.clip(
            1.0 - (distance[touched] - radius) / falloff, 0.0, 1.0
        )
        major_gain = distortion.major_mm / hole_diameter_mm - 1.0
        minor_gain = distortion.minor_mm / hole_diameter_mm - 1.0

        u = radial[touched] @ along
        v = radial[touched] @ across
        points[touched] += (
            np.outer(u * major_gain * weight, along)
            + np.outer(v * minor_gain * weight, across)
        )
        if distortion.skew_deg:
            shear = np.tan(np.radians(distortion.skew_deg))
            points[touched] += np.outer(v * shear * weight, along)
    return points
