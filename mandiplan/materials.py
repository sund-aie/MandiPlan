"""Implant alloys, and what they do when you bend them.

Every number here is a **published standard minimum or a nominal handbook
value**, not a measurement of the plate in your hand. ASTM specifications give
minima; a real lot is usually stronger. Treat these as the starting point for
a bending prediction, not as certified properties of a specific device.

Why the crystal structure is recorded
-------------------------------------
It is not decoration. How a plate behaves under an iron follows from it:

* **CP titanium** (grades 1-4) is single-phase alpha, hexagonal close-packed.
  HCP has few independent slip systems, so it work-hardens quickly, is
  anisotropic, and is prone to springback. Ductile, but it resents repeated
  reverse bending.
* **Ti-6Al-4V ELI** is two-phase alpha+beta, HCP plus body-centred cubic. The
  beta phase adds slip systems and strength but the alloy is markedly less
  ductile than CP titanium; it springs back hardest and cracks soonest.
* **316LVM steel** is austenitic, face-centred cubic. FCC has twelve slip
  systems, which is why it is by far the most forgiving to bend — high
  elongation, low springback for its stiffness.

Springback
----------
:meth:`Alloy.springback_ratio` is the classical sheet-bending relation

    Ri/Rf = 4 (Ri sigma_y / (E t))^3 - 3 (Ri sigma_y / (E t)) + 1

relating the radius under load to the radius after release. It is a
first-order elastic-recovery model for a bent strip, not a finite-element
result, and it ignores the Bauschinger effect and any prior cold work. It is
the right order of magnitude and the right *direction*, which is what a
bending plan needs; it is not a substitute for bending the plate and looking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Alloy:
    """An implant alloy, as its standard specifies it."""

    id: str
    name: str
    standard: str
    #: alpha / alpha-beta / austenitic
    phase: str
    #: The space lattice: HCP, BCC, FCC, or a combination.
    lattice: str
    density_g_cm3: float
    modulus_gpa: float
    yield_mpa: float
    ultimate_mpa: float
    elongation_pct: float
    poisson: float
    #: Minimum bend radius as a multiple of the plate thickness.
    min_bend_radius_ratio: float
    #: Rough relative resistance to repeated reverse bending, 0-1.
    reverse_bend_tolerance: float
    notes: str = ""

    @property
    def modulus_mpa(self) -> float:
        return self.modulus_gpa * 1000.0

    def min_bend_radius_mm(self, thickness_mm: float) -> float:
        return self.min_bend_radius_ratio * float(thickness_mm)

    def springback_ratio(self, radius_mm: float, thickness_mm: float) -> float:
        """``Ri / Rf`` — loaded radius over released radius.

        Returns a value in (0, 1]. 1.0 means the bend holds exactly; 0.8 means
        the plate relaxes to a radius 25% larger than it was bent to.
        """
        if radius_mm <= 0 or thickness_mm <= 0:
            raise ValueError("radius and thickness must be positive")
        x = radius_mm * self.yield_mpa / (self.modulus_mpa * thickness_mm)
        ratio = 4.0 * x**3 - 3.0 * x + 1.0
        return float(min(max(ratio, 1e-3), 1.0))

    def overbend_deg(self, target_deg: float, radius_mm: float, thickness_mm: float) -> float:
        """How far past the target to bend so it relaxes onto the target.

        Angle and radius are inversely related at fixed arc length, so the
        angle scales as the inverse of the springback ratio.
        """
        ratio = self.springback_ratio(radius_mm, thickness_mm)
        return float(target_deg / ratio)

    def strain_at_bend(self, radius_mm: float, thickness_mm: float) -> float:
        """Peak outer-fibre strain of a bend, as a fraction."""
        if radius_mm <= 0:
            return float("inf")
        return float(thickness_mm / (2.0 * radius_mm + thickness_mm))

    def strain_headroom(self, radius_mm: float, thickness_mm: float) -> float:
        """Outer-fibre strain as a fraction of the alloy's elongation limit.

        Above 1.0 the bend is past the material's rated ductility and is a
        crack risk, not merely a tight bend.
        """
        limit = self.elongation_pct / 100.0
        return float(self.strain_at_bend(radius_mm, thickness_mm) / limit)

    def summary_lines(self) -> list[str]:
        return [
            f"{self.name} ({self.standard})",
            f"{self.phase}, {self.lattice} lattice",
            f"Yield {self.yield_mpa:.0f} MPa, tensile {self.ultimate_mpa:.0f} MPa, "
            f"elongation {self.elongation_pct:.0f}%",
            f"Modulus {self.modulus_gpa:.0f} GPa, density {self.density_g_cm3:.2f} g/cm³",
            f"Minimum bend radius {self.min_bend_radius_ratio:.1f} x thickness",
        ]


#: Published standard minima and nominal handbook values. Not lot-certified.
ALLOYS: tuple[Alloy, ...] = (
    Alloy(
        id="ti-6al-4v-eli",
        name="Ti-6Al-4V ELI (Grade 23)",
        standard="ASTM F136",
        phase="alpha-beta",
        lattice="HCP + BCC",
        density_g_cm3=4.43,
        modulus_gpa=114.0,
        yield_mpa=795.0,
        ultimate_mpa=860.0,
        elongation_pct=10.0,
        poisson=0.34,
        min_bend_radius_ratio=4.0,
        reverse_bend_tolerance=0.35,
        notes=(
            "Strongest of the three and the least forgiving. Springs back "
            "hardest, cracks soonest, and tolerates the fewest reverse bends. "
            "Used where a plate must be thin and still load-bearing."
        ),
    ),
    Alloy(
        id="cp-ti-grade-4",
        name="Commercially pure titanium, Grade 4",
        standard="ASTM F67",
        phase="alpha",
        lattice="HCP",
        density_g_cm3=4.51,
        modulus_gpa=105.0,
        yield_mpa=483.0,
        ultimate_mpa=550.0,
        elongation_pct=15.0,
        poisson=0.34,
        min_bend_radius_ratio=3.0,
        reverse_bend_tolerance=0.55,
        notes=(
            "The usual choice for reconstruction bars that have to be "
            "contoured on the table: strong enough to bear load, ductile "
            "enough to shape."
        ),
    ),
    Alloy(
        id="cp-ti-grade-2",
        name="Commercially pure titanium, Grade 2",
        standard="ASTM F67",
        phase="alpha",
        lattice="HCP",
        density_g_cm3=4.51,
        modulus_gpa=103.0,
        yield_mpa=275.0,
        ultimate_mpa=345.0,
        elongation_pct=20.0,
        poisson=0.34,
        min_bend_radius_ratio=2.0,
        reverse_bend_tolerance=0.70,
        notes=(
            "Soft and very formable. Thin adaptation and trauma plates, not "
            "load-bearing reconstruction."
        ),
    ),
    Alloy(
        id="316lvm",
        name="316LVM stainless steel",
        standard="ASTM F138",
        phase="austenitic",
        lattice="FCC",
        density_g_cm3=7.95,
        modulus_gpa=193.0,
        yield_mpa=190.0,
        ultimate_mpa=490.0,
        elongation_pct=40.0,
        poisson=0.30,
        min_bend_radius_ratio=1.5,
        reverse_bend_tolerance=0.80,
        notes=(
            "Twelve slip systems, so by far the most forgiving under an iron: "
            "high elongation and the least springback for its stiffness. "
            "Stiffer than titanium and not MRI-friendly."
        ),
    ),
)


def alloy_by_id(alloy_id: str) -> Alloy:
    for alloy in ALLOYS:
        if alloy.id == alloy_id:
            return alloy
    raise KeyError(f"no alloy with id {alloy_id!r}")


def alloy_choices() -> list[tuple[str, str]]:
    return [(alloy.id, alloy.name) for alloy in ALLOYS]
