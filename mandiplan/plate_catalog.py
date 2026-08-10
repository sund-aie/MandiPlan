"""Plate systems, bending kits, and whether a chosen plate actually fits.

The catalogue in ``mandiplan/data`` is a set of generic profiles by size class,
not a manufacturer catalogue: the numbers are defaults to be replaced with the
ones on the specification sheet of the system in the theatre. Everything here
reads that file rather than hard-coding a dimension.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class PlateOption:
    """One length of one plate system."""

    holes: int
    length_mm: float

    def __str__(self) -> str:
        return f"{self.holes} holes, {self.length_mm:.1f} mm"


@dataclass(frozen=True)
class PlateSystem:
    id: str
    name: str
    shape: str
    hole_pitch_mm: float
    end_margin_mm: float
    width_mm: float
    thickness_mm: float
    hole_counts: tuple[int, ...]
    bend_warning_deg: float
    min_bend_radius_mm: float
    min_holes_per_side: int
    notes: str = ""
    preformed_angle_deg: float | None = None

    @property
    def is_custom(self) -> bool:
        return self.shape == "custom"

    def length_for(self, holes: int) -> float:
        """Overall plate length for a hole count, in millimetres."""
        return (holes - 1) * self.hole_pitch_mm + 2 * self.end_margin_mm

    def options(self) -> list[PlateOption]:
        return [PlateOption(h, self.length_for(h)) for h in self.hole_counts]

    def smallest_option_for(self, required_mm: float) -> PlateOption | None:
        for option in self.options():
            if option.length_mm >= required_mm:
                return option
        return None

    def longest_option(self) -> PlateOption | None:
        options = self.options()
        return options[-1] if options else None


@dataclass(frozen=True)
class BendingKit:
    id: str
    name: str
    instruments: tuple[str, ...]
    handles: tuple[str, ...]
    grip_span_mm: float
    max_angle_per_pass_deg: float
    can_twist: bool
    marking: str
    notes: str = ""

    def can_make(self, bend_kind: str) -> bool:
        return bend_kind in self.handles


@lru_cache(maxsize=1)
def load_systems() -> tuple[PlateSystem, ...]:
    raw = json.loads((DATA_DIR / "plate_systems.json").read_text(encoding="utf-8"))
    return tuple(
        PlateSystem(
            id=entry["id"],
            name=entry["name"],
            shape=entry["shape"],
            hole_pitch_mm=float(entry["hole_pitch_mm"]),
            end_margin_mm=float(entry["end_margin_mm"]),
            width_mm=float(entry["width_mm"]),
            thickness_mm=float(entry["thickness_mm"]),
            hole_counts=tuple(int(h) for h in entry["hole_counts"]),
            bend_warning_deg=float(entry["bend_warning_deg"]),
            min_bend_radius_mm=float(entry["min_bend_radius_mm"]),
            min_holes_per_side=int(entry["min_holes_per_side"]),
            notes=entry.get("notes", ""),
            preformed_angle_deg=(
                float(entry["preformed_angle_deg"])
                if "preformed_angle_deg" in entry
                else None
            ),
        )
        for entry in raw["systems"]
    )


@lru_cache(maxsize=1)
def load_kits() -> tuple[BendingKit, ...]:
    raw = json.loads((DATA_DIR / "bending_kits.json").read_text(encoding="utf-8"))
    return tuple(
        BendingKit(
            id=entry["id"],
            name=entry["name"],
            instruments=tuple(entry["instruments"]),
            handles=tuple(entry["handles"]),
            grip_span_mm=float(entry["grip_span_mm"]),
            max_angle_per_pass_deg=float(entry["max_angle_per_pass_deg"]),
            can_twist=bool(entry["can_twist"]),
            marking=entry["marking"],
            notes=entry.get("notes", ""),
        )
        for entry in raw["kits"]
    )


def system_by_id(system_id: str) -> PlateSystem:
    for system in load_systems():
        if system.id == system_id:
            return system
    raise KeyError(f"no plate system with id {system_id!r}")


def kit_by_id(kit_id: str) -> BendingKit:
    for kit in load_kits():
        if kit.id == kit_id:
            return kit
    raise KeyError(f"no bending kit with id {kit_id!r}")


@dataclass
class FitReport:
    """Whether the chosen plate can be made to fit the drawn path."""

    system: PlateSystem
    required_mm: float
    option: PlateOption | None
    holes_proximal: int = 0
    holes_distal: int = 0
    holes_over_defect: int = 0
    uncovered_mm: float = 0.0
    trim_holes: int = 0
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        if self.problems:
            return False
        # A custom plate is made to the path, so it has no stock length.
        return self.system.is_custom or self.option is not None

    @property
    def verdict(self) -> str:
        if self.system.is_custom:
            return (
                f"Custom plate: made to the planned path, {self.required_mm:.1f} mm, "
                "no chairside bending."
            )
        if self.option is None:
            longest = self.system.longest_option()
            longest_text = f"{longest.length_mm:.1f} mm" if longest else "none listed"
            return (
                f"No length in this system reaches {self.required_mm:.1f} mm "
                f"(longest is {longest_text}). Shorten the path, choose another "
                "system, or order a custom plate."
            )
        text = (
            f"Use the {self.option.holes}-hole length "
            f"({self.option.length_mm:.1f} mm) for a {self.required_mm:.1f} mm path."
        )
        if self.trim_holes:
            text += f" Trim {self.trim_holes} hole(s) to length."
        return text


def fit_check(plan, system: PlateSystem, defect_span=None, frames=None) -> FitReport:
    """Check a plate path against one plate system.

    ``defect_span`` is ``(entry_mm, exit_mm)`` along the arch curve and
    ``frames`` the arch frames, so that screw holes can be counted on the
    retained bone either side of the resection.
    """
    required = float(plan.total_length_mm)
    report = FitReport(
        system=system,
        required_mm=required,
        option=None if system.is_custom else system.smallest_option_for(required),
    )

    if not system.is_custom:
        if report.option is None:
            longest = system.longest_option()
            report.uncovered_mm = required - (longest.length_mm if longest else 0.0)
            report.problems.append(
                f"{report.uncovered_mm:.1f} mm of the path has no plate over it."
            )
        else:
            report.trim_holes = _trim_holes(system, report.option, required)

    if abs(plan.pitch_mm - system.hole_pitch_mm) > 1e-6:
        report.warnings.append(
            f"The path was resampled at {plan.pitch_mm:.1f} mm but this system's "
            f"screw-hole pitch is {system.hole_pitch_mm:.1f} mm."
        )

    if defect_span is not None and frames is not None:
        proximal, over, distal = _count_holes(plan, frames, defect_span)
        report.holes_proximal = proximal
        report.holes_over_defect = over
        report.holes_distal = distal
        for label, count in (("proximal", proximal), ("distal", distal)):
            if count < system.min_holes_per_side:
                report.problems.append(
                    f"Only {count} screw hole(s) land on retained bone {label} to the "
                    f"defect; this system's working minimum is "
                    f"{system.min_holes_per_side}. Extend the path on that side."
                )

    steep = [
        node
        for node in plan.bends
        if np.isfinite(node.in_plane_deg)
        and max(abs(node.in_plane_deg), abs(node.out_of_plane_deg))
        > system.bend_warning_deg
    ]
    for node in steep:
        report.warnings.append(
            f"Node {node.index} at {node.cumulative_mm:.1f} mm turns "
            f"{max(abs(node.in_plane_deg), abs(node.out_of_plane_deg)):.1f}°, above the "
            f"{system.bend_warning_deg:.0f}° working limit set for this system."
        )
    return report


def _trim_holes(system: PlateSystem, option: PlateOption, required_mm: float) -> int:
    """How many holes can be cut off and still cover the path."""
    trim = 0
    for holes in range(option.holes - 1, 1, -1):
        if system.length_for(holes) >= required_mm:
            trim = option.holes - holes
        else:
            break
    return trim


def _count_holes(plan, frames, defect_span) -> tuple[int, int, int]:
    entry, exit_ = sorted(float(v) for v in defect_span)
    proximal = over = distal = 0
    for node in plan.nodes:
        s = _arc_position(frames, node)
        if s < entry:
            proximal += 1
        elif s > exit_:
            distal += 1
        else:
            over += 1
    return proximal, over, distal


def _arc_position(frames, point) -> float:
    d = np.linalg.norm(frames.points - np.asarray(point, dtype=float), axis=1)
    return float(frames.s[int(np.argmin(d))])
