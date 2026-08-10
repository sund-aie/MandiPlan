"""Turn a plate plan into step-by-step bending instructions for a chosen kit.

Distances are given from the **proximal cut end of the plate**, because that is
what a person measures with a ruler at the bench, not from the first screw
hole. Directions are given in patient anatomy (anterior, superior, patient's
left...) rather than in the frame conventions, so a step can be followed
without reading the geometry documentation first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .plate_catalog import BendingKit, FitReport, PlateSystem

#: Bends smaller than this are not worth a step.
NEGLIGIBLE_DEG = 0.5

_AXIS_LABELS = (
    ("toward the patient's left", "toward the patient's right"),
    ("posteriorly", "anteriorly"),
    ("superiorly", "inferiorly"),
)


@dataclass
class BendStep:
    order: int
    kind: str  # "cut", "bend", "twist", "check"
    distance_mm: float  # from the proximal cut end of the plate, NaN when n/a
    node_index: int
    angle_deg: float
    instrument: str
    text: str


def anatomical_direction(vector) -> str:
    """Name the dominant direction of an LPS vector in anatomical words."""
    v = np.asarray(vector, dtype=float)
    axis = int(np.argmax(np.abs(v)))
    return _AXIS_LABELS[axis][0 if v[axis] > 0 else 1]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _passes(angle_deg: float, kit: BendingKit) -> int:
    if kit.max_angle_per_pass_deg <= 0:
        return 1
    return max(1, math.ceil(abs(angle_deg) / kit.max_angle_per_pass_deg))


def generate_steps(
    plan, system: PlateSystem, kit: BendingKit, fit: FitReport | None = None
) -> list[BendStep]:
    """Ordered bench instructions for bending ``plan`` out of ``system``."""
    steps: list[BendStep] = []
    order = 1
    offset = system.end_margin_mm  # hole 0 sits this far from the cut end

    if system.is_custom:
        steps.append(
            BendStep(
                order,
                "check",
                float("nan"),
                -1,
                float("nan"),
                "—",
                "Patient-specific plate: it arrives contoured to this path. "
                "Verify it against the printed template before use; no bending steps.",
            )
        )
        return steps

    if fit is not None and fit.option is not None:
        length = fit.option.length_mm
        cut = (
            f" Trim {fit.trim_holes} hole(s) from the distal end to "
            f"{system.length_for(fit.option.holes - fit.trim_holes):.1f} mm."
            if fit.trim_holes
            else ""
        )
        steps.append(
            BendStep(
                order,
                "cut",
                length,
                -1,
                float("nan"),
                "—",
                f"Take the {fit.option.holes}-hole {system.name} "
                f"({length:.1f} mm).{cut}",
            )
        )
        order += 1

    if system.preformed_angle_deg is not None:
        steps.append(
            BendStep(
                order,
                "check",
                float("nan"),
                -1,
                system.preformed_angle_deg,
                "—",
                f"This bar is pre-formed to {system.preformed_angle_deg:.0f}°. "
                "Lay it on the template first and bend only the residual "
                "difference given below.",
            )
        )
        order += 1

    for node in plan.bends:
        index = node.index
        if index == 0 or index >= len(plan.nodes) - 1:
            continue
        distance = node.cumulative_mm + offset
        tangent = plan.tangents[index]
        normal = plan.normals[index]
        binormal = plan.binormals[index]

        for kind, angle, axis in (
            ("in_plane", node.in_plane_deg, normal),
            ("out_of_plane", node.out_of_plane_deg, binormal),
        ):
            if not np.isfinite(angle) or abs(angle) < NEGLIGIBLE_DEG:
                continue
            heading = _unit(np.cross(axis, tangent)) * np.sign(angle)
            towards = anatomical_direction(heading)
            plane = (
                "in the plate's own plane"
                if kind == "in_plane"
                else "across the plate's face"
            )
            if not kit.can_make(kind):
                steps.append(
                    BendStep(
                        order,
                        "bend",
                        distance,
                        index,
                        angle,
                        "—",
                        f"At {distance:.1f} mm (hole {index}): {abs(angle):.1f}° "
                        f"{plane}, {towards}. {kit.name} does not make this bend — "
                        "use an instrument that does.",
                    )
                )
                order += 1
                continue
            passes = _passes(angle, kit)
            pass_text = (
                f" in {passes} passes of about {abs(angle) / passes:.1f}° each"
                if passes > 1
                else ""
            )
            grip = (
                f" Grip {kit.grip_span_mm / 2:.0f} mm either side of the mark."
                if kit.grip_span_mm > 0
                else ""
            )
            steps.append(
                BendStep(
                    order,
                    "bend",
                    distance,
                    index,
                    angle,
                    kit.instruments[0],
                    f"Mark at {distance:.1f} mm from the proximal end (hole {index}). "
                    f"Bend {abs(angle):.1f}° {plane}, so the distal end moves "
                    f"{towards}{pass_text}.{grip}",
                )
            )
            order += 1

        twist = node.twist_deg
        if np.isfinite(twist) and abs(twist) >= NEGLIGIBLE_DEG:
            sense = "clockwise" if twist > 0 else "counter-clockwise"
            if kit.can_twist:
                steps.append(
                    BendStep(
                        order,
                        "twist",
                        distance,
                        index,
                        twist,
                        kit.instruments[-1],
                        f"At {distance:.1f} mm (hole {index}): twist {abs(twist):.1f}° "
                        f"{sense}, sighting along the plate from the proximal end.",
                    )
                )
            else:
                steps.append(
                    BendStep(
                        order,
                        "twist",
                        distance,
                        index,
                        twist,
                        "—",
                        f"At {distance:.1f} mm (hole {index}): {abs(twist):.1f}° of "
                        f"twist {sense} is needed; {kit.name} cannot twist. Use "
                        "twisting forceps here.",
                    )
                )
            order += 1

    steps.append(
        BendStep(
            order,
            "check",
            float("nan"),
            -1,
            float("nan"),
            "—",
            f"{kit.marking} Lay the bent plate on the printed template and correct "
            "any step that does not sit down before taking it to the field.",
        )
    )
    return steps


def steps_as_rows(steps: list[BendStep]) -> list[list[str]]:
    """Step list as display strings, units included."""
    rows = []
    for step in steps:
        rows.append(
            [
                str(step.order),
                step.kind,
                "—" if not np.isfinite(step.distance_mm) else f"{step.distance_mm:.1f} mm",
                "—" if not np.isfinite(step.angle_deg) else f"{step.angle_deg:+.1f}°",
                step.text,
            ]
        )
    return rows
