"""The planning workflow as data: what is done, what is next, what is blocked.

Both of the commercial planners present the work as a numbered sequence rather
than a pile of tools, which is the right shape for a procedure that has to be
done in order. MandiPlan's steps are its own — a segmental resection is not an
orthognathic case — but the idea of the application knowing where you are, and
saying what is missing before a step can be taken, is worth borrowing.

This module holds no Qt: the status of each step is derived from the session
and can be tested without a window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StepStatus:
    index: int  # zero-based
    title: str
    done: bool
    blocked_by: str  # empty when the step can be worked on
    summary: str  # what has been achieved, or what to do next

    @property
    def available(self) -> bool:
        return not self.blocked_by


def _volume_step(session) -> tuple[bool, str, str]:
    if session.volume is None:
        return False, "", "Open a DICOM folder to begin."
    points = session.surface.GetNumberOfPoints() if session.surface is not None else 0
    if points == 0:
        return False, "", "No bone at this threshold — move the slider."
    spacing = session.volume.spacing
    return (
        True,
        "",
        f"Volume loaded at {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f} mm, "
        f"threshold {session.threshold:.0f}.",
    )


def _arch_step(session) -> tuple[bool, str, str]:
    if session.volume is None:
        return False, "a volume", "Load a volume first."
    if session.arch_curve is None:
        needed = 2 - len(session.arch_seeds)
        return False, "", f"Place at least {max(needed, 1)} more arch point(s) in the axial view."
    return (
        True,
        "",
        f"Arch curve {session.arch_curve.length_mm:.1f} mm long, "
        f"{len(session.arch_seeds)} seed points.",
    )


def _resection_step(session) -> tuple[bool, str, str]:
    if session.surface is None:
        return False, "a bone surface", "Load a volume first."
    if not session.planes:
        return False, "", "Add a cutting plane."
    report = session.report
    if report is None or not np.isfinite(report.arc_length_mm):
        return False, "", "Position the planes so they cross the arch curve."
    state = "cut executed" if session.cut_applied else "cut not executed yet"
    return (
        True,
        "",
        f"{len(session.planes)} plane(s), {report.arc_length_mm:.1f} mm segment, {state}.",
    )


def _reconstruction_step(session) -> tuple[bool, str, str]:
    if not session.planes:
        return False, "a resection", "Place the cutting planes first."
    if session.symmetry_plane is None:
        return False, "", "Estimate the mid-sagittal plane."
    if session.graft_surface is None:
        return False, "", "Mirror the healthy side into the defect."
    coverage = session.coverage
    detail = ""
    if coverage is not None and coverage.crosses_midline:
        detail = f", {coverage.uncovered_mm:.1f} mm not mirrorable"
    return True, "", f"Graft {session.graft_volume_mm3:.0f} mm³{detail}."


def _plate_step(session) -> tuple[bool, str, str]:
    if session.surface is None:
        return False, "a bone surface", "Load a volume first."
    if session.plate_plan is None:
        return False, "", "Draw a plate path on the bone."
    fit = session.fit
    if fit is not None and fit.problems:
        return False, "", fit.problems[0]
    return (
        True,
        "",
        f"{session.plate_plan.total_length_mm:.1f} mm path, "
        f"{len(session.plate_plan.nodes)} screw holes.",
    )


STEPS = (
    ("Volume and bone threshold", _volume_step),
    ("Arch curve and reformat", _arch_step),
    ("Resection planning", _resection_step),
    ("Mirror reconstruction", _reconstruction_step),
    ("Plate path and bends", _plate_step),
)


def workflow_status(session) -> list[StepStatus]:
    """Status of every planning step for the current session state."""
    statuses = []
    for index, (title, rule) in enumerate(STEPS):
        done, blocked_by, summary = rule(session)
        statuses.append(
            StepStatus(
                index=index,
                title=title,
                done=done,
                blocked_by=blocked_by,
                summary=summary,
            )
        )
    return statuses


def current_step(session) -> StepStatus:
    """The first step that is not finished, or the last one when all are."""
    statuses = workflow_status(session)
    for status in statuses:
        if not status.done:
            return status
    return statuses[-1]
