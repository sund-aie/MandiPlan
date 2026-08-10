"""Shared fixtures.  The phantom is built once per session; it takes seconds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_phantom import PhantomSpec, make_phantom  # noqa: E402

from mandiplan.geometry import cpr  # noqa: E402
from mandiplan.geometry.spline import ArchCurve  # noqa: E402


@pytest.fixture(scope="session")
def spec() -> PhantomSpec:
    """Default phantom: anisotropic 0.3 x 0.3 x 0.6 mm voxels."""
    return PhantomSpec()


@pytest.fixture(scope="session")
def phantom(spec):
    return make_phantom(spec)


@pytest.fixture(scope="session")
def skewed_spec() -> PhantomSpec:
    """A second phantom whose three voxel dimensions all differ."""
    return PhantomSpec(spacing=(0.25, 0.5, 0.8), seed=7)


@pytest.fixture(scope="session")
def skewed_phantom(skewed_spec):
    return make_phantom(skewed_spec)


@pytest.fixture(scope="session")
def arch_curve(spec) -> ArchCurve:
    """Arch curve fitted through 7 seed points on the phantom centre-line."""
    return ArchCurve(spec.centre_line(7))


@pytest.fixture(scope="session")
def arch_frames(arch_curve):
    return cpr.build_frames(arch_curve, step_mm=0.2)


@pytest.fixture(scope="session")
def wide_arch_frames(spec):
    """Arch curve drawn past both ends of the bone, as a user would draw it."""
    curve = ArchCurve(spec.centre_line(9, extend_deg=15.0))
    return cpr.build_frames(curve, step_mm=0.2)


@pytest.fixture(scope="session")
def bone_surface(phantom, spec):
    from mandiplan.render.surface import extract_isosurface

    return extract_isosurface(phantom, spec.half_max_value)
