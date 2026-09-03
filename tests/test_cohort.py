"""Dataset registry, licence policy, and the field-of-view filter."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mandiplan.cohort import (
    cohort_lines,
    installed_case_total,
    licence_is_allowed,
    load_cohort,
    load_registry,
    source_is_allowed,
)
from mandiplan.geometry.truncation import check_field_of_view, mask_from_volume


def test_the_registry_lists_the_requested_datasets():
    names = {entry["name"] for entry in load_registry()["datasets"]}
    assert {
        "margo",
        "dental3d-multimodal",
        "pmcanalseg",
        "cipriano-mandibular-canal",
        "toothfairy2",
    } <= names


def test_nothing_in_the_registry_claims_to_be_verified():
    """Every record is a user-supplied claim until the fetcher checks it."""
    registry = load_registry()
    for entry in registry["datasets"] + registry["external_models"]:
        assert entry["metadata_verified"] is False


def test_toothfairy2_is_flagged_and_filtered():
    entry = next(e for e in load_registry()["datasets"] if e["name"] == "toothfairy2")
    assert "license unclear" in entry["licence_status"]
    assert entry["pool"] == "secondary"
    assert entry["requires_truncation_filter"] is True


def test_the_model_weights_are_not_vendored():
    model = load_registry()["external_models"][0]
    assert model["vendored"] is False
    assert model["labels"]["2"] == "mandible"
    assert model["labels"]["5"] == "mandibular canal"


@pytest.mark.parametrize(
    "licence",
    ["CC-BY-NC-ND-4.0", "CC-BY-NC-4.0", "CC-BY-ND-4.0", "MIT", ""],
)
def test_excluded_licences_are_rejected(licence):
    allowed, reason = licence_is_allowed(licence)
    assert not allowed
    assert reason


@pytest.mark.parametrize("licence", ["CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0"])
def test_permitted_licences_are_accepted(licence):
    assert licence_is_allowed(licence)[0]


def test_tcia_sources_are_rejected_by_pattern():
    allowed, reason = source_is_allowed("https://www.cancerimagingarchive.net/x")
    assert not allowed
    assert "excluded" in reason
    assert source_is_allowed("https://zenodo.org/records/7882821")[0]


def test_an_uninstalled_cohort_says_so(tmp_path):
    cohort = load_cohort(tmp_path)
    assert all(not record.installed for record in cohort)
    assert installed_case_total(cohort) == 0
    text = "\n".join(cohort_lines(cohort))
    assert "not installed" in text
    assert "metadata unverified" in text


def test_the_cohort_states_which_populations_are_missing(tmp_path):
    text = "\n".join(cohort_lines(load_cohort(tmp_path)))
    for population in ("Middle Eastern", "African", "South Asian"):
        assert population in text


def test_an_installed_manifest_is_counted(tmp_path):
    folder = tmp_path / "margo"
    folder.mkdir()
    (folder / "manifest.json").write_text(
        json.dumps({"name": "margo", "case_count": 42}), encoding="utf-8"
    )
    cohort = load_cohort(tmp_path)
    margo = next(record for record in cohort if record.name == "margo")
    assert margo.installed
    assert margo.installed_cases == 42
    assert installed_case_total(cohort) == 42


def test_demographics_are_never_invented(tmp_path):
    for record in load_cohort(tmp_path):
        assert record.sex_split == "not published"
        assert record.age_range == "not published"


# -- field of view ----------------------------------------------------------


def _labels(shape=(24, 24, 24)) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def test_a_mandible_inside_the_scan_is_usable():
    labels = _labels()
    labels[6:18, 6:18, 6:18] = 2
    report = check_field_of_view(labels)
    assert report.usable
    assert not report.faces


@pytest.mark.parametrize(
    "slicer,face",
    [
        ((slice(0, 12), slice(6, 18), slice(6, 18)), "inferior"),
        ((slice(12, 24), slice(6, 18), slice(6, 18)), "superior"),
        ((slice(6, 18), slice(12, 24), slice(6, 18)), "posterior"),
        ((slice(6, 18), slice(6, 18), slice(12, 24)), "patient left"),
    ],
)
def test_a_mandible_touching_any_face_is_rejected(slicer, face):
    labels = _labels()
    labels[slicer] = 2
    report = check_field_of_view(labels)
    assert not report.usable
    assert face in report.faces
    assert "cut off" in report.summary()


def test_a_case_with_no_mandible_label_is_rejected():
    report = check_field_of_view(_labels())
    assert not report.usable
    assert "no mandible label" in report.faces[0]


def test_only_the_mandible_label_is_considered():
    labels = _labels()
    labels[6:18, 6:18, 6:18] = 2  # mandible, well inside
    labels[0:24, 0:2, 0:24] = 1  # maxilla running off the edge
    assert check_field_of_view(labels).usable


def test_the_filter_also_runs_on_a_raw_volume(phantom, spec):
    """A dataset with no labels yet can still be screened by threshold."""
    labels = mask_from_volume(phantom, spec.half_max_value)
    assert check_field_of_view(labels).usable
