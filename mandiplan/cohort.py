"""Which datasets back the shape library, and who is in them.

Reads the registry and whatever manifests have been written under
``data/cbct/<dataset>/manifest.json``. Makes no network calls: fetching is
tools/fetch_datasets.py, which is not part of the application.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "data" / "datasets.json"
DEFAULT_DATA_ROOT = Path("data/cbct")

UNKNOWN = "not published"

#: Populations that none of the registered datasets is known to cover. Stated
#: in the interface rather than left for the user to work out, because a shape
#: library built entirely on other populations is a limitation of every number
#: derived from it.
UNDER_REPRESENTED = (
    "Middle Eastern",
    "African",
    "South Asian",
)


@dataclass
class DatasetRecord:
    name: str
    title: str
    kind: str
    licence: str
    licence_status: str
    case_count: int | None
    pool: str
    metadata_verified: bool
    records: list[str] = field(default_factory=list)
    notes: str = ""
    demographics: dict = field(default_factory=dict)
    installed_cases: int = 0
    installed: bool = False

    @property
    def sex_split(self) -> str:
        return self.demographics.get("sex_split") or UNKNOWN

    @property
    def age_range(self) -> str:
        return self.demographics.get("age_range") or UNKNOWN

    @property
    def country_of_origin(self) -> str:
        return self.demographics.get("country_of_origin") or UNKNOWN


def load_registry(path: Path | None = None) -> dict:
    return json.loads((path or REGISTRY_PATH).read_text(encoding="utf-8"))


def licence_is_allowed(licence: str, registry: dict | None = None) -> tuple[bool, str]:
    """Whether a licence may be ingested, and why not when it may not."""
    registry = registry or load_registry()
    text = (licence or "").strip()
    if not text:
        return False, "no licence stated"
    for pattern in registry["licence_denylist_patterns"]:
        if pattern.lower() in text.lower():
            return False, f"licence {text} matches the excluded pattern {pattern}"
    if text not in registry["licence_allowlist"]:
        return False, f"licence {text} is not on the allowlist"
    return True, ""


def source_is_allowed(source: str, registry: dict | None = None) -> tuple[bool, str]:
    registry = registry or load_registry()
    for pattern in registry["source_denylist_patterns"]:
        if pattern.lower() in (source or "").lower():
            return False, f"source {source} matches the excluded pattern {pattern}"
    return True, ""


def load_cohort(
    data_root: Path | None = None, registry: dict | None = None
) -> list[DatasetRecord]:
    """The registry, annotated with what is actually installed on this machine."""
    registry = registry or load_registry()
    data_root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT

    cohort = []
    for entry in registry["datasets"]:
        record = DatasetRecord(
            name=entry["name"],
            title=entry["title"],
            kind=entry["kind"],
            licence=entry["licence"],
            licence_status=entry.get("licence_status", ""),
            case_count=entry.get("case_count"),
            pool=entry.get("pool", "primary"),
            metadata_verified=bool(entry.get("metadata_verified", False)),
            records=list(entry.get("records", [])),
            notes=entry.get("notes", ""),
            demographics=dict(entry.get("demographics", {})),
        )
        manifest = data_root / record.name / "manifest.json"
        if manifest.exists():
            installed = json.loads(manifest.read_text(encoding="utf-8"))
            record.installed = True
            record.installed_cases = int(installed.get("case_count", 0))
        cohort.append(record)
    return cohort


def cohort_lines(cohort: list[DatasetRecord]) -> list[str]:
    """One line per dataset for the info panel, plus what is missing overall."""
    lines = []
    for record in cohort:
        state = (
            f"{record.installed_cases} case(s) installed"
            if record.installed
            else "not installed"
        )
        expected = "" if record.case_count is None else f" of {record.case_count}"
        licence = record.licence_status or record.licence
        flag = "" if record.metadata_verified else " · metadata unverified"
        lines.append(
            f"{record.title} — {state}{expected} · {licence}{flag}\n"
            f"    sex split: {record.sex_split} · age: {record.age_range} · "
            f"origin: {record.country_of_origin}"
        )
    lines.append(
        "No registered dataset is known to include "
        + ", ".join(UNDER_REPRESENTED)
        + " populations. A shape library built on these datasets carries that gap "
        "into every reference shape and every mirrored reconstruction taken from it."
    )
    return lines


def installed_case_total(cohort: list[DatasetRecord]) -> int:
    return sum(record.installed_cases for record in cohort)
