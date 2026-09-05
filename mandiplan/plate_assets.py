"""The plate asset catalogue: real meshes, their holes, and their provenance.

Two kinds of asset live here and the difference is never hidden:

**Generic** — the parametric reconstruction plates generated in this
repository by ``tools/make_plate_assets.py``. Real solids with real
through-holes and real thickness, sized from published dimensional classes,
but not any manufacturer's implant. They are labelled "Generic parametric
approximation" everywhere they appear.

**Exact** — a licensed manufacturer asset the operator has placed in
``mandiplan/data/plates/`` with metadata of their own. An asset is only
described as exact when its own metadata says so *and* carries provenance
saying where it came from; :func:`_validate` refuses the claim otherwise, so
a copied-and-edited entry cannot quietly promote itself.

Nothing here downloads anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .geometry.mesh_io import Mesh, check_scale, load_mesh, scale_for_units

DATA_DIR = Path(__file__).resolve().parent / "data" / "plates"

GENERIC_LABEL = "Generic parametric approximation"
EXACT_LABEL = "Exact licensed asset"
GENERIC_NOTICE = (
    "Generic parametric approximation - not manufacturer-specific and not for "
    "clinical device selection."
)


class PlateAssetError(ValueError):
    """A catalogue entry is malformed, or claims more than it can support."""


@dataclass(frozen=True)
class DeformationLimits:
    """What this plate design may have done to it."""

    min_bend_radius_mm: float = 15.0
    max_bend_deg_per_node: float = 15.0
    #: Around each hole, the radius held rigid so the hole cannot deform.
    protected_radius_mm: float = 3.5
    bendable: str = "inter-hole bridges only"


@dataclass(frozen=True)
class PlateAsset:
    """One plate model: its mesh, its holes, and where it came from."""

    id: str
    name: str
    family: str
    category: str
    exact: bool
    mesh_name: str
    mesh_format: str
    units: str
    unit_scale: float
    coordinate_convention: str
    thickness_mm: float
    width_mm: float
    length_mm: float
    hole_count: int
    hole_pitch_mm: float
    hole_diameter_mm: float
    hole_centres_mm: np.ndarray
    hole_axes: np.ndarray
    centreline_mm: np.ndarray
    side: str
    mirrorable: bool
    provenance: str
    licence: str
    deformation: DeformationLimits
    notes: str = ""
    system_id: str | None = None
    preform_radius_mm: float | None = None
    preform_angle_deg: float | None = None
    #: Alloy id from mandiplan.materials.
    material_id: str = "cp-ti-grade-4"
    #: False for adaptation and trauma profiles that cannot bridge a defect.
    load_bearing: bool = True
    dimensional_class: str = ""
    audit: dict = field(default_factory=dict)
    root: Path = DATA_DIR

    @property
    def status_label(self) -> str:
        return EXACT_LABEL if self.exact else GENERIC_LABEL

    @property
    def is_generic(self) -> bool:
        return not self.exact

    @property
    def mesh_path(self) -> Path:
        return self.root / self.mesh_name

    def __str__(self) -> str:
        return f"{self.name} ({self.status_label})"

    @property
    def alloy(self):
        from .materials import alloy_by_id

        return alloy_by_id(self.material_id)

    def summary_lines(self) -> list[str]:
        """What the inspector shows under Plate Properties."""
        alloy = self.alloy
        lines = [
            f"{self.name}",
            f"{self.status_label}",
            f"Thickness {self.thickness_mm:.1f} mm, width {self.width_mm:.1f} mm",
            f"{self.hole_count} holes at {self.hole_pitch_mm:.1f} mm pitch, "
            f"{self.hole_diameter_mm:.1f} mm diameter",
            f"Nominal length {self.length_mm:.1f} mm",
            "",
            *alloy.summary_lines(),
            f"Springback: overbend {alloy.overbend_deg(10.0, 30.0, self.thickness_mm):.1f}° "
            f"to hold 10° at a 30 mm radius",
        ]
        if not self.load_bearing:
            lines.append(
                "NOT load-bearing — trauma and adaptation only; this profile "
                "will not carry a mandible across a continuity defect."
            )
        if self.dimensional_class:
            lines.append(f"Class: {self.dimensional_class}")
        if self.preform_angle_deg:
            lines.append(f"Preformed angle {self.preform_angle_deg:.0f}°")
        elif self.preform_radius_mm:
            lines.append(f"Preformed radius {self.preform_radius_mm:.0f} mm")
        if self.is_generic:
            lines.append(GENERIC_NOTICE)
        lines.append(f"Source: {self.provenance}")
        lines.append(f"Licence: {self.licence}")
        return lines

    def export_metadata(self) -> dict:
        """Provenance fields written into every export that uses this plate."""
        return {
            "plate_asset_id": self.id,
            "plate_asset_name": self.name,
            "plate_asset_status": self.status_label,
            "plate_asset_exact": str(self.exact).lower(),
            "plate_asset_source": self.provenance,
            "plate_asset_licence": self.licence,
            "plate_asset_units": "mm",
            "plate_hole_count": str(self.hole_count),
            "plate_thickness_mm": f"{self.thickness_mm:.3f}",
            "plate_material": self.alloy.name,
            "plate_material_standard": self.alloy.standard,
            "plate_material_lattice": self.alloy.lattice,
            "plate_load_bearing": str(self.load_bearing).lower(),
        }


def _validate(entry: dict) -> None:
    """Refuse a claim the entry's own metadata does not support."""
    required = ("id", "name", "mesh", "thickness_mm", "hole_centres_mm")
    missing = [key for key in required if key not in entry]
    if missing:
        raise PlateAssetError(
            f"plate entry {entry.get('id', '<no id>')!r} is missing {missing}"
        )
    if entry.get("exact"):
        provenance = str(entry.get("provenance", "")).strip()
        licence = str(entry.get("licence", "")).strip()
        if not provenance or not licence:
            raise PlateAssetError(
                f"plate {entry['id']!r} claims to be an exact licensed asset but "
                "carries no provenance and licence. An exact claim needs both; "
                "set exact to false or supply them."
            )
    holes = entry.get("hole_count")
    centres = entry.get("hole_centres_mm") or []
    if holes is not None and len(centres) != int(holes):
        raise PlateAssetError(
            f"plate {entry['id']!r} declares {holes} holes but lists "
            f"{len(centres)} hole centres"
        )


def _asset_from_entry(entry: dict, root: Path) -> PlateAsset:
    _validate(entry)
    units = entry.get("units", "mm")
    scale = entry.get("unit_scale")
    scale = scale_for_units(units) if scale is None else float(scale)
    limits = entry.get("deformation", {})
    centres = np.asarray(entry["hole_centres_mm"], dtype=float).reshape(-1, 3)
    axes = np.asarray(
        entry.get("hole_axes") or [[0.0, 0.0, 1.0]] * len(centres), dtype=float
    ).reshape(-1, 3)
    centreline = np.asarray(
        entry.get("centreline_mm") or entry["hole_centres_mm"], dtype=float
    ).reshape(-1, 3)
    return PlateAsset(
        id=entry["id"],
        name=entry["name"],
        family=entry.get("family", entry["id"]),
        category=entry.get("category", ""),
        exact=bool(entry.get("exact", False)),
        mesh_name=entry["mesh"],
        mesh_format=entry.get("mesh_format", Path(entry["mesh"]).suffix.lstrip(".")),
        units=units,
        unit_scale=scale,
        coordinate_convention=entry.get("coordinate_convention", ""),
        thickness_mm=float(entry["thickness_mm"]),
        width_mm=float(entry.get("width_mm", 0.0)),
        length_mm=float(entry.get("length_mm", 0.0)),
        hole_count=int(entry.get("hole_count", len(centres))),
        hole_pitch_mm=float(entry.get("hole_pitch_mm", 0.0)),
        hole_diameter_mm=float(entry.get("hole_diameter_mm", 0.0)),
        hole_centres_mm=centres * scale,
        hole_axes=axes / np.linalg.norm(axes, axis=1, keepdims=True),
        centreline_mm=centreline * scale,
        side=entry.get("side", "universal"),
        mirrorable=bool(entry.get("mirrorable", True)),
        provenance=entry.get("provenance", ""),
        licence=entry.get("licence", ""),
        deformation=DeformationLimits(
            min_bend_radius_mm=float(limits.get("min_bend_radius_mm", 15.0)),
            max_bend_deg_per_node=float(limits.get("max_bend_deg_per_node", 15.0)),
            protected_radius_mm=float(limits.get("protected_radius_mm", 3.5)),
            bendable=limits.get("bendable", "inter-hole bridges only"),
        ),
        notes=entry.get("notes", ""),
        system_id=entry.get("system_id"),
        preform_radius_mm=entry.get("preform_radius_mm"),
        preform_angle_deg=entry.get("preform_angle_deg"),
        material_id=entry.get("material_id", "cp-ti-grade-4"),
        load_bearing=bool(entry.get("load_bearing", True)),
        dimensional_class=entry.get("dimensional_class", ""),
        audit=entry.get("audit", {}),
        root=root,
    )


def load_catalogue(root: Path | None = None) -> tuple[PlateAsset, ...]:
    """Every plate asset installed, generic and exact alike."""
    root = Path(root) if root is not None else DATA_DIR
    index = root / "plates.json"
    if not index.exists():
        return ()
    raw = json.loads(index.read_text(encoding="utf-8"))
    return tuple(_asset_from_entry(entry, root) for entry in raw.get("plates", []))


@lru_cache(maxsize=1)
def _default_catalogue() -> tuple[PlateAsset, ...]:
    return load_catalogue()


def load_assets(root: Path | None = None) -> tuple[PlateAsset, ...]:
    return _default_catalogue() if root is None else load_catalogue(root)


def asset_by_id(asset_id: str, root: Path | None = None) -> PlateAsset:
    for asset in load_assets(root):
        if asset.id == asset_id:
            return asset
    raise KeyError(f"no plate asset with id {asset_id!r}")


def families(root: Path | None = None) -> list[tuple[str, str]]:
    """``(family id, human label)`` for each installed family, in order."""
    seen: dict[str, str] = {}
    for asset in load_assets(root):
        seen.setdefault(asset.family, asset.name.split(",")[0])
    return list(seen.items())


def assets_in_family(family: str, root: Path | None = None) -> list[PlateAsset]:
    return [a for a in load_assets(root) if a.family == family]


def best_asset_for(
    required_mm: float, family: str | None = None, root: Path | None = None
) -> PlateAsset | None:
    """The shortest installed asset that still covers a path of this length."""
    candidates = [
        asset
        for asset in load_assets(root)
        if family is None or asset.family == family
    ]
    covering = sorted(
        (a for a in candidates if a.length_mm >= required_mm),
        key=lambda a: a.length_mm,
    )
    return covering[0] if covering else None


@lru_cache(maxsize=32)
def _cached_mesh(path: str, unit_scale: float, expected_mm: float) -> Mesh:
    mesh = load_mesh(path, unit_scale)
    if expected_mm > 0:
        check_scale(mesh, expected_mm)
    return mesh


def load_asset_mesh(asset: PlateAsset) -> Mesh:
    """The asset's mesh in millimetres, size-checked against its metadata."""
    return _cached_mesh(str(asset.mesh_path), asset.unit_scale, asset.length_mm)
