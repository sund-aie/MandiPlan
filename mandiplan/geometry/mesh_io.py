"""Reading triangle meshes, in millimetres.

STL, PLY and OBJ, with numpy only — no VTK here, so the geometry package
stays unit-testable on its own.

Units are the whole point of this module. A mesh authored in metres or inches
looks perfectly valid and is silently a thousand times too big or twenty-five
times too small; a plate that has quietly become the wrong size is the kind of
error nobody catches by eye. So every loader takes an explicit ``unit_scale``,
and :func:`check_scale` refuses a mesh whose size is nowhere near what its
catalogue entry claims.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Multipliers onto millimetres for the unit names a sidecar may carry.
UNIT_SCALES = {
    "mm": 1.0,
    "millimetre": 1.0,
    "millimeter": 1.0,
    "cm": 10.0,
    "centimetre": 10.0,
    "centimeter": 10.0,
    "m": 1000.0,
    "metre": 1000.0,
    "meter": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "um": 0.001,
    "micron": 0.001,
}


class MeshLoadError(ValueError):
    """A mesh file could not be read, or is not in the units it claims."""


@dataclass
class Mesh:
    """A triangle mesh in millimetres."""

    points: np.ndarray  # (N, 3) float, mm
    triangles: np.ndarray  # (M, 3) int

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float).reshape(-1, 3)
        self.triangles = np.asarray(self.triangles, dtype=np.int64).reshape(-1, 3)

    @property
    def bounds_mm(self) -> tuple[np.ndarray, np.ndarray]:
        return self.points.min(axis=0), self.points.max(axis=0)

    @property
    def extent_mm(self) -> np.ndarray:
        lo, hi = self.bounds_mm
        return hi - lo

    def transformed(self, rotation: np.ndarray, translation: np.ndarray) -> "Mesh":
        """A copy under a rigid transform: ``x -> R x + t``."""
        rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        translation = np.asarray(translation, dtype=float).reshape(3)
        return Mesh(self.points @ rotation.T + translation, self.triangles.copy())

    def with_points(self, points) -> "Mesh":
        return Mesh(points, self.triangles.copy())

    def edge_count(self) -> int:
        edges = np.vstack(
            [
                self.triangles[:, [0, 1]],
                self.triangles[:, [1, 2]],
                self.triangles[:, [2, 0]],
            ]
        )
        edges = np.sort(edges, axis=1)
        return len(np.unique(edges, axis=0))

    def euler_characteristic(self) -> int:
        return len(self.points) - self.edge_count() + len(self.triangles)

    def genus(self) -> int:
        """Handles through the solid — one per genuine through-hole."""
        return (2 - self.euler_characteristic()) // 2

    def is_watertight(self) -> bool:
        """Every edge shared by exactly two triangles."""
        edges = np.vstack(
            [
                self.triangles[:, [0, 1]],
                self.triangles[:, [1, 2]],
                self.triangles[:, [2, 0]],
            ]
        )
        edges = np.sort(edges, axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return bool(np.all(counts == 2))

    def volume_mm3(self) -> float:
        a = self.points[self.triangles[:, 0]]
        b = self.points[self.triangles[:, 1]]
        c = self.points[self.triangles[:, 2]]
        return abs(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0))


def _merge_vertices(points: np.ndarray, tolerance: float = 1e-6):
    """Weld coincident vertices, as an STL's per-facet vertices always are."""
    keys = np.round(points / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return points[first], inverse.reshape(-1)


def load_stl(path, unit_scale: float = 1.0, merge: bool = True) -> Mesh:
    """Read a binary or ASCII STL."""
    data = Path(path).read_bytes()
    if _looks_ascii(data):
        points, triangles = _parse_ascii_stl(data.decode("utf-8", "replace"))
    else:
        points, triangles = _parse_binary_stl(data)
    if merge and len(points):
        merged, inverse = _merge_vertices(points)
        points, triangles = merged, inverse[triangles]
    return Mesh(points * float(unit_scale), triangles)


def _looks_ascii(data: bytes) -> bool:
    if not data[:5].lower().startswith(b"solid"):
        return False
    # A binary STL may still begin with "solid" in its 80-byte header, so the
    # declared triangle count is checked against the actual file length.
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + count * 50:
            return False
    return b"facet" in data[:1024].lower()


def _parse_binary_stl(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(data) < 84:
        raise MeshLoadError("binary STL is too short to hold a header")
    count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + count * 50
    if len(data) < expected:
        raise MeshLoadError(
            f"binary STL declares {count} triangles but is {len(data)} bytes, "
            f"short of the {expected} needed"
        )
    record = np.dtype(
        [("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    facets = np.frombuffer(data, dtype=record, count=count, offset=84)
    points = facets["vertices"].reshape(-1, 3).astype(float)
    triangles = np.arange(count * 3, dtype=np.int64).reshape(count, 3)
    return points, triangles


_ASCII_VERTEX = re.compile(
    r"vertex\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)\s+(-?[\d.eE+-]+)"
)


def _parse_ascii_stl(text: str) -> tuple[np.ndarray, np.ndarray]:
    values = _ASCII_VERTEX.findall(text)
    if not values or len(values) % 3:
        raise MeshLoadError("ASCII STL has no complete triangles")
    points = np.array(values, dtype=float)
    triangles = np.arange(len(points), dtype=np.int64).reshape(-1, 3)
    return points, triangles


def load_obj(path, unit_scale: float = 1.0) -> Mesh:
    """Read a Wavefront OBJ. Faces are fanned, so quads and n-gons are fine."""
    points: list[list[float]] = []
    triangles: list[tuple[int, int, int]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            points.append([float(v) for v in parts[1:4]])
        elif parts[0] == "f":
            corners = []
            for token in parts[1:]:
                index = int(token.split("/")[0])
                # OBJ indices are 1-based, and negative means relative to the
                # end of the vertex list so far.
                corners.append(index - 1 if index > 0 else len(points) + index)
            for i in range(1, len(corners) - 1):
                triangles.append((corners[0], corners[i], corners[i + 1]))
    if not triangles:
        raise MeshLoadError("OBJ file contains no faces")
    return Mesh(np.array(points, dtype=float) * float(unit_scale), triangles)


def load_ply(path, unit_scale: float = 1.0) -> Mesh:
    """Read an ASCII or little-endian binary PLY."""
    raw = Path(path).read_bytes()
    marker = b"end_header"
    end = raw.find(marker)
    if not raw[:3].lower().startswith(b"ply") or end < 0:
        raise MeshLoadError("not a PLY file")
    header = raw[:end].decode("ascii", "replace").splitlines()
    body = raw[raw.index(b"\n", end) + 1 :]

    fmt = ""
    counts: dict[str, int] = {}
    properties: dict[str, list[tuple[str, str]]] = {}
    element = ""
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            element = parts[1]
            counts[element] = int(parts[2])
            properties[element] = []
        elif parts[0] == "property" and element:
            properties[element].append((parts[1], parts[-1]))

    if fmt == "ascii":
        return _parse_ascii_ply(body, counts, properties, unit_scale)
    if fmt == "binary_little_endian":
        return _parse_binary_ply(body, counts, properties, unit_scale)
    raise MeshLoadError(f"unsupported PLY format {fmt!r}")


_PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "i2", "ushort": "u2", "int16": "i2", "uint16": "u2",
    "int": "i4", "uint": "u4", "int32": "i4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def _parse_ascii_ply(body, counts, properties, unit_scale) -> Mesh:
    lines = body.decode("ascii", "replace").split("\n")
    names = [name for _, name in properties.get("vertex", [])]
    try:
        cols = [names.index(axis) for axis in ("x", "y", "z")]
    except ValueError as error:
        raise MeshLoadError("PLY vertices have no x/y/z properties") from error

    n = counts.get("vertex", 0)
    points = np.array(
        [[float(lines[i].split()[c]) for c in cols] for i in range(n)], dtype=float
    )
    triangles = []
    for i in range(n, n + counts.get("face", 0)):
        parts = [int(v) for v in lines[i].split()]
        corners = parts[1 : 1 + parts[0]]
        for k in range(1, len(corners) - 1):
            triangles.append((corners[0], corners[k], corners[k + 1]))
    if not triangles:
        raise MeshLoadError("PLY file contains no faces")
    return Mesh(points * float(unit_scale), triangles)


def _parse_binary_ply(body, counts, properties, unit_scale) -> Mesh:
    vertex_dtype = np.dtype(
        [(name, "<" + _PLY_TYPES[kind]) for kind, name in properties["vertex"]]
    )
    n = counts.get("vertex", 0)
    vertices = np.frombuffer(body, dtype=vertex_dtype, count=n)
    points = np.column_stack(
        [vertices["x"], vertices["y"], vertices["z"]]
    ).astype(float)

    offset = vertex_dtype.itemsize * n
    count_kind, index_kind = properties["face"][0][0], properties["face"][0][1]
    # PLY lists declare "property list <count type> <index type> vertex_indices".
    count_np = np.dtype("<" + _PLY_TYPES.get(count_kind, "u1"))
    index_np = np.dtype("<" + _PLY_TYPES.get(index_kind, "i4"))
    triangles = []
    for _ in range(counts.get("face", 0)):
        size = int(np.frombuffer(body, count_np, 1, offset)[0])
        offset += count_np.itemsize
        corners = np.frombuffer(body, index_np, size, offset).astype(int)
        offset += index_np.itemsize * size
        for k in range(1, size - 1):
            triangles.append((corners[0], corners[k], corners[k + 1]))
    if not triangles:
        raise MeshLoadError("PLY file contains no faces")
    return Mesh(points * float(unit_scale), triangles)


_LOADERS = {".stl": load_stl, ".obj": load_obj, ".ply": load_ply}


def load_mesh(path, unit_scale: float = 1.0) -> Mesh:
    """Read a mesh by file extension, scaled into millimetres."""
    path = Path(path)
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise MeshLoadError(
            f"unsupported mesh format {path.suffix!r}; "
            f"expected one of {', '.join(sorted(_LOADERS))}"
        )
    if not path.exists():
        raise MeshLoadError(f"mesh file not found: {path}")
    return loader(path, unit_scale)


def scale_for_units(units: str) -> float:
    """Millimetres per unit for a unit name from a metadata sidecar."""
    key = str(units).strip().lower()
    if key not in UNIT_SCALES:
        raise MeshLoadError(
            f"unknown mesh units {units!r}; expected one of "
            f"{', '.join(sorted(set(UNIT_SCALES)))}"
        )
    return UNIT_SCALES[key]


def check_scale(mesh: Mesh, expected_mm: float, tolerance: float = 0.05) -> None:
    """Refuse a mesh whose longest dimension is not the length it claims.

    This is the guard against a metre- or inch-authored asset arriving with a
    ``unit_scale`` of 1: the file loads perfectly and is the wrong size.
    """
    longest = float(mesh.extent_mm.max())
    if expected_mm <= 0:
        raise MeshLoadError("expected length must be positive")
    error = abs(longest - expected_mm) / expected_mm
    if error > tolerance:
        raise MeshLoadError(
            f"mesh is {longest:.2f} mm along its longest axis but its metadata "
            f"claims {expected_mm:.2f} mm ({error:.0%} out). Check the unit "
            "scale in the asset's catalogue entry."
        )
