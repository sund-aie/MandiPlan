"""Where the jaws are in a CT that shows much more than the head.

A CBCT is taken of the jaws; a medical CT may run from the top of the head
to the feet. Everything MandiPlan does happens on the mandible, so for a
large CT the jaws are found first and only a box round them is kept, at the
scan's full resolution. That box is also small enough to be resampled to
cubic voxels, which is what makes the bone surface of a thick-slice CT as
smooth as a CBCT's (see ``dicom_io``).

The head is found by its skull vault: in an axial slice through the
cranium, bone encloses a large region of soft tissue (the brain). Nothing
else in the body does that for several centimetres on end — the ribs are cut
obliquely and never close a ring, the lungs inside them are air, and the
pelvis closes a ring for a few slices at most. The vault's top is close to
the vertex, and the mandible hangs a known distance below it, so the
mandible's U (``mandible.find_arch``) is searched for only in that band,
where the clavicles and the pelvis cannot be mistaken for it.

No Qt, no VTK. Coordinates are world millimetres (LPS).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .threshold import otsu_threshold
from .volume import Volume

#: The coarse grid the head is looked for on, mm.
COARSE_MM = 3.0
#: An axial slice through the brain encloses at least this much, cm².
MIN_BRAIN_AREA_CM2 = 40.0
#: ... and the vault keeps doing so for at least this far, mm.
MIN_VAULT_HEIGHT_MM = 24.0
#: The mandible is searched for this far below the top of the vault, mm.
#: The vertex to the chin is 200–250 mm in adults; the condyles are about
#: 110–130 mm below the vertex.
JAW_SEARCH_MM = (60.0, 300.0)
#: The box kept round the jaws, relative to the level the mandibular body is
#: found at and to the middle of the arch, mm: (below, above), (either side),
#: (in front, behind the front of the arch).
BOX_BELOW_MM = 45.0
BOX_ABOVE_MM = 150.0
BOX_HALF_WIDTH_MM = 85.0
BOX_FRONT_MM = 30.0
BOX_BACK_MM = 135.0
#: The mandibular body is never wider than this at the level it is found
#: at; the shoulder girdle and the ribs are.
MAX_ARCH_WIDTH_MM = 150.0


@dataclass
class JawRegion:
    """The box round the jaws, and how it was found."""

    lo: np.ndarray  # world mm, voxel-centre bounds
    hi: np.ndarray
    vault_top_mm: float  # z of the top of the skull vault; nan if not seen
    arch_level_mm: float  # z the mandibular body was found at

    @property
    def size_mm(self) -> np.ndarray:
        return self.hi - self.lo


def coarse_view(array: np.ndarray, spacing, origin, mm: float = COARSE_MM) -> Volume:
    """The scan averaged over blocks about ``mm`` across: a cheap look at it.

    Averaged, not sampled every n-th voxel: on a sharp (bone) kernel the
    voxels are noisy, and a sample every 3 mm breaks a 2 mm cortex into
    pieces, so the mandible's outline is no longer one U. One output slice
    is made at a time, so the scan is never converted to float whole.
    """
    spacing = np.asarray(spacing, dtype=float)
    fx, fy, fz = (int(v) for v in np.maximum(np.round(mm / spacing).astype(int), 1))
    nz, ny, nx = array.shape[0] // fz, array.shape[1] // fy, array.shape[2] // fx
    if min(nz, ny, nx) < 1:
        view = np.asarray(array, dtype=np.float32)
        return Volume(view, spacing, np.asarray(origin, dtype=float))
    out = np.empty((nz, ny, nx), dtype=np.float32)
    for k in range(nz):
        slab = np.asarray(array[k * fz : (k + 1) * fz, : ny * fy, : nx * fx], dtype=np.float32)
        out[k] = slab.reshape(fz, ny, fy, nx, fx).mean(axis=(0, 2, 4))
    factor = np.array([fx, fy, fz], dtype=float)
    # A block's value belongs at the centre of the voxels it averaged.
    centre = np.asarray(origin, dtype=float) + 0.5 * (factor - 1) * spacing
    return Volume(out, spacing * factor, centre)


def tissue_thresholds(volume: Volume) -> tuple[float, float]:
    """Air/tissue and tissue/bone splits of a scan's gray values.

    Two Otsu splits, the second among the tissue only: in a scan of the
    whole body the soft tissue so outnumbers the bone that one three-class
    split puts its upper threshold among the fat.
    """
    values = volume.array[::2, ::2, ::2].ravel()
    lo, hi = (float(v) for v in np.percentile(values, [0.5, 99.9]))
    counts, edges = np.histogram(np.clip(values, lo, hi), 256, (lo, hi))
    body = otsu_threshold(counts, edges)
    tissue = values[values > body]
    if tissue.size < 100:
        return body, hi
    top = float(np.percentile(tissue, 99.9))
    counts, edges = np.histogram(np.clip(tissue, body, top), 256, (body, top))
    return body, otsu_threshold(counts, edges)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(mask.astype(np.uint8))
    return sitk.GetArrayFromImage(sitk.BinaryFillhole(image)) > 0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    import SimpleITK as sitk

    labels = sitk.GetArrayFromImage(
        sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype(np.uint8)))
    )
    if labels.max() == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def vault_top(volume: Volume, body: float, bone: float) -> float | None:
    """z of the top of the skull vault, mm; None if there is no vault.

    Per axial slice, the largest region bone closes off; a brain slice is one
    where that region is large and soft tissue inside. The vault is the run
    of brain slices enclosing the most.
    """
    pixel_cm2 = float(volume.spacing[0] * volume.spacing[1]) / 100.0
    brain = np.zeros(volume.array.shape[0], dtype=float)
    for k, section in enumerate(volume.array):
        bone_mask = section >= bone
        if not bone_mask.any():
            continue
        enclosed = _largest_component(_fill_holes(bone_mask) & ~bone_mask)
        area = float(enclosed.sum()) * pixel_cm2
        if area >= MIN_BRAIN_AREA_CM2 and body < float(section[enclosed].mean()) < bone:
            brain[k] = area
    best, best_volume, k = None, 0.0, 0
    while k < len(brain):
        if brain[k] <= 0:
            k += 1
            continue
        start = k
        while k < len(brain) and brain[k] > 0:
            k += 1
        height = (k - start) * volume.spacing[2]
        enclosed = float(brain[start:k].sum())
        if height >= MIN_VAULT_HEIGHT_MM and enclosed > best_volume:
            best, best_volume = (start, k - 1), enclosed
    if best is None:
        return None
    return float(volume.origin[2] + best[1] * volume.spacing[2])


def find_jaw_region(volume: Volume) -> JawRegion | None:
    """The box round the mandible in a CT of any extent, or None.

    ``volume`` may be a coarse view of the scan (see :func:`coarse_view`);
    the box is in world millimetres, to be cut from the full scan.
    """
    from .mandible import arch_axes, find_arch

    body, bone = tissue_thresholds(volume)
    top = vault_top(volume, body, bone)
    lo_z, hi_z = volume.bounds_mm[0][2], volume.bounds_mm[1][2]
    if top is not None:
        window = (max(lo_z, top - JAW_SEARCH_MM[1]), min(hi_z, top - JAW_SEARCH_MM[0]))
    else:
        window = (lo_z, hi_z)
    k0 = int(np.floor((window[0] - volume.origin[2]) / volume.spacing[2]))
    k1 = int(np.ceil((window[1] - volume.origin[2]) / volume.spacing[2])) + 1
    k0, k1 = max(k0, 0), min(k1, volume.array.shape[0])
    if k1 - k0 < 4:
        return None
    band = Volume(
        volume.array[k0:k1], volume.spacing, volume.origin + [0.0, 0.0, k0 * volume.spacing[2]]
    )
    arch = find_arch(band, bone, max_width_mm=MAX_ARCH_WIDTH_MM, search_fraction=1.0)
    if arch is None:
        return None
    level = float(np.mean(arch[:, 2]))
    # The box is laid in the arch's own frame, then taken axis-aligned, so a
    # head lying turned still has both rami in it.
    front, back, lateral = arch_axes(arch)
    across = arch[:, :2] @ lateral
    middle = front + (0.5 * (across.min() + across.max()) - float(front @ lateral)) * lateral
    corners = np.array(
        [
            middle + side * BOX_HALF_WIDTH_MM * lateral + depth * back
            for side in (-1.0, 1.0)
            for depth in (-BOX_FRONT_MM, BOX_BACK_MM)
        ]
    )
    lo = np.array([*corners.min(axis=0), level - BOX_BELOW_MM])
    hi = np.array([*corners.max(axis=0), level + BOX_ABOVE_MM])
    scan_lo, scan_hi = volume.bounds_mm
    lo, hi = np.maximum(lo, scan_lo), np.minimum(hi, scan_hi)
    return JawRegion(lo=lo, hi=hi, vault_top_mm=np.nan if top is None else top, arch_level_mm=level)
