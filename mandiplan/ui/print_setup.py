"""Printer and protected-side selection for the mandible print kit."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)


def protection_points(session):
    """World points bounding the *actual* resection, including oblique cuts."""
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from ..render.surface import clip_closed

    points = [np.asarray(point) for _, point in session.landmarks]
    if session.planes:
        portion = session.fragment_surface if session.cut_applied else None
        if portion is None:
            portion = clip_closed(session.surface, session.planes, keep_resected=True)
        if portion is None or not portion.GetNumberOfPoints():
            raise ValueError("The planned cut does not intersect the mandible. Adjust the cut before printing.")
        verts = vtk_to_numpy(portion.GetPoints().GetData())
        # Retain the extent on every axis for protection and orientation.
        points.extend(verts[np.r_[verts.argmin(axis=0), verts.argmax(axis=0)]])
        if session.frames is not None:
            points.extend(session.frames.point_at(session.plane_arc_position(i))
                          for i in range(len(session.planes)))
    return np.asarray(points, dtype=float).reshape(-1, 3)


def _mm(lo, hi, value, decimals=2):
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(lo, hi)
    box.setValue(value)
    box.setSuffix(" mm")
    return box


def working_boundary(session, surface, side, cut_points):
    """Protect the whole anatomical hemisphere even when its plane is tilted."""
    from vtkmodules.util.numpy_support import vtk_to_numpy

    plane = session.symmetry_plane
    if plane is None:
        return side, None
    normal = np.asarray(plane.normal) * (1 if plane.normal[0] >= 0 else -1)
    if side == "auto":
        if not len(cut_points):
            return side, float(plane.point[0])
        offset = float(np.mean((cut_points - plane.point) @ normal))
        if abs(offset) < 1.0:
            raise ValueError("The resection is centred on the midline. Choose the working side explicitly.")
        side = "left" if offset > 0 else "right"
    vertices = vtk_to_numpy(surface.GetPoints().GetData())
    sign = 1 if side == "left" else -1
    half = vertices[((vertices - plane.point) @ normal) * sign >= 0]
    if not len(half):
        raise ValueError("The selected working side is outside the model. Check the symmetry plane.")
    return side, float(half[:, 0].min() if sign == 1 else half[:, 0].max())


class PrintSetupDialog(QDialog):
    def __init__(self, session, parent=None):
        from ..printing import PRINTERS
        super().__init__(parent)
        self.setWindowTitle("Prepare mandible for printing")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.model = QComboBox()
        if session.reconstructed_surface is not None:
            self.model.addItem("Reconstructed mandible", "reconstructed")
        self.model.addItem("Separated mandible", "original")
        form.addRow("Model", self.model)
        self.printer = QComboBox()
        for name, size in PRINTERS.items():
            self.printer.addItem(name, tuple(size))
        self.printer.addItem("Custom build volume", None)
        form.addRow("Printer", self.printer)
        dims = QWidget()
        row = QHBoxLayout(dims)
        row.setContentsMargins(0, 0, 0, 0)
        self.dims = [_mm(10.01, 2000, 180) for _ in range(3)]
        for axis, spin in zip("XYZ", self.dims):
            spin.setAccessibleName(f"Build volume {axis}")
            row.addWidget(QLabel(axis))
            row.addWidget(spin)
        form.addRow("Build volume", dims)
        def select_printer():
            size = self.printer.currentData()
            for i, spin in enumerate(self.dims):
                if size:
                    spin.setValue(size[i])
                spin.setEnabled(size is None)
        self.printer.currentIndexChanged.connect(select_printer)
        select_printer()
        self.side = QComboBox()
        for label, key in [("From planned resection", "auto"),
                           ("Patient left", "left"), ("Patient right", "right")]:
            self.side.addItem(label, key)
        form.addRow("Protect working side", self.side)
        self.clearance = _mm(0.05, 0.5, 0.2)
        form.addRow("Peg clearance", self.clearance)
        self.supports = QCheckBox("Add removable support stems")
        self.supports.setChecked(True)
        form.addRow("", self.supports)
        layout.addLayout(form)
        note = QLabel("Parts stay at actual size and connect with pegs on flat faces. The resected section "
                      "stays whole in one part, laid with its outer face up and no support on that face. "
                      "If no cut is planned, choose a side before a split is needed.")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Choose output folder…")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
