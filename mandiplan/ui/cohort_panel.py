"""A small dismissible panel naming the datasets behind the shape library."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .theme import set_role

from ..cohort import cohort_lines, installed_case_total, load_cohort


class CohortPanel(QWidget):
    """Which data backs the shape library, how much of it, and who is missing.

    Kept in front of the user rather than in a document: a reference library
    drawn entirely from populations that do not include the patient in front of
    you is a limitation of every measurement taken from it.
    """

    dismissed = pyqtSignal()

    def __init__(self, data_root=None, parent=None):
        super().__init__(parent)
        self._data_root = data_root
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.heading = QLabel("Cohort")
        set_role(self.heading, "section")
        # Collapsed by default. The provenance matters, but it is reference
        # material: it must not push the planning controls off the panel.
        self.disclosure = QPushButton("Show")
        self.disclosure.setCheckable(True)
        self.disclosure.setFlat(True)
        self.disclosure.setToolTip("Show the datasets behind the shape library")
        self.close_button = QPushButton("\u2715")
        self.close_button.setFixedSize(22, 22)
        self.close_button.setToolTip("Dismiss")
        self.body = QLabel("")
        self.body.setWordWrap(True)
        self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.body.setVisible(False)
        set_role(self.body, "hint")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        row = QHBoxLayout()
        row.addWidget(self.heading, 1)
        row.addWidget(self.disclosure)
        row.addWidget(self.close_button)
        layout.addLayout(row)
        layout.addWidget(self.body)

        self.disclosure.toggled.connect(self._on_disclosure)
        self.close_button.clicked.connect(self._dismiss)
        self.refresh()

    def _on_disclosure(self, shown: bool) -> None:
        self.body.setVisible(shown)
        self.disclosure.setText("Hide" if shown else "Show")

    def expand(self) -> None:
        """Open the panel, e.g. when reached from the Help menu."""
        self.show()
        self.disclosure.setChecked(True)

    def _dismiss(self) -> None:
        self.hide()
        self.dismissed.emit()

    def refresh(self) -> None:
        cohort = load_cohort(self._data_root)
        total = installed_case_total(cohort)
        installed = sum(1 for record in cohort if record.installed)
        self.heading.setText(
            f"Cohort — {installed} of {len(cohort)} datasets installed, "
            f"{total} case(s)"
            if installed
            else "Cohort — no dataset installed yet"
        )
        self.body.setText("\n".join(cohort_lines(cohort)))
