"""Top exclusive workbench switcher (icon/text buttons)."""

from __future__ import annotations

from typing import List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QHBoxLayout, QPushButton, QSizePolicy, QWidget

# (id, label, tooltip)
WORKBENCHES: List[Tuple[str, str, str]] = [
    ("preprocessing", "Preprocess", "Prepare videos and draw markers"),
    ("detector", "Detector", "Train/test detector, detect+track, review IDs"),
    ("categorizer", "Categorizer", "Ethograms, training pairs, train, process"),
    ("results", "Results", "Export and figures (coming soon)"),
]


class WorkbenchBar(QWidget):
    """Horizontal exclusive button group for workbench selection."""

    workbench_changed = Signal(str)  # workbench_id

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

        for wid, label, tip in WORKBENCHES:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(tip)
            btn.setMinimumHeight(32)
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            btn.setStyleSheet(
                "QPushButton { padding: 6px 14px; font-weight: 600; }"
                "QPushButton:checked { background: #3a6ea5; color: white; }"
            )
            self._group.addButton(btn)
            self._buttons[wid] = btn
            layout.addWidget(btn)
            btn.clicked.connect(lambda checked=False, i=wid: self._on_click(i))

        layout.addStretch(1)
        # Default
        if "categorizer" in self._buttons:
            self._buttons["categorizer"].setChecked(True)
        elif self._buttons:
            next(iter(self._buttons.values())).setChecked(True)

    def _on_click(self, workbench_id: str) -> None:
        self.workbench_changed.emit(workbench_id)

    def current_id(self) -> str:
        for wid, btn in self._buttons.items():
            if btn.isChecked():
                return wid
        return WORKBENCHES[0][0]

    def set_current(self, workbench_id: str) -> None:
        btn = self._buttons.get(workbench_id)
        if btn is None:
            return
        btn.setChecked(True)
        self.workbench_changed.emit(workbench_id)

    def ids(self) -> List[str]:
        return [w[0] for w in WORKBENCHES]
