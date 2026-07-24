"""Placeholder / not-yet-ported tab content."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PlaceholderTab(QWidget):
    """Empty state for tabs not yet ported to PySide."""

    launch_legacy = Signal()

    def __init__(
        self,
        title: str,
        body: str,
        *,
        phase_note: str = "",
        show_legacy: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        h = QLabel(title)
        h.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(h)

        desc = QLabel(body)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        if phase_note:
            note = QLabel(phase_note)
            note.setWordWrap(True)
            note.setStyleSheet("color: #9ab; margin-top: 8px;")
            layout.addWidget(note)

        if show_legacy:
            btn = QPushButton("Open LabGym (legacy wx) — temporary")
            btn.setToolTip(
                "Launches the classic wxPython GUI in a separate process. "
                "Removed when this tab is fully ported."
            )
            btn.clicked.connect(self.launch_legacy.emit)
            layout.addWidget(btn)

        layout.addStretch(1)
