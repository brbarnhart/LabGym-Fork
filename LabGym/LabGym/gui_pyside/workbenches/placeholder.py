"""Placeholder / not-yet-ported tab content."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)


class PlaceholderTab(QWidget):
    """Empty state for tabs not yet ported to PySide."""

    def __init__(
        self,
        title: str,
        body: str,
        *,
        phase_note: str = "",
        show_legacy: bool = False,  # Phase 8: unused; legacy is Tools menu only
        parent=None,
    ):
        super().__init__(parent)
        # show_legacy kept for call-site compatibility; no longer adds a button.
        _ = show_legacy
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

        layout.addStretch(1)
