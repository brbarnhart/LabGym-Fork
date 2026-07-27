"""Directory browse helpers shared by workbench forms."""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)


def browse_existing_directory(
    parent: Optional[QWidget] = None,
    start: str = "",
    caption: str = "Select folder",
) -> Optional[str]:
    """Open a folder picker; return path or None if cancelled."""
    d = QFileDialog.getExistingDirectory(parent, caption, start or "")
    return d or None


def set_line_edit_directory(
    parent: Optional[QWidget],
    edit: QLineEdit,
    *,
    caption: str = "Select folder",
) -> bool:
    """Browse and write into *edit* if the user accepts. Returns True if set."""
    d = browse_existing_directory(parent, edit.text(), caption)
    if d:
        edit.setText(d)
        return True
    return False


def path_edit_row(edit: QLineEdit, button: QPushButton) -> QWidget:
    """Horizontal row: stretch path field + button (common form pattern)."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(edit, 1)
    h.addWidget(button)
    return w
