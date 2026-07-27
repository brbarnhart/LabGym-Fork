"""Shared base for long-running job pop-out progress dialogs."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QDialog, QWidget


class JobProgressDialogBase(QDialog):
    """Independent window-style dialog with cancel signal and show helpers.

    Subclasses own their bars/labels; this base only standardizes:
    - true top-level window flag
    - cancel_requested signal
    - show_as_window / running state helpers
    """

    cancel_requested = Signal()

    def __init__(
        self,
        title: str,
        parent: Optional[QWidget] = None,
        *,
        min_width: int = 480,
        min_height: Optional[int] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(min_width)
        if min_height is not None:
            self.setMinimumHeight(min_height)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self._running = False

    @property
    def is_job_running(self) -> bool:
        return self._running

    def set_job_running(self, running: bool) -> None:
        self._running = bool(running)

    def show_as_window(self) -> None:
        """Show, raise, and activate (use after filling initial progress state)."""
        self.show()
        self.raise_()
        self.activateWindow()
