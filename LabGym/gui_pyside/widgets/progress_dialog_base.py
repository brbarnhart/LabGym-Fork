"""Shared base for long-running job pop-out progress dialogs."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class JobProgressDialogBase(QDialog):
    """Independent window-style progress dialog with shared chrome.

    Standardizes:
    - top-level window flag + show/raise/activate
    - running-state flag
    - cancel_requested signal
    - optional Cancel / Close action bar
    - optional phase label, muted status label, log pane
    - optional "confirm close while running" behavior

    Subclasses add domain-specific bars/metrics into :attr:`content_layout`.
    """

    cancel_requested = Signal()

    def __init__(
        self,
        title: str,
        parent: Optional[QWidget] = None,
        *,
        min_width: int = 480,
        min_height: Optional[int] = None,
        cancel_text: str = "Cancel",
        cancel_tooltip: str = "",
        show_close_button: bool = False,
        confirm_close_while_running: bool = False,
        close_while_running_title: str = "Job in progress",
        close_while_running_message: str = (
            "A job is still running.\n\n"
            "Cancel the job and close?\n"
            "Choose No to keep this window open."
        ),
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(min_width)
        if min_height is not None:
            self.setMinimumHeight(min_height)
        self.setWindowFlag(Qt.WindowType.Window, True)

        self._running = False
        self._confirm_close_while_running = bool(confirm_close_while_running)
        self._close_while_running_title = close_while_running_title
        self._close_while_running_message = close_while_running_message

        self._root = QVBoxLayout(self)
        self.content_layout = QVBoxLayout()
        self._root.addLayout(self.content_layout, 1)

        self.lbl_phase: Optional[QLabel] = None
        self.lbl_status: Optional[QLabel] = None
        self.log: Optional[QTextEdit] = None
        self.btn_cancel: Optional[QPushButton] = None
        self.btn_close: Optional[QPushButton] = None

        self._cancel_text = cancel_text
        self._cancel_tooltip = cancel_tooltip
        self._show_close_button = show_close_button
        self._actions_built = False

    # --- lifecycle ---

    @property
    def is_job_running(self) -> bool:
        return self._running

    def set_job_running(self, running: bool) -> None:
        self._running = bool(running)
        if self.btn_cancel is not None:
            self.btn_cancel.setEnabled(self._running)
        if self.btn_close is not None:
            self.btn_close.setEnabled(not self._running)

    def show_as_window(self) -> None:
        """Show, raise, and activate (use after filling initial progress state)."""
        self.show()
        self.raise_()
        self.activateWindow()

    def begin_job(self) -> None:
        """Mark running, enable cancel, clear phase/log, show window."""
        self._ensure_action_bar()
        self.set_job_running(True)
        if self.lbl_phase is not None:
            self.lbl_phase.setText("Starting…")
        if self.lbl_status is not None:
            self.lbl_status.setText("")
        if self.log is not None:
            self.log.clear()
        self.show_as_window()

    def mark_finished(
        self,
        *,
        cancelled: bool = False,
        failed: bool = False,
        status: str = "",
    ) -> None:
        """Mark job idle; enable Close when present."""
        self.set_job_running(False)
        if status:
            self.set_phase(status)
            self.set_status_message(status)
        elif failed:
            self.set_phase("Failed")
        elif cancelled:
            self.set_phase("Cancelled")
        else:
            self.set_phase("Finished")

    # --- shared widgets ---

    def add_phase_label(self, text: str = "Starting…") -> QLabel:
        """Primary status line at the top of the content area."""
        self.lbl_phase = QLabel(text)
        self.lbl_phase.setWordWrap(True)
        self.content_layout.addWidget(self.lbl_phase)
        return self.lbl_phase

    def add_status_label(self, text: str = "") -> QLabel:
        """Secondary muted status line."""
        self.lbl_status = QLabel(text)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #9ab;")
        self.content_layout.addWidget(self.lbl_status)
        return self.lbl_status

    def add_log(self, stretch: bool = True) -> QTextEdit:
        """Read-only log pane."""
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.content_layout.addWidget(self.log, 1 if stretch else 0)
        return self.log

    def add_progress_bar(
        self,
        *,
        format_str: str = "%p%",
        tooltip: str = "",
        determinate: bool = True,
    ) -> QProgressBar:
        bar = QProgressBar()
        if determinate:
            bar.setRange(0, 100)
            bar.setValue(0)
        else:
            bar.setRange(0, 0)
        bar.setFormat(format_str)
        if tooltip:
            bar.setToolTip(tooltip)
        self.content_layout.addWidget(bar)
        return bar

    def set_phase(self, msg: str) -> None:
        if self.lbl_phase is not None:
            self.lbl_phase.setText(msg)

    def set_status_message(self, msg: str) -> None:
        if self.lbl_status is not None:
            self.lbl_status.setText(msg)

    def append_log(self, msg: str) -> None:
        if self.log is not None:
            self.log.append(msg)

    def _ensure_action_bar(self) -> None:
        if self._actions_built:
            return
        self._actions_built = True
        row = QHBoxLayout()
        self.btn_cancel = QPushButton(self._cancel_text)
        if self._cancel_tooltip:
            self.btn_cancel.setToolTip(self._cancel_tooltip)
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        row.addWidget(self.btn_cancel)
        row.addStretch(1)
        if self._show_close_button:
            self.btn_close = QPushButton("Close")
            self.btn_close.setEnabled(False)
            self.btn_close.setToolTip("Close this window (enabled when the job finishes).")
            self.btn_close.clicked.connect(self.accept)
            row.addWidget(self.btn_close)
        self._root.addLayout(row)

    def finish_building_ui(self) -> None:
        """Call after subclass has filled ``content_layout`` to attach the action bar."""
        self._ensure_action_bar()

    # --- cancel / close ---

    def _on_cancel_clicked(self) -> None:
        """Disable cancel and emit; subclasses may override for extra messaging."""
        if self.btn_cancel is not None:
            self.btn_cancel.setEnabled(False)
        self.cancel_requested.emit()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.is_job_running and self._confirm_close_while_running:
            r = QMessageBox.question(
                self,
                self._close_while_running_title,
                self._close_while_running_message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if r == QMessageBox.StandardButton.Yes:
                self._on_cancel_clicked()
            # Keep open until the job ends or user declines cancel.
            event.ignore()
            return
        # Without confirm: allow hide while job continues (cancel via button).
        event.accept()
