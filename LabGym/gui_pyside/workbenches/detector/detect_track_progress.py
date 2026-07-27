"""Pop-out progress for batch Detect + track (files + frames)."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)


class DetectTrackProgressDialog(QDialog):
    """Pop-out progress for batch detect+track (files + frames in current file)."""

    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detect + track progress")
        self.setMinimumWidth(480)
        self.setWindowFlag(Qt.WindowType.Window, True)

        layout = QVBoxLayout(self)
        self.lbl_overall = QLabel("Starting batch…")
        self.lbl_overall.setWordWrap(True)
        layout.addWidget(self.lbl_overall)

        self.bar_files = QProgressBar()
        self.bar_files.setRange(0, 1)
        self.bar_files.setValue(0)
        self.bar_files.setFormat("Videos: %v / %m")
        self.bar_files.setToolTip("How many videos in this batch have finished.")
        layout.addWidget(self.bar_files)

        self.lbl_current = QLabel("Current video: —")
        self.lbl_current.setWordWrap(True)
        layout.addWidget(self.lbl_current)

        self.bar_frames = QProgressBar()
        self.bar_frames.setRange(0, 0)  # indeterminate until first frame report
        self.bar_frames.setFormat("Frames: waiting…")
        self.bar_frames.setToolTip(
            "Frames processed in the video currently being tracked."
        )
        layout.addWidget(self.bar_frames)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #9ab;")
        layout.addWidget(self.lbl_status)

        self.btn_cancel = QPushButton("Cancel queue")
        self.btn_cancel.clicked.connect(self.cancel_requested.emit)
        layout.addWidget(self.btn_cancel)
        layout.addStretch(1)

        self._n_files = 0
        self._done_files = 0

    def begin_batch(self, n_files: int) -> None:
        self._n_files = max(0, int(n_files))
        self._done_files = 0
        bar_max = max(1, self._n_files)
        self.bar_files.setRange(0, bar_max)
        self.bar_files.setValue(0)
        self.bar_files.setFormat(f"Videos: 0 / {self._n_files}")
        self.lbl_overall.setText(f"Processing {self._n_files} video(s)…")
        self.bar_frames.setRange(0, 0)
        self.bar_frames.setFormat("Frames: waiting…")
        self.lbl_current.setText("Current video: —")
        self.lbl_status.setText("")
        self.btn_cancel.setEnabled(True)
        self.show()
        self.raise_()
        self.activateWindow()

    def set_current_video(self, label: str, index: int) -> None:
        # index is 0-based among jobs; show 1-based position
        n = max(1, self._n_files)
        self.lbl_current.setText(f"Current video ({index + 1} of {n}): {label}")
        self.bar_frames.setRange(0, 0)
        self.bar_frames.setFormat("Frames: preparing…")
        self.lbl_status.setText("Starting…")

    def set_frame_progress(self, current: int, total: int) -> None:
        current = max(0, int(current))
        total = max(0, int(total))
        if total <= 0:
            self.bar_frames.setRange(0, 0)
            self.bar_frames.setFormat(f"Frames: {current}…")
        else:
            self.bar_frames.setRange(0, total)
            self.bar_frames.setValue(min(current, total))
            self.bar_frames.setFormat(f"Frames: {current} / {total} (%p%)")

    def set_status_message(self, msg: str) -> None:
        self.lbl_status.setText(msg)

    def mark_file_finished(self) -> None:
        self._done_files = min(self._n_files, self._done_files + 1)
        self.bar_files.setValue(self._done_files)
        self.bar_files.setFormat(f"Videos: {self._done_files} / {self._n_files}")
        self.lbl_overall.setText(
            f"Completed {self._done_files} of {self._n_files} video(s)."
        )

    def finish_batch(self) -> None:
        self.btn_cancel.setEnabled(False)
        self.lbl_status.setText("Batch finished.")
        if self.bar_frames.maximum() > 0:
            self.bar_frames.setValue(self.bar_frames.maximum())
