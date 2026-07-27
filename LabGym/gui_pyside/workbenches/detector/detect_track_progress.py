"""Pop-out progress for batch Detect + track (files + frames)."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel

from LabGym.gui_pyside.widgets.progress_dialog_base import JobProgressDialogBase


class DetectTrackProgressDialog(JobProgressDialogBase):
    """Pop-out progress for batch detect+track (files + frames in current file)."""

    def __init__(self, parent=None):
        super().__init__(
            "Detect + track progress",
            parent,
            min_width=480,
            cancel_text="Cancel queue",
            cancel_tooltip="Stop the remaining queue after the current video finishes cooperatively.",
        )

        self.lbl_overall = self.add_phase_label("Starting batch…")

        self.bar_files = self.add_progress_bar(
            format_str="Videos: %v / %m",
            tooltip="How many videos in this batch have finished.",
        )
        self.bar_files.setRange(0, 1)
        self.bar_files.setValue(0)

        self.lbl_current = QLabel("Current video: —")
        self.lbl_current.setWordWrap(True)
        self.content_layout.addWidget(self.lbl_current)

        self.bar_frames = self.add_progress_bar(
            format_str="Frames: waiting…",
            tooltip="Frames processed in the video currently being tracked.",
            determinate=False,
        )

        self.add_status_label("")
        self.content_layout.addStretch(1)
        self.finish_building_ui()

        self._n_files = 0
        self._done_files = 0

    def begin_batch(self, n_files: int) -> None:
        self.set_job_running(True)
        self._n_files = max(0, int(n_files))
        self._done_files = 0
        bar_max = max(1, self._n_files)
        self.bar_files.setRange(0, bar_max)
        self.bar_files.setValue(0)
        self.bar_files.setFormat(f"Videos: 0 / {self._n_files}")
        self.set_phase(f"Processing {self._n_files} video(s)…")
        self.bar_frames.setRange(0, 0)
        self.bar_frames.setFormat("Frames: waiting…")
        self.lbl_current.setText("Current video: —")
        self.set_status_message("")
        if self.btn_cancel is not None:
            self.btn_cancel.setEnabled(True)
        self.show_as_window()

    def set_current_video(self, label: str, index: int) -> None:
        n = max(1, self._n_files)
        self.lbl_current.setText(f"Current video ({index + 1} of {n}): {label}")
        self.bar_frames.setRange(0, 0)
        self.bar_frames.setFormat("Frames: preparing…")
        self.set_status_message("Starting…")

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

    def mark_file_finished(self) -> None:
        self._done_files = min(self._n_files, self._done_files + 1)
        self.bar_files.setValue(self._done_files)
        self.bar_files.setFormat(f"Videos: {self._done_files} / {self._n_files}")
        self.set_phase(
            f"Completed {self._done_files} of {self._n_files} video(s)."
        )

    def finish_batch(self) -> None:
        self.mark_finished(status="Batch finished.")
        if self.bar_frames.maximum() > 0:
            self.bar_frames.setValue(self.bar_frames.maximum())
