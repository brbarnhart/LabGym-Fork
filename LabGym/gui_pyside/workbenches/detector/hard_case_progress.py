"""Pop-out progress for Review IDs hard-case training image export."""

from __future__ import annotations

from LabGym.gui_pyside.widgets.progress_dialog_base import JobProgressDialogBase


class HardCaseProgressDialog(JobProgressDialogBase):
    """Pop-out progress for exporting detector training frames from selected ranges."""

    def __init__(self, parent=None):
        super().__init__(
            "Hard-case training images",
            parent,
            min_width=480,
            cancel_text="Cancel",
            cancel_tooltip=(
                "Stop after the current frame is written. Already-written images are kept."
            ),
            show_close_button=True,
            confirm_close_while_running=True,
            close_while_running_title="Export in progress",
            close_while_running_message=(
                "Frame export is still running.\n\n"
                "Cancel the export and close?\n"
                "Choose No to keep this window open."
            ),
        )

        self.add_phase_label("Starting…")

        self.bar_frames = self.add_progress_bar(
            format_str="Frames: waiting…",
            tooltip="How many of the planned training frames have been written.",
            determinate=False,
        )

        self.add_status_label("")
        self.add_log(stretch=True)
        self.finish_building_ui()

        self._total = 0

    def begin_export(self, n_frames: int, *, out_path: str = "", video_label: str = "") -> None:
        self.set_job_running(True)
        self._total = max(0, int(n_frames))
        total = max(1, self._total)
        self.bar_frames.setRange(0, total)
        self.bar_frames.setValue(0)
        if self._total > 0:
            self.bar_frames.setFormat(f"Frames: 0 / {self._total} (%p%)")
        else:
            self.bar_frames.setRange(0, 0)
            self.bar_frames.setFormat("Frames: preparing…")
        phase = f"Exporting {self._total} frame(s)…"
        if video_label:
            phase = f"Exporting {self._total} frame(s) from {video_label}…"
        self.set_phase(phase)
        self.set_status_message(out_path or "")
        if self.log is not None:
            self.log.clear()
            if out_path:
                self.append_log(f"Output: {out_path}")
            if video_label:
                self.append_log(f"Video: {video_label}")
            self.append_log(f"Planned frames: {self._total}")
        if self.btn_cancel is not None:
            self.btn_cancel.setEnabled(True)
        if self.btn_close is not None:
            self.btn_close.setEnabled(False)
        self.show_as_window()

    def set_frame_progress(self, current: int, total: int, msg: str = "") -> None:
        current = max(0, int(current))
        total = max(0, int(total))
        if total <= 0:
            self.bar_frames.setRange(0, 0)
            self.bar_frames.setFormat(f"Frames: {current}…")
        else:
            self._total = total
            self.bar_frames.setRange(0, total)
            self.bar_frames.setValue(min(current, total))
            self.bar_frames.setFormat(f"Frames: {current} / {total} (%p%)")
        if msg:
            self.set_status_message(msg)
            # Avoid flooding the log: append every frame for small jobs, sample for large.
            if total <= 40 or current == 1 or current == total or current % max(1, total // 20) == 0:
                self.append_log(msg)

    def finish_export(
        self,
        *,
        cancelled: bool = False,
        failed: bool = False,
        status: str = "",
    ) -> None:
        if not failed and not cancelled and self.bar_frames.maximum() > 0:
            self.bar_frames.setValue(self.bar_frames.maximum())
        self.mark_finished(cancelled=cancelled, failed=failed, status=status)
        if status:
            self.append_log(status)

    def _on_cancel_clicked(self) -> None:
        if self.btn_cancel is not None:
            self.btn_cancel.setEnabled(False)
        self.set_phase("Cancel requested… finishing current frame")
        self.append_log("Cancel requested — will stop after the current frame.")
        self.cancel_requested.emit()
