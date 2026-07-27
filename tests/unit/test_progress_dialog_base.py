"""Unit tests for JobProgressDialogBase shared chrome."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication


def _app():
    return QApplication.instance() or QApplication(sys.argv)


def test_base_begin_mark_finished_and_widgets():
    _app()
    from LabGym.gui_pyside.widgets.progress_dialog_base import JobProgressDialogBase

    dlg = JobProgressDialogBase(
        "Unit job",
        cancel_text="Stop",
        show_close_button=True,
    )
    dlg.add_phase_label("Ready")
    dlg.add_status_label("idle")
    dlg.add_log()
    bar = dlg.add_progress_bar(format_str="Work: %p%")
    dlg.finish_building_ui()

    assert dlg.btn_cancel is not None
    assert dlg.btn_close is not None
    assert dlg.btn_cancel.text() == "Stop"

    dlg.begin_job()
    assert dlg.is_job_running
    assert dlg.btn_cancel.isEnabled()
    assert not dlg.btn_close.isEnabled()
    assert dlg.lbl_phase.text() == "Starting…"

    dlg.append_log("hello")
    dlg.set_phase("Working")
    dlg.set_status_message("50%")
    bar.setValue(50)

    dlg.mark_finished()
    assert not dlg.is_job_running
    assert not dlg.btn_cancel.isEnabled()
    assert dlg.btn_close.isEnabled()
    assert "Finished" in dlg.lbl_phase.text()
    dlg.close()


def test_detect_and_train_dialogs_use_shared_chrome():
    _app()
    from LabGym.gui_pyside.workbenches.categorizer.train_progress_dialog import (
        TrainProgressDialog,
    )
    from LabGym.gui_pyside.workbenches.detector.detect_track_progress import (
        DetectTrackProgressDialog,
    )

    dt = DetectTrackProgressDialog()
    assert dt.btn_cancel is not None
    assert dt.lbl_phase is not None
    assert dt.lbl_status is not None
    dt.begin_batch(2)
    assert dt.is_job_running
    dt.set_frame_progress(3, 9)
    dt.finish_batch()
    assert not dt.is_job_running
    dt.close()

    tr = TrainProgressDialog()
    assert tr.btn_close is not None
    assert tr.log is not None
    tr.begin_job()
    tr.on_status("exporting")
    tr.mark_finished(failed=True)
    assert not tr.is_job_running
    assert tr.btn_close.isEnabled()
    tr.close()
