"""AUD-TEST-002: light integration / smoke tests (no GPU, no long training).

These chain multiple layers (project → tabs → jobs helpers → package IO)
without requiring real detectors or video decoding success.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication

pytestmark = pytest.mark.integration


def _process_events(app: QApplication, ms: int = 50) -> None:
    app.processEvents()
    QTimer.singleShot(ms, app.quit)
    app.exec()


@pytest.mark.gui
def test_smoke_shell_navigates_all_workbenches(qapp):
    from LabGym.gui_pyside.main_window import WorkbenchMainWindow

    w = WorkbenchMainWindow()
    assert w.wb_preprocessing is not None
    assert w.wb_detector is not None
    assert w.wb_categorizer is not None
    assert w.wb_results is not None

    assert w.wb_preprocessing.set_current_tab("preprocess")
    assert w.wb_preprocessing.set_current_tab("markers")
    assert w.wb_detector.set_current_tab("detect_track")
    assert w.wb_detector.set_current_tab("review_ids")
    assert w.wb_detector.set_current_tab("train")
    assert w.wb_detector.set_current_tab("test")
    assert w.wb_categorizer.set_current_tab("process")
    assert w.wb_categorizer.set_current_tab("train")
    assert w.wb_categorizer.set_current_tab("test")
    assert w.wb_results.set_current_tab("mine")
    assert w.wb_results.set_current_tab("plot")
    assert w.wb_results.set_current_tab("distances")
    w.close()


@pytest.mark.gui
def test_smoke_project_save_reload_drives_detect_table(
    qapp, tmp_path: Path, project_controller_with_videos
):
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.project.model import Project
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab

    ctrl = project_controller_with_videos
    assert len(ctrl.project.videos) == 2

    save_path = tmp_path / "smoke.labproj.json"
    ctrl.project.file_path = str(save_path)
    out = ctrl.save(str(save_path))
    assert Path(out).is_file()

    ctrl2 = ProjectController()
    ctrl2.load_from_path(save_path)
    assert len(ctrl2.project.videos) == 2

    tab = DetectTrackTab(ctrl2)
    assert tab.table.rowCount() == 2
    # mark_dirty → refresh preserves sticky status after batch-active ends
    path0 = str(tab.table.item(0, 0).data(Qt.ItemDataRole.UserRole))
    tab._set_status(0, path0, "done", "pkg")
    tab._batch_active = True
    ctrl2.mark_dirty()
    assert tab.table.item(0, 2).text() == "done"
    tab._batch_active = False
    tab.refresh_videos()
    statuses = [
        tab.table.item(r, 2).text()
        for r in range(tab.table.rowCount())
        if str(tab.table.item(r, 0).data(Qt.ItemDataRole.UserRole)) == path0
    ]
    assert statuses == ["done"]


@pytest.mark.gui
def test_smoke_detect_and_process_queue_start_path_ids(
    qapp, tmp_path: Path, project_controller_with_videos, monkeypatch
):
    from LabGym.gui_pyside.jobs.sequential_queue import SequentialJobQueue
    from LabGym.gui_pyside.workbenches.categorizer.process_videos_tab import (
        ProcessVideosTab,
    )
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab

    ctrl = project_controller_with_videos
    det = tmp_path / "det"
    det.mkdir()
    (det / "model_parameters.txt").write_text(
        '{"animal_names": ["mouse"]}', encoding="utf-8"
    )
    cat = tmp_path / "cat"
    cat.mkdir()
    (cat / "model_parameters.txt").write_text(
        "classnames,network,time_step\napproach,2,15\n", encoding="utf-8"
    )

    captured = {}

    def fake_start(self, items, runner):
        captured.setdefault("batches", []).append(list(items))

    monkeypatch.setattr(SequentialJobQueue, "start", fake_start)

    det_tab = DetectTrackTab(ctrl)
    det_tab.ed_detector.setEditText(str(det))
    det_tab._start_batch()
    assert det_tab._batch_active is True
    assert len(captured["batches"][0]) == 2
    assert all(it.job_id == it.payload for it in captured["batches"][0])
    det_tab._batch_active = False

    proc = ProcessVideosTab(ctrl)
    proc.ed_detector.setText(str(det))
    proc.ed_categorizer.setText(str(cat))
    proc.ed_out.setText(str(tmp_path / "analysis"))
    proc._start()
    assert proc._batch_active is True
    assert len(captured["batches"][1]) == 2
    assert all(it.job_id == it.payload for it in captured["batches"][1])
    proc._batch_active = False


@pytest.mark.gui
def test_smoke_review_ids_loads_package(
    qapp, project_controller_with_videos, minimal_id_review_package: Path
):
    from LabGym.gui_pyside.workbenches.detector.review_ids_tab import ReviewIdsTab

    tab = ReviewIdsTab(project_controller_with_videos)
    ok = tab.load_package(str(minimal_id_review_package))
    assert ok is True
    assert tab.review_dir
    assert tab.n_frames >= 1
    assert "mouse" in tab._stores
    assert tab.markers_panel.table.rowCount() == 0
    tab.close()


@pytest.mark.gui
def test_smoke_sequential_queue_threaded_soft_fail(qapp):
    """Run SequentialJobQueue on a real QThread with soft-fail results."""
    from dataclasses import dataclass

    from LabGym.gui_pyside.jobs.sequential_queue import (
        JobItem,
        JobProgress,
        SequentialJobQueue,
        summarize_job_statuses,
    )

    @dataclass
    class R:
        ok: bool = True
        error: str = ""

    items = [
        JobItem(job_id="ok", label="ok", payload="ok"),
        JobItem(job_id="bad", label="bad", payload="bad"),
    ]
    done = []
    failed = []
    finished_queue = []

    def runner(job: JobItem, prog: JobProgress):
        prog(f"run {job.job_id}")
        if job.job_id == "bad":
            return R(ok=False, error="nope")
        return R(ok=True)

    queue = SequentialJobQueue()
    queue.job_finished.connect(lambda jid, res: done.append(jid))
    queue.job_failed.connect(lambda jid, err: failed.append((jid, err)))
    queue.queue_finished.connect(lambda: finished_queue.append(True))
    queue.start(items, runner)

    # Wait for worker thread (bounded)
    for _ in range(200):
        qapp.processEvents()
        if finished_queue:
            break
        qapp.thread().msleep(10)
    assert finished_queue, "queue did not finish in time"
    assert done == ["ok"]
    assert failed == [("bad", "nope")]
    assert summarize_job_statuses(queue.items) == (1, 1, 0)


@pytest.mark.gui
def test_smoke_progress_dialogs_and_train_dialog_construct(qapp):
    from LabGym.gui_pyside.workbenches.categorizer.train_progress_dialog import (
        TrainProgressDialog,
    )
    from LabGym.gui_pyside.workbenches.detector.detect_track_progress import (
        DetectTrackProgressDialog,
    )
    from LabGym.gui_pyside.widgets.progress_dialog_base import JobProgressDialogBase

    base = JobProgressDialogBase("Smoke")
    base.set_job_running(True)
    base.show_as_window()
    assert base.isVisible()
    base.close()

    dt = DetectTrackProgressDialog()
    dt.begin_batch(2)
    dt.set_frame_progress(1, 5)
    dt.mark_file_finished()
    dt.finish_batch()
    dt.close()

    tr = TrainProgressDialog()
    tr.begin_job()
    tr.on_aug_progress(1, 2, "aug")
    tr.on_train_progress(1, {"loss": 0.5, "val_loss": 0.6})
    tr.mark_finished()
    tr.close()
