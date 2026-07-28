"""Unit tests for detector training progress helpers (no full Detectron2 train)."""

from __future__ import annotations

from LabGym.detection.train_progress import (
    ProgressCallbackHook,
    collect_training_metrics,
    primary_total_loss,
    progress_report_period,
)
from LabGym.detectron2.utils.events import EventStorage


def test_progress_report_period_short_runs_every_step():
    assert progress_report_period(10) == 1
    assert progress_report_period(20) == 1
    assert progress_report_period(21) == 20
    assert progress_report_period(500) == 20


def test_primary_total_loss_prefers_total():
    assert primary_total_loss({"total_loss": 1.5, "loss_cls": 0.2}) == 1.5
    assert primary_total_loss({"loss_cls": 0.2, "loss_box": 0.3}) == 0.5
    assert primary_total_loss({"lr": 0.001}) is None


def test_collect_training_metrics_from_event_storage():
    with EventStorage(0) as storage:
        for i in range(5):
            storage.iter = i
            storage.put_scalars(total_loss=2.0 - 0.1 * i, loss_cls=0.5, lr=0.001)
        metrics = collect_training_metrics(storage, window_size=5)
    assert "total_loss" in metrics
    assert "loss_cls" in metrics
    assert "lr" in metrics
    assert metrics["total_loss"] > 0


def test_progress_callback_hook_period_and_final():
    seen = []

    def cb(it, mx, metrics):
        seen.append((it, mx, dict(metrics)))

    hook = ProgressCallbackHook(cb, period=3, window_size=3)

    class _T:
        def __init__(self):
            self.iter = 0
            self.max_iter = 7
            self.storage = None

    trainer = _T()
    hook.trainer = trainer

    with EventStorage(0) as storage:
        trainer.storage = storage
        for i in range(7):
            trainer.iter = i
            storage.iter = i
            storage.put_scalars(total_loss=1.0 / (i + 1), lr=1e-3)
            hook.after_step()

    # period 3 → report at 3, 6; also final max_iter 7
    iters = [s[0] for s in seen]
    assert 3 in iters
    assert 6 in iters
    assert 7 in iters
    assert all(s[1] == 7 for s in seen)
    assert all("total_loss" in s[2] for s in seen)


def test_train_detector_progress_dialog_metrics_update():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from LabGym.gui_pyside.workbenches.detector.train_progress_dialog import (
        TrainDetectorProgressDialog,
    )

    dlg = TrainDetectorProgressDialog()
    dlg.begin_job(max_iter=100)
    assert dlg.progress.maximum() == 100
    dlg.on_train_progress(20, 100, {"total_loss": 1.25, "lr": 0.001, "loss_cls": 0.4})
    assert dlg.lbl_iter.text().startswith("20")
    assert "1.2500" in dlg.lbl_total_loss.text() or dlg.lbl_total_loss.text().startswith("1.25")
    assert len(dlg._loss) == 1
    dlg.on_train_progress(40, 100, {"total_loss": 0.9, "lr": 0.001})
    assert len(dlg._loss) == 2
    dlg.mark_finished()
    dlg.close()
