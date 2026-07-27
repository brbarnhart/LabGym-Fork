"""Unit tests for SequentialJobQueue helpers and JobProgress (no GPU)."""

from __future__ import annotations

from dataclasses import dataclass

from LabGym.gui_pyside.jobs.sequential_queue import (
    JobItem,
    JobProgress,
    as_frame_callback,
    soft_error_from_result,
    summarize_job_statuses,
)


@dataclass
class _OkResult:
    ok: bool = True
    error: str = ""


@dataclass
class _FailResult:
    ok: bool = False
    error: str = "boom"


def test_soft_error_from_result_none_and_ok():
    assert soft_error_from_result(None) is None
    assert soft_error_from_result(_OkResult()) is None
    assert soft_error_from_result("plain") is None
    assert soft_error_from_result({"ok": True}) is None


def test_soft_error_from_result_dataclass_and_dict():
    assert soft_error_from_result(_FailResult()) == "boom"
    assert soft_error_from_result(_FailResult(error="")) == "failed"
    assert soft_error_from_result({"ok": False, "error": "nope"}) == "nope"
    assert soft_error_from_result({"ok": False}) == "failed"


def test_summarize_job_statuses():
    items = [
        JobItem(job_id="a", label="a", status="done"),
        JobItem(job_id="b", label="b", status="error"),
        JobItem(job_id="c", label="c", status="error"),
        JobItem(job_id="d", label="d", status="cancelled"),
        JobItem(job_id="e", label="e", status="pending"),
    ]
    assert summarize_job_statuses(items) == (1, 2, 1)


def test_job_progress_message_and_frame():
    messages = []
    frames = []
    prog = JobProgress("vid", messages.append, lambda c, t: frames.append((c, t)))
    prog("hello")
    prog.message("world")
    prog.frame(3, 10)
    assert messages == ["hello", "world"]
    assert frames == [(3, 10)]
    assert as_frame_callback(prog) is not None
    as_frame_callback(prog)(1, 2)
    assert frames[-1] == (1, 2)
    assert as_frame_callback(None) is None
    assert as_frame_callback(lambda m: None) is None


def test_worker_soft_fail_status(qapp=None):
    """Run worker logic synchronously by invoking _Worker.run on this thread."""
    from LabGym.gui_pyside.jobs.sequential_queue import _Worker

    items = [
        JobItem(job_id="/a.mp4", label="a", payload="/a.mp4"),
        JobItem(job_id="/b.mp4", label="b", payload="/b.mp4"),
    ]

    def runner(job: JobItem, prog: JobProgress):
        if job.job_id.endswith("a.mp4"):
            return _OkResult()
        return _FailResult(error="track failed")

    worker = _Worker(items, runner)
    finished = []
    failed = []
    worker.finished_one.connect(lambda jid, res: finished.append(jid))
    worker.failed_one.connect(lambda jid, err: failed.append((jid, err)))
    worker.run()

    assert items[0].status == "done"
    assert items[1].status == "error"
    assert items[1].error == "track failed"
    assert finished == ["/a.mp4"]
    assert failed == [("/b.mp4", "track failed")]
    assert summarize_job_statuses(items) == (1, 1, 0)


def test_worker_cancel_skips_remaining():
    from LabGym.gui_pyside.jobs.sequential_queue import _Worker

    items = [
        JobItem(job_id="1", label="1"),
        JobItem(job_id="2", label="2"),
        JobItem(job_id="3", label="3"),
    ]
    worker = _Worker(items, lambda job, prog: _OkResult())

    def runner(job: JobItem, prog: JobProgress):
        if job.job_id == "1":
            worker.cancel()
            return _OkResult()
        return _OkResult()

    worker.runner = runner
    worker.run()
    assert items[0].status == "done"
    assert items[1].status == "cancelled"
    assert items[2].status == "cancelled"
    assert summarize_job_statuses(items) == (1, 0, 2)


def test_worker_exception_continues_queue():
    from LabGym.gui_pyside.jobs.sequential_queue import _Worker

    items = [
        JobItem(job_id="x", label="x"),
        JobItem(job_id="y", label="y"),
    ]

    def runner_ex(job: JobItem, prog: JobProgress):
        if job.job_id == "x":
            raise RuntimeError("hard fail")
        return _OkResult()

    w3 = _Worker(items, runner_ex)
    failed = []
    w3.failed_one.connect(lambda jid, err: failed.append((jid, err)))
    w3.run()
    assert items[0].status == "error"
    assert "hard fail" in items[0].error
    assert items[1].status == "done"
    assert failed[0][0] == "x"


def test_worker_emits_frame_and_message_progress():
    from LabGym.gui_pyside.jobs.sequential_queue import _Worker

    items = [JobItem(job_id="/v.mp4", label="v")]

    def runner(job: JobItem, prog: JobProgress):
        prog("starting")
        prog.frame(2, 10)
        prog.frame(10, 10)
        return _OkResult()

    worker = _Worker(items, runner)
    messages = []
    frames = []
    worker.progress.connect(lambda jid, msg: messages.append((jid, msg)))
    worker.frame_progress.connect(
        lambda jid, c, t: frames.append((jid, c, t))
    )
    started = []
    worker.started_one.connect(started.append)
    worker.run()
    assert started == ["/v.mp4"]
    assert messages == [("/v.mp4", "starting")]
    assert frames == [("/v.mp4", 2, 10), ("/v.mp4", 10, 10)]
