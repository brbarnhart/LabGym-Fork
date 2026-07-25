"""One-at-a-time background job queue for batch detect / process."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from PySide6.QtCore import QObject, QThread, Signal


@dataclass
class JobItem:
    """One unit of work in the queue."""

    job_id: str
    label: str
    payload: Any = None
    status: str = "pending"  # pending | running | done | error | cancelled
    error: str = ""
    result: Any = None


class _Worker(QObject):
    finished_one = Signal(str, object)  # job_id, result
    failed_one = Signal(str, str)  # job_id, error
    progress = Signal(str, str)  # job_id, message
    queue_finished = Signal()

    def __init__(
        self,
        items: List[JobItem],
        runner: Callable[[JobItem, Callable[[str], None]], Any],
    ):
        super().__init__()
        self.items = items
        self.runner = runner
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        for item in self.items:
            if self._cancel:
                item.status = "cancelled"
                continue
            item.status = "running"

            def _prog(msg: str, jid=item.job_id) -> None:
                self.progress.emit(jid, msg)

            try:
                result = self.runner(item, _prog)
                item.result = result
                item.status = "done"
                self.finished_one.emit(item.job_id, result)
            except Exception as exc:
                item.status = "error"
                item.error = str(exc)
                self.failed_one.emit(item.job_id, str(exc))
        self.queue_finished.emit()


class SequentialJobQueue(QObject):
    """Run jobs sequentially on a worker thread."""

    job_progress = Signal(str, str)
    job_finished = Signal(str, object)
    job_failed = Signal(str, str)
    queue_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None
        self.items: List[JobItem] = []

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def start(
        self,
        items: List[JobItem],
        runner: Callable[[JobItem, Callable[[str], None]], Any],
    ) -> None:
        if self.is_running:
            raise RuntimeError("Queue already running")
        self.items = list(items)
        self._thread = QThread(self)
        self._worker = _Worker(self.items, runner)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.job_progress.emit)
        self._worker.finished_one.connect(self.job_finished.emit)
        self._worker.failed_one.connect(self.job_failed.emit)
        self._worker.queue_finished.connect(self._on_queue_finished)
        self._worker.queue_finished.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()

    def cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _on_queue_finished(self) -> None:
        self.queue_finished.emit()

    def _cleanup(self) -> None:
        self._worker = None
        self._thread = None
