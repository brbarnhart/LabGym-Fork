"""Detectron2 training progress hooks for live loss reporting in the UI."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from LabGym.detectron2.engine.train_loop import HookBase
from LabGym.detectron2.utils.events import get_event_storage

# Callback: (iteration_1based_or_next, max_iter, metrics_dict) -> None
TrainProgressCallback = Callable[[int, int, Dict[str, float]], None]

# Match Detectron2 PeriodicWriter default period.
DEFAULT_PROGRESS_PERIOD = 20
DEFAULT_SMOOTH_WINDOW = 20


def progress_report_period(max_iter: int, preferred: int = DEFAULT_PROGRESS_PERIOD) -> int:
    """How often (in iterations) to emit progress for *max_iter* total steps."""
    mi = max(1, int(max_iter))
    pref = max(1, int(preferred))
    # For very short runs, report more often so the curve is not empty.
    if mi <= pref:
        return 1
    return pref


def collect_training_metrics(
    storage: Any,
    *,
    window_size: int = DEFAULT_SMOOTH_WINDOW,
) -> Dict[str, float]:
    """Read smoothed loss scalars (and lr when available) from EventStorage."""
    metrics: Dict[str, float] = {}
    try:
        histories = storage.histories()
    except Exception:
        return metrics

    for key, buf in histories.items():
        if "loss" not in key:
            continue
        value = _smoothed_value(storage, key, buf, window_size)
        if value is not None:
            metrics[key] = value

    try:
        metrics["lr"] = float(storage.history("lr").latest())
    except Exception:
        pass
    return metrics


def _smoothed_value(storage: Any, key: str, buf: Any, window_size: int) -> Optional[float]:
    try:
        n = storage.count_samples(key, window_size)
        return float(buf.median(n))
    except Exception:
        pass
    try:
        return float(buf.latest())
    except Exception:
        return None


def primary_total_loss(metrics: Mapping[str, float]) -> Optional[float]:
    """Prefer ``total_loss``; otherwise sum of keys containing ``loss``."""
    if "total_loss" in metrics:
        try:
            return float(metrics["total_loss"])
        except (TypeError, ValueError):
            return None
    parts = []
    for k, v in metrics.items():
        if k == "lr" or "loss" not in k:
            continue
        try:
            parts.append(float(v))
        except (TypeError, ValueError):
            continue
    if not parts:
        return None
    return float(sum(parts))


class ProgressCallbackHook(HookBase):
    """Call *callback* periodically with iteration index and loss metrics.

    Detectron2 stores scalars after each step; this hook reads them on a
    cadence similar to :class:`~LabGym.detectron2.engine.hooks.PeriodicWriter`
    so the GUI can update a progress bar and loss curve without flooding Qt.
    """

    def __init__(
        self,
        callback: TrainProgressCallback,
        *,
        period: int = DEFAULT_PROGRESS_PERIOD,
        window_size: int = DEFAULT_SMOOTH_WINDOW,
    ):
        if callback is None:
            raise ValueError("callback is required")
        self._callback = callback
        self._period = max(1, int(period))
        self._window_size = max(1, int(window_size))

    def after_step(self) -> None:
        # After step N, trainer.iter is still N; report as completed iter N+1
        # to match CommonMetricPrinter / user-facing 1-based progress.
        next_iter = int(self.trainer.iter) + 1
        max_iter = int(self.trainer.max_iter)
        if next_iter % self._period != 0 and next_iter != max_iter:
            return
        try:
            storage = get_event_storage()
        except Exception:
            storage = getattr(self.trainer, "storage", None)
        if storage is None:
            return
        metrics = collect_training_metrics(storage, window_size=self._window_size)
        try:
            self._callback(next_iter, max_iter, metrics)
        except Exception:
            # Never let UI callback failures abort Detectron2 training.
            pass
