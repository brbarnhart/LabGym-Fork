"""Keras callbacks for live categorizer training progress and cancel (UI / logging)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union


class TrainingCancelled(Exception):
	"""Raised when the user aborts augmentation or training."""


CancelToken = Any  # threading.Event, multiprocessing.Event, or zero-arg callable -> bool


def is_cancelled(cancel_event: Optional[CancelToken]) -> bool:
	"""Return True if a cancel token is set / reports True."""
	if cancel_event is None:
		return False
	try:
		if hasattr(cancel_event, "is_set"):
			return bool(cancel_event.is_set())
		if callable(cancel_event):
			return bool(cancel_event())
	except Exception:
		return False
	return False


def raise_if_cancelled(cancel_event: Optional[CancelToken], message: str = "Cancelled by user") -> None:
	if is_cancelled(cancel_event):
		raise TrainingCancelled(message)


def _as_float(value: Any) -> Optional[float]:
	try:
		if value is None:
			return None
		return float(value)
	except (TypeError, ValueError):
		return None


def sanitize_logs(logs: Optional[dict]) -> Dict[str, float]:
	"""Return a plain dict of float metrics suitable for Qt signals / JSON."""
	out: Dict[str, float] = {}
	if not logs:
		return out
	for key, value in logs.items():
		f = _as_float(value)
		if f is not None:
			out[str(key)] = f
	return out


class EpochProgressCallback:
	"""Keras-compatible callback: ``on_progress(epoch_1based, logs_dict)`` each epoch.

	Constructed without importing keras at module import time so unit tests can
	load helpers without TensorFlow. Keras Callback is mixed in at init.
	"""

	def __init__(
		self,
		on_progress: Optional[Callable[[int, Dict[str, float]], None]] = None,
		*,
		history_sink: Optional[Dict[str, List[float]]] = None,
	):
		from keras.callbacks import Callback

		# Dynamically subclass so isinstance checks / fit() accept us.
		class _Impl(Callback):  # type: ignore[misc, valid-type]
			def __init__(self_inner):
				super().__init__()
				self_inner._on_progress = on_progress
				self_inner._history_sink = history_sink if history_sink is not None else {}
				for k in ("loss", "val_loss", "accuracy", "val_accuracy"):
					self_inner._history_sink.setdefault(k, [])

			def on_epoch_end(self_inner, epoch, logs=None):
				clean = sanitize_logs(logs)
				# Keras epoch is 0-based
				epoch_1 = int(epoch) + 1
				for k in ("loss", "val_loss", "accuracy", "val_accuracy"):
					if k in clean:
						self_inner._history_sink.setdefault(k, []).append(clean[k])
				if self_inner._on_progress is not None:
					try:
						self_inner._on_progress(epoch_1, clean)
					except Exception:
						pass

		self._cb = _Impl()

	@property
	def callback(self):
		return self._cb


def make_epoch_progress_callback(
	on_progress: Optional[Callable[[int, Dict[str, float]], None]] = None,
	*,
	history_sink: Optional[Dict[str, List[float]]] = None,
):
	"""Return a real Keras Callback instance, or None if no progress handler."""
	if on_progress is None and history_sink is None:
		return None
	return EpochProgressCallback(on_progress, history_sink=history_sink).callback


def make_cancel_callback(cancel_event: Optional[CancelToken] = None):
	"""Return a Keras Callback that sets ``model.stop_training`` when cancelled.

	Cooperative: takes effect at epoch/batch boundaries (not mid-batch hard kill).
	"""
	if cancel_event is None:
		return None
	from keras.callbacks import Callback

	class _CancelCallback(Callback):  # type: ignore[misc, valid-type]
		def __init__(self_inner):
			super().__init__()
			self_inner._token = cancel_event

		def _check(self_inner):
			if is_cancelled(self_inner._token) and self_inner.model is not None:
				self_inner.model.stop_training = True

		def on_train_begin(self_inner, logs=None):
			self_inner._check()

		def on_epoch_begin(self_inner, epoch, logs=None):
			self_inner._check()

		def on_epoch_end(self_inner, epoch, logs=None):
			self_inner._check()

		def on_batch_end(self_inner, batch, logs=None):
			# Faster response during long epochs
			if batch is not None and int(batch) % 5 == 0:
				self_inner._check()

	return _CancelCallback()
