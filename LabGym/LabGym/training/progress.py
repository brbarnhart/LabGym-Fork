"""Keras callbacks for live categorizer training progress (UI / logging)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


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
