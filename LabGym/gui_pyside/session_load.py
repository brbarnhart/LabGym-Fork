"""Shared helpers for cancellable Review IDs / Annotate **session load**.

Session load is not an analysis job-queue item. Callers run I/O off the UI
thread, skip callbacks after cancel, and join parallel kind/frame workers.
"""

from __future__ import annotations

import threading
import zipfile
from typing import Any, Callable, Optional, Sequence, Tuple

# User-visible npz / path / OpenCV failures. Do not swallow programming errors.
SESSION_IO_ERRORS: Tuple[type, ...] = (
    OSError,
    ValueError,
    FileNotFoundError,
    KeyError,
    TypeError,
    RuntimeError,
    zipfile.BadZipFile,
)
try:
    import cv2

    SESSION_IO_ERRORS = SESSION_IO_ERRORS + (cv2.error,)
except ImportError:
    pass


def load_cancelled(cancel_event: Optional[threading.Event]) -> bool:
    """True when a session-load cancel event has been set."""
    return cancel_event is not None and cancel_event.is_set()


def run_session_io(
    cancel_event: Optional[threading.Event],
    work: Callable[[], Any],
    *,
    on_ok: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
) -> None:
    """Run one I/O call unless cancelled; report I/O errors without raising.

    Args:
        cancel_event: Set to skip work and callbacks.
        work: Blocking load (npz, video, …).
        on_ok: Called with *work*'s return value after a successful load.
        on_error: Called with ``str(exc)`` for :data:`SESSION_IO_ERRORS`.
    """
    if load_cancelled(cancel_event):
        return
    try:
        result = work()
    except SESSION_IO_ERRORS as exc:
        if not load_cancelled(cancel_event) and on_error is not None:
            on_error(str(exc))
        return
    if load_cancelled(cancel_event):
        return
    if on_ok is not None:
        on_ok(result)


def load_session_frame0(
    *,
    skip_video: bool,
    cancel_event: Optional[threading.Event],
    read_frame_fn: Callable[..., tuple],
    video_path: Optional[str],
    store_meta: Optional[dict],
    fps: float,
    on_frame: Optional[Callable[[Any], None]] = None,
    on_video_error: Optional[Callable[[str], None]] = None,
) -> None:
    """Read analysis frame 0 unless skipped or cancelled.

    Args:
        skip_video: When True, do not open the video (kind-only load).
        cancel_event: Set to abort remaining callbacks.
        read_frame_fn: Returns ``(frame, error)``; frame is None on failure.
        video_path: Preview video path.
        store_meta: Store metadata used to map analysis frame 0 to a video index.
        fps: Analysis FPS for that mapping.
        on_frame: Called with the BGR (or converted) frame array.
        on_video_error: Called when open/read fails or *read_frame_fn* errors.
    """
    if skip_video or load_cancelled(cancel_event):
        return

    def _read() -> tuple:
        return read_frame_fn(
            video_path=video_path,
            store_meta=store_meta,
            fps=fps,
        )

    def _got(outcome: tuple) -> None:
        frame, err = outcome
        if err:
            if on_video_error is not None:
                on_video_error(str(err))
            return
        if on_frame is not None:
            on_frame(frame)

    run_session_io(
        cancel_event,
        _read,
        on_ok=_got,
        on_error=on_video_error,
    )


def join_session_workers(
    tasks: Sequence[Tuple[str, Callable[[], None]]],
    cancel_event: Optional[threading.Event],
) -> str:
    """Start named threads, join them, and report cancel vs completion.

    Args:
        tasks: ``(thread_name, target)`` pairs.
        cancel_event: Set while workers run to return ``"cancelled"``.

    Returns:
        ``"cancelled"`` if the event is set after join, else ``"ok"``.
    """
    workers = [
        threading.Thread(target=target, name=name) for name, target in tasks
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    if load_cancelled(cancel_event):
        return "cancelled"
    return "ok"
