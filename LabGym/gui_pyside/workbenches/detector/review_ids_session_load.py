"""Cancellable Review IDs **session load** (not an analysis job-queue item).

Unlocks on package JSON, then loads the visible **animal kind** and frame 0
independently off the UI thread. Cancel or a newer load replaces in-flight work.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal

from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
    ReviewPackageMetadata,
    load_review_kind_store,
    load_review_package_metadata,
    resolve_video_path,
)
from LabGym.id_review.samples import analysis_frame_to_video_frame

LoadKindFn = Callable[[str, str], Any]
ReadFrameFn = Callable[..., tuple]
OnMetadata = Callable[[ReviewPackageMetadata], None]
OnKind = Callable[[str, Any], None]
OnFrame = Callable[[Any], None]
OnKindError = Callable[[str, str], None]
OnVideoError = Callable[[str], None]
OnError = Callable[[str], None]


def _cancelled(cancel_event: Optional[threading.Event]) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def read_session_frame0(
    video_path: Optional[str],
    store_meta: Optional[dict],
    fps: float,
) -> tuple:
    """Open the video, read analysis frame 0, and release the capture.

    Decoder strategy matches Review IDs preview (``CAP_PROP_POS_FRAMES`` +
    read). The live play capture is not used here.

    Returns:
        ``(bgr_frame, error)``. Frame is None when opening or reading fails.
    """
    import cv2

    path = (video_path or "").strip()
    if not path:
        return None, "video unavailable"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, f"video unavailable: {path}"
    try:
        v_idx = analysis_frame_to_video_frame(dict(store_meta or {}), 0, float(fps or 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(v_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None, f"Failed to read video frame {v_idx}"
        return frame, None
    finally:
        cap.release()


def run_review_session_load(
    review_dir: str,
    *,
    visible_kind: Optional[str] = None,
    video_hint: Optional[str] = None,
    skip_metadata: bool = False,
    skip_video: bool = False,
    cancel_event: Optional[threading.Event] = None,
    on_metadata: Optional[OnMetadata] = None,
    on_kind: Optional[OnKind] = None,
    on_frame: Optional[OnFrame] = None,
    on_kind_error: Optional[OnKindError] = None,
    on_video_error: Optional[OnVideoError] = None,
    on_error: Optional[OnError] = None,
    load_kind: Optional[LoadKindFn] = None,
    read_frame: Optional[ReadFrameFn] = None,
    metadata: Optional[ReviewPackageMetadata] = None,
) -> str:
    """Run one Review IDs session load to completion or cancel.

    Metadata (unless skipped) is applied before geometry or pixels. The
    visible kind's packed npz and frame 0 are loaded in parallel so either
    can arrive first. This is not an analysis job-queue item.

    Args:
        review_dir: Identity package folder.
        visible_kind: **Animal kind** to load; default is the preferred kind
            from package JSON.
        video_hint: Project video path used to resolve the preview file.
        skip_metadata: When True, do not re-read package JSON (kind change).
        skip_video: When True, do not open the video (kind-only load).
        cancel_event: Set to abort; in-flight npz/video may still finish I/O
            but callbacks after cancel are skipped.
        on_metadata: Called with JSON metadata before npz/video work.
        on_kind: Called with ``(kind, TrackletStore)`` when packed arrays are
            ready.
        on_frame: Called with the BGR frame-0 array.
        on_kind_error: Called with ``(kind, message)`` on corrupt/missing npz.
        on_video_error: Called with a message when the video cannot be opened
            or frame 0 cannot be read.
        on_error: Called when package JSON cannot be read (no metadata).
        load_kind: Injectable kind loader (tests).
        read_frame: Injectable frame-0 reader (tests).
        metadata: Preloaded metadata when *skip_metadata* is True.

    Returns:
        ``"ok"``, ``"cancelled"``, or ``"error"``.
    """
    load_kind_fn = load_kind or load_review_kind_store
    read_frame_fn = read_frame or (
        lambda **kwargs: read_session_frame0(
            kwargs.get("video_path"),
            kwargs.get("store_meta"),
            float(kwargs.get("fps") or 0),
        )
    )

    meta = metadata
    if not skip_metadata:
        try:
            meta = load_review_package_metadata(
                review_dir, video_hint=video_hint
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            if _cancelled(cancel_event):
                return "cancelled"
            if on_error is not None:
                on_error(str(exc))
            else:
                raise
            return "error"
        if _cancelled(cancel_event):
            return "cancelled"
        if on_metadata is not None:
            on_metadata(meta)
        if _cancelled(cancel_event):
            return "cancelled"
    elif meta is None:
        try:
            meta = load_review_package_metadata(
                review_dir, video_hint=video_hint
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            if _cancelled(cancel_event):
                return "cancelled"
            if on_error is not None:
                on_error(str(exc))
            elif on_kind_error is not None:
                on_kind_error(visible_kind or "", str(exc))
            else:
                raise
            return "error"

    kind = (visible_kind or (meta.animal_kind if meta else "") or "").strip()
    if not kind:
        return "error"

    info = meta.kind_info.get(kind) if meta is not None else None
    store_meta = dict(info.store_meta) if info is not None else {}
    fps = float(
        (info.fps if info is not None else 0)
        or (meta.fps if meta is not None else 10)
        or 10
    )
    video_path = None
    if meta is not None:
        video_path = resolve_video_path(
            meta.review_dir,
            store_meta,
            meta.events,
            video_hint or meta.video_path,
        )

    def _load_kind() -> None:
        if _cancelled(cancel_event):
            return
        try:
            store = load_kind_fn(review_dir, kind)
        except Exception as exc:
            if not _cancelled(cancel_event) and on_kind_error is not None:
                on_kind_error(kind, str(exc))
            return
        if _cancelled(cancel_event):
            return
        if on_kind is not None:
            on_kind(kind, store)

    def _load_frame() -> None:
        if skip_video or _cancelled(cancel_event):
            return
        try:
            frame, err = read_frame_fn(
                video_path=video_path,
                store_meta=store_meta,
                fps=fps,
            )
        except Exception as exc:
            if not _cancelled(cancel_event) and on_video_error is not None:
                on_video_error(str(exc))
            return
        if _cancelled(cancel_event):
            return
        if err:
            if on_video_error is not None:
                on_video_error(str(err))
            return
        if on_frame is not None:
            on_frame(frame)

    workers = [threading.Thread(target=_load_kind, name="review-ids-kind")]
    if not skip_video:
        workers.append(threading.Thread(target=_load_frame, name="review-ids-frame"))
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    if _cancelled(cancel_event):
        return "cancelled"
    return "ok"


class ReviewSessionLoadThread(QThread):
    """Background **session load**. Callers apply results on the UI thread."""

    metadata_ready = Signal(object)
    kind_ready = Signal(str, object)
    frame_ready = Signal(object)
    kind_error = Signal(str, str)
    video_error = Signal(str)
    load_error = Signal(str)

    def __init__(
        self,
        review_dir: str,
        *,
        visible_kind: Optional[str] = None,
        video_hint: Optional[str] = None,
        skip_metadata: bool = False,
        skip_video: bool = False,
        metadata: Optional[ReviewPackageMetadata] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._review_dir = review_dir
        self._visible_kind = visible_kind
        self._video_hint = video_hint
        self._skip_metadata = skip_metadata
        self._skip_video = skip_video
        self._metadata = metadata
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Abort remaining callbacks; I/O may still finish."""
        self._cancel.set()

    def run(self) -> None:
        run_review_session_load(
            self._review_dir,
            visible_kind=self._visible_kind,
            video_hint=self._video_hint,
            skip_metadata=self._skip_metadata,
            skip_video=self._skip_video,
            cancel_event=self._cancel,
            metadata=self._metadata,
            on_metadata=self.metadata_ready.emit,
            on_kind=self.kind_ready.emit,
            on_frame=self.frame_ready.emit,
            on_kind_error=self.kind_error.emit,
            on_video_error=self.video_error.emit,
            on_error=self.load_error.emit,
        )
