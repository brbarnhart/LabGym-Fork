"""Cancellable Annotate ethogram **session load** (not an analysis job-queue item).

Unlocks on package JSON, then loads **every animal kind** needed for overlays
and the video independently. Cancel or a newer load replaces in-flight work.
Geometry is published remapped tracklets (ADR-0006 / ADR-0009), not raw stores.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

from PySide6.QtCore import QThread, Signal

from LabGym.annotator.core.tracklets_bridge import (
    LoadedTracklets,
    discover_tracklet_kinds,
    load_tracklets_for_annotator,
)
from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
    ReviewPackageMetadata,
    load_review_package_metadata,
)
from LabGym.identity.package import SubjectRecord

LoadTrackletsFn = Callable[..., Any]
ReadFrameFn = Callable[..., tuple]
OnMetadata = Callable[["AnnotateSessionMetadata"], None]
OnTracklets = Callable[[Any], None]
OnFrame = Callable[[Any], None]
OnTrackletsError = Callable[[str], None]
OnVideoError = Callable[[str], None]
OnError = Callable[[str], None]


def _cancelled(cancel_event: Optional[threading.Event]) -> bool:
    return cancel_event is not None and cancel_event.is_set()


@dataclass
class AnnotateSessionMetadata:
    """Package JSON / project paths that unlock Annotate before pixels."""

    video_path: str
    annotations_path: str
    review_dir: str
    n_frames: int
    fps: float
    kinds: List[str]
    subjects: List[SubjectRecord] = field(default_factory=list)
    behavior_mode: int = 0
    exclusive_mode: bool = False
    accepted: bool = False
    package: Optional[ReviewPackageMetadata] = None


def load_annotate_session_metadata(
    review_dir: str,
    *,
    video_hint: Optional[str] = None,
    annotations_path: str = "",
    behavior_mode: int = 0,
    exclusive_mode: bool = False,
) -> AnnotateSessionMetadata:
    """Load package JSON (n_frames, subjects, kinds) without opening npz.

    Overlay kinds come from published remapped ``*_tracklets_meta.json`` at
    the package root. Raw snapshot JSON is used only as a fallback for
    n_frames when public meta is missing.

    Args:
        review_dir: Identity package folder.
        video_hint: Project video path for this package.
        annotations_path: Sidecar ethogram path (may not exist yet).
        behavior_mode: Detect-world behavior mode.
        exclusive_mode: Ethogram exclusive-mode flag.

    Returns:
        Metadata sufficient to unlock the annotator session.

    Raises:
        FileNotFoundError: *review_dir* is not a folder.
        ValueError: No kind meta JSON in the package.
    """
    pkg = load_review_package_metadata(review_dir, video_hint=video_hint)
    published = discover_tracklet_kinds(review_dir)
    kinds = list(published) if published else list(pkg.kinds)
    return AnnotateSessionMetadata(
        video_path=str(video_hint or pkg.video_path or ""),
        annotations_path=str(annotations_path or ""),
        review_dir=pkg.review_dir,
        n_frames=int(pkg.n_frames),
        fps=float(pkg.fps),
        kinds=kinds,
        subjects=list(pkg.subjects),
        behavior_mode=int(behavior_mode),
        exclusive_mode=bool(exclusive_mode),
        accepted=bool(pkg.accepted),
        package=pkg,
    )


def loaded_tracklets_placeholder(
    directory: str,
    subjects: Sequence[SubjectRecord],
) -> LoadedTracklets:
    """Subject list for session unlock before packed arrays are in RAM."""
    from LabGym.annotator.core.data_models import Subject

    recs = list(subjects)
    subj_models: List[Subject] = []
    mapping = {}
    for rec in recs:
        sid = int(rec.subject_id)
        kind = str(rec.animal_kind or "animal")
        tid = int(rec.track_id if rec.track_id is not None else rec.subject_id)
        subj_models.append(
            Subject(
                subject_id=sid,
                animal_kind=kind,
                display_name=str(rec.display_name or ""),
                color=str(rec.color or ""),
            )
        )
        mapping[sid] = (kind, tid)
    return LoadedTracklets(
        directory=str(directory or ""),
        stores={},
        analysis_start_frame=0,
        subjects=subj_models,
        subject_to_track=mapping,
    )


def _default_read_frame(
    *,
    video_path: Optional[str] = None,
    store_meta: Optional[dict] = None,
    fps: float = 0,
    **_kwargs,
) -> tuple:
    from LabGym.gui_pyside.workbenches.detector.review_ids_session_load import (
        read_session_frame0,
    )

    return read_session_frame0(video_path, store_meta, float(fps or 0))


def _default_load_tracklets(review_dir: str, **kwargs) -> LoadedTracklets:
    return load_tracklets_for_annotator(
        review_dir,
        video_total_frames=kwargs.get("video_total_frames"),
    )


def run_annotate_session_load(
    review_dir: str = "",
    *,
    video_path: Optional[str] = None,
    annotations_path: Optional[str] = None,
    skip_metadata: bool = False,
    skip_video: bool = False,
    cancel_event: Optional[threading.Event] = None,
    on_metadata: Optional[OnMetadata] = None,
    on_tracklets: Optional[OnTracklets] = None,
    on_frame: Optional[OnFrame] = None,
    on_tracklets_error: Optional[OnTrackletsError] = None,
    on_video_error: Optional[OnVideoError] = None,
    on_error: Optional[OnError] = None,
    load_tracklets: Optional[LoadTrackletsFn] = None,
    read_frame: Optional[ReadFrameFn] = None,
    metadata: Optional[AnnotateSessionMetadata] = None,
    behavior_mode: int = 0,
    exclusive_mode: bool = False,
) -> str:
    """Run one Annotate session load to completion or cancel.

    Metadata (unless skipped) is applied before geometry or pixels. Packed
    remapped npz for **every overlay kind** and frame 0 are loaded in
    parallel so either can arrive first. This is not an analysis job-queue
    item.

    Args:
        review_dir: Identity package folder (published remapped tracklets).
        video_path: Video to open via the annotator decoder (or preview
            reader when *read_frame* is used).
        annotations_path: Sidecar ethogram path.
        skip_metadata: When True, do not re-read package JSON.
        skip_video: When True, do not open the video (UI uses the live
            annotator decoder instead).
        cancel_event: Set to abort; in-flight npz/video may still finish I/O
            but callbacks after cancel are skipped.
        on_metadata: Called with JSON metadata before npz/video work.
        on_tracklets: Called with ``LoadedTracklets`` (all overlay kinds).
        on_frame: Called with a BGR/RGB frame-0 array when not skipped.
        on_tracklets_error: Called with a message on corrupt/missing npz.
        on_video_error: Called when the video cannot be opened.
        on_error: Called when package JSON cannot be read (no metadata).
        load_tracklets: Injectable loader (tests); default is
            ``load_tracklets_for_annotator``.
        read_frame: Injectable frame-0 reader (tests).
        metadata: Preloaded metadata when *skip_metadata* is True.
        behavior_mode: Passed through to metadata.
        exclusive_mode: Passed through to metadata.

    Returns:
        ``"ok"``, ``"cancelled"``, or ``"error"``.
    """
    load_fn = load_tracklets or _default_load_tracklets
    read_frame_fn = read_frame or _default_read_frame

    meta = metadata
    directory = str(review_dir or "").strip()
    if not skip_metadata:
        if directory:
            try:
                meta = load_annotate_session_metadata(
                    directory,
                    video_hint=video_path,
                    annotations_path=str(annotations_path or ""),
                    behavior_mode=behavior_mode,
                    exclusive_mode=exclusive_mode,
                )
            except (FileNotFoundError, ValueError, OSError) as exc:
                if _cancelled(cancel_event):
                    return "cancelled"
                if on_error is not None:
                    on_error(str(exc))
                else:
                    raise
                return "error"
        else:
            meta = AnnotateSessionMetadata(
                video_path=str(video_path or ""),
                annotations_path=str(annotations_path or ""),
                review_dir="",
                n_frames=0,
                fps=10.0,
                kinds=[],
                behavior_mode=int(behavior_mode),
                exclusive_mode=bool(exclusive_mode),
                accepted=False,
            )
        if video_path:
            meta.video_path = str(video_path)
        if annotations_path:
            meta.annotations_path = str(annotations_path)
        if _cancelled(cancel_event):
            return "cancelled"
        if on_metadata is not None:
            on_metadata(meta)
        if _cancelled(cancel_event):
            return "cancelled"
    elif meta is None and directory:
        try:
            meta = load_annotate_session_metadata(
                directory,
                video_hint=video_path,
                annotations_path=str(annotations_path or ""),
                behavior_mode=behavior_mode,
                exclusive_mode=exclusive_mode,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            if _cancelled(cancel_event):
                return "cancelled"
            if on_error is not None:
                on_error(str(exc))
            else:
                raise
            return "error"

    resolved_video = str(
        video_path or (meta.video_path if meta is not None else "") or ""
    )
    n_frames = int(meta.n_frames) if meta is not None else 0
    fps = float(meta.fps) if meta is not None else 10.0
    store_meta = {}
    if meta is not None and meta.package is not None:
        kind = meta.package.animal_kind
        info = meta.package.kind_info.get(kind)
        if info is not None:
            store_meta = dict(info.store_meta)

    def _load_tracklets() -> None:
        if not directory or _cancelled(cancel_event):
            return
        try:
            loaded = load_fn(
                directory, video_total_frames=n_frames or None
            )
        except Exception as exc:
            if not _cancelled(cancel_event) and on_tracklets_error is not None:
                on_tracklets_error(str(exc))
            return
        if _cancelled(cancel_event):
            return
        if on_tracklets is not None:
            on_tracklets(loaded)

    def _load_frame() -> None:
        if skip_video or _cancelled(cancel_event):
            return
        try:
            frame, err = read_frame_fn(
                video_path=resolved_video,
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

    workers = [threading.Thread(target=_load_tracklets, name="annotate-tracklets")]
    if not skip_video:
        workers.append(threading.Thread(target=_load_frame, name="annotate-frame"))
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    if _cancelled(cancel_event):
        return "cancelled"
    return "ok"


class AnnotateSessionLoadThread(QThread):
    """Background **session load**. Callers apply results on the UI thread."""

    metadata_ready = Signal(object)
    tracklets_ready = Signal(object)
    frame_ready = Signal(object)
    tracklets_error = Signal(str)
    video_error = Signal(str)
    load_error = Signal(str)

    def __init__(
        self,
        review_dir: str = "",
        *,
        video_path: Optional[str] = None,
        annotations_path: Optional[str] = None,
        skip_metadata: bool = False,
        skip_video: bool = True,
        metadata: Optional[AnnotateSessionMetadata] = None,
        behavior_mode: int = 0,
        exclusive_mode: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self._review_dir = review_dir
        self._video_path = video_path
        self._annotations_path = annotations_path
        self._skip_metadata = skip_metadata
        self._skip_video = skip_video
        self._metadata = metadata
        self._behavior_mode = behavior_mode
        self._exclusive_mode = exclusive_mode
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Abort remaining callbacks; I/O may still finish."""
        self._cancel.set()

    def run(self) -> None:
        run_annotate_session_load(
            self._review_dir,
            video_path=self._video_path,
            annotations_path=self._annotations_path,
            skip_metadata=self._skip_metadata,
            skip_video=self._skip_video,
            cancel_event=self._cancel,
            metadata=self._metadata,
            behavior_mode=self._behavior_mode,
            exclusive_mode=self._exclusive_mode,
            on_metadata=self.metadata_ready.emit,
            on_tracklets=self.tracklets_ready.emit,
            on_frame=self.frame_ready.emit,
            on_tracklets_error=self.tracklets_error.emit,
            on_video_error=self.video_error.emit,
            on_error=self.load_error.emit,
        )
