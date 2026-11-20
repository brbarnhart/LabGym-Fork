"""Cancellable Annotate ethogram **session load** (not an analysis job-queue item).

Unlocks on package JSON, then loads **every animal kind** needed for overlays
and the video independently. Cancel or a newer load replaces in-flight work.
Geometry is published remapped tracklets (ADR-0006 / ADR-0009), not raw stores.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QObject, QThread, Signal

from LabGym.annotator.core.tracklets_bridge import (
    LoadedTracklets,
    discover_tracklet_kinds,
    load_tracklets_for_annotator,
)
from LabGym.gui_pyside.session_load import (
    join_session_workers,
    load_cancelled,
    load_session_frame0,
    run_session_io,
)
from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
    ReviewPackageMetadata,
    load_review_package_metadata,
)
from LabGym.identity.package import SubjectRecord

LoadTrackletsFn = Callable[..., Any]
ReadFrameFn = Callable[..., Tuple[Any, Optional[str]]]
Frame0Result = Tuple[Any, Optional[str]]
OnMetadata = Callable[["AnnotateSessionMetadata"], None]
OnTracklets = Callable[[Any], None]
OnFrame = Callable[[Any], None]
OnTrackletsError = Callable[[str], None]
OnVideoError = Callable[[str], None]
OnError = Callable[[str], None]
OnGateFailed = Callable[[str], None]


@dataclass
class AnnotateSessionMetadata:
    """Package JSON / project paths that unlock Annotate before pixels."""

    video_path: str
    annotations_path: str
    review_dir: str
    n_frames: Optional[int]
    fps: Optional[float]
    kinds: List[str]
    subjects: List[SubjectRecord] = field(default_factory=list)
    behavior_mode: int = 0
    exclusive_mode: bool = False
    accepted: bool = False
    package: Optional[ReviewPackageMetadata] = None
    annotations_data: Optional[Dict[str, Any]] = None


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
        n_frames=int(pkg.n_frames) if pkg.n_frames else None,
        fps=float(pkg.fps) if pkg.fps else None,
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


def overlay_subject_display(session: Any, records: Sequence[SubjectRecord]) -> None:
    """Copy names/colors onto existing subjects; do not replace bout maps."""
    if not records or session is None:
        return
    by_sid = {int(r.subject_id): r for r in records}
    by_key: Dict[tuple, SubjectRecord] = {}
    for rec in records:
        tid = int(rec.track_id if rec.track_id is not None else rec.subject_id)
        by_key[(str(rec.animal_kind), tid)] = rec
    for subj in list(getattr(session, "subjects", None) or []):
        rec = by_sid.get(int(subj.subject_id))
        if rec is None:
            rec = by_key.get((str(getattr(subj, "animal_kind", "")), int(subj.subject_id)))
        if rec is None:
            continue
        if rec.display_name:
            subj.display_name = rec.display_name
        if rec.color:
            subj.color = rec.color


def read_sidecar_annotations(path: str) -> Optional[Dict[str, Any]]:
    """Parse an ethogram JSON file; return None if missing or invalid."""
    raw_path = str(path or "").strip()
    if not raw_path:
        return None
    file_path = Path(raw_path)
    if not file_path.is_file():
        return None
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _attach_sidecar(meta: AnnotateSessionMetadata) -> None:
    data = read_sidecar_annotations(meta.annotations_path)
    if data is None:
        return
    meta.annotations_data = data
    if not meta.n_frames:
        try:
            n = int(data.get("total_frames") or 0)
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            meta.n_frames = n
    if not meta.fps:
        try:
            fps = float(data.get("fps") or 0)
        except (TypeError, ValueError):
            fps = 0.0
        if fps > 0:
            meta.fps = fps


def _default_read_frame(
    *,
    video_path: Optional[str] = None,
    store_meta: Optional[dict] = None,
    fps: float = 0,
    **_kwargs,
) -> Frame0Result:
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
    cancel_event: Optional[threading.Event] = None,
    on_metadata: Optional[OnMetadata] = None,
    on_tracklets: Optional[OnTracklets] = None,
    on_frame: Optional[OnFrame] = None,
    on_tracklets_error: Optional[OnTrackletsError] = None,
    on_video_error: Optional[OnVideoError] = None,
    on_error: Optional[OnError] = None,
    on_gate_failed: Optional[OnGateFailed] = None,
    load_tracklets: Optional[LoadTrackletsFn] = None,
    read_frame: Optional[ReadFrameFn] = None,
    behavior_mode: int = 0,
    exclusive_mode: bool = False,
    project: Optional[Any] = None,
) -> str:
    """Run one Annotate session load to completion or cancel.

    Metadata is applied before geometry or pixels. Packed remapped npz for
    **every overlay kind** and frame 0 are loaded in parallel so either can
    arrive first. This is not an analysis job-queue item. Annotate does not
    support Review IDs kind-only skip; every load reads metadata and frame 0.

    Args:
        review_dir: Identity package folder (published remapped tracklets).
        video_path: Video path for the throwaway frame-0 preview reader.
        annotations_path: Sidecar ethogram path (parsed on this thread).
        cancel_event: Set to abort; in-flight npz/video may still finish I/O
            but callbacks after cancel are skipped.
        on_metadata: Called with JSON metadata before npz/video work.
        on_tracklets: Called with ``LoadedTracklets`` (all overlay kinds).
        on_frame: Called with a BGR/RGB frame-0 array.
        on_tracklets_error: Called with a message on corrupt/missing npz.
        on_video_error: Called when the video cannot be opened.
        on_error: Called when package JSON cannot be read (no metadata).
        on_gate_failed: Called when **accepted identities** refuse Load.
        load_tracklets: Injectable loader (tests); default is
            ``load_tracklets_for_annotator``.
        read_frame: Injectable frame-0 reader (tests).
        behavior_mode: Passed through to metadata.
        exclusive_mode: Passed through to metadata.
        project: When set, resolve the video context and apply the
            **accepted identities** gate on this worker thread.

    Returns:
        ``"ok"``, ``"cancelled"``, or ``"error"``.
    """
    load_fn = load_tracklets or _default_load_tracklets
    read_frame_fn = read_frame or _default_read_frame

    if project is not None:
        from LabGym.gui_pyside.project.paths import resolve_video_context
        from LabGym.identity.downstream import (
            may_use_downstream,
            review_ids_required_message,
        )
        from LabGym.identity.package import has_identity_package

        if load_cancelled(cancel_event):
            return "cancelled"
        ctx = resolve_video_context(project, video_path=video_path)
        tracks = str(getattr(ctx, "tracklets_dir", "") or "")
        package = bool(tracks) and has_identity_package(tracks)
        if not may_use_downstream(
            int(ctx.behavior_mode),
            bool(ctx.accepted_identities),
            identity_package=package,
        ):
            if load_cancelled(cancel_event):
                return "cancelled"
            msg = review_ids_required_message(tracks)
            if on_gate_failed is not None:
                on_gate_failed(msg)
            elif on_error is not None:
                on_error(msg)
            return "error"
        if not review_dir:
            review_dir = tracks if (package and bool(ctx.accepted_identities)) else ""
        if not video_path:
            video_path = str(getattr(ctx, "video_path", "") or "")
        if annotations_path is None:
            ann = getattr(ctx, "annotations_path", None) or ""
            annotations_path = str(ann) if getattr(ctx, "annotations_exists", False) else ""
        behavior_mode = int(ctx.behavior_mode)
        exclusive_mode = bool(getattr(ctx, "exclusive_mode", exclusive_mode))

    directory = str(review_dir or "").strip()
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
            if load_cancelled(cancel_event):
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
            n_frames=None,
            fps=None,
            kinds=[],
            behavior_mode=int(behavior_mode),
            exclusive_mode=bool(exclusive_mode),
            accepted=False,
        )
    if video_path:
        meta.video_path = str(video_path)
    if annotations_path:
        meta.annotations_path = str(annotations_path)
    _attach_sidecar(meta)
    if load_cancelled(cancel_event):
        return "cancelled"
    if on_metadata is not None:
        on_metadata(meta)
    if load_cancelled(cancel_event):
        return "cancelled"

    resolved_video = str(video_path or meta.video_path or "")
    fps = float(meta.fps) if meta.fps else 10.0
    store_meta = {}
    if meta.package is not None:
        kind = meta.package.animal_kind
        info = meta.package.kind_info.get(kind)
        if info is not None:
            store_meta = dict(info.store_meta)

    def _load_tracklets() -> None:
        if not directory:
            return
        run_session_io(
            cancel_event,
            lambda: load_fn(directory),
            on_ok=on_tracklets,
            on_error=on_tracklets_error,
        )

    def _load_frame() -> None:
        load_session_frame0(
            skip_video=False,
            cancel_event=cancel_event,
            read_frame_fn=read_frame_fn,
            video_path=resolved_video,
            store_meta=store_meta,
            fps=fps,
            on_frame=on_frame,
            on_video_error=on_video_error,
        )

    return join_session_workers(
        [
            ("annotate-tracklets", _load_tracklets),
            ("annotate-frame", _load_frame),
        ],
        cancel_event,
    )


class AnnotateSessionLoadThread(QThread):
    """Background **session load**. Callers apply results on the UI thread."""

    metadata_ready = Signal(object)
    tracklets_ready = Signal(object)
    frame_ready = Signal(object)
    tracklets_error = Signal(str)
    video_error = Signal(str)
    load_error = Signal(str)

    gate_failed = Signal(str)

    def __init__(
        self,
        review_dir: str = "",
        *,
        video_path: Optional[str] = None,
        annotations_path: Optional[str] = None,
        behavior_mode: int = 0,
        exclusive_mode: bool = False,
        project: Optional[Any] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        """Create a worker for one Annotate **session load**.

        Args:
            review_dir: Identity package folder. Empty when *project* should hunt.
            video_path: Video path for the throwaway frame-0 preview reader.
            annotations_path: Sidecar ethogram path (parsed on this thread).
            behavior_mode: Detect-world behavior mode fallback.
            exclusive_mode: Ethogram exclusive-mode fallback.
            project: When set, resolve context and the accepted-identities gate
                off the UI thread.
            parent: Qt owner (usually the Annotate ethogram tab).
        """
        super().__init__(parent)
        self._review_dir = review_dir
        self._video_path = video_path
        self._annotations_path = annotations_path
        self._behavior_mode = behavior_mode
        self._exclusive_mode = exclusive_mode
        self._project = project
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Abort remaining callbacks; I/O may still finish."""
        self._cancel.set()

    def run(self) -> None:
        run_annotate_session_load(
            self._review_dir,
            video_path=self._video_path,
            annotations_path=self._annotations_path,
            cancel_event=self._cancel,
            behavior_mode=self._behavior_mode,
            exclusive_mode=self._exclusive_mode,
            project=self._project,
            on_metadata=self.metadata_ready.emit,
            on_tracklets=self.tracklets_ready.emit,
            on_frame=self.frame_ready.emit,
            on_tracklets_error=self.tracklets_error.emit,
            on_video_error=self.video_error.emit,
            on_error=self.load_error.emit,
            on_gate_failed=self.gate_failed.emit,
        )
