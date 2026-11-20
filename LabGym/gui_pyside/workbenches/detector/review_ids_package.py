"""Load / save identity packages for Review IDs (no Qt widgets)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from LabGym.annotator.core.tracklets_bridge import discover_tracklet_kinds
from LabGym.gui_pyside.project.model import Project
from LabGym.gui_pyside.project.paths import (
    annotations_path_for,
    discover_tracklets_dir,
    examples_out_dir_for,
    list_project_video_choices,
)
from LabGym.id_review.dataset import (
    finalize_switch_annotations,
    load_events,
    load_switches,
)
from LabGym.id_review.raw_store import (
    has_accepted_identities,
    has_raw_snapshot,
    load_kind_stores,
    raw_dir,
)
from LabGym.id_review.types import ContactEvent, SwitchMarker
from LabGym.identity.package import (
    SubjectRecord,
    apply_decisions_and_save_tracklets,
    clone_store,
    load_subjects,
    save_subjects,
    subjects_from_track_ids,
)

_log = logging.getLogger(__name__)


@dataclass
class KindTrackletInfo:
    """JSON-only facts for one **animal kind** (no npz)."""

    animal_kind: str
    ids: List[int]
    n_frames: int
    fps: float
    store_meta: Dict[str, Any] = field(default_factory=dict)
    directory: str = ""
    npz_file: str = ""


@dataclass
class ReviewPackageMetadata:
    """Identity-package JSON that can unlock Review IDs before pixels/outlines."""

    review_dir: str
    events: List[ContactEvent]
    markers: List[SwitchMarker]
    subjects: List[SubjectRecord]
    kinds: List[str]
    kind_info: Dict[str, KindTrackletInfo]
    animal_kind: str
    n_frames: int
    fps: float
    already_corrected: bool
    has_raw: bool
    accepted: bool
    video_path: Optional[str] = None


@dataclass
class LoadedReviewPackage:
    """In-memory state after opening an id_review folder."""

    review_dir: str
    events: List[ContactEvent]
    markers: List[SwitchMarker]
    stores: Dict[str, Any]
    baseline_stores: Dict[str, Any]
    already_corrected: bool
    has_raw: bool
    accepted: bool
    animal_kind: str
    n_frames: int
    fps: float
    subjects: List[SubjectRecord] = field(default_factory=list)


@dataclass
class SavePackageResult:
    """Outcome of writing remapped tracklets and subjects from Review IDs.

    ``ok`` is False when save refused (no raw, I/O error). ``n_remap`` counts
    switch decisions that changed geometry; empty-switch accept still sets
    ``accepted`` when the write succeeded.
    """

    ok: bool
    error: str = ""
    n_remap: int = 0
    remap_note: str = ""
    n_subjects: int = 0
    already_corrected: bool = False
    has_raw: bool = False
    accepted: bool = False
    stores: Dict[str, Any] = field(default_factory=dict)
    baseline_stores: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DownstreamArtifactCheck:
    """Outcome of looking up ethogram / example-store files before remap save.

    ``check_failed`` is the fail-closed signal: the caller must confirm save
    and must not treat the result as “nothing downstream.”
    """

    check_failed: bool
    error: str = ""
    ethogram_path: Optional[str] = None
    examples_path: Optional[str] = None

    @property
    def requires_confirm(self) -> bool:
        """True when artifacts exist or the lookup failed (never an all-clear)."""
        return bool(self.check_failed or self.ethogram_path or self.examples_path)

    def note_lines(self) -> List[str]:
        """Human-readable artifact paths or the lookup error (not dialog copy)."""
        if self.check_failed:
            return [
                "Could not check for an existing ethogram or example store: "
                f"{self.error or 'unknown error'}"
            ]
        lines: List[str] = []
        if self.ethogram_path:
            lines.append(f"Ethogram: {self.ethogram_path}")
        if self.examples_path:
            lines.append(f"Examples: {self.examples_path}")
        return lines


def should_confirm_stale_downstream(
    *,
    will_rebuild: bool,
    check: DownstreamArtifactCheck,
) -> bool:
    """True when Save must ask because remapped tracklets will be rebuilt.

    Names/roles-only saves do not rebuild remapped geometry and must not
    warn as if they will.
    """
    return bool(will_rebuild and check.requires_confirm)


def video_path_for_review_package(
    review_dir: str,
    *,
    project: Optional[Project] = None,
    hinted_video: Optional[str] = None,
    store_meta: Optional[dict] = None,
    events: Optional[Sequence[ContactEvent]] = None,
) -> Optional[str]:
    """Video this identity package belongs to (not the project's current video).

    Prefers a caller hint (Load from a project video), then
    ``detect_track_job.json`` / tracklet meta, then a project video whose
    discovered package folder is *review_dir*.
    """
    hint = (hinted_video or "").strip()
    if hint:
        return hint

    from LabGym.identity.package import read_detect_job_video

    recorded = read_detect_job_video(review_dir)
    if recorded:
        if Path(recorded).is_file():
            return recorded
        resolved = resolve_video_path(
            review_dir, {"video": recorded}, events or (), None
        )
        if resolved:
            return resolved

    from_meta = resolve_video_path(
        review_dir, dict(store_meta or {}), events or (), None
    )
    if from_meta and Path(from_meta).is_file():
        return from_meta

    if project is not None:
        matched = _match_package_to_project_video(project, review_dir)
        if matched:
            return matched
    return from_meta


def _match_package_to_project_video(project: Project, review_dir: str) -> Optional[str]:
    try:
        want = Path(review_dir).resolve()
    except OSError:
        want = Path(review_dir)
    for _label, video in list_project_video_choices(project):
        tracks = discover_tracklets_dir(project, video)
        if not tracks:
            continue
        try:
            got = Path(tracks).resolve()
        except OSError:
            got = Path(tracks)
        if got == want:
            return video
    return None


def check_downstream_artifacts(
    project: Optional[Project] = None,
    *,
    video_path: Optional[str] = None,
    annotations_path: Optional[Union[str, PathLike]] = None,
    examples_dir: Optional[Union[str, PathLike]] = None,
) -> DownstreamArtifactCheck:
    """Look up an ethogram or example store that may go stale after remap.

    Args:
        project: Open project used to resolve ethogram / example paths for
            *video_path*.
        video_path: Video this package belongs to. Required when *project*
            is used and explicit paths are omitted. The project's current
            video is never consulted.
        annotations_path: Optional ethogram path override.
        examples_dir: Optional example-store directory override.

    Returns:
        A result whose ``check_failed`` flag is True if lookup/stat raises
        or the package's video cannot be identified. That case is never an
        all-clear: ``requires_confirm`` is True.
    """
    try:
        if (
            annotations_path is None
            and examples_dir is None
            and project is not None
        ):
            video = (video_path or "").strip()
            if not video:
                return DownstreamArtifactCheck(
                    check_failed=True,
                    error=(
                        "Could not identify which video this identity "
                        "package belongs to"
                    ),
                )
            annotations_path = annotations_path_for(project, video)
            examples_dir = examples_out_dir_for(project, video)
        ethogram: Optional[str] = None
        examples: Optional[str] = None
        if annotations_path:
            p = Path(annotations_path)
            if p.is_file():
                ethogram = str(p)
        if examples_dir:
            d = Path(examples_dir)
            if d.is_dir() and any(d.iterdir()):
                examples = str(d)
        return DownstreamArtifactCheck(
            check_failed=False,
            ethogram_path=ethogram,
            examples_path=examples,
        )
    except Exception as exc:
        _log.exception("Ethogram/example-store lookup failed")
        return DownstreamArtifactCheck(check_failed=True, error=str(exc))


def _tracklet_json_dir(review_dir: str, *, has_raw: Optional[bool] = None) -> Path:
    """Folder whose ``*_tracklets_meta.json`` files describe the preview source."""
    root = Path(review_dir)
    if has_raw is None:
        has_raw = has_raw_snapshot(root)
    return raw_dir(root) if has_raw else root


def _read_kind_tracklet_info(directory: Path, kind: str) -> Optional[KindTrackletInfo]:
    """Parse ``{kind}_tracklets_meta.json`` without opening the npz."""
    path = directory / f"{kind}_tracklets_meta.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    ids_raw = raw.get("ids") or []
    try:
        ids = [int(x) for x in ids_raw]
    except (TypeError, ValueError):
        ids = []
    try:
        n_frames = int(raw.get("n_frames") or 0)
    except (TypeError, ValueError):
        n_frames = 0
    store_meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    try:
        fps = float(store_meta.get("fps") or 0.0)
    except (TypeError, ValueError):
        fps = 0.0
    return KindTrackletInfo(
        animal_kind=str(raw.get("animal_kind") or kind),
        ids=ids,
        n_frames=max(0, n_frames),
        fps=fps,
        store_meta=dict(store_meta),
        directory=str(directory),
        npz_file=str(raw.get("npz_file") or f"{kind}_tracklets.npz"),
    )


def _preferred_animal_kind(kind_info: Dict[str, KindTrackletInfo]) -> str:
    return max(
        kind_info.keys(),
        key=lambda k: (len(kind_info[k].ids), kind_info[k].n_frames, k),
    )


def load_review_package_metadata(
    review_dir: str,
    *,
    video_hint: Optional[str] = None,
) -> ReviewPackageMetadata:
    """Load package JSON (events, switches, subjects, kind meta) without npz.

    Raises ``FileNotFoundError`` / ``ValueError`` with user-facing messages.
    """
    review_dir = str(Path(review_dir).resolve())
    if not Path(review_dir).is_dir():
        raise FileNotFoundError(f"Not a folder:\n{review_dir}")

    events = load_events(review_dir)
    markers = load_switches(review_dir)
    accepted = has_accepted_identities(review_dir)
    has_raw = has_raw_snapshot(review_dir)
    json_dir = _tracklet_json_dir(review_dir, has_raw=has_raw)
    kinds = discover_tracklet_kinds(json_dir)
    kind_info: Dict[str, KindTrackletInfo] = {}
    for kind in kinds:
        info = _read_kind_tracklet_info(json_dir, kind)
        if info is not None:
            kind_info[kind] = info
    if not kind_info:
        raise ValueError(f"No *_tracklets_meta.json in:\n{review_dir}")

    animal_kind = _preferred_animal_kind(kind_info)
    preferred = kind_info[animal_kind]
    n_frames = max(1, preferred.n_frames)
    fps = float(
        preferred.fps
        or (events[0].fps if events else 10)
        or 10
    )
    recs = load_subjects(review_dir)
    if not recs:
        recs = subjects_from_track_ids({k: list(v.ids) for k, v in kind_info.items()})
    already_corrected = bool(accepted and not has_raw)
    video_path = video_path_for_review_package(
        review_dir,
        hinted_video=video_hint,
        store_meta=preferred.store_meta,
        events=events,
    )
    return ReviewPackageMetadata(
        review_dir=review_dir,
        events=events,
        markers=markers,
        subjects=list(recs),
        kinds=sorted(kind_info.keys()),
        kind_info=kind_info,
        animal_kind=animal_kind,
        n_frames=n_frames,
        fps=fps,
        already_corrected=already_corrected,
        has_raw=has_raw,
        accepted=accepted,
        video_path=video_path,
    )


def load_review_kind_store(review_dir: str, animal_kind: str) -> Any:
    """Load packed tracklets for one **animal kind** (ADR-0009).

    Uses the raw snapshot when present; otherwise public package files.
    Other kinds in the package are not opened.

    Raises ``FileNotFoundError`` / ``ValueError`` / ``OSError`` on I/O or
    corrupt npz.
    """
    review_dir = str(Path(review_dir).resolve())
    if not Path(review_dir).is_dir():
        raise FileNotFoundError(f"Not a folder:\n{review_dir}")
    directory = _tracklet_json_dir(review_dir)
    loaded = load_kind_stores(directory, animal_kinds=[animal_kind])
    store = loaded.get(animal_kind)
    if store is None:
        raise ValueError(
            f"No tracklets for animal kind {animal_kind!r} in:\n{directory}"
        )
    return store


def load_review_package(review_dir: str) -> LoadedReviewPackage:
    """Load events, switches, tracklets, and subjects from *review_dir*.

    Does not migrate public tracklets into raw. Callers that want that
    must confirm and call ``migrate_uncorrected_public_to_raw`` first.

    Loads every discovered **animal kind**. Review IDs session load uses
    :func:`load_review_package_metadata` plus :func:`load_review_kind_store`
    so the visible kind can arrive first.

    Raises ``FileNotFoundError`` / ``ValueError`` with user-facing messages.
    """
    meta = load_review_package_metadata(review_dir)
    stores: Dict[str, Any] = {}
    baseline: Dict[str, Any] = {}
    for kind in meta.kinds:
        store = load_review_kind_store(meta.review_dir, kind)
        stores[kind] = store
        baseline[kind] = store
    return LoadedReviewPackage(
        review_dir=meta.review_dir,
        events=meta.events,
        markers=meta.markers,
        stores=stores,
        baseline_stores=baseline,
        already_corrected=meta.already_corrected,
        has_raw=meta.has_raw,
        accepted=meta.accepted,
        animal_kind=meta.animal_kind,
        n_frames=meta.n_frames,
        fps=meta.fps,
        subjects=list(meta.subjects),
    )


def resolve_video_path(
    review_dir: str,
    store_meta: dict,
    events: Sequence[ContactEvent],
    project_video: Optional[str] = None,
) -> Optional[str]:
    """Resolve the source video path for preview playback."""
    meta = dict(store_meta or {})
    if events and events[0].video:
        meta.setdefault("video", events[0].video)
    video = meta.get("video")
    if not video:
        return None
    if Path(str(video)).is_file():
        return str(video)
    candidates = [
        Path(review_dir) / video,
        Path(review_dir).parent / Path(str(video)).name,
    ]
    if project_video:
        candidates.insert(0, Path(project_video))
    for c in candidates:
        if c.is_file():
            return str(c)
    return str(video)


def save_review_package(
    review_dir: str,
    markers: Sequence[SwitchMarker],
    events: Sequence[ContactEvent],
    subjects: Sequence[SubjectRecord],
    *,
    already_corrected: bool,
    baseline_stores: Dict[str, Any],
) -> SavePackageResult:
    """Finalize switches, publish remapped tracklets from raw, write subjects."""
    try:
        decisions = finalize_switch_annotations(
            review_dir,
            list(markers),
            events=list(events),
            export_samples=True,
        )
        n = 0
        can_rebuild = has_raw_snapshot(review_dir)
        stores = {k: clone_store(s) for k, s in baseline_stores.items()}
        baselines = dict(baseline_stores)
        accepted = False
        if can_rebuild:
            n = apply_decisions_and_save_tracklets(
                review_dir,
                decisions,
                source="pyside_id_review",
            )
            accepted = True
            # Keep editor baselines as raw; preview applies markers live.
            remap_note = f"Remap applications: {n}\n"
        else:
            remap_note = (
                "No raw snapshot — skipped rebuild "
                "(subjects + switches updated).\n"
            )
        save_subjects(review_dir, list(subjects))
        return SavePackageResult(
            ok=True,
            n_remap=n,
            remap_note=remap_note,
            n_subjects=len(subjects),
            already_corrected=already_corrected,
            has_raw=can_rebuild,
            accepted=accepted or has_accepted_identities(review_dir),
            stores=stores,
            baseline_stores=baselines,
        )
    except Exception as exc:
        return SavePackageResult(ok=False, error=str(exc))


def clone_markers(markers: Sequence[SwitchMarker]) -> List[SwitchMarker]:
    return [SwitchMarker.from_dict(m.to_dict()) for m in markers]


def events_for_kind(
    events: Sequence[ContactEvent], animal_kind: str
) -> List[ContactEvent]:
    return [e for e in events if e.animal_kind == animal_kind]
