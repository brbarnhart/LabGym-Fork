"""Load / save identity packages for Review IDs (no Qt widgets)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from LabGym.gui_pyside.project.model import Project
from LabGym.gui_pyside.project.paths import (
    annotations_path_for,
    current_video_path,
    examples_out_dir_for,
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
    load_raw_tracklets,
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


def check_downstream_artifacts(
    project: Optional[Project] = None,
    *,
    annotations_path: Optional[Union[str, PathLike]] = None,
    examples_dir: Optional[Union[str, PathLike]] = None,
) -> DownstreamArtifactCheck:
    """Look up an ethogram or example store that may go stale after remap.

    Args:
        project: Open project used to resolve the current video's paths when
            explicit paths are not supplied.
        annotations_path: Optional ethogram path override.
        examples_dir: Optional example-store directory override.

    Returns:
        A result whose ``check_failed`` flag is True if lookup/stat raises.
        That case is never an all-clear: ``requires_confirm`` is True.
    """
    try:
        if (
            annotations_path is None
            and examples_dir is None
            and project is not None
        ):
            video = current_video_path(project)
            if video:
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


def load_review_package(review_dir: str) -> LoadedReviewPackage:
    """Load events, switches, tracklets, and subjects from *review_dir*.

    Does not migrate public tracklets into raw. Callers that want that
    must confirm and call ``migrate_uncorrected_public_to_raw`` first.

    Raises ``FileNotFoundError`` / ``ValueError`` with user-facing messages.
    """
    review_dir = str(Path(review_dir).resolve())
    if not Path(review_dir).is_dir():
        raise FileNotFoundError(f"Not a folder:\n{review_dir}")

    events = load_events(review_dir)
    markers = load_switches(review_dir)
    accepted = has_accepted_identities(review_dir)
    has_raw = has_raw_snapshot(review_dir)
    stores: Dict[str, Any] = {}
    baseline: Dict[str, Any] = {}
    if has_raw:
        loaded = load_raw_tracklets(review_dir)
    else:
        loaded = load_kind_stores(review_dir)
    for kind, store in loaded.items():
        baseline[kind] = store
        stores[kind] = clone_store(store)

    if not stores:
        raise ValueError(f"No *_tracklets_meta.json in:\n{review_dir}")

    animal_kind = max(
        stores.keys(),
        key=lambda k: (len(stores[k].ids), stores[k].n_frames),
    )
    store = stores[animal_kind]
    n_frames = max(1, store.n_frames)
    fps = float(
        store.meta.get("fps")
        or (events[0].fps if events else 10)
        or 10
    )

    recs = load_subjects(review_dir)
    if not recs:
        kind_ids = {k: list(s.ids) for k, s in stores.items()}
        recs = subjects_from_track_ids(kind_ids)

    # Legacy baked packs have remapped geometry and no raw — freeze switch edits.
    already_corrected = bool(accepted and not has_raw)

    return LoadedReviewPackage(
        review_dir=review_dir,
        events=events,
        markers=markers,
        stores=stores,
        baseline_stores=baseline,
        already_corrected=already_corrected,
        has_raw=has_raw,
        accepted=accepted,
        animal_kind=animal_kind,
        n_frames=n_frames,
        fps=fps,
        subjects=list(recs),
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
