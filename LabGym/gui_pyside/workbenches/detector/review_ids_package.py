"""Load / save identity packages for Review IDs (no Qt widgets)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from LabGym.annotator.core.tracklets_bridge import discover_tracklet_kinds
from LabGym.id_review.dataset import (
    finalize_switch_annotations,
    load_events,
    load_switches,
)
from LabGym.id_review.raw_store import (
    has_accepted_identities,
    has_raw_snapshot,
    load_raw_tracklets,
    snapshot_uncorrected_root_to_raw,
)
from LabGym.id_review.tracklets import load_tracklets
from LabGym.id_review.types import ContactEvent, SwitchMarker
from LabGym.identity.package import (
    SubjectRecord,
    apply_decisions_and_save_tracklets,
    clone_store,
    load_subjects,
    save_subjects,
    subjects_from_track_ids,
)


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


def load_review_package(review_dir: str) -> LoadedReviewPackage:
    """Load events, switches, tracklets, and subjects from *review_dir*.

    Raises ``FileNotFoundError`` / ``ValueError`` with user-facing messages.
    """
    review_dir = str(Path(review_dir).resolve())
    if not Path(review_dir).is_dir():
        raise FileNotFoundError(f"Not a folder:\n{review_dir}")

    events = load_events(review_dir)
    markers = load_switches(review_dir)
    accepted = has_accepted_identities(review_dir)
    if not has_raw_snapshot(review_dir) and discover_tracklet_kinds(review_dir) and not accepted:
        snapshot_uncorrected_root_to_raw(review_dir)

    has_raw = has_raw_snapshot(review_dir)
    stores: Dict[str, Any] = {}
    baseline: Dict[str, Any] = {}
    if has_raw:
        raw = load_raw_tracklets(review_dir)
        for kind, store in raw.items():
            baseline[kind] = store
            stores[kind] = clone_store(store)
    else:
        kinds = discover_tracklet_kinds(review_dir)
        if not kinds:
            raise ValueError(f"No *_tracklets_meta.json in:\n{review_dir}")
        for kind in kinds:
            store = load_tracklets(review_dir, kind)
            stores[kind] = store
            baseline[kind] = clone_store(store)

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
    has_raw: Optional[bool] = None,
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
        can_rebuild = has_raw if has_raw is not None else (not already_corrected)
        stores = {k: clone_store(s) for k, s in baseline_stores.items()}
        baselines = dict(baseline_stores)
        corrected = already_corrected
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
            already_corrected=corrected,
            has_raw=bool(can_rebuild) or has_raw_snapshot(review_dir),
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
