"""Load / save identity packages for Review IDs (no Qt widgets)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from LabGym.annotator.core.tracklets_bridge import discover_tracklet_kinds
from LabGym.id_review.apply import read_tracklets_identity_status
from LabGym.id_review.dataset import (
    finalize_switch_annotations,
    load_events,
    load_switches,
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
    kinds = discover_tracklet_kinds(review_dir)
    if not kinds:
        raise ValueError(f"No *_tracklets_meta.json in:\n{review_dir}")

    status = read_tracklets_identity_status(review_dir)
    already_corrected = bool(status.get("corrected"))

    stores: Dict[str, Any] = {}
    baseline: Dict[str, Any] = {}
    for kind in kinds:
        store = load_tracklets(review_dir, kind)
        stores[kind] = store
        baseline[kind] = clone_store(store)

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

    return LoadedReviewPackage(
        review_dir=review_dir,
        events=events,
        markers=markers,
        stores=stores,
        baseline_stores=baseline,
        already_corrected=already_corrected,
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
    """Finalize switches, optionally remap tracklets, write subjects.json."""
    try:
        decisions = finalize_switch_annotations(
            review_dir,
            list(markers),
            events=list(events),
            export_samples=True,
        )
        n = 0
        stores = dict(baseline_stores)
        baselines = dict(baseline_stores)
        corrected = already_corrected
        if not already_corrected:
            n = apply_decisions_and_save_tracklets(
                review_dir,
                decisions,
                baseline_stores=baseline_stores,
                source="pyside_id_review",
            )
            corrected = True
            stores = {}
            baselines = {}
            for kind in list(baseline_stores.keys()):
                stores[kind] = load_tracklets(review_dir, kind)
                baselines[kind] = clone_store(stores[kind])
            remap_note = f"Remap applications: {n}\n"
        else:
            remap_note = (
                "Tracklets already corrected — skipped re-apply "
                "(subjects + switches updated).\n"
            )
        save_subjects(review_dir, list(subjects))
        return SavePackageResult(
            ok=True,
            n_remap=n,
            remap_note=remap_note,
            n_subjects=len(subjects),
            already_corrected=corrected,
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
