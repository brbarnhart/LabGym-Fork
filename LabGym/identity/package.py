"""Durable identity package: tracklets folder + subjects.json (+ review status)."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from LabGym.annotator.core.tracklets_bridge import subject_color_for_index

SUBJECTS_FILENAME = "subjects.json"


@dataclass
class SubjectRecord:
    """Experimental identity for one track ID (within an animal kind)."""

    subject_id: int
    animal_kind: str = "animal"
    display_name: str = ""
    role: str = ""
    color: str = ""
    track_id: Optional[int] = None  # original tracker id; defaults to subject_id

    def __post_init__(self) -> None:
        if self.track_id is None:
            self.track_id = int(self.subject_id)
        if not self.display_name:
            self.display_name = f"{self.animal_kind}_{self.subject_id}"
        if not self.color:
            self.color = subject_color_for_index(int(self.subject_id))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": int(self.subject_id),
            "animal_kind": str(self.animal_kind),
            "display_name": str(self.display_name),
            "role": str(self.role or ""),
            "color": str(self.color or ""),
            "track_id": int(self.track_id if self.track_id is not None else self.subject_id),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SubjectRecord":
        return cls(
            subject_id=int(d["subject_id"]),
            animal_kind=str(d.get("animal_kind") or "animal"),
            display_name=str(d.get("display_name") or ""),
            role=str(d.get("role") or ""),
            color=str(d.get("color") or ""),
            track_id=(
                int(d["track_id"])
                if d.get("track_id") is not None
                else int(d["subject_id"])
            ),
        )


def subjects_from_track_ids(
    kind_to_ids: Dict[str, Sequence[int]],
) -> List[SubjectRecord]:
    """Build default subject records from tracklet id lists."""
    multi = len(kind_to_ids) > 1
    records: List[SubjectRecord] = []
    next_sid = 0
    for kind in sorted(kind_to_ids.keys()):
        for tid in kind_to_ids[kind]:
            if multi:
                sid = next_sid
                next_sid += 1
                display = f"{kind}_{tid}"
            else:
                sid = int(tid)
                display = f"{kind}_{tid}"
            records.append(
                SubjectRecord(
                    subject_id=sid,
                    animal_kind=str(kind),
                    display_name=display,
                    track_id=int(tid),
                    color=subject_color_for_index(len(records)),
                )
            )
    return records


def subjects_path(directory: str | Path) -> Path:
    return Path(directory) / SUBJECTS_FILENAME


def load_subjects(directory: str | Path) -> List[SubjectRecord]:
    path = subjects_path(directory)
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "subjects" in raw:
        items = raw["subjects"]
    elif isinstance(raw, list):
        items = raw
    else:
        return []
    return [SubjectRecord.from_dict(x) for x in items]


def save_subjects(directory: str | Path, subjects: Sequence[SubjectRecord]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = subjects_path(directory)
    payload = {
        "schema_version": 1,
        "subjects": [s.to_dict() for s in subjects],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def merge_subjects_into_loaded(loaded, subjects: Sequence[SubjectRecord]) -> None:
    """Update LoadedTracklets.subjects display_name / role / color from subjects.json.

    Matches by (animal_kind, track_id) when multi-kind; else by subject_id == track_id.
    """
    if not subjects:
        return
    from LabGym.annotator.core.data_models import Subject

    by_key: Dict[Tuple[str, int], SubjectRecord] = {}
    by_sid: Dict[int, SubjectRecord] = {}
    for rec in subjects:
        tid = int(rec.track_id if rec.track_id is not None else rec.subject_id)
        by_key[(str(rec.animal_kind), tid)] = rec
        by_sid[int(rec.subject_id)] = rec

    new_subjects: List[Subject] = []
    for subj in loaded.subjects:
        kind, track_id = loaded.subject_to_track.get(
            subj.subject_id, (subj.animal_kind, subj.subject_id)
        )
        rec = by_key.get((str(kind), int(track_id))) or by_sid.get(int(subj.subject_id))
        if rec is None:
            new_subjects.append(subj)
            continue
        new_subjects.append(
            Subject(
                subject_id=int(subj.subject_id),
                animal_kind=str(kind),
                display_name=rec.display_name or subj.display_name,
                color=rec.color or subj.color,
            )
        )
        # Stash role on a dynamic attribute for UI that wants it
        new_subjects[-1].role = rec.role  # type: ignore[attr-defined]
    loaded.subjects = new_subjects


def clone_store(store):
    """Deep-ish copy of a TrackletStore for remap baselines."""
    from LabGym.id_review.types import TrackletStore

    return TrackletStore(
        schema_version=store.schema_version,
        animal_kind=store.animal_kind,
        ids=list(store.ids),
        n_frames=int(store.n_frames),
        centers=store.centers.copy(),
        valid=store.valid.copy(),
        heights=store.heights.copy(),
        contours=copy.deepcopy(store.contours),
        meta=dict(store.meta or {}),
    )


def apply_decisions_and_save_tracklets(
    directory: str | Path,
    decisions: Sequence,
    *,
    baseline_stores: Optional[Dict[str, Any]] = None,
    source: str = "pyside_id_review",
) -> int:
    """Apply remap decisions to tracklets and write corrected npz + identity status.

    If ``baseline_stores`` is provided (pre-remap geometry), those are used as the
    starting point so re-saving never double-applies. Otherwise loads from disk.

    Returns number of decision applications that remapped geometry.
    """
    from LabGym.id_review.apply import (
        apply_decisions_to_store,
        write_tracklets_identity_status,
    )
    from LabGym.id_review.tracklets import load_tracklets, save_tracklets
    from LabGym.annotator.core.tracklets_bridge import discover_tracklet_kinds

    directory = Path(directory)
    kinds = discover_tracklet_kinds(directory)
    if not kinds and baseline_stores:
        kinds = sorted(baseline_stores.keys())

    n_total = 0
    for kind in kinds:
        if baseline_stores and kind in baseline_stores:
            store = clone_store(baseline_stores[kind])
        else:
            store = load_tracklets(str(directory), kind)
        n = apply_decisions_to_store(store, decisions, animal_kind=kind)
        n_total += n
        save_tracklets(store, str(directory))

    write_tracklets_identity_status(
        str(directory),
        corrected=True,
        n_decisions=n_total,
        source=source,
    )
    return n_total
