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


def switch_edits_allowed(directory: str | Path) -> bool:
    """True when the switch-marker list may be mutated.

    Add, delete, remove-at-frame, undo, and reorder all require raw tracklets
    so remapped geometry can be rebuilt. Names and roles do not use this gate.
    """
    from LabGym.id_review.raw_store import has_raw_snapshot

    return has_raw_snapshot(directory)


def needs_uncorrected_raw_migrate(directory: str | Path) -> bool:
    """True when opening should ask before moving public tracklets into raw.

    Offered only for an uncorrected pack that has public tracklets and no raw.
    Accepted identities (including legacy corrected packs) are never offered.
    """
    from LabGym.annotator.core.tracklets_bridge import discover_tracklet_kinds
    from LabGym.id_review.raw_store import has_accepted_identities, has_raw_snapshot

    directory = Path(directory)
    if has_raw_snapshot(directory):
        return False
    if has_accepted_identities(directory):
        return False
    return bool(discover_tracklet_kinds(directory))


def migrate_uncorrected_public_to_raw(directory: str | Path) -> bool:
    """Move unpublished public tracklets into raw after the caller confirmed.

    Returns True if files were moved. No-op (False) when migrate is not
    offered — including accepted / legacy corrected packs, so remapped
    geometry is never copied in as raw.
    """
    from LabGym.id_review.raw_store import snapshot_uncorrected_root_to_raw

    if not needs_uncorrected_raw_migrate(directory):
        return False
    return bool(snapshot_uncorrected_root_to_raw(directory))


def apply_decisions_and_save_tracklets(
    directory: str | Path,
    decisions: Sequence,
    *,
    source: str = "pyside_id_review",
) -> int:
    """Publish remapped tracklets from raw plus the current switch decisions.

    Rebuilds every kind in the raw snapshot. Empty decisions still write public
    remapped files equal to raw and record accepted identities. Refuses when
    raw is missing — public remapped files are never the remap baseline.

    Returns number of decision applications that remapped geometry.
    """
    from LabGym.id_review.apply import (
        apply_decisions_to_store,
        write_tracklets_identity_status,
    )
    from LabGym.id_review.raw_store import has_raw_snapshot, load_raw_tracklets
    from LabGym.id_review.tracklets import save_tracklets

    directory = Path(directory)
    if not has_raw_snapshot(directory):
        raise FileNotFoundError(
            f"Cannot rebuild remapped tracklets without raw tracklets in {directory}"
        )

    n_total = 0
    for kind, raw_store in load_raw_tracklets(directory).items():
        store = clone_store(raw_store)
        n_total += apply_decisions_to_store(store, decisions, animal_kind=kind)
        save_tracklets(store, str(directory))

    write_tracklets_identity_status(
        str(directory),
        corrected=True,
        accepted=True,
        has_raw=True,
        n_decisions=n_total,
        source=source,
    )
    return n_total
