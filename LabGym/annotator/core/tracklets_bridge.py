"""Load LabGym id_review tracklets and map them into annotator subjects / overlays.

Tracklet frames are *analysis* frames. Video frame mapping:

    video_frame = analysis_frame + analysis_start_frame
    analysis_frame = video_frame - analysis_start_frame

``analysis_start_frame`` defaults to 0, or can be inferred from tracklet meta
(``start_t * fps``) when the tracklet span is shorter than the full video.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from LabGym.annotator.core.data_models import Subject, TracksRef

# Distinct, high-contrast subject colors (hex)
DEFAULT_SUBJECT_COLORS: Tuple[str, ...] = (
    "#4FC3F7",
    "#FF8A65",
    "#81C784",
    "#CE93D8",
    "#FFD54F",
    "#4DB6AC",
    "#F06292",
    "#90A4AE",
    "#A1887F",
    "#E57373",
)


def subject_color_for_index(index: int) -> str:
    return DEFAULT_SUBJECT_COLORS[int(index) % len(DEFAULT_SUBJECT_COLORS)]


@dataclass
class TrackOverlay:
    """Geometry for one subject at one frame (image coordinates)."""

    subject_id: int
    animal_kind: str
    center: Optional[Tuple[float, float]]  # (x, y) or None if invalid
    contour: Optional[np.ndarray]  # shape (N, 1, 2) or (N, 2), int
    color: str
    valid: bool = True


@dataclass
class LoadedTracklets:
    """In-memory multi-kind tracklet bundle for the annotator."""

    directory: str
    stores: Dict[str, Any] = field(default_factory=dict)  # animal_kind -> TrackletStore
    analysis_start_frame: int = 0
    # subject_id is unique across kinds when possible; for multi-kind we
    # encode as sequential slots with (kind, track_id) mapping.
    subjects: List[Subject] = field(default_factory=list)
    # subject_id -> (animal_kind, track_id)
    subject_to_track: Dict[int, Tuple[str, int]] = field(default_factory=dict)

    def tracks_ref(self) -> TracksRef:
        # Prefer first store's files as the primary path reference
        path = None
        meta_path = None
        kinds = sorted(self.stores.keys())
        if kinds:
            kind = kinds[0]
            prefix = f"{kind}_"
            path = str(Path(self.directory) / f"{prefix}tracklets.npz")
            meta_path = str(Path(self.directory) / f"{prefix}tracklets_meta.json")
        return TracksRef(
            path=path,
            meta_path=meta_path,
            analysis_start_frame=int(self.analysis_start_frame),
        )


def discover_tracklet_kinds(directory: str | Path) -> List[str]:
    """Find animal kinds that have ``{kind}_tracklets_meta.json`` in directory."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    kinds: List[str] = []
    for p in directory.glob("*_tracklets_meta.json"):
        name = p.name
        # strip suffix
        if name.endswith("_tracklets_meta.json"):
            kind = name[: -len("_tracklets_meta.json")]
            if kind:
                kinds.append(kind)
    return sorted(kinds)


def infer_analysis_start_frame(
    meta: Dict[str, Any],
    n_track_frames: int,
    video_total_frames: Optional[int] = None,
) -> int:
    """Best-effort analysis→video offset.

    Prefer 0 when tracklet length matches the video (typical processed clips).
    Otherwise use round(start_t * fps) when available.
    """
    if video_total_frames is not None and video_total_frames > 0:
        if n_track_frames >= video_total_frames - 1:
            return 0
    fps = meta.get("fps")
    start_t = meta.get("start_t")
    if fps and start_t is not None:
        try:
            return max(0, int(round(float(start_t) * float(fps))))
        except (TypeError, ValueError):
            pass
    return 0


def load_tracklets_for_annotator(
    directory: str | Path,
    animal_kinds: Optional[Sequence[str]] = None,
    analysis_start_frame: Optional[int] = None,
    video_total_frames: Optional[int] = None,
) -> LoadedTracklets:
    """Load one or more tracklet stores and build Subject list.

    subject_id is assigned as the track id when a single animal kind is present.
    With multiple kinds, subjects get sequential unique ids while still storing
    the original track id in ``subject_to_track``.
    """
    from LabGym.id_review.tracklets import load_tracklets, save_tracklets
    from LabGym.id_review.apply import (
        apply_decisions_to_store,
        load_decisions,
        read_tracklets_identity_status,
        write_tracklets_identity_status,
    )
    from LabGym.id_review.dataset import load_switches, switches_to_decisions

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Tracklet directory not found: {directory}")

    kinds = list(animal_kinds) if animal_kinds else discover_tracklet_kinds(directory)
    if not kinds:
        raise FileNotFoundError(
            f"No *_tracklets_meta.json found in {directory}"
        )

    stores = {}
    for kind in kinds:
        stores[kind] = load_tracklets(str(directory), kind)

    # Ensure ID remaps from review are reflected in geometry.
    # New analysis runs re-save corrected npz; older packs only had switches on disk.
    status = read_tracklets_identity_status(str(directory))
    if not status.get("corrected"):
        markers = load_switches(str(directory))
        if markers:
            decisions = switches_to_decisions(markers)
        else:
            decisions = load_decisions(
                str(directory / "decisions.jsonl")
            )
        if decisions:
            n_total = 0
            for kind, store in stores.items():
                n_total += apply_decisions_to_store(store, decisions, animal_kind=kind)
                try:
                    save_tracklets(store, str(directory))
                except Exception:
                    pass
            if n_total > 0:
                write_tracklets_identity_status(
                    str(directory),
                    corrected=True,
                    n_decisions=n_total,
                    source="annotator_lazy_apply",
                )

    # Infer start frame from first store meta if not provided
    first = stores[kinds[0]]
    if analysis_start_frame is None:
        analysis_start_frame = infer_analysis_start_frame(
            first.meta or {},
            first.n_frames,
            video_total_frames=video_total_frames,
        )

    subjects: List[Subject] = []
    subject_to_track: Dict[int, Tuple[str, int]] = {}
    multi_kind = len(kinds) > 1
    next_sid = 0

    for kind in kinds:
        store = stores[kind]
        for track_id in store.ids:
            if multi_kind:
                sid = next_sid
                next_sid += 1
                display = f"{kind}_{track_id}"
            else:
                sid = int(track_id)
                display = f"{kind}_{track_id}"
            color = subject_color_for_index(len(subjects))
            subjects.append(
                Subject(
                    subject_id=sid,
                    animal_kind=str(kind),
                    display_name=display,
                    color=color,
                )
            )
            subject_to_track[sid] = (str(kind), int(track_id))

    loaded = LoadedTracklets(
        directory=str(directory.resolve()),
        stores=stores,
        analysis_start_frame=int(analysis_start_frame),
        subjects=subjects,
        subject_to_track=subject_to_track,
    )
    # Merge experimental names/roles/colors from subjects.json when present
    try:
        from LabGym.identity.package import load_subjects, merge_subjects_into_loaded

        recs = load_subjects(directory)
        if recs:
            merge_subjects_into_loaded(loaded, recs)
    except Exception:
        pass
    return loaded


def video_to_analysis_frame(video_frame: int, analysis_start_frame: int) -> int:
    return int(video_frame) - int(analysis_start_frame)


def analysis_to_video_frame(analysis_frame: int, analysis_start_frame: int) -> int:
    return int(analysis_frame) + int(analysis_start_frame)


def overlays_at_video_frame(
    loaded: LoadedTracklets,
    video_frame: int,
    subject_colors: Optional[Dict[int, str]] = None,
) -> List[TrackOverlay]:
    """Return track overlays for all subjects at the given video frame."""
    a_frame = video_to_analysis_frame(video_frame, loaded.analysis_start_frame)
    overlays: List[TrackOverlay] = []
    for subj in loaded.subjects:
        kind, track_id = loaded.subject_to_track[subj.subject_id]
        store = loaded.stores.get(kind)
        if store is None:
            continue
        if a_frame < 0 or a_frame >= store.n_frames:
            overlays.append(
                TrackOverlay(
                    subject_id=subj.subject_id,
                    animal_kind=kind,
                    center=None,
                    contour=None,
                    color=(subject_colors or {}).get(subj.subject_id, subj.color),
                    valid=False,
                )
            )
            continue
        try:
            row = store.id_index(track_id)
        except ValueError:
            continue
        valid = bool(store.valid[row, a_frame])
        center = None
        contour = None
        if valid:
            c = store.centers[row, a_frame]
            center = (float(c[0]), float(c[1]))
            cnt = store.contours[row][a_frame]
            if cnt is not None:
                contour = np.asarray(cnt)
        color = (subject_colors or {}).get(subj.subject_id, subj.color)
        overlays.append(
            TrackOverlay(
                subject_id=subj.subject_id,
                animal_kind=kind,
                center=center,
                contour=contour,
                color=color,
                valid=valid and center is not None,
            )
        )
    return overlays


def apply_subjects_to_session(session, loaded: LoadedTracklets) -> None:
    """Replace session subjects from loaded tracklets; preserve matching bout maps."""
    from LabGym.annotator.core.data_models import AnnotationSession

    assert isinstance(session, AnnotationSession)
    old_bouts = dict(session.bouts)
    session.subjects = list(loaded.subjects)
    session.tracks_ref = loaded.tracks_ref()
    # Rebuild bout maps for new subjects, keep data for same subject_id keys
    new_bouts: Dict[str, dict] = {}
    names = [b.name for b in session.behaviors]
    for subj in session.subjects:
        key = str(subj.subject_id)
        if key in old_bouts:
            bmap = old_bouts[key]
            for n in names:
                bmap.setdefault(n, [])
            new_bouts[key] = bmap
        else:
            new_bouts[key] = {n: [] for n in names}
    session.bouts = new_bouts
    if session.subjects:
        if session.get_subject(session.active_subject_id) is None:
            session.active_subject_id = session.subjects[0].subject_id
    session._ensure_bout_maps()


def try_autoload_id_review(
    video_path: str | Path,
    video_total_frames: Optional[int] = None,
) -> Optional[LoadedTracklets]:
    """Search common sibling locations for an id_review folder with tracklets."""
    video_path = Path(video_path)
    candidates = [
        video_path.parent / "id_review",
        video_path.with_suffix("") / "id_review",
        video_path.parent / f"{video_path.stem}" / "id_review",
        video_path.parent / f"{video_path.stem}_processed" / "id_review",
    ]
    # Also: results folder named like the video next to it
    for c in candidates:
        if c.is_dir() and discover_tracklet_kinds(c):
            try:
                return load_tracklets_for_annotator(
                    c, video_total_frames=video_total_frames
                )
            except Exception:
                continue
    return None
