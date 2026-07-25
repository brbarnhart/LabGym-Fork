"""Subject-aware sorting of LabGym behavior examples from annotations / CSVs."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# LabGym detector examples often look like:
#   video_mouse_0_1234_len15....avi
#   video_0_1234_len15.avi
# Frame is the token immediately before _len
_LEN_RE = re.compile(r"_(\d+)_len(\d+)", re.IGNORECASE)
# Optional animal_kind + id before frame: _kind_id_frame_len or _id_frame_len
_KIND_ID_FRAME_RE = re.compile(
    r"_([A-Za-z][A-Za-z0-9]*)_(\d+)_(\d+)_len\d+", re.IGNORECASE
)
_ID_FRAME_RE = re.compile(r"_(\d+)_(\d+)_len\d+", re.IGNORECASE)


def parse_labgym_example_basename(filename: str) -> Dict[str, Any]:
    """Parse LabGym example filename for frame, length, subject/track id.

    Returns dict with keys: frame, length, subject_id, animal_kind (any may be None).
    """
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    out: Dict[str, Any] = {
        "frame": None,
        "length": None,
        "subject_id": None,
        "animal_kind": None,
        "basename": stem,
    }
    m = _LEN_RE.search(stem)
    if m:
        out["frame"] = int(m.group(1))
        out["length"] = int(m.group(2))

    m2 = _KIND_ID_FRAME_RE.search(stem)
    if m2:
        out["animal_kind"] = m2.group(1)
        out["subject_id"] = int(m2.group(2))
        # Prefer frame from this match if present
        out["frame"] = int(m2.group(3))
        return out

    m3 = _ID_FRAME_RE.search(stem)
    if m3:
        out["subject_id"] = int(m3.group(1))
        out["frame"] = int(m3.group(2))
    return out


def _behavior_at_frame(
    annotation_df: pd.DataFrame,
    frame: int,
    subject_id: Optional[int] = None,
) -> List[str]:
    """Return behavior names with score==1 at frame (optionally for subject_id)."""
    df = annotation_df
    if "subject_id" in df.columns and subject_id is not None:
        rows = df[(df["frame"] == frame) & (df["subject_id"] == subject_id)]
    elif "frame" in df.columns:
        rows = df[df["frame"] == frame]
    else:
        # index is frame
        if frame in df.index:
            rows = df.loc[[frame]]
        else:
            return []
    if rows.empty:
        return []
    row = rows.iloc[0]
    skip = {"frame", "subject_id", "Unnamed: 0"}
    hits = []
    for col, val in row.items():
        if str(col) in skip:
            continue
        try:
            if float(val) == 1.0:
                hits.append(str(col))
        except (TypeError, ValueError):
            continue
    return hits


def sort_examples_from_csv_subject_aware(
    path_to_examples: Union[str, Path],
    out_path: Union[str, Path],
    csv_path: Optional[Union[str, Path]] = None,
    *,
    copy: bool = False,
) -> Dict[str, int]:
    """Sort LabGym examples using a frame_labels CSV (optionally multi-subject).

    Supports:
    - classic: columns frame, beh1, beh2, ... (frame-global)
    - multi-subject: frame, subject_id, beh1, ...
    - multi-file: frame_labels_subject{N}.csv in the examples folder

    Returns counts moved per behavior.
    """
    path_to_examples = Path(path_to_examples)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    animations = [
        p for p in path_to_examples.iterdir() if p.suffix.lower() == ".avi"
    ]
    if not animations:
        print("No behavior example .avi files found!")
        return {}

    # Load annotation table(s)
    frames_df: Optional[pd.DataFrame] = None
    if csv_path is not None:
        frames_df = pd.read_csv(csv_path)
    else:
        # Prefer all-subjects combined
        candidates = [
            path_to_examples / "frame_labels_all_subjects.csv",
            path_to_examples / "frame_labels.csv",
        ]
        for c in candidates:
            if c.is_file():
                frames_df = pd.read_csv(c)
                break
        if frames_df is None:
            # merge per-subject files
            parts = []
            for c in sorted(path_to_examples.glob("frame_labels_subject*.csv")):
                part = pd.read_csv(c)
                m = re.search(r"subject(\d+)", c.name)
                if m and "subject_id" not in part.columns:
                    part.insert(1, "subject_id", int(m.group(1)))
                parts.append(part)
            if parts:
                frames_df = pd.concat(parts, ignore_index=True)

    if frames_df is None:
        print("No .csv frame labels found!")
        return {}

    # Ensure frame column
    if "frame" not in frames_df.columns and "Unnamed: 0" in frames_df.columns:
        frames_df = frames_df.rename(columns={"Unnamed: 0": "frame"})

    behavior_cols = [
        c
        for c in frames_df.columns
        if c not in ("frame", "subject_id", "Unnamed: 0")
    ]
    for b in behavior_cols:
        os.makedirs(out_path / str(b), exist_ok=True)

    op = shutil.copy2 if copy else shutil.move
    counts: Dict[str, int] = {b: 0 for b in behavior_cols}

    for anim in animations:
        info = parse_labgym_example_basename(anim.name)
        frame = info.get("frame")
        if frame is None:
            continue
        subject_id = info.get("subject_id")
        behaviors = _behavior_at_frame(frames_df, int(frame), subject_id)
        if not behaviors:
            # fallback: ignore subject filter
            behaviors = _behavior_at_frame(frames_df, int(frame), None)
        pattern = anim.with_suffix(".jpg")
        for beh in behaviors:
            if beh not in counts:
                os.makedirs(out_path / beh, exist_ok=True)
                counts[beh] = 0
            dest_a = out_path / beh / anim.name
            if anim.exists():
                op(str(anim), str(dest_a))
                counts[beh] += 1
            if pattern.exists():
                op(str(pattern), str(out_path / beh / pattern.name))
            break  # one primary folder (first matching behavior)

    print("Subject-aware sorting completed!")
    return counts


def sort_examples_from_annotations(
    annotations_path: Union[str, Path],
    path_to_examples: Union[str, Path],
    out_path: Union[str, Path],
    *,
    copy: bool = False,
    exclusive: bool = True,
) -> Dict[str, int]:
    """Sort examples using a LabGym annotator session JSON (schema v1 or v2)."""
    from LabGym.annotator.core.data_models import (
        BEHAVIOR_MODE_INTERACTIVE_BASIC,
        AnnotationSession,
    )
    from LabGym.training.soft_labels import dense_frame_labels_from_session

    annotations_path = Path(annotations_path)
    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    session = AnnotationSession.from_dict(data)

    path_to_examples = Path(path_to_examples)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Precompute dense labels
    use_group = int(session.behavior_mode) == BEHAVIOR_MODE_INTERACTIVE_BASIC
    classnames = [b.name for b in session.behaviors]
    for b in classnames:
        os.makedirs(out_path / b, exist_ok=True)

    subject_arrays: Dict[Optional[int], Any] = {}
    if use_group:
        _, arr = dense_frame_labels_from_session(session, use_group=True)
        subject_arrays[None] = arr
    else:
        for subj in session.subjects:
            _, arr = dense_frame_labels_from_session(
                session, subject_id=subj.subject_id, use_group=False
            )
            subject_arrays[int(subj.subject_id)] = arr

    op = shutil.copy2 if copy else shutil.move
    counts: Dict[str, int] = {b: 0 for b in classnames}
    name_to_i = {n: i for i, n in enumerate(classnames)}

    animations = [
        p for p in path_to_examples.iterdir() if p.suffix.lower() == ".avi"
    ]
    for anim in animations:
        info = parse_labgym_example_basename(anim.name)
        frame = info.get("frame")
        if frame is None:
            continue
        frame = int(frame)
        sid = info.get("subject_id")
        if use_group:
            arr = subject_arrays[None]
        elif sid is not None and sid in subject_arrays:
            arr = subject_arrays[sid]
        elif session.active_subject_id in subject_arrays:
            arr = subject_arrays[int(session.active_subject_id)]
        else:
            arr = next(iter(subject_arrays.values()))

        if frame < 0 or frame >= arr.shape[0]:
            continue
        row = arr[frame]
        active = [classnames[i] for i, v in enumerate(row) if v >= 0.5]
        if not active:
            continue
        if exclusive and len(active) > 1:
            # prefer single: if exclusive session, shouldn't happen; take first
            active = [active[0]]
        beh = active[0]
        pattern = anim.with_suffix(".jpg")
        if anim.exists():
            op(str(anim), str(out_path / beh / anim.name))
            counts[beh] = counts.get(beh, 0) + 1
        if pattern.exists():
            op(str(pattern), str(out_path / beh / pattern.name))

    print("Annotation-based sorting completed!")
    return counts
