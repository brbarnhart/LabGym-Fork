"""Soft label construction from dense frame-by-frame ethograms.

LabGym training examples are short windows of length ``time_step`` ending
(or centered) at a center frame. Soft targets ``q`` are class occupancy
fractions over that window, optionally edge-smoothed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

LABEL_MODE_HARD_ONLY = "hard_only"
LABEL_MODE_HARD_SOFT_AUX = "hard_soft_aux"
LABEL_MODE_SOFT_PRIMARY = "soft_primary"

DEFAULT_LABEL_MODE = LABEL_MODE_HARD_SOFT_AUX
DEFAULT_LAMBDA_SOFT = 0.4


@dataclass
class SoftLabelTable:
    """Soft targets keyed by example basename (without extension)."""

    classnames: List[str]
    # basename -> (hard_label, soft_vector)
    rows: Dict[str, Tuple[str, np.ndarray]]

    def soft_matrix(self, basenames: Sequence[str], classnames: Optional[Sequence[str]] = None) -> np.ndarray:
        names = list(classnames) if classnames is not None else self.classnames
        name_to_i = {n: i for i, n in enumerate(names)}
        out = np.zeros((len(basenames), len(names)), dtype=np.float32)
        for r, base in enumerate(basenames):
            if base not in self.rows:
                continue
            hard, soft = self.rows[base]
            # Reorder if class order differs
            if list(names) == self.classnames:
                out[r] = soft
            else:
                for i, n in enumerate(self.classnames):
                    if n in name_to_i:
                        out[r, name_to_i[n]] = soft[i]
        return out

    def to_dataframe(self) -> pd.DataFrame:
        records = []
        for base, (hard, soft) in sorted(self.rows.items()):
            row = {"basename": base, "hard_label": hard}
            for i, c in enumerate(self.classnames):
                row[f"q_{c}"] = float(soft[i])
            records.append(row)
        return pd.DataFrame(records)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "SoftLabelTable":
        q_cols = [c for c in df.columns if c.startswith("q_")]
        classnames = [c[2:] for c in q_cols]
        rows: Dict[str, Tuple[str, np.ndarray]] = {}
        for _, r in df.iterrows():
            base = str(r["basename"])
            hard = str(r.get("hard_label", ""))
            soft = np.array([float(r[c]) for c in q_cols], dtype=np.float32)
            rows[base] = (hard, soft)
        return cls(classnames=classnames, rows=rows)

    def save_csv(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        return path

    @classmethod
    def load_csv(cls, path: Union[str, Path]) -> "SoftLabelTable":
        return cls.from_dataframe(pd.read_csv(path))


def _edge_smooth_1d(labels: np.ndarray, edge_k: int) -> np.ndarray:
    """labels: (T, C) multi-hot floats. Linear ramp near bout edges per class."""
    if edge_k <= 0:
        return labels.astype(np.float32)
    T, C = labels.shape
    out = labels.astype(np.float32).copy()
    for c in range(C):
        col = labels[:, c]
        # Find transitions
        for t in range(T):
            if col[t] < 0.5:
                continue
            # distance to nearest outside frame
            left = t
            while left > 0 and col[left - 1] >= 0.5:
                left -= 1
            right = t
            while right < T - 1 and col[right + 1] >= 0.5:
                right += 1
            dist_edge = min(t - left, right - t)
            if dist_edge < edge_k:
                out[t, c] = (dist_edge + 1) / float(edge_k + 1)
    return out


def build_soft_targets_for_window(
    frame_labels: np.ndarray,
    center_frame: int,
    window_len: int,
    *,
    classnames: Sequence[str],
    exclusive: bool = True,
    edge_smooth: int = 0,
    end_aligned: bool = True,
) -> Tuple[str, np.ndarray]:
    """Compute hard label + soft vector for one training window.

    Parameters
    ----------
    frame_labels : (total_frames, C) multi-hot 0/1
    center_frame : frame index associated with the example (LabGym convention)
    window_len : time_step / animation length
    exclusive : if True, normalize soft mass to sum to 1 (or all-zero if empty)
    edge_smooth : ramp width in frames at bout boundaries
    end_aligned : if True, window is [center-window_len+1, center]; else centered
    """
    T, C = frame_labels.shape
    if window_len < 1:
        window_len = 1
    if end_aligned:
        start = center_frame - window_len + 1
        end = center_frame
    else:
        half = window_len // 2
        start = center_frame - half
        end = start + window_len - 1
    start = max(0, start)
    end = min(T - 1, end)
    if end < start:
        soft = np.zeros(C, dtype=np.float32)
        return ("", soft)

    window = frame_labels[start : end + 1].astype(np.float32)
    if edge_smooth > 0:
        window = _edge_smooth_1d(window, edge_smooth)

    occupancy = window.mean(axis=0)
    if exclusive:
        s = float(occupancy.sum())
        if s > 1e-8:
            soft = (occupancy / s).astype(np.float32)
        else:
            soft = occupancy.astype(np.float32)
    else:
        soft = occupancy.astype(np.float32)

    # Hard label: center frame class if exclusive; else argmax of soft
    if 0 <= center_frame < T:
        center_row = frame_labels[center_frame]
        active = np.where(center_row >= 0.5)[0]
        if len(active) == 1:
            hard = classnames[int(active[0])]
        elif len(active) > 1:
            hard = classnames[int(active[np.argmax(soft[active])])]
        else:
            # fall back to soft argmax if any mass
            if soft.sum() > 1e-8:
                hard = classnames[int(np.argmax(soft))]
            else:
                hard = ""
    else:
        hard = classnames[int(np.argmax(soft))] if soft.sum() > 1e-8 else ""

    return hard, soft


def dense_frame_labels_from_session(
    session,
    subject_id: Optional[int] = None,
    *,
    use_group: Optional[bool] = None,
) -> Tuple[List[str], np.ndarray]:
    """Return (classnames, labels) with labels shape (total_frames, C)."""
    from LabGym.annotator.core.data_models import BEHAVIOR_MODE_INTERACTIVE_BASIC

    classnames = [b.name for b in session.behaviors]
    total = int(session.total_frames)
    C = len(classnames)
    labels = np.zeros((total, C), dtype=np.float32)
    name_to_i = {n: i for i, n in enumerate(classnames)}

    if use_group is None:
        use_group = int(session.behavior_mode) == BEHAVIOR_MODE_INTERACTIVE_BASIC

    if use_group:
        bmap = session.interaction_bouts.get("group", {})
    else:
        sid = session.active_subject_id if subject_id is None else subject_id
        bmap = session.bouts_for_subject(sid)

    for name, blist in bmap.items():
        if name not in name_to_i:
            continue
        ci = name_to_i[name]
        for bout in blist:
            a = max(0, int(bout.start_frame))
            b = min(total - 1, int(bout.end_frame))
            if b >= a:
                labels[a : b + 1, ci] = 1.0
    return classnames, labels


def write_soft_labels_sidecar(
    examples_dir: Union[str, Path],
    session,
    *,
    window_len: int = 15,
    edge_smooth: int = 2,
    exclusive: bool = True,
    end_aligned: bool = True,
    filename: str = "soft_labels.csv",
) -> Path:
    """Build soft_labels.csv for LabGym examples in *examples_dir* using session ethograms.

    Matches examples by parsing basename for subject id + center frame when possible;
    falls back to frame-only matching against the active/group ethogram.
    """
    from LabGym.training.example_sort import parse_labgym_example_basename

    examples_dir = Path(examples_dir)
    classnames = [b.name for b in session.behaviors]
    # Precompute dense labels per subject + group
    per_subject: Dict[Optional[int], np.ndarray] = {}
    for subj in session.subjects:
        _, arr = dense_frame_labels_from_session(session, subject_id=subj.subject_id, use_group=False)
        per_subject[int(subj.subject_id)] = arr
    _, group_arr = dense_frame_labels_from_session(session, use_group=True)

    rows: Dict[str, Tuple[str, np.ndarray]] = {}
    for path in sorted(examples_dir.iterdir()):
        if path.suffix.lower() not in (".avi", ".jpg", ".mp4"):
            continue
        base = path.stem
        # Prefer .avi basenames once
        if path.suffix.lower() == ".jpg" and (examples_dir / f"{base}.avi").exists():
            continue
        info = parse_labgym_example_basename(path.name)
        frame = info.get("frame")
        subject_id = info.get("subject_id")
        if frame is None:
            continue
        if subject_id is not None and subject_id in per_subject:
            fl = per_subject[subject_id]
        elif int(getattr(session, "behavior_mode", 0)) == 1:
            fl = group_arr
        elif subject_id is None and session.active_subject_id in per_subject:
            fl = per_subject[int(session.active_subject_id)]
        else:
            fl = next(iter(per_subject.values()), group_arr)

        hard, soft = build_soft_targets_for_window(
            fl,
            int(frame),
            int(window_len),
            classnames=classnames,
            exclusive=exclusive,
            edge_smooth=edge_smooth,
            end_aligned=end_aligned,
        )
        rows[base] = (hard, soft)

    table = SoftLabelTable(classnames=classnames, rows=rows)
    return table.save_csv(examples_dir / filename)


def attach_soft_to_hard_labels(
    hard_onehot: np.ndarray,
    soft: np.ndarray,
) -> np.ndarray:
    """Stack hard and soft along last axis for custom loss: (N, 2C)."""
    hard_onehot = np.asarray(hard_onehot, dtype=np.float32)
    soft = np.asarray(soft, dtype=np.float32)
    if hard_onehot.ndim == 1:
        hard_onehot = hard_onehot.reshape(-1, 1)
    if soft.shape != hard_onehot.shape:
        raise ValueError(f"soft shape {soft.shape} != hard shape {hard_onehot.shape}")
    return np.concatenate([hard_onehot, soft], axis=-1)
