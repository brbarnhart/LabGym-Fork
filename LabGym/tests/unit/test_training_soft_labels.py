"""Tests for soft labels, parsing, and subject-aware sorting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from LabGym.annotator.core.annotation_manager import AnnotationManager
from LabGym.annotator.core.data_models import (
    AnnotationSession,
    Behavior,
    Bout,
    Subject,
)
from LabGym.training.example_sort import (
    parse_labgym_example_basename,
    sort_examples_from_annotations,
    sort_examples_from_csv_subject_aware,
)
from LabGym.training.soft_labels import (
    LABEL_MODE_HARD_SOFT_AUX,
    SoftLabelTable,
    attach_soft_to_hard_labels,
    build_soft_targets_for_window,
    dense_frame_labels_from_session,
    write_soft_labels_sidecar,
)


def test_parse_labgym_example_basename():
    info = parse_labgym_example_basename("clip_mouse_0_1234_len15_x.avi")
    assert info["frame"] == 1234
    assert info["subject_id"] == 0
    assert info["animal_kind"] == "mouse"
    assert info["length"] == 15

    info2 = parse_labgym_example_basename("foo_1_99_len10.avi")
    assert info2["frame"] == 99
    assert info2["subject_id"] == 1


def test_build_soft_targets_occupancy():
    # 20 frames, class0 on 0-9, class1 on 10-19
    labels = np.zeros((20, 2), dtype=np.float32)
    labels[:10, 0] = 1
    labels[10:, 1] = 1
    hard, soft = build_soft_targets_for_window(
        labels, center_frame=14, window_len=10, classnames=["a", "b"], exclusive=True
    )
    # window [5..14]: 5 frames a, 5 frames b
    assert hard in ("a", "b")
    assert soft.shape == (2,)
    assert abs(float(soft.sum()) - 1.0) < 1e-5
    assert soft[0] == pytest.approx(0.5, abs=0.05)


def test_dense_and_soft_sidecar(tmp_path: Path):
    sess = AnnotationSession(
        video_path="v.avi",
        fps=10,
        total_frames=30,
        behaviors=[Behavior("grooming"), Behavior("rearing")],
        subjects=[Subject(0), Subject(1)],
    )
    mgr = AnnotationManager(sess)
    mgr.add_bout("grooming", 0, 14, subject_id=0)
    mgr.add_bout("rearing", 10, 25, subject_id=1)

    names, arr = dense_frame_labels_from_session(sess, subject_id=0)
    assert names == ["grooming", "rearing"]
    assert arr[5, 0] == 1
    assert arr[5, 1] == 0

    # fake examples
    (tmp_path / "vid_mouse_0_10_len15.avi").write_bytes(b"x")
    (tmp_path / "vid_mouse_0_10_len15.jpg").write_bytes(b"x")
    (tmp_path / "vid_mouse_1_20_len15.avi").write_bytes(b"x")
    (tmp_path / "vid_mouse_1_20_len15.jpg").write_bytes(b"x")

    path = write_soft_labels_sidecar(tmp_path, sess, window_len=15)
    assert path.is_file()
    table = SoftLabelTable.load_csv(path)
    assert len(table.rows) >= 1


def test_sort_from_annotations(tmp_path: Path):
    sess = AnnotationSession(
        video_path="v.avi",
        fps=10,
        total_frames=50,
        behaviors=[Behavior("approach"), Behavior("fight")],
        subjects=[Subject(0), Subject(1)],
    )
    mgr = AnnotationManager(sess)
    mgr.add_bout("approach", 0, 20, subject_id=0)
    mgr.add_bout("fight", 0, 20, subject_id=1)

    ann = tmp_path / "sess.annotations.json"
    mgr.save_to_json(ann)

    ex = tmp_path / "examples"
    ex.mkdir()
    out = tmp_path / "sorted"
    # two examples
    (ex / "v_mouse_0_10_len15.avi").write_bytes(b"a")
    (ex / "v_mouse_0_10_len15.jpg").write_bytes(b"a")
    (ex / "v_mouse_1_10_len15.avi").write_bytes(b"b")
    (ex / "v_mouse_1_10_len15.jpg").write_bytes(b"b")

    counts = sort_examples_from_annotations(ann, ex, out, copy=True)
    assert counts.get("approach", 0) >= 1
    assert counts.get("fight", 0) >= 1
    assert (out / "approach").is_dir()
    assert (out / "fight").is_dir()


def test_sort_csv_subject_aware(tmp_path: Path):
    ex = tmp_path / "ex"
    ex.mkdir()
    out = tmp_path / "out"
    df = pd.DataFrame(
        {
            "frame": [10, 10],
            "subject_id": [0, 1],
            "grooming": [1, 0],
            "rearing": [0, 1],
        }
    )
    df.to_csv(ex / "frame_labels_all_subjects.csv", index=False)
    (ex / "clip_mouse_0_10_len15.avi").write_bytes(b"a")
    (ex / "clip_mouse_0_10_len15.jpg").write_bytes(b"a")
    (ex / "clip_mouse_1_10_len15.avi").write_bytes(b"b")
    (ex / "clip_mouse_1_10_len15.jpg").write_bytes(b"b")

    counts = sort_examples_from_csv_subject_aware(ex, out, copy=True)
    assert counts.get("grooming", 0) == 1
    assert counts.get("rearing", 0) == 1


def test_attach_soft_stack():
    hard = np.eye(3, dtype=np.float32)
    soft = np.full((3, 3), 1 / 3, dtype=np.float32)
    stacked = attach_soft_to_hard_labels(hard, soft)
    assert stacked.shape == (3, 6)


def test_make_label_loss_hard_only_string():
    from LabGym.training.losses import make_label_loss

    assert make_label_loss("hard_only", binary=False) == "categorical_crossentropy"
    assert make_label_loss("hard_only", binary=True) == "binary_crossentropy"
