"""Tests for ethogram-driven window sampling and example generation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from LabGym.annotator.core.annotation_manager import AnnotationManager
from LabGym.annotator.core.data_models import (
    BEHAVIOR_MODE_INTERACTIVE_BASIC,
    BEHAVIOR_MODE_NON_INTERACTIVE,
    AnnotationSession,
    Behavior,
    Bout,
    Subject,
)
from LabGym.annotator.core.tracklets_bridge import LoadedTracklets
from LabGym.id_review.types import TrackletStore, SCHEMA_VERSION
from LabGym.training.ethogram_examples import (
    GenerationConfig,
    collect_windows,
    generate_examples_from_ethogram,
    sample_windows_from_bout,
)


def test_sample_bout_end():
    bout = Bout(10, 40)
    specs = sample_windows_from_bout(
        bout, "grooming", 0, length=10, sampling="bout_end", stride=1,
        min_bout_frames=1, total_frames=100,
    )
    assert len(specs) == 1
    assert specs[0].center_frame == 40
    assert specs[0].start_frame == 31


def test_sample_dense_stride():
    bout = Bout(10, 40)
    specs = sample_windows_from_bout(
        bout, "grooming", 0, length=10, sampling="dense_in_bout", stride=10,
        min_bout_frames=1, total_frames=100,
    )
    centers = [s.center_frame for s in specs]
    assert centers[0] == 10  # max(b0, length-1)=10
    assert 40 in centers
    assert all(s.end_frame - s.start_frame + 1 == 10 for s in specs)


def test_sample_short_bout_min_filter():
    bout = Bout(5, 8)
    specs = sample_windows_from_bout(
        bout, "x", 0, length=10, sampling="bout_end", stride=1,
        min_bout_frames=5, total_frames=100,
    )
    assert specs == []


def test_collect_windows_mode0_vs_mode1():
    sess = AnnotationSession(
        video_path="v.avi",
        fps=10,
        total_frames=50,
        behaviors=[Behavior("approach"), Behavior("fight")],
        subjects=[Subject(0), Subject(1)],
        behavior_mode=BEHAVIOR_MODE_NON_INTERACTIVE,
    )
    mgr = AnnotationManager(sess)
    mgr.add_bout("approach", 10, 30, subject_id=0)
    mgr.add_bout("fight", 10, 30, subject_id=1)
    w0 = collect_windows(sess, length=5, sampling="bout_end", stride=1, min_bout_frames=1)
    assert len(w0) == 2
    assert {w.subject_id for w in w0} == {0, 1}

    sess.behavior_mode = BEHAVIOR_MODE_INTERACTIVE_BASIC
    sess.interaction_bouts = {
        "group": {"fight": [Bout(10, 30)]}
    }
    w1 = collect_windows(sess, length=5, sampling="bout_end", stride=1, min_bout_frames=1)
    assert len(w1) == 1
    assert w1[0].subject_id is None
    assert w1[0].behavior == "fight"


def _synthetic_loaded(n_frames=60, n_ids=1, h=64, w=64) -> LoadedTracklets:
    ids = list(range(n_ids))
    centers = np.zeros((n_ids, n_frames, 2), dtype=np.float64)
    valid = np.ones((n_ids, n_frames), dtype=bool)
    heights = np.full((n_ids, n_frames), 10.0)
    contours = []
    for i in range(n_ids):
        row = []
        for f in range(n_frames):
            cx, cy = 20 + i * 15, 20 + (f % 5)
            centers[i, f] = (cx, cy)
            # small square contour
            pts = np.array(
                [
                    [[cx - 5, cy - 5]],
                    [[cx + 5, cy - 5]],
                    [[cx + 5, cy + 5]],
                    [[cx - 5, cy + 5]],
                ],
                dtype=np.int32,
            )
            row.append(pts)
        contours.append(row)
    store = TrackletStore(
        schema_version=SCHEMA_VERSION,
        animal_kind="mouse",
        ids=ids,
        n_frames=n_frames,
        centers=centers,
        valid=valid,
        heights=heights,
        contours=contours,
        meta={"fps": 10},
    )
    subjects = [
        Subject(subject_id=i, animal_kind="mouse", display_name=f"mouse_{i}")
        for i in ids
    ]
    return LoadedTracklets(
        directory=".",
        stores={"mouse": store},
        analysis_start_frame=0,
        subjects=subjects,
        subject_to_track={i: ("mouse", i) for i in ids},
    )


def test_generate_mode0_writes_pairs(tmp_path: Path):
    n_frames = 40
    h, w = 48, 48
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h), True
    )
    for f in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[15:30, 15:30] = (200, 200, 200)
        writer.write(frame)
    writer.release()

    sess = AnnotationSession(
        video_path=str(video),
        fps=10.0,
        total_frames=n_frames,
        behaviors=[Behavior("grooming"), Behavior("other")],
        subjects=[Subject(0, animal_kind="mouse")],
        behavior_mode=BEHAVIOR_MODE_NON_INTERACTIVE,
        exclusive_mode=True,
    )
    mgr = AnnotationManager(sess)
    mgr.add_bout("grooming", 10, 30)

    loaded = _synthetic_loaded(n_frames=n_frames, n_ids=1, h=h, w=w)
    out = tmp_path / "examples"
    cfg = GenerationConfig(
        video_path=str(video),
        annotations_path=str(tmp_path / "a.json"),
        tracklets_dir=str(tmp_path),
        output_dir=str(out),
        length=5,
        sampling="bout_end",
        min_bout_frames=1,
        write_soft_labels=True,
        analysis_start_frame=0,
    )
    result = generate_examples_from_ethogram(
        cfg, session=sess, loaded_tracklets=loaded
    )
    assert result["written"] >= 1
    assert (out / "grooming").is_dir()
    avis = list((out / "grooming").glob("*.avi"))
    jpgs = list((out / "grooming").glob("*.jpg"))
    assert len(avis) == len(jpgs) >= 1
    assert (out / "generation_config.json").is_file()
    assert (out / "soft_labels.csv").is_file()
    # basename pattern includes len
    assert "len5" in avis[0].name
    assert "_0_" in avis[0].name or "mouse_0" in avis[0].name or "mouse" in avis[0].name
