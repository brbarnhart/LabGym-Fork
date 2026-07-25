"""Tests for tracklets bridge and multi-subject annotation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from LabGym.annotator.core.annotation_manager import AnnotationManager
from LabGym.annotator.core.data_models import (
    BEHAVIOR_MODE_INTERACTIVE_BASIC,
    AnnotationSession,
    Behavior,
    Subject,
)
from LabGym.annotator.core.example_generator import ExampleGenerator
from LabGym.annotator.core.tracklets_bridge import (
    apply_subjects_to_session,
    discover_tracklet_kinds,
    infer_analysis_start_frame,
    load_tracklets_for_annotator,
    overlays_at_video_frame,
    video_to_analysis_frame,
)


FIXTURE_ID_REVIEW = (
    Path(__file__).resolve().parents[2]
    / "testing_ground"
    / "J5904-M-DCZ_processed"
    / "id_review"
)


@pytest.mark.skipif(not FIXTURE_ID_REVIEW.is_dir(), reason="id_review fixture missing")
def test_discover_and_load_tracklets():
    kinds = discover_tracklet_kinds(FIXTURE_ID_REVIEW)
    assert "mouse" in kinds
    loaded = load_tracklets_for_annotator(FIXTURE_ID_REVIEW, video_total_frames=2000)
    assert len(loaded.subjects) == 2
    assert loaded.subjects[0].subject_id == 0
    assert loaded.subjects[1].subject_id == 1
    assert loaded.analysis_start_frame == 0  # matches full clip length
    ov = overlays_at_video_frame(loaded, 0)
    assert len(ov) == 2
    assert any(o.valid and o.center is not None for o in ov)


def test_infer_start_frame_short_tracklets():
    meta = {"fps": 10, "start_t": 5.0}
    # tracklets shorter than video → use start_t * fps
    assert infer_analysis_start_frame(meta, n_track_frames=100, video_total_frames=1000) == 50
    # matching length → 0
    assert infer_analysis_start_frame(meta, n_track_frames=1000, video_total_frames=1000) == 0


def test_video_analysis_frame_roundtrip():
    assert video_to_analysis_frame(100, 10) == 90


def test_apply_subjects_preserves_matching_bouts():
    sess = AnnotationSession(
        video_path="x.avi",
        fps=10,
        total_frames=100,
        behaviors=[Behavior("grooming")],
        subjects=[Subject(0), Subject(1)],
    )
    mgr = AnnotationManager(sess)
    mgr.add_bout("grooming", 1, 5, subject_id=0)

    # Fake loaded tracklets structure without real files
    from LabGym.annotator.core.tracklets_bridge import LoadedTracklets

    loaded = LoadedTracklets(
        directory=".",
        stores={},
        analysis_start_frame=0,
        subjects=[
            Subject(0, display_name="mouse_0", color="#4FC3F7"),
            Subject(1, display_name="mouse_1", color="#FF8A65"),
        ],
        subject_to_track={0: ("mouse", 0), 1: ("mouse", 1)},
    )
    apply_subjects_to_session(sess, loaded)
    assert len(mgr.get_bouts_for_behavior("grooming", subject_id=0)) == 1
    assert mgr.get_bouts_for_behavior("grooming", subject_id=1) == []
    assert sess.tracks_ref is not None
    assert sess.tracks_ref.analysis_start_frame == 0


def test_interactive_basic_stores_group_bouts():
    sess = AnnotationSession(
        video_path="x.avi",
        fps=10,
        total_frames=100,
        behaviors=[Behavior("fight"), Behavior("approach")],
        behavior_mode=BEHAVIOR_MODE_INTERACTIVE_BASIC,
        exclusive_mode=True,
    )
    mgr = AnnotationManager(sess)
    assert mgr.uses_group_ethogram()
    mgr.toggle_bout("fight", 10)
    mgr.toggle_bout("fight", 20)
    assert len(mgr.get_bouts_for_behavior("fight")) == 1
    assert "group" in sess.interaction_bouts
    assert len(sess.interaction_bouts["group"]["fight"]) == 1
    # Per-subject maps should stay empty
    assert all(
        len(sess.bouts_for_subject(s.subject_id).get("fight", [])) == 0
        for s in sess.subjects
    )


def test_export_frame_labels_all_subjects(tmp_path: Path):
    sess = AnnotationSession(
        video_path="x.avi",
        fps=10,
        total_frames=30,
        behaviors=[Behavior("grooming"), Behavior("rearing")],
        subjects=[Subject(0), Subject(1)],
    )
    mgr = AnnotationManager(sess)
    mgr.add_bout("grooming", 0, 9, subject_id=0)
    mgr.add_bout("rearing", 10, 19, subject_id=1)
    gen = ExampleGenerator(sess, video_path="x.avi")
    paths = gen.export_frame_labels_all_subjects(tmp_path, combined=True)
    names = {p.name for p in paths}
    assert "frame_labels_subject0.csv" in names
    assert "frame_labels_subject1.csv" in names
    assert "frame_labels_all_subjects.csv" in names
    import pandas as pd

    all_df = pd.read_csv(tmp_path / "frame_labels_all_subjects.csv")
    assert "subject_id" in all_df.columns
    assert set(all_df["subject_id"].unique()) == {0, 1}
    s0 = all_df[all_df["subject_id"] == 0]
    assert int(s0.loc[s0["frame"] == 5, "grooming"].iloc[0]) == 1
    assert int(s0.loc[s0["frame"] == 5, "rearing"].iloc[0]) == 0
