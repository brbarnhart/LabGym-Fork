"""Unit tests for LabGym annotator schema v2 and bout management."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from LabGym.annotator.core.annotation_manager import AnnotationManager
from LabGym.annotator.core.data_models import (
    SCHEMA_VERSION,
    AnnotationSession,
    Behavior,
    Bout,
    Subject,
)


def test_v1_migration_to_subject_zero(tmp_path: Path):
    v1 = {
        "video_path": "clip.avi",
        "fps": 30.0,
        "total_frames": 100,
        "behaviors": [
            {"name": "grooming", "color": "#00AAFF", "hotkey": "1"},
            {"name": "rearing", "color": "#FFAA00", "hotkey": "2"},
        ],
        "bouts": {
            "grooming": [{"start_frame": 10, "end_frame": 20}],
            "rearing": [],
        },
        "exclusive_mode": True,
    }
    path = tmp_path / "v1.json"
    path.write_text(json.dumps(v1), encoding="utf-8")

    mgr = AnnotationManager.load_from_json(path)
    assert mgr.session.schema_version == SCHEMA_VERSION
    assert len(mgr.session.subjects) == 1
    assert mgr.session.subjects[0].subject_id == 0
    assert mgr.session.active_subject_id == 0
    bouts = mgr.get_bouts_for_behavior("grooming")
    assert len(bouts) == 1
    assert bouts[0].start_frame == 10 and bouts[0].end_frame == 20
    assert mgr.session.exclusive_mode is True


def test_v2_roundtrip_multi_subject(tmp_path: Path):
    sess = AnnotationSession(
        video_path="vid.mp4",
        fps=10.0,
        total_frames=50,
        behaviors=[Behavior("approach", "#00aaff", "1")],
        subjects=[
            Subject(subject_id=0, animal_kind="mouse", display_name="m0"),
            Subject(subject_id=1, animal_kind="mouse", display_name="m1"),
        ],
        exclusive_mode=True,
        behavior_mode=0,
        active_subject_id=0,
    )
    sess.add_bout("approach", Bout(5, 15), subject_id=0)
    sess.add_bout("approach", Bout(20, 25), subject_id=1)

    path = tmp_path / "v2.json"
    mgr = AnnotationManager(sess)
    mgr.save_to_json(path)

    loaded = AnnotationManager.load_from_json(path)
    assert loaded.session.schema_version == SCHEMA_VERSION
    assert len(loaded.session.subjects) == 2
    assert len(loaded.get_bouts_for_behavior("approach", subject_id=0)) == 1
    assert len(loaded.get_bouts_for_behavior("approach", subject_id=1)) == 1
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "0" in raw["bouts"] and "1" in raw["bouts"]
    assert raw["schema_version"] == SCHEMA_VERSION


def test_toggle_bout_and_overlap():
    sess = AnnotationSession(
        video_path="x.avi",
        fps=30.0,
        total_frames=100,
        behaviors=[Behavior("grooming", hotkey="1"), Behavior("rearing", hotkey="2")],
    )
    mgr = AnnotationManager(sess)
    action, bout = mgr.toggle_bout("grooming", 10)
    assert action == "started" and bout is None
    action, bout = mgr.toggle_bout("grooming", 20)
    assert action == "closed"
    assert bout is not None
    assert bout.start_frame == 10 and bout.end_frame == 20

    with pytest.raises(ValueError, match="Overlap"):
        mgr.add_bout("grooming", 15, 18)


def test_exclusive_mode_switches_close_previous():
    sess = AnnotationSession(
        video_path="x.avi",
        fps=30.0,
        total_frames=100,
        behaviors=[Behavior("a", hotkey="1"), Behavior("b", hotkey="2")],
        exclusive_mode=True,
    )
    mgr = AnnotationManager(sess)
    mgr.set_exclusive_mode(True)
    mgr.toggle_bout("a", 5)
    mgr.toggle_bout("b", 10)  # should auto-close a at frame 9
    assert mgr.is_behavior_active("b")
    assert not mgr.is_behavior_active("a")
    a_bouts = mgr.get_bouts_for_behavior("a")
    assert len(a_bouts) == 1
    assert a_bouts[0].end_frame == 9


def test_per_subject_independence():
    sess = AnnotationSession(
        video_path="x.avi",
        fps=30.0,
        total_frames=100,
        behaviors=[Behavior("grooming", hotkey="1")],
        subjects=[
            Subject(0, display_name="s0"),
            Subject(1, display_name="s1"),
        ],
    )
    mgr = AnnotationManager(sess)
    mgr.set_active_subject(0)
    mgr.add_bout("grooming", 0, 10)
    mgr.set_active_subject(1)
    mgr.add_bout("grooming", 0, 10)  # same frames, different subject OK
    assert len(mgr.get_bouts_for_behavior("grooming", subject_id=0)) == 1
    assert len(mgr.get_bouts_for_behavior("grooming", subject_id=1)) == 1


def test_undo_restores_bouts():
    sess = AnnotationSession(
        video_path="x.avi",
        fps=30.0,
        total_frames=100,
        behaviors=[Behavior("grooming")],
    )
    mgr = AnnotationManager(sess)
    mgr.add_bout("grooming", 1, 5)
    assert not mgr.is_empty()
    assert mgr.can_undo()
    mgr.undo()
    assert mgr.is_empty()


def test_annotated_at_frame_active_subject():
    sess = AnnotationSession(
        video_path="x.avi",
        fps=30.0,
        total_frames=100,
        behaviors=[Behavior("grooming")],
        subjects=[Subject(0), Subject(1)],
    )
    mgr = AnnotationManager(sess)
    mgr.add_bout("grooming", 10, 20, subject_id=0)
    mgr.set_active_subject(0)
    assert "grooming" in mgr.get_annotated_behaviors_at_frame(15)
    mgr.set_active_subject(1)
    assert mgr.get_annotated_behaviors_at_frame(15) == []
