"""Tests for identity package (subjects.json + remapped tracklets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from LabGym.identity.package import (
    SUBJECTS_FILENAME,
    SubjectRecord,
    apply_decisions_and_save_tracklets,
    clone_store,
    load_subjects,
    save_subjects,
    subjects_from_track_ids,
)
from LabGym.id_review.apply import read_tracklets_identity_status
from LabGym.id_review.dataset import make_swap_marker, switches_to_decisions
from LabGym.id_review.tracklets import load_tracklets, save_tracklets
from LabGym.id_review.types import SCHEMA_VERSION, TrackletStore


def _store(n_frames=20, kind="mouse"):
    ids = [0, 1]
    centers = np.zeros((2, n_frames, 2), dtype=np.float64)
    valid = np.ones((2, n_frames), dtype=bool)
    heights = np.full((2, n_frames), 10.0)
    contours = []
    for row, tid in enumerate(ids):
        row_c = []
        for f in range(n_frames):
            centers[row, f] = [tid * 100 + f, tid * 10]
            x, y = centers[row, f]
            cnt = np.array(
                [
                    [[int(x), int(y)]],
                    [[int(x) + 5, int(y)]],
                    [[int(x) + 5, int(y) + 5]],
                    [[int(x), int(y) + 5]],
                ],
                dtype=np.int32,
            )
            row_c.append(cnt)
        contours.append(row_c)
    return TrackletStore(
        schema_version=SCHEMA_VERSION,
        animal_kind=kind,
        ids=ids,
        n_frames=n_frames,
        centers=centers,
        valid=valid,
        heights=heights,
        contours=contours,
        meta={"video": "x.mp4", "fps": 30},
    )


def test_subjects_roundtrip(tmp_path: Path):
    recs = [
        SubjectRecord(0, "mouse", "resident", "R", "#ff0000", track_id=0),
        SubjectRecord(1, "mouse", "intruder", "I", "#00ff00", track_id=1),
    ]
    save_subjects(tmp_path, recs)
    assert (tmp_path / SUBJECTS_FILENAME).is_file()
    loaded = load_subjects(tmp_path)
    assert len(loaded) == 2
    assert loaded[0].display_name == "resident"
    assert loaded[1].role == "I"
    assert loaded[0].color == "#ff0000"


def test_subjects_from_track_ids():
    recs = subjects_from_track_ids({"mouse": [0, 2]})
    assert [r.subject_id for r in recs] == [0, 2]
    assert recs[0].display_name == "mouse_0"


def test_apply_decisions_from_baseline(tmp_path: Path):
    store = _store(n_frames=15)
    save_tracklets(store, str(tmp_path))
    baseline = {"mouse": clone_store(store)}
    marker = make_swap_marker(8, "mouse", [0, 1], fps=30.0)
    decisions = switches_to_decisions([marker])
    n = apply_decisions_and_save_tracklets(
        tmp_path, decisions, baseline_stores=baseline
    )
    assert n >= 1
    status = read_tracklets_identity_status(str(tmp_path))
    assert status["corrected"] is True
    loaded = load_tracklets(str(tmp_path), "mouse")
    # after frame 8, ids swapped relative to original
    assert np.allclose(loaded.centers[0, 8:], store.centers[1, 8:])
    assert np.allclose(loaded.centers[1, 8:], store.centers[0, 8:])
    # before unchanged
    assert np.allclose(loaded.centers[0, :8], store.centers[0, :8])


def test_no_double_apply_when_using_baseline(tmp_path: Path):
    store = _store(n_frames=15)
    save_tracklets(store, str(tmp_path))
    baseline = {"mouse": clone_store(store)}
    marker = make_swap_marker(5, "mouse", [0, 1], fps=30.0)
    decisions = switches_to_decisions([marker])
    apply_decisions_and_save_tracklets(tmp_path, decisions, baseline_stores=baseline)
    # re-apply from same baseline → same result (not double)
    apply_decisions_and_save_tracklets(tmp_path, decisions, baseline_stores=baseline)
    loaded = load_tracklets(str(tmp_path), "mouse")
    assert np.allclose(loaded.centers[0, 5:], store.centers[1, 5:])


def test_merge_subjects_into_loaded(tmp_path: Path):
    from LabGym.annotator.core.tracklets_bridge import load_tracklets_for_annotator

    store = _store()
    save_tracklets(store, str(tmp_path))
    save_subjects(
        tmp_path,
        [
            SubjectRecord(0, "mouse", "Alice", "alpha", "#112233", track_id=0),
            SubjectRecord(1, "mouse", "Bob", "beta", "#445566", track_id=1),
        ],
    )
    loaded = load_tracklets_for_annotator(tmp_path)
    names = {s.subject_id: s.display_name for s in loaded.subjects}
    assert names[0] == "Alice"
    assert names[1] == "Bob"
    colors = {s.subject_id: s.color for s in loaded.subjects}
    assert colors[0] == "#112233"
