"""Tests for identity package (subjects.json + remapped tracklets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from LabGym.identity.package import (
    SUBJECTS_FILENAME,
    SubjectRecord,
    apply_decisions_and_save_tracklets,
    load_subjects,
    migrate_uncorrected_public_to_raw,
    needs_uncorrected_raw_migrate,
    save_subjects,
    subjects_from_track_ids,
    switch_edits_allowed,
)
from LabGym.id_review.apply import (
    read_tracklets_identity_status,
    write_tracklets_identity_status,
)
from LabGym.id_review.dataset import make_swap_marker, switches_to_decisions
from LabGym.id_review.raw_store import has_accepted_identities, save_raw_tracklets
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


def test_empty_switch_save_publishes_remapped_equal_to_raw(tmp_path: Path):
    raw = _store(n_frames=12)
    save_raw_tracklets(tmp_path, {"mouse": raw})
    n = apply_decisions_and_save_tracklets(tmp_path, [])
    assert n == 0
    published = load_tracklets(str(tmp_path), "mouse")
    assert np.allclose(published.centers, raw.centers)
    assert np.array_equal(published.valid, raw.valid)
    status = read_tracklets_identity_status(str(tmp_path))
    assert status["accepted"] is True
    assert has_accepted_identities(tmp_path) is True


def test_second_save_same_switch_markers_does_not_double_apply(tmp_path: Path):
    raw = _store(n_frames=15)
    save_raw_tracklets(tmp_path, {"mouse": raw})
    marker = make_swap_marker(5, "mouse", [0, 1], fps=30.0)
    decisions = switches_to_decisions([marker])
    apply_decisions_and_save_tracklets(tmp_path, decisions)
    apply_decisions_and_save_tracklets(tmp_path, decisions)
    published = load_tracklets(str(tmp_path), "mouse")
    assert np.allclose(published.centers[0, 5:], raw.centers[1, 5:])
    assert np.allclose(published.centers[1, 5:], raw.centers[0, 5:])
    assert np.allclose(published.centers[0, :5], raw.centers[0, :5])


def test_publish_without_raw_refuses_and_does_not_use_public_files(tmp_path: Path):
    public = _store(n_frames=15)
    save_tracklets(public, str(tmp_path))
    before = load_tracklets(str(tmp_path), "mouse")
    marker = make_swap_marker(5, "mouse", [0, 1], fps=30.0)
    decisions = switches_to_decisions([marker])
    with pytest.raises(FileNotFoundError, match="raw"):
        apply_decisions_and_save_tracklets(tmp_path, decisions)
    after = load_tracklets(str(tmp_path), "mouse")
    assert np.allclose(after.centers, before.centers)
    status = read_tracklets_identity_status(str(tmp_path))
    assert status["accepted"] is False


def test_apply_decisions_from_raw_writes_swapped_geometry(tmp_path: Path):
    store = _store(n_frames=15)
    save_raw_tracklets(tmp_path, {"mouse": store})
    marker = make_swap_marker(8, "mouse", [0, 1], fps=30.0)
    decisions = switches_to_decisions([marker])
    n = apply_decisions_and_save_tracklets(tmp_path, decisions)
    assert n >= 1
    status = read_tracklets_identity_status(str(tmp_path))
    assert status["accepted"] is True
    loaded = load_tracklets(str(tmp_path), "mouse")
    assert np.allclose(loaded.centers[0, 8:], store.centers[1, 8:])
    assert np.allclose(loaded.centers[1, 8:], store.centers[0, 8:])
    assert np.allclose(loaded.centers[0, :8], store.centers[0, :8])


def test_analyzer_resave_publishes_empty_switch_accept(tmp_path: Path):
    from types import SimpleNamespace

    from LabGym.id_review.dataset import run_id_review_pipeline
    from LabGym.id_review.raw_store import load_raw_tracklets

    n = 8
    kind = "mouse"
    analyzer = SimpleNamespace(
        results_path=str(tmp_path / "clip"),
        animal_kinds=[kind],
        animal_centers={kind: {0: [(0.0, 0.0)] * n, 1: [(10.0, 0.0)] * n}},
        animal_heights={kind: {0: [8.0] * n, 1: [8.0] * n}},
        animal_contours={kind: {0: [None] * n, 1: [None] * n}},
        animal_area={kind: 10.0},
        fps=10,
        t=0,
        length=0,
        path_to_video="clip.avi",
        framewidth=None,
        frameheight=None,
        duration=0,
        all_time=list(range(n)),
    )
    out_dir, _events, decisions = run_id_review_pipeline(
        analyzer, extract_samples=False, auto_load_existing_decisions=False
    )
    assert decisions == []
    raw = load_raw_tracklets(out_dir)["mouse"]
    published = load_tracklets(out_dir, "mouse")
    assert np.allclose(published.centers, raw.centers)
    assert has_accepted_identities(out_dir) is True
    status = read_tracklets_identity_status(out_dir)
    assert status["accepted"] is True


def test_merge_subjects_into_loaded(tmp_path: Path):
    from LabGym.annotator.core.tracklets_bridge import load_tracklets_for_annotator

    store = _store()
    save_tracklets(store, str(tmp_path))
    from LabGym.id_review.apply import write_tracklets_identity_status

    write_tracklets_identity_status(str(tmp_path), corrected=True, accepted=True)
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


def test_switch_edits_not_allowed_without_raw(tmp_path: Path):
    save_tracklets(_store(), str(tmp_path))
    assert switch_edits_allowed(tmp_path) is False


def test_switch_edits_allowed_with_raw(tmp_path: Path):
    save_raw_tracklets(tmp_path, {"mouse": _store()})
    assert switch_edits_allowed(tmp_path) is True


def test_uncorrected_public_pack_offers_migrate_without_moving_files(tmp_path: Path):
    save_tracklets(_store(), str(tmp_path))
    write_tracklets_identity_status(str(tmp_path), corrected=False)
    public = tmp_path / "mouse_tracklets.npz"
    assert public.is_file()
    assert needs_uncorrected_raw_migrate(tmp_path) is True
    assert public.is_file()
    assert not (tmp_path / "raw" / "mouse_tracklets.npz").is_file()
    assert switch_edits_allowed(tmp_path) is False


def test_declining_migrate_leaves_files_and_status_unchanged(tmp_path: Path):
    save_tracklets(_store(), str(tmp_path))
    write_tracklets_identity_status(str(tmp_path), corrected=False, accepted=False)
    public = tmp_path / "mouse_tracklets.npz"
    meta = tmp_path / "mouse_tracklets_meta.json"
    public_bytes = public.read_bytes()
    meta_text = meta.read_text(encoding="utf-8")
    status_before = read_tracklets_identity_status(str(tmp_path))
    assert needs_uncorrected_raw_migrate(tmp_path) is True
    # Decline: do not call migrate. Pack must be unchanged.
    assert public.read_bytes() == public_bytes
    assert meta.read_text(encoding="utf-8") == meta_text
    assert read_tracklets_identity_status(str(tmp_path)) == status_before
    assert switch_edits_allowed(tmp_path) is False
    assert not (tmp_path / "raw" / "mouse_tracklets.npz").is_file()


def test_accepting_migrate_moves_public_tracklets_into_raw(tmp_path: Path):
    save_tracklets(_store(), str(tmp_path))
    write_tracklets_identity_status(str(tmp_path), corrected=False)
    assert migrate_uncorrected_public_to_raw(tmp_path) is True
    assert needs_uncorrected_raw_migrate(tmp_path) is False
    assert switch_edits_allowed(tmp_path) is True
    assert not (tmp_path / "mouse_tracklets.npz").is_file()
    assert (tmp_path / "raw" / "mouse_tracklets.npz").is_file()
    assert has_accepted_identities(tmp_path) is False


def test_accepted_pack_is_not_offered_migrate(tmp_path: Path):
    save_tracklets(_store(), str(tmp_path))
    write_tracklets_identity_status(
        str(tmp_path), corrected=True, accepted=True, n_decisions=1
    )
    public = tmp_path / "mouse_tracklets.npz"
    assert needs_uncorrected_raw_migrate(tmp_path) is False
    assert migrate_uncorrected_public_to_raw(tmp_path) is False
    assert public.is_file()
    assert not (tmp_path / "raw" / "mouse_tracklets.npz").is_file()
    assert switch_edits_allowed(tmp_path) is False


def test_legacy_corrected_pack_is_not_offered_migrate(tmp_path: Path):
    save_tracklets(_store(), str(tmp_path))
    write_tracklets_identity_status(str(tmp_path), corrected=True, n_decisions=2)
    public = tmp_path / "mouse_tracklets.npz"
    assert needs_uncorrected_raw_migrate(tmp_path) is False
    assert migrate_uncorrected_public_to_raw(tmp_path) is False
    assert public.is_file()
    assert not (tmp_path / "raw" / "mouse_tracklets.npz").is_file()
    assert switch_edits_allowed(tmp_path) is False


def test_names_and_roles_save_without_raw(tmp_path: Path):
    save_tracklets(_store(), str(tmp_path))
    write_tracklets_identity_status(str(tmp_path), corrected=True)
    assert switch_edits_allowed(tmp_path) is False
    recs = [
        SubjectRecord(0, "mouse", "resident", "R", "#ff0000", track_id=0),
        SubjectRecord(1, "mouse", "intruder", "I", "#00ff00", track_id=1),
    ]
    save_subjects(tmp_path, recs)
    loaded = load_subjects(tmp_path)
    assert loaded[0].display_name == "resident"
    assert loaded[1].role == "I"
