"""Raw tracklets snapshot and accepted-identities status."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from LabGym.annotator.core.tracklets_bridge import discover_tracklet_kinds
from LabGym.id_review.apply import write_tracklets_identity_status
from LabGym.id_review.raw_store import (
    has_accepted_identities,
    has_raw_snapshot,
    load_raw_tracklets,
    save_raw_tracklets,
    snapshot_uncorrected_root_to_raw,
    unpublish_remapped_tracklets,
)
from LabGym.id_review.tracklets import save_tracklets
from LabGym.id_review.types import SCHEMA_VERSION, TrackletStore


def _store(n_frames: int = 8, kind: str = "mouse") -> TrackletStore:
    ids = [0, 1]
    centers = np.zeros((2, n_frames, 2), dtype=np.float64)
    valid = np.ones((2, n_frames), dtype=bool)
    heights = np.full((2, n_frames), 10.0)
    contours = []
    for row, tid in enumerate(ids):
        row_c = []
        for f in range(n_frames):
            centers[row, f] = [tid * 50.0 + f, float(tid)]
            x, y = centers[row, f]
            cnt = np.array(
                [
                    [[int(x), int(y)]],
                    [[int(x) + 4, int(y)]],
                    [[int(x) + 4, int(y) + 4]],
                    [[int(x), int(y) + 4]],
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
        meta={"video": "x.mp4", "fps": 10.0},
    )


def test_unsaved_detect_pack_is_not_accepted(tmp_path: Path):
    save_tracklets(_store(), str(tmp_path))
    write_tracklets_identity_status(str(tmp_path), corrected=False)
    assert has_accepted_identities(tmp_path) is False


def test_legacy_corrected_pack_is_accepted(tmp_path: Path):
    save_tracklets(_store(), str(tmp_path))
    write_tracklets_identity_status(str(tmp_path), corrected=True, n_decisions=1)
    assert has_accepted_identities(tmp_path) is True


def test_raw_snapshot_is_not_a_discovered_kind(tmp_path: Path):
    store = _store()
    save_raw_tracklets(tmp_path, {"mouse": store})
    assert discover_tracklet_kinds(tmp_path) == []
    loaded = load_raw_tracklets(tmp_path)
    assert list(loaded) == ["mouse"]
    assert loaded["mouse"].n_frames == 8
    assert has_accepted_identities(tmp_path) is False


def test_snapshot_moves_uncorrected_root_to_raw(tmp_path: Path):
    save_tracklets(_store(), str(tmp_path))
    write_tracklets_identity_status(str(tmp_path), corrected=False)
    snapshot_uncorrected_root_to_raw(tmp_path)
    assert has_raw_snapshot(tmp_path) is True
    assert discover_tracklet_kinds(tmp_path) == []
    assert has_accepted_identities(tmp_path) is False
    assert not (tmp_path / "mouse_tracklets.npz").is_file()
    assert (tmp_path / "raw" / "mouse_tracklets.npz").is_file()


def test_unpublish_removes_root_tracklets_not_raw(tmp_path: Path):
    save_raw_tracklets(tmp_path, {"mouse": _store()})
    save_tracklets(_store(), str(tmp_path))
    write_tracklets_identity_status(
        str(tmp_path), corrected=True, n_decisions=1, source="test"
    )
    unpublish_remapped_tracklets(tmp_path)
    assert has_raw_snapshot(tmp_path) is True
    assert discover_tracklet_kinds(tmp_path) == []
    assert has_accepted_identities(tmp_path) is False


def test_publish_from_raw_does_not_double_apply(tmp_path: Path):
    from LabGym.identity.package import apply_decisions_and_save_tracklets
    from LabGym.id_review.dataset import make_swap_marker, switches_to_decisions
    from LabGym.id_review.tracklets import load_tracklets

    raw = _store(n_frames=12)
    save_raw_tracklets(tmp_path, {"mouse": raw})
    marker = make_swap_marker(5, "mouse", [0, 1], fps=10.0)
    apply_decisions_and_save_tracklets(
        tmp_path, switches_to_decisions([marker])
    )
    apply_decisions_and_save_tracklets(
        tmp_path, switches_to_decisions([marker])
    )
    published = load_tracklets(str(tmp_path), "mouse")
    assert np.allclose(published.centers[0, 5:], raw.centers[1, 5:])
    assert np.allclose(published.centers[0, :5], raw.centers[0, :5])
    still_raw = load_raw_tracklets(tmp_path)["mouse"]
    assert np.allclose(still_raw.centers, raw.centers)
    assert has_accepted_identities(tmp_path) is True


def test_export_review_pack_writes_raw_not_published(tmp_path: Path):
    from types import SimpleNamespace

    from LabGym.id_review.dataset import export_review_pack

    n = 6
    kind = "mouse"
    centers = {0: [(0.0, 0.0)] * n, 1: [(10.0, 0.0)] * n}
    heights = {0: [8.0] * n, 1: [8.0] * n}
    contours = {0: [None] * n, 1: [None] * n}
    analyzer = SimpleNamespace(
        results_path=str(tmp_path / "clip"),
        animal_kinds=[kind],
        animal_centers={kind: centers},
        animal_heights={kind: heights},
        animal_contours={kind: contours},
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
    out_dir, _events = export_review_pack(analyzer, extract_samples=False)
    assert has_raw_snapshot(out_dir) is True
    assert discover_tracklet_kinds(out_dir) == []
    assert has_accepted_identities(out_dir) is False
    assert (Path(out_dir) / "raw" / "mouse_tracklets.npz").is_file()
