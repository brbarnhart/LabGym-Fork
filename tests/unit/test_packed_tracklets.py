"""Packed contour arrays on session load (ADR-0009)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from LabGym.id_review.raw_store import load_kind_stores, save_raw_tracklets
from LabGym.id_review.tracklets import (
    apply_mapping_to_store,
    load_tracklets,
    save_tracklets,
)
from LabGym.id_review.types import SCHEMA_VERSION, TrackletStore, swap_mapping
from LabGym.identity.package import (
    apply_decisions_and_save_tracklets,
    clone_store,
)

_NPZ_KEYS = {
    "centers",
    "valid",
    "heights",
    "ids",
    "contour_lengths",
    "contour_points",
    "n_frames",
}


def _square(x: int, y: int, size: int = 5) -> np.ndarray:
    return np.array(
        [
            [[x, y]],
            [[x + size, y]],
            [[x + size, y + size]],
            [[x, y + size]],
        ],
        dtype=np.int32,
    )


def _store(*, n_frames: int = 12, kind: str = "mouse", ids=(0, 1)) -> TrackletStore:
    ids = [int(i) for i in ids]
    n_ids = len(ids)
    centers = np.zeros((n_ids, n_frames, 2), dtype=np.float64)
    valid = np.ones((n_ids, n_frames), dtype=bool)
    heights = np.full((n_ids, n_frames), 10.0)
    contours = []
    for row, tid in enumerate(ids):
        row_c = []
        for f in range(n_frames):
            centers[row, f] = [tid * 100 + f, tid * 10]
            x, y = int(centers[row, f, 0]), int(centers[row, f, 1])
            row_c.append(_square(x, y))
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


def _legacy_expand(lengths, points, n_ids: int, n_frames: int):
    """Reproduce the pre-ADR-0009 load_tracklets contour expansion."""
    contours = []
    cursor = 0
    li = 0
    for _row in range(n_ids):
        row_c = []
        for _f in range(n_frames):
            L = int(lengths[li])
            li += 1
            if L == 0:
                row_c.append(None)
            else:
                pts = points[cursor : cursor + L].copy()
                cursor += L
                row_c.append(pts.reshape(L, 1, 2).astype(np.int32))
        contours.append(row_c)
    return contours


def test_load_tracklets_keeps_packed_arrays_without_expanding(tmp_path: Path):
    original = _store(n_frames=10)
    save_tracklets(original, str(tmp_path))
    loaded = load_tracklets(str(tmp_path), "mouse")

    assert loaded.contours is None
    n_cells = len(loaded.ids) * loaded.n_frames
    assert loaded.contour_lengths is not None
    assert loaded.contour_lengths.shape == (n_cells,)
    assert loaded.contour_points is not None
    assert loaded.contour_points.ndim == 2
    assert loaded.contour_points.shape[1] == 2


def test_contour_at_matches_legacy_whole_video_expand(tmp_path: Path):
    original = _store(n_frames=9)
    save_tracklets(original, str(tmp_path))
    npz = np.load(tmp_path / "mouse_tracklets.npz", allow_pickle=False)
    legacy = _legacy_expand(
        npz["contour_lengths"].astype(int),
        npz["contour_points"],
        n_ids=len(original.ids),
        n_frames=original.n_frames,
    )
    npz.close()

    loaded = load_tracklets(str(tmp_path), "mouse")
    for row in range(len(loaded.ids)):
        for frame in (0, 3, loaded.n_frames - 1):
            got = loaded.contour_at(row, frame)
            want = legacy[row][frame]
            assert got is not None and want is not None
            assert np.array_equal(got, want)
            assert np.array_equal(got, original.contours[row][frame])


def test_load_one_kind_does_not_require_other_kinds(tmp_path: Path):
    save_tracklets(_store(kind="mouse"), str(tmp_path))
    save_tracklets(_store(kind="object", ids=(0,)), str(tmp_path))
    (tmp_path / "object_tracklets.npz").unlink()
    (tmp_path / "object_tracklets_meta.json").unlink()

    loaded = load_tracklets(str(tmp_path), "mouse")
    assert loaded.animal_kind == "mouse"
    assert loaded.ids == [0, 1]
    stores = load_kind_stores(tmp_path, animal_kinds=["mouse"])
    assert list(stores) == ["mouse"]


def test_clone_store_of_packed_load_does_not_expand(tmp_path: Path):
    save_tracklets(_store(n_frames=8), str(tmp_path))
    loaded = load_tracklets(str(tmp_path), "mouse")
    cloned = clone_store(loaded)
    assert cloned.contours is None
    assert cloned is not loaded
    assert cloned.centers is not loaded.centers
    cloned.centers[0, 0] = [999.0, 999.0]
    assert loaded.centers[0, 0, 0] != 999.0
    cloned.contour_points[0, 0] = -1
    assert loaded.contour_points[0, 0] != -1


def test_load_review_package_does_not_clone_at_load(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
        load_review_package,
    )

    review = tmp_path / "id_review"
    review.mkdir()
    save_raw_tracklets(review, {"mouse": _store(n_frames=6)})
    pkg = load_review_package(str(review))
    assert pkg.stores["mouse"] is pkg.baseline_stores["mouse"]
    assert pkg.stores["mouse"].contours is None


def test_npz_layout_unchanged(tmp_path: Path):
    save_tracklets(_store(), str(tmp_path))
    with np.load(tmp_path / "mouse_tracklets.npz", allow_pickle=False) as data:
        assert set(data.files) == _NPZ_KEYS


def test_play_seek_contour_from_ram_after_load(tmp_path: Path):
    from LabGym.annotator.core.tracklets_bridge import overlays_at_video_frame
    from LabGym.id_review.apply import write_tracklets_identity_status
    from LabGym.annotator.core.tracklets_bridge import load_tracklets_for_annotator

    save_tracklets(_store(n_frames=7), str(tmp_path))
    write_tracklets_identity_status(str(tmp_path), corrected=True, accepted=True)
    loaded = load_tracklets_for_annotator(tmp_path, animal_kinds=["mouse"])
    store = loaded.stores["mouse"]
    assert store.contours is None
    ov0 = overlays_at_video_frame(loaded, 0)
    ov3 = overlays_at_video_frame(loaded, 3)
    assert any(o.valid and o.contour is not None for o in ov0)
    assert any(o.valid and o.contour is not None for o in ov3)
    assert not np.array_equal(ov0[0].contour, ov3[0].contour)


def test_apply_mapping_on_packed_store_swaps_outlines(tmp_path: Path):
    original = _store(n_frames=15)
    save_tracklets(original, str(tmp_path))
    loaded = load_tracklets(str(tmp_path), "mouse")
    f0 = 8
    before_0 = loaded.contour_at(0, f0).copy()
    before_1 = loaded.contour_at(1, f0).copy()
    apply_mapping_to_store(loaded, swap_mapping([0, 1]), f0)
    assert np.array_equal(loaded.contour_at(0, f0), before_1)
    assert np.array_equal(loaded.contour_at(1, f0), before_0)
    assert np.array_equal(loaded.contour_at(0, f0 - 1), original.contours[0][f0 - 1])


def test_save_rebuilds_remapped_from_raw_not_public(tmp_path: Path):
    from LabGym.id_review.dataset import make_swap_marker, switches_to_decisions

    raw = _store(n_frames=15)
    save_raw_tracklets(tmp_path, {"mouse": raw})
    public_before = _store(n_frames=15)
    public_before.centers[0, :] = 0.0
    save_tracklets(public_before, str(tmp_path))
    marker = make_swap_marker(5, "mouse", [0, 1], fps=30.0)
    apply_decisions_and_save_tracklets(tmp_path, switches_to_decisions([marker]))
    published = load_tracklets(str(tmp_path), "mouse")
    assert np.allclose(published.centers[0, 5:], raw.centers[1, 5:])
    assert np.allclose(published.centers[1, 5:], raw.centers[0, 5:])
    assert np.allclose(published.centers[0, :5], raw.centers[0, :5])
    assert np.array_equal(published.contour_at(0, 5), raw.contours[1][5])


def test_annotator_can_load_one_kind_when_package_has_two(tmp_path: Path):
    from LabGym.annotator.core.tracklets_bridge import load_tracklets_for_annotator
    from LabGym.id_review.apply import write_tracklets_identity_status

    save_tracklets(_store(kind="mouse"), str(tmp_path))
    save_tracklets(_store(kind="object", ids=(0,)), str(tmp_path))
    write_tracklets_identity_status(str(tmp_path), corrected=True, accepted=True)
    loaded = load_tracklets_for_annotator(tmp_path, animal_kinds=["mouse"])
    assert list(loaded.stores) == ["mouse"]
    assert loaded.stores["mouse"].contours is None
