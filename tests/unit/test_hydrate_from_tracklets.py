"""Hydrate analyzer geometry from remapped tracklets."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from LabGym.analysis.hydrate_from_tracklets import (
    fill_geometry_from_stores,
    rebuild_categorizer_inputs,
    resolve_package_kinds,
)
from LabGym.tools import generate_patternimage, generate_patternimage_interact
from LabGym.id_review.types import SCHEMA_VERSION, TrackletStore


def _square(x: int, y: int, size: int = 8) -> np.ndarray:
    return np.array(
        [
            [[x, y]],
            [[x + size, y]],
            [[x + size, y + size]],
            [[x, y + size]],
        ],
        dtype=np.int32,
    )


def _moving_store(*, n: int = 5, kind: str = "mouse", tid: int = 0) -> TrackletStore:
    centers = np.zeros((1, n, 2), dtype=np.float64)
    valid = np.ones((1, n), dtype=bool)
    heights = np.full((1, n), 12.0)
    row = []
    for f in range(n):
        x = 4 + 10 * f
        y = 8
        centers[0, f] = [float(x + 4), float(y + 4)]
        row.append(_square(x, y))
    return TrackletStore(
        schema_version=SCHEMA_VERSION,
        animal_kind=kind,
        ids=[tid],
        n_frames=n,
        centers=centers,
        valid=valid,
        heights=heights,
        contours=[row],
        meta={"fps": 10.0},
    )


def _empty_store(*, n: int = 5, kind: str = "object") -> TrackletStore:
    return TrackletStore(
        schema_version=SCHEMA_VERSION,
        animal_kind=kind,
        ids=[],
        n_frames=n,
        centers=np.zeros((0, n, 2), dtype=np.float64),
        valid=np.zeros((0, n), dtype=bool),
        heights=np.zeros((0, n), dtype=np.float64),
        contours=[],
        meta={"fps": 10.0},
    )


def _write_video(path: Path, n: int, h: int = 48, w: int = 80) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (w, h), True
    )
    assert writer.isOpened(), "VideoWriter failed to open"
    for f in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = 4 + 10 * f
        y = 8
        frame[y : y + 8, x : x + 8] = (220, 220, 220)
        writer.write(frame)
    writer.release()


def _analyzer(
    n: int,
    *,
    kinds: list[str] | None = None,
    animation_analyzer: bool = False,
    include_bodyparts: bool = False,
    behavior_mode: int = 0,
    length: int = 3,
    dim_conv: int = 8,
    dim_tconv: int = 8,
    channel: int = 1,
) -> SimpleNamespace:
    kinds = list(kinds or ["mouse"])
    zero_anim = np.zeros((length, dim_tconv, dim_tconv, channel), dtype="uint8")
    zero_pat = np.zeros((dim_conv, dim_conv, 3), dtype="uint8")
    animal_centers = {}
    animal_contours = {}
    animal_heights = {}
    animal_existingcenters = {}
    register_counts = {}
    pattern_images = {}
    animations = {}
    animal_inners = {}
    animal_other_inners = {}
    for kind in kinds:
        animal_centers[kind] = {0: [None] * n}
        animal_contours[kind] = {0: [None] * n}
        animal_heights[kind] = {0: [None] * n}
        animal_existingcenters[kind] = {0: (-10000, -10000)}
        register_counts[kind] = {0: None}
        pattern_images[kind] = {0: [zero_pat.copy() for _ in range(n)]}
        if animation_analyzer:
            animations[kind] = {0: [zero_anim.copy() for _ in range(n)]}
        if include_bodyparts:
            animal_inners[kind] = {0: []}
            if behavior_mode == 2:
                animal_other_inners[kind] = {0: []}
    return SimpleNamespace(
        total_analysis_framecount=n,
        fps=10.0,
        framewidth=None,
        animal_kinds=kinds,
        animal_centers=animal_centers,
        animal_contours=animal_contours,
        animal_heights=animal_heights,
        animal_existingcenters=animal_existingcenters,
        register_counts=register_counts,
        animation_analyzer=animation_analyzer,
        animations=animations,
        pattern_images=pattern_images,
        include_bodyparts=include_bodyparts,
        animal_inners=animal_inners,
        animal_other_inners=animal_other_inners,
        behavior_mode=behavior_mode,
        length=length,
        dim_conv=dim_conv,
        dim_tconv=dim_tconv,
        channel=channel,
        std=0,
    )


def test_fill_geometry_copies_centers_and_contours():
    n = 5
    store = _moving_store(n=n)
    analyzer = _analyzer(n, animation_analyzer=False)
    fill_geometry_from_stores(analyzer, {"mouse": store})
    assert analyzer.animal_centers["mouse"][0][3] == (38.0, 12.0)
    assert analyzer.animal_contours["mouse"][0][3] is not None
    assert analyzer.animal_heights["mouse"][0][3] == 12.0


def test_rebuild_animation_uses_zeros_when_id_absent(tmp_path: Path):
    n = 4
    length = 3
    video = tmp_path / "clip.avi"
    _write_video(video, n)
    store = _moving_store(n=n)
    store.valid[0, 2] = False
    store.contours[0][2] = None
    analyzer = _analyzer(n, animation_analyzer=True, length=length)
    fill_geometry_from_stores(analyzer, {"mouse": store})
    rebuild_categorizer_inputs(analyzer, video_path=str(video), store_meta=store.meta)
    clip = np.asarray(analyzer.animations["mouse"][0][2])
    assert np.array_equal(clip[-1], np.zeros_like(clip[-1]))
    assert not np.array_equal(clip[-2], np.zeros_like(clip[-2]))


def test_rebuild_animation_is_temporal_window_not_frozen_still(tmp_path: Path):
    n = 5
    length = 3
    video = tmp_path / "clip.avi"
    _write_video(video, n)
    store = _moving_store(n=n)
    analyzer = _analyzer(n, animation_analyzer=True, length=length)
    fill_geometry_from_stores(analyzer, {"mouse": store})
    rebuild_categorizer_inputs(analyzer, video_path=str(video), store_meta=store.meta)
    clip = np.asarray(analyzer.animations["mouse"][0][n - 1])
    assert clip.shape[0] == length
    slices = [clip[i] for i in range(length)]
    assert not all(np.array_equal(slices[0], s) for s in slices[1:])


def _inners_from_calls(mock_fn):
    found = []
    for call in mock_fn.call_args_list:
        if "inners" in call.kwargs:
            found.append(call.kwargs["inners"])
        elif len(call.args) > 2:
            found.append(call.args[2])
        else:
            found.append(None)
    return found


def test_rebuild_passes_inners_when_bodyparts(tmp_path: Path):
    n = 4
    video = tmp_path / "clip.avi"
    _write_video(video, n)
    store = _moving_store(n=n)
    analyzer = _analyzer(n, include_bodyparts=True, animation_analyzer=False)
    fill_geometry_from_stores(analyzer, {"mouse": store})
    with patch(
        "LabGym.analysis.hydrate_from_tracklets.generate_patternimage",
        wraps=generate_patternimage,
    ) as mock_gp:
        rebuild_categorizer_inputs(
            analyzer, video_path=str(video), store_meta=store.meta
        )
    inners_args = _inners_from_calls(mock_gp)
    assert inners_args
    assert all(item is not None for item in inners_args)


def test_rebuild_omits_inners_without_bodyparts(tmp_path: Path):
    n = 4
    video = tmp_path / "clip.avi"
    _write_video(video, n)
    store = _moving_store(n=n)
    analyzer = _analyzer(n, include_bodyparts=False, animation_analyzer=False)
    fill_geometry_from_stores(analyzer, {"mouse": store})
    with patch(
        "LabGym.analysis.hydrate_from_tracklets.generate_patternimage",
        wraps=generate_patternimage,
    ) as mock_gp:
        rebuild_categorizer_inputs(
            analyzer, video_path=str(video), store_meta=store.meta
        )
    inners_args = _inners_from_calls(mock_gp)
    assert inners_args
    assert all(item is None for item in inners_args)


def test_rebuild_passes_other_inners_in_interactive_advanced(tmp_path: Path):
    n = 4
    video = tmp_path / "clip.avi"
    _write_video(video, n)
    store_a = _moving_store(n=n, tid=0)
    store_b_centers = np.zeros((1, n, 2), dtype=np.float64)
    store_b_valid = np.ones((1, n), dtype=bool)
    store_b_heights = np.full((1, n), 12.0)
    row_b = []
    for f in range(n):
        x = 40
        y = 24
        store_b_centers[0, f] = [float(x + 4), float(y + 4)]
        row_b.append(_square(x, y))
    store_b = TrackletStore(
        schema_version=SCHEMA_VERSION,
        animal_kind="mouse",
        ids=[1],
        n_frames=n,
        centers=store_b_centers,
        valid=store_b_valid,
        heights=store_b_heights,
        contours=[row_b],
        meta={"fps": 10.0},
    )
    analyzer = _analyzer(
        n, include_bodyparts=True, behavior_mode=2, animation_analyzer=False
    )
    analyzer.animal_centers["mouse"][1] = [None] * n
    analyzer.animal_contours["mouse"][1] = [None] * n
    analyzer.animal_heights["mouse"][1] = [None] * n
    analyzer.animal_existingcenters["mouse"][1] = (-10000, -10000)
    analyzer.register_counts["mouse"][1] = None
    analyzer.pattern_images["mouse"][1] = [
        np.zeros((8, 8, 3), dtype="uint8") for _ in range(n)
    ]
    analyzer.animal_inners["mouse"][1] = []
    analyzer.animal_other_inners["mouse"][1] = []
    fill_geometry_from_stores(analyzer, {"mouse": store_a})
    fill_geometry_from_stores(analyzer, {"mouse": store_b})
    with patch(
        "LabGym.analysis.hydrate_from_tracklets.generate_patternimage_interact",
        wraps=generate_patternimage_interact,
    ) as mock_gpi:
        rebuild_categorizer_inputs(
            analyzer, video_path=str(video), store_meta=store_a.meta
        )
    assert mock_gpi.called
    for call in mock_gpi.call_args_list:
        assert call.kwargs.get("inners") is not None
        assert call.kwargs.get("other_inners") is not None


def test_empty_kind_hydrates_without_error(tmp_path: Path):
    n = 4
    empty = _empty_store(n=n, kind="object")
    analyzer = _analyzer(n, kinds=["object"], animation_analyzer=True)
    analyzer.animal_centers["object"] = {}
    analyzer.animal_contours["object"] = {}
    analyzer.animal_heights["object"] = {}
    analyzer.animal_existingcenters["object"] = {}
    analyzer.register_counts["object"] = {}
    analyzer.pattern_images["object"] = {}
    analyzer.animations["object"] = {}
    fill_geometry_from_stores(analyzer, {"object": empty})
    kinds = resolve_package_kinds({"object": empty}, requested=["object"])
    assert kinds == ["object"]
    video = tmp_path / "clip.avi"
    _write_video(video, n)
    rebuild_categorizer_inputs(analyzer, video_path=str(video), store_meta=empty.meta)


def test_missing_kind_raises():
    store = _moving_store()
    with pytest.raises(ValueError, match="missing kind"):
        resolve_package_kinds({"mouse": store}, requested=["mouse", "fly"])


def test_remapped_kind_not_in_analyzer_raises():
    n = 4
    mouse = _moving_store(n=n, kind="mouse")
    obj = _moving_store(n=n, kind="object")
    analyzer = _analyzer(n, kinds=["mouse"], animation_analyzer=False)
    with pytest.raises(ValueError, match="cannot be loaded"):
        fill_geometry_from_stores(analyzer, {"mouse": mouse, "object": obj})


def test_resolve_package_kinds_never_drops_remapped():
    mouse = _moving_store(kind="mouse")
    obj = _empty_store(kind="object")
    kinds = resolve_package_kinds({"mouse": mouse, "object": obj}, requested=["mouse"])
    assert kinds == ["mouse", "object"]
