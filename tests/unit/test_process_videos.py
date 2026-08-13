"""Unit tests for headless process_video config / metadata (no GPU)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import sys
import types

import pandas as pd
import pytest

from LabGym.analysis.process_videos import (
    ProcessVideoConfig,
    load_categorizer_metadata,
    process_video,
    _behavior_colors,
)


def test_behavior_colors():
    c = _behavior_colors(["a", "b"])
    assert "a" in c and "b" in c
    assert c["a"][0] == "#ffffff"
    assert c["a"][1].startswith("#")


def test_load_categorizer_metadata(tmp_path: Path):
    cat = tmp_path / "cat"
    cat.mkdir()
    df = pd.DataFrame(
        {
            "classnames": ["approach", "fight"],
            "dim_conv": [32, 32],
            "dim_tconv": [32, 32],
            "channel": [1, 1],
            "time_step": [15, 15],
            "network": [2, 2],
            "inner_code": [1, 1],
            "std": [0, 0],
            "background_free": [0, 0],
            "black_background": [0, 0],
            "behavior_kind": [0, 0],
            "social_distance": [0, 0],
        }
    )
    df.to_csv(cat / "model_parameters.txt", index=False)
    meta = load_categorizer_metadata(cat)
    assert "approach" in meta["classnames"]
    assert int(meta["time_step"]) == 15
    assert int(meta["network"]) == 2


def test_process_video_missing_paths(tmp_path: Path):
    cfg = ProcessVideoConfig(
        video_path=str(tmp_path / "no.avi"),
        categorizer_path=str(tmp_path / "cat"),
        results_root=str(tmp_path / "out"),
        id_review_dir=str(tmp_path / "id_review"),
    )
    r = process_video(cfg)
    assert r.ok is False
    assert "not found" in r.error.lower()


def test_process_video_requires_accepted_identities(tmp_path: Path):
    video = tmp_path / "clip.avi"
    video.write_bytes(b"x")
    cat = tmp_path / "cat"
    cat.mkdir()
    cfg = ProcessVideoConfig(
        video_path=str(video),
        categorizer_path=str(cat),
        results_root=str(tmp_path / "out"),
        id_review_dir="",
    )
    r = process_video(cfg)
    assert r.ok is False
    assert "Review IDs" in r.error


def test_process_video_mocked(tmp_path: Path):
    video = tmp_path / "clip.avi"
    video.write_bytes(b"x")
    review = tmp_path / "id_review"
    review.mkdir()
    from LabGym.id_review.apply import write_tracklets_identity_status
    from LabGym.id_review.tracklets import save_tracklets
    from LabGym.id_review.types import SCHEMA_VERSION, TrackletStore
    import numpy as np

    n = 4
    store = TrackletStore(
        schema_version=SCHEMA_VERSION,
        animal_kind="mouse",
        ids=[0],
        n_frames=n,
        centers=np.zeros((1, n, 2)),
        valid=np.ones((1, n), dtype=bool),
        heights=np.ones((1, n)),
        contours=[[None] * n],
        meta={"fps": 10},
    )
    save_tracklets(store, str(review))
    write_tracklets_identity_status(str(review), corrected=True, accepted=True)
    cat = tmp_path / "cat"
    cat.mkdir()
    pd.DataFrame(
        {
            "classnames": ["beh"],
            "dim_conv": [32],
            "dim_tconv": [32],
            "channel": [1],
            "time_step": [15],
            "network": [0],
            "inner_code": [1],
            "std": [0],
            "background_free": [0],
            "black_background": [0],
            "behavior_kind": [0],
            "social_distance": [0],
        }
    ).to_csv(cat / "model_parameters.txt", index=False)
    out = tmp_path / "analysis"
    out.mkdir()
    results = out / "clip"
    results.mkdir()

    fake_aad = MagicMock()
    fake_aad.results_path = str(results)
    fake_aad.prepare_analysis = MagicMock()
    fake_aad.categorize_behaviors = MagicMock()
    fake_aad.annotate_video = MagicMock()
    fake_aad.export_results = MagicMock()

    fake_mod = types.ModuleType("LabGym.analyzebehavior_dt")
    fake_mod.AnalyzeAnimalDetector = MagicMock(return_value=fake_aad)
    prev = sys.modules.get("LabGym.analyzebehavior_dt")
    sys.modules["LabGym.analyzebehavior_dt"] = fake_mod
    try:
        with patch(
            "LabGym.analysis.hydrate_from_tracklets.fill_geometry_from_stores"
        ), patch(
            "LabGym.analysis.hydrate_from_tracklets.rebuild_categorizer_inputs"
        ):
            cfg = ProcessVideoConfig(
                video_path=str(video),
                categorizer_path=str(cat),
                results_root=str(out),
                animal_kinds=["mouse"],
                animal_number={"mouse": 1},
                id_review_dir=str(review),
            )
            r = process_video(cfg)
    finally:
        if prev is None:
            sys.modules.pop("LabGym.analyzebehavior_dt", None)
        else:
            sys.modules["LabGym.analyzebehavior_dt"] = prev

    assert r.ok is True, r.error
    assert r.results_path == str(results)
    assert fake_aad.prepare_analysis.call_args[0][0] is None
    fake_aad.categorize_behaviors.assert_called_once()
    fake_aad.export_results.assert_called_once()
    assert (results / "process_video_job.json").is_file()


def _accepted_review(tmp_path: Path, kinds: list[str]):
    from LabGym.id_review.apply import write_tracklets_identity_status
    from LabGym.id_review.tracklets import save_tracklets
    from LabGym.id_review.types import SCHEMA_VERSION, TrackletStore
    import numpy as np

    review = tmp_path / "id_review"
    review.mkdir()
    n = 4
    for kind in kinds:
        ids = [] if kind.endswith("_empty") else [0]
        real_kind = kind.replace("_empty", "")
        n_ids = len(ids)
        store = TrackletStore(
            schema_version=SCHEMA_VERSION,
            animal_kind=real_kind,
            ids=ids,
            n_frames=n,
            centers=np.zeros((n_ids, n, 2)),
            valid=np.ones((n_ids, n), dtype=bool) if n_ids else np.zeros((0, n), dtype=bool),
            heights=np.ones((n_ids, n)) if n_ids else np.zeros((0, n)),
            contours=[[None] * n] if n_ids else [],
            meta={"fps": 10},
        )
        save_tracklets(store, str(review))
    write_tracklets_identity_status(str(review), corrected=True, accepted=True)
    return review


def _minimal_categorizer(tmp_path: Path):
    cat = tmp_path / "cat"
    cat.mkdir()
    pd.DataFrame(
        {
            "classnames": ["beh"],
            "dim_conv": [32],
            "dim_tconv": [32],
            "channel": [1],
            "time_step": [15],
            "network": [0],
            "inner_code": [1],
            "std": [0],
            "background_free": [0],
            "black_background": [0],
            "behavior_kind": [0],
            "social_distance": [0],
        }
    ).to_csv(cat / "model_parameters.txt", index=False)
    return cat


def test_process_video_missing_kind_fails(tmp_path: Path):
    video = tmp_path / "clip.avi"
    video.write_bytes(b"x")
    review = _accepted_review(tmp_path, ["mouse"])
    cat = _minimal_categorizer(tmp_path)
    cfg = ProcessVideoConfig(
        video_path=str(video),
        categorizer_path=str(cat),
        results_root=str(tmp_path / "out"),
        animal_kinds=["mouse", "fly"],
        id_review_dir=str(review),
    )
    r = process_video(cfg)
    assert r.ok is False
    assert "missing kind" in r.error.lower()


def test_process_video_keeps_package_kinds(tmp_path: Path):
    video = tmp_path / "clip.avi"
    video.write_bytes(b"x")
    review = _accepted_review(tmp_path, ["mouse", "object_empty"])
    cat = _minimal_categorizer(tmp_path)
    out = tmp_path / "analysis"
    out.mkdir()
    results = out / "clip"
    results.mkdir()
    fake_aad = MagicMock()
    fake_aad.results_path = str(results)
    fake_mod = types.ModuleType("LabGym.analyzebehavior_dt")
    fake_mod.AnalyzeAnimalDetector = MagicMock(return_value=fake_aad)
    prev = sys.modules.get("LabGym.analyzebehavior_dt")
    sys.modules["LabGym.analyzebehavior_dt"] = fake_mod
    try:
        with patch(
            "LabGym.analysis.hydrate_from_tracklets.fill_geometry_from_stores"
        ), patch(
            "LabGym.analysis.hydrate_from_tracklets.rebuild_categorizer_inputs"
        ):
            cfg = ProcessVideoConfig(
                video_path=str(video),
                categorizer_path=str(cat),
                results_root=str(out),
                animal_kinds=["mouse"],
                animal_number={"mouse": 1},
                id_review_dir=str(review),
            )
            r = process_video(cfg)
    finally:
        if prev is None:
            sys.modules.pop("LabGym.analyzebehavior_dt", None)
        else:
            sys.modules["LabGym.analyzebehavior_dt"] = prev
    assert r.ok is True, r.error
    prepared_kinds = fake_aad.prepare_analysis.call_args[0][4]
    assert "mouse" in prepared_kinds
    assert "object" in prepared_kinds


def test_process_tab_constructs():
    from PySide6.QtWidgets import QApplication
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.workbenches.categorizer.process_videos_tab import (
        ProcessVideosTab,
    )
    from LabGym.gui_pyside.main_window import WorkbenchMainWindow

    p = ProjectController()
    assert ProcessVideosTab(p) is not None
    w = WorkbenchMainWindow()
    assert w.wb_categorizer.set_current_tab("process")
    w._goto_process_videos()
    assert w.host.current_id() == "categorizer"
