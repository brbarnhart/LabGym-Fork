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
        detector_path=str(tmp_path / "det"),
        categorizer_path=str(tmp_path / "cat"),
        results_root=str(tmp_path / "out"),
    )
    r = process_video(cfg)
    assert r.ok is False
    assert "not found" in r.error.lower()


def test_process_video_mocked(tmp_path: Path):
    video = tmp_path / "clip.avi"
    video.write_bytes(b"x")
    det = tmp_path / "det"
    det.mkdir()
    (det / "model_parameters.txt").write_text(
        '{"animal_names":["mouse"],"inferencing_framesize":100}',
        encoding="utf-8",
    )
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
    fake_aad.acquire_information = MagicMock()
    fake_aad.craft_data = MagicMock()
    fake_aad.categorize_behaviors = MagicMock()
    fake_aad.annotate_video = MagicMock()
    fake_aad.export_results = MagicMock()

    fake_mod = types.ModuleType("LabGym.analyzebehavior_dt")
    fake_mod.AnalyzeAnimalDetector = MagicMock(return_value=fake_aad)
    prev = sys.modules.get("LabGym.analyzebehavior_dt")
    sys.modules["LabGym.analyzebehavior_dt"] = fake_mod
    try:
        cfg = ProcessVideoConfig(
            video_path=str(video),
            detector_path=str(det),
            categorizer_path=str(cat),
            results_root=str(out),
            animal_kinds=["mouse"],
            animal_number={"mouse": 1},
        )
        r = process_video(cfg)
    finally:
        if prev is None:
            sys.modules.pop("LabGym.analyzebehavior_dt", None)
        else:
            sys.modules["LabGym.analyzebehavior_dt"] = prev

    assert r.ok is True, r.error
    assert r.results_path == str(results)
    fake_aad.categorize_behaviors.assert_called_once()
    fake_aad.export_results.assert_called_once()
    assert (results / "process_video_job.json").is_file()


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
