"""Unit tests for shared model path discovery (no GPU / GUI dialogs)."""

from __future__ import annotations

import json
from pathlib import Path

from LabGym.gui_pyside.model_paths import (
    list_categorizer_folders,
    model_search_roots,
    scan_categorizer_paths,
    scan_detector_paths,
)


def test_list_categorizer_folders(tmp_path: Path):
    cat = tmp_path / "MyCat"
    cat.mkdir()
    (cat / "model_parameters.txt").write_text(
        "classnames,network,time_step\napproach,2,15\n", encoding="utf-8"
    )
    det = tmp_path / "MyDet"
    det.mkdir()
    (det / "model_parameters.txt").write_text(
        json.dumps({"animal_names": ["mouse"]}), encoding="utf-8"
    )
    found = list_categorizer_folders(tmp_path)
    assert cat in found
    assert det not in found


def test_scan_detector_paths(tmp_path: Path):
    det = tmp_path / "det"
    det.mkdir()
    (det / "model_parameters.txt").write_text(
        json.dumps({"animal_names": ["mouse"], "inferencing_framesize": 100}),
        encoding="utf-8",
    )
    paths = scan_detector_paths(roots=[tmp_path])
    assert str(det) in paths


def test_scan_categorizer_paths(tmp_path: Path):
    cat = tmp_path / "cat"
    cat.mkdir()
    (cat / "model_parameters.txt").write_text(
        "classnames,network\na,2\n", encoding="utf-8"
    )
    paths = scan_categorizer_paths(roots=[tmp_path])
    assert str(cat) in paths


def test_model_search_roots_empty_project():
    # No project: still may include bundled roots if present on disk
    roots = model_search_roots(None)
    assert isinstance(roots, list)
