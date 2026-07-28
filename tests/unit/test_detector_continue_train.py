"""Unit tests for detector continue-training helpers (no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from LabGym.detection.continue_train import (
    CONTINUE_BASE_LR,
    annotation_animal_names,
    class_lists_compatible,
    plan_continue_training,
    suggest_continued_detector_name,
)


def _write_detector(folder: Path, names: list[str], *, infer: int = 480) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model_parameters.txt").write_text(
        json.dumps(
            {
                "animal_names": names,
                "animal_mapping": {str(i): n for i, n in enumerate(names)},
                "inferencing_framesize": infer,
            }
        ),
        encoding="utf-8",
    )
    (folder / "model_final.pth").write_bytes(b"fake-weights")
    (folder / "config.yaml").write_text("MODEL: {}\n", encoding="utf-8")
    return folder


def _write_ann(path: Path, names: list[str]) -> Path:
    cats = [{"id": 0, "name": "background"}] + [
        {"id": i + 1, "name": n} for i, n in enumerate(names)
    ]
    path.write_text(
        json.dumps({"images": [], "annotations": [], "categories": cats}),
        encoding="utf-8",
    )
    return path


def test_annotation_animal_names_skips_background(tmp_path: Path):
    ann = _write_ann(tmp_path / "a.json", ["mouse", "object"])
    assert annotation_animal_names(ann) == ["mouse", "object"]


def test_class_lists_compatible_order_sensitive():
    assert class_lists_compatible(["mouse"], ["mouse"])
    assert not class_lists_compatible(["mouse", "object"], ["object", "mouse"])
    assert not class_lists_compatible(["mouse"], ["mouse", "object"])


def test_suggest_continued_detector_name():
    assert suggest_continued_detector_name("/x/social_v3") == "social_v3_ft"
    assert suggest_continued_detector_name("social_v3_ft") == "social_v3_ft2"


def test_plan_continue_training_ok(tmp_path: Path):
    det = _write_detector(tmp_path / "det", ["mouse"], infer=512)
    ann = _write_ann(tmp_path / "ann.json", ["mouse"])
    plan = plan_continue_training(det, ann)
    assert plan.animal_names == ["mouse"]
    assert plan.inference_size == 512
    assert plan.base_lr == CONTINUE_BASE_LR
    assert Path(plan.weights_path).name == "model_final.pth"
    assert Path(plan.weights_path).is_file()


def test_plan_continue_training_class_mismatch(tmp_path: Path):
    det = _write_detector(tmp_path / "det", ["mouse"])
    ann = _write_ann(tmp_path / "ann.json", ["mouse", "object"])
    with pytest.raises(ValueError, match="do not match"):
        plan_continue_training(det, ann)


def test_plan_continue_training_rejects_categorizer(tmp_path: Path):
    cat = tmp_path / "cat"
    cat.mkdir()
    (cat / "model_parameters.txt").write_text(
        "classnames,dim_tconv,network\nFollow,32,2\n",
        encoding="utf-8",
    )
    (cat / "model.keras").write_bytes(b"x")
    ann = _write_ann(tmp_path / "ann.json", ["mouse"])
    with pytest.raises(ValueError, match="categorizer"):
        plan_continue_training(cat, ann)


def test_plan_custom_base_lr(tmp_path: Path):
    det = _write_detector(tmp_path / "det", ["mouse"])
    ann = _write_ann(tmp_path / "ann.json", ["mouse"])
    plan = plan_continue_training(det, ann, base_lr=5e-5)
    assert plan.base_lr == pytest.approx(5e-5)
