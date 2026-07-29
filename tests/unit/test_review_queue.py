"""Unit tests for review queue and train-time effective labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from LabGym.training.dataset_manifest import (
    DatasetManifest,
    apply_manifest_to_path_list,
    example_id_from_path,
    rebuild_classmapping,
)
from LabGym.training.evaluation import (
    compute_evaluation_metrics,
    rank_high_loss_examples,
    write_evaluation_run,
)
from LabGym.training.review_queue import (
    SOURCE_HIGH_LOSS,
    SOURCE_MISCLASSIFIED,
    available_categories,
    build_review_queue,
    ensure_queue_in_manifest,
    normalize_example_id,
)


def _write_run(model: Path, run_id: str, *, with_high_loss: bool = True) -> Path:
    classnames = ["groom", "rear", "walk"]
    y_true = ["groom", "groom", "rear", "walk"]
    y_pred = ["groom", "rear", "rear", "groom"]  # 2 misclassified
    m = compute_evaluation_metrics(
        y_true,
        classnames,
        y_pred=y_pred,
        example_ids=["e0_groom", "e1_groom", "e2_rear", "e3_walk"],
        confidences=[0.9, 0.55, 0.8, 0.4],
    )
    hl = None
    if with_high_loss:
        hl = rank_high_loss_examples(
            ["e0_groom", "e4_groom"],
            [0.2, 3.5],
            ["groom", "groom"],
            pred_labels=["groom", "walk"],
            top_k=2,
        )
    return write_evaluation_run(
        model,
        m,
        run_id=run_id,
        source="test",
        high_loss=hl,
    )


def test_normalize_example_id():
    assert normalize_example_id("groom/clip_0.avi") == "clip_0"
    assert normalize_example_id("e1_groom.jpg") == "e1_groom"
    assert normalize_example_id("e1_groom") == "e1_groom"


def test_build_review_queue_misclassified_and_high_loss(tmp_path: Path):
    model = tmp_path / "model"
    run = _write_run(model, "r1")
    q = build_review_queue([run], include_misclassified=True, include_high_loss=True)
    ids = {it.example_id for it in q}
    # misclassified: e1_groom, e3_walk; high-loss: e0_groom, e4_groom
    assert "e1_groom" in ids
    assert "e3_walk" in ids
    assert "e4_groom" in ids
    # e0 may appear from high-loss (and not misclassified)
    assert any(SOURCE_HIGH_LOSS in it.sources for it in q)
    assert any(SOURCE_MISCLASSIFIED in it.sources for it in q)
    # high-loss first when sorted
    assert q[0].source == SOURCE_HIGH_LOSS or SOURCE_HIGH_LOSS in q[0].sources


def test_build_review_queue_dedupe_merges_sources(tmp_path: Path):
    model = tmp_path / "model"
    # Craft run where same id is both high-loss and misclassified
    classnames = ["a", "b"]
    m = compute_evaluation_metrics(
        ["a", "b"],
        classnames,
        y_pred=["b", "b"],
        example_ids=["x", "y"],
        confidences=[0.6, 0.9],
    )
    hl = rank_high_loss_examples(["x"], [2.0], ["a"], pred_labels=["b"], top_k=1)
    run = write_evaluation_run(model, m, run_id="dup", high_loss=hl)
    q = build_review_queue([run], dedupe=True)
    xs = [it for it in q if it.example_id == "x"]
    assert len(xs) == 1
    assert set(xs[0].sources) == {SOURCE_HIGH_LOSS, SOURCE_MISCLASSIFIED}


def test_ensure_queue_and_keep_exclude(tmp_path: Path):
    store = tmp_path / "store"
    store.mkdir()
    model = tmp_path / "model"
    run = _write_run(model, "r2", with_high_loss=False)
    q = build_review_queue([run], include_high_loss=False)
    m = DatasetManifest.load_or_create(store)
    n = ensure_queue_in_manifest(m, q)
    assert n >= 1
    eid = q[0].example_id
    m.keep(eid)
    assert m.examples[eid].excluded is False
    m.exclude(eid)
    assert m.examples[eid].excluded is True
    m.recategorize(eid, "walk")
    assert m.examples[eid].active_label == "walk"
    assert m.undo()  # recategorize
    assert m.undo()  # exclude
    assert m.examples[eid].excluded is False


def test_apply_manifest_to_path_list_overrides(tmp_path: Path):
    store = tmp_path / "data"
    store.mkdir()
    paths = []
    for name in ("a_groom.jpg", "b_rear.jpg", "c_groom.jpg"):
        p = store / name
        p.write_bytes(b"x")
        paths.append(str(p))
    m = DatasetManifest.load_or_create(store)
    m.sync_from_scan(
        [
            ("a_groom", "groom", "a_groom.jpg"),
            ("b_rear", "rear", "b_rear.jpg"),
            ("c_groom", "groom", "c_groom.jpg"),
        ]
    )
    m.exclude("a_groom")
    m.recategorize("b_rear", "groom")
    m.examples["c_groom"].split = "sealed_test"
    kept, labels, n_drop = apply_manifest_to_path_list(paths, m)
    assert n_drop == 2  # a excluded + c sealed
    assert len(kept) == 1
    assert labels[kept[0]] == "groom"
    assert example_id_from_path(kept[0]) == "b_rear"

    classnames, mapping = rebuild_classmapping(labels.values())
    assert classnames == ["groom"]
    assert "groom" in mapping


def test_available_categories(tmp_path: Path):
    store = tmp_path / "s"
    (store / "groom").mkdir(parents=True)
    (store / "rear").mkdir()
    (store / "groom" / "x.jpg").write_bytes(b"x")
    (store / "rear" / "y.jpg").write_bytes(b"x")
    cats = available_categories(store, extra=["walk"])
    assert "groom" in cats and "rear" in cats and "walk" in cats
