"""Unit tests for soft projection and taxonomy merge/exclude."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from LabGym.training.dataset_manifest import DatasetManifest
from LabGym.training.soft_labels import SoftLabelTable
from LabGym.training.soft_projection import (
    compose_merge_map,
    effective_soft_for_basename,
    effective_soft_matrix,
    excluded_categories_from_ops,
    project_soft_vector,
)


def test_project_soft_merge_and_exclude():
    soft = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    src = ["a", "b", "c"]
    # merge a+b -> ab, drop c
    out = project_soft_vector(
        soft,
        src,
        ["ab"],
        merge_map={"a": "ab", "b": "ab"},
        excluded={"c"},
    )
    assert out.shape == (1,)
    assert out[0] == pytest.approx(1.0)  # 0.2+0.3 renormalized


def test_compose_merge_map_chain():
    ops = [
        {"op": "merge", "sources": ["a", "b"], "target": "ab"},
        {"op": "merge", "sources": ["ab", "c"], "target": "all"},
    ]
    m = compose_merge_map(ops)
    assert m["a"] == "all"
    assert m["b"] == "all"
    assert m["c"] == "all" or m.get("ab") == "all"


def test_excluded_categories_from_ops():
    ops = [
        {"op": "exclude_category", "category": "rare", "excluded": True},
        {"op": "exclude_category", "category": "noise", "excluded": True},
        {"op": "include_category", "category": "rare", "excluded": False},
    ]
    excl = excluded_categories_from_ops(ops)
    assert "noise" in excl
    assert "rare" not in excl


def test_manifest_merge_exclude_undo_and_soft(tmp_path: Path):
    m = DatasetManifest(store_root=tmp_path)
    m.sync_from_scan(
        [
            ("e0", "groom", "e0.jpg"),
            ("e1", "rear", "e1.jpg"),
            ("e2", "walk", "e2.jpg"),
            ("e3", "walk", "e3.jpg"),
        ]
    )
    n = m.merge_categories(["groom", "rear"], "stationary")
    assert n == 2
    assert m.examples["e0"].active_label == "stationary"
    assert m.examples["e1"].active_label == "stationary"
    assert m.examples["e2"].active_label == "walk"
    assert any(op.get("op") == "merge" for op in m.taxonomy_ops)

    n_ex = m.exclude_category("walk")
    assert n_ex == 2
    assert m.examples["e2"].excluded is True
    assert "walk" in m.excluded_categories()

    # Soft table in original space
    table = SoftLabelTable(
        classnames=["groom", "rear", "walk"],
        rows={
            "e0": ("groom", np.array([0.7, 0.2, 0.1], dtype=np.float32)),
            "e2": ("walk", np.array([0.1, 0.1, 0.8], dtype=np.float32)),
        },
    )
    targets = ["stationary", "walk"]
    soft0 = effective_soft_for_basename("e0", table, targets, manifest=m)
    assert soft0 is not None
    # groom+rear mass 0.9 onto stationary, walk 0.1 — walk not excluded from soft
    # but walk category is taxonomy-excluded so walk mass dropped → stationary only
    assert soft0[0] == pytest.approx(1.0)
    assert soft0[1] == pytest.approx(0.0)

    soft2 = effective_soft_for_basename("e2", table, targets, manifest=m)
    # walk mass dropped; residual groom/rear mass projects onto stationary
    assert soft2 is not None
    assert soft2[0] == pytest.approx(1.0)
    assert soft2[1] == pytest.approx(0.0)

    # Pure walk soft becomes empty when walk is taxonomy-excluded
    table_pure = SoftLabelTable(
        classnames=["groom", "rear", "walk"],
        rows={"e2": ("walk", np.array([0.0, 0.0, 1.0], dtype=np.float32))},
    )
    assert effective_soft_for_basename("e2", table_pure, targets, manifest=m) is None

    assert m.undo()  # include walk
    assert m.examples["e2"].excluded is False
    assert m.undo()  # unmerge
    assert m.examples["e0"].label_override is None
    assert m.examples["e0"].active_label == "groom"
    assert m.taxonomy_ops == [] or not any(
        op.get("op") == "merge" for op in m.taxonomy_ops
    )


def test_effective_soft_matrix_shape(tmp_path: Path):
    m = DatasetManifest(store_root=tmp_path)
    m.sync_from_scan([("a", "x", None), ("b", "y", None)])
    m.merge_categories(["x", "y"], "xy")
    table = SoftLabelTable(
        classnames=["x", "y"],
        rows={
            "a": ("x", np.array([1.0, 0.0], dtype=np.float32)),
            "b": ("y", np.array([0.25, 0.75], dtype=np.float32)),
        },
    )
    mat, usable = effective_soft_matrix(table, ["a", "b"], ["xy"], manifest=m)
    assert mat.shape == (2, 1)
    assert all(usable)
    assert mat[0, 0] == pytest.approx(1.0)
    assert mat[1, 0] == pytest.approx(1.0)


def test_category_summary(tmp_path: Path):
    m = DatasetManifest(store_root=tmp_path)
    m.sync_from_scan([(f"e{i}", "a" if i < 2 else "b", None) for i in range(4)])
    m.ensure_train_val_split(val_fraction=0.25, seed=1)
    rows = m.category_summary()
    names = {r["category"] for r in rows}
    assert names == {"a", "b"}
    assert sum(r["n_total"] for r in rows) == 4
