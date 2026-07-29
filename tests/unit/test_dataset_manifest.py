"""Unit tests for dataset manifest, splits, and effective view."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from LabGym.training.dataset_manifest import (
    MANIFEST_FILENAME,
    SPLIT_SEALED_TEST,
    SPLIT_TRAIN,
    SPLIT_UNASSIGNED,
    SPLIT_VALIDATION,
    DatasetManifest,
    example_id_from_path,
    original_label_from_flat_name,
    resolve_train_val_paths,
    scan_behavior_folder_store,
    scan_flat_example_store,
)
from LabGym.training.evaluation import (
    high_loss_from_predictions,
    per_example_cross_entropy,
)


def _touch_flat(store: Path, name: str) -> Path:
    p = store / name
    p.write_bytes(b"x")
    return p


def test_example_id_and_label_helpers():
    assert example_id_from_path("a/b/clip_mouse_0_10_len15_groom.jpg") == "clip_mouse_0_10_len15_groom"
    assert original_label_from_flat_name("clip_mouse_0_10_len15_groom.jpg") == "groom"


def test_exclude_recategorize_undo(tmp_path: Path):
    m = DatasetManifest(store_root=tmp_path)
    m.sync_from_scan(
        [
            ("e1", "groom", "e1_groom.jpg"),
            ("e2", "rear", "e2_rear.jpg"),
        ]
    )
    m.exclude("e1")
    assert m.examples["e1"].excluded is True
    eff = m.effective_examples()
    assert [e.example_id for e in eff] == ["e2"]

    m.recategorize("e2", "groom")
    assert m.examples["e2"].active_label == "groom"
    assert m.examples["e2"].original_label == "rear"

    assert m.undo()  # undo recategorize
    assert m.examples["e2"].label_override is None
    assert m.examples["e2"].active_label == "rear"

    assert m.undo()  # undo exclude
    assert m.examples["e1"].excluded is False
    assert len(m.effective_examples()) == 2


def test_keep_and_ensure_example(tmp_path: Path):
    m = DatasetManifest(store_root=tmp_path)
    rec = m.ensure_example("new1", "groom", path_hint="new1_groom.jpg")
    assert rec.example_id == "new1"
    assert "new1" in m.examples
    m.exclude("new1")
    m.keep("new1")
    assert m.examples["new1"].excluded is False
    assert m.undo()  # undo keep restores excluded
    assert m.examples["new1"].excluded is True


def test_train_val_split_stable_and_persists(tmp_path: Path):
    m = DatasetManifest(store_root=tmp_path)
    scanned = []
    for i in range(10):
        lab = "a" if i < 5 else "b"
        scanned.append((f"ex{i}", lab, f"ex{i}_{lab}.jpg"))
    m.sync_from_scan(scanned)
    counts1 = m.ensure_train_val_split(val_fraction=0.2, seed=7)
    assert counts1[SPLIT_TRAIN] + counts1[SPLIT_VALIDATION] == 10
    assert counts1[SPLIT_VALIDATION] >= 2  # stratified ~1 per class
    assignment1 = {eid: r.split for eid, r in m.examples.items()}

    # Second call without regenerate keeps membership
    m.ensure_train_val_split(val_fraction=0.2, seed=7, regenerate=False)
    assignment2 = {eid: r.split for eid, r in m.examples.items()}
    assert assignment1 == assignment2

    # New example stays unassigned until assign_new
    m.sync_from_scan(scanned + [("ex_new", "a", "ex_new_a.jpg")])
    m.ensure_train_val_split(val_fraction=0.2, seed=7, assign_new=False)
    assert m.examples["ex_new"].split == SPLIT_UNASSIGNED
    m.ensure_train_val_split(val_fraction=0.2, seed=7, assign_new=True)
    assert m.examples["ex_new"].split in (SPLIT_TRAIN, SPLIT_VALIDATION)

    path = m.save()
    assert path.name == MANIFEST_FILENAME
    loaded = DatasetManifest.load(tmp_path)
    assert loaded.examples["ex0"].split == assignment1["ex0"]


def test_sealed_test_never_in_train_time(tmp_path: Path):
    m = DatasetManifest(store_root=tmp_path)
    m.sync_from_scan([(f"e{i}", "a" if i % 2 == 0 else "b", f"e{i}.jpg") for i in range(20)])
    m.ensure_train_val_split(val_fraction=0.2, seed=1)
    sealed = m.assign_sealed_test(fraction=0.2, seed=1)
    assert len(sealed) >= 1
    train_time = set(m.train_time_example_ids())
    for eid in sealed:
        assert eid not in train_time
        assert m.examples[eid].split == SPLIT_SEALED_TEST
    # sealed not in effective train/val
    for e in m.effective_examples(splits=[SPLIT_TRAIN, SPLIT_VALIDATION]):
        assert e.example_id not in set(sealed)


def test_scan_flat_and_behavior(tmp_path: Path):
    flat = tmp_path / "flat"
    flat.mkdir()
    _touch_flat(flat, "clip_0_groom.jpg")
    _touch_flat(flat, "clip_1_rear.jpg")
    (flat / "train").mkdir()
    _touch_flat(flat / "train", "clip_2_groom.jpg")
    hits = scan_flat_example_store(flat)
    ids = {h[0] for h in hits}
    assert "clip_0_groom" in ids
    assert "clip_2_groom" in ids

    beh = tmp_path / "beh"
    (beh / "groom").mkdir(parents=True)
    (beh / "rear").mkdir()
    (beh / "groom" / "a.jpg").write_bytes(b"x")
    (beh / "rear" / "b.jpg").write_bytes(b"x")
    bhits = scan_behavior_folder_store(beh)
    assert {h[1] for h in bhits} == {"groom", "rear"}


def test_resolve_train_val_paths_respects_sealed_and_override(tmp_path: Path):
    paths = []
    labels = []
    for i in range(12):
        lab = "groom" if i < 6 else "rear"
        name = f"ex{i}_{lab}.jpg"
        p = _touch_flat(tmp_path, name)
        paths.append(str(p))
        labels.append(lab)

    tr, va, ytr, yva, m = resolve_train_val_paths(
        tmp_path, paths, labels, seed=0, val_fraction=0.25, persist=True
    )
    assert m is not None
    assert len(tr) + len(va) == 12
    assert set(ytr + yva) <= {"groom", "rear"}

    # Seal two train examples and exclude one
    train_ids = [example_id_from_path(p) for p in tr]
    m.assign_sealed_test(example_ids=train_ids[:2])
    m.exclude(example_id_from_path(tr[2]))
    m.recategorize(example_id_from_path(tr[3]), "rear")
    m.save()

    tr2, va2, ytr2, yva2, m2 = resolve_train_val_paths(
        tmp_path, paths, labels, seed=0, val_fraction=0.25, persist=True
    )
    sealed = set(m2.sealed_test_example_ids())
    for p in tr2 + va2:
        assert example_id_from_path(p) not in sealed
    assert example_id_from_path(tr[2]) not in [example_id_from_path(p) for p in tr2 + va2]
    # Override applied for surviving example
    eid3 = example_id_from_path(tr[3])
    if eid3 in [example_id_from_path(p) for p in tr2]:
        idx = [example_id_from_path(p) for p in tr2].index(eid3)
        assert ytr2[idx] == "rear"


def test_per_example_cross_entropy_and_high_loss():
    # Multiclass: true class 0 vs confident wrong
    y_true = [0, 1, 2]
    proba = np.array(
        [
            [0.05, 0.9, 0.05],  # high loss
            [0.1, 0.8, 0.1],  # low loss
            [0.2, 0.2, 0.6],
        ]
    )
    losses = per_example_cross_entropy(y_true, proba, 3)
    assert losses[0] > losses[1]
    ranked = high_loss_from_predictions(
        ["a", "b", "c"], y_true, proba, ["x", "y", "z"], top_k=2
    )
    assert ranked[0].example_id == "a"
    assert ranked[0].rank == 1
    assert ranked[0].true_label == "x"
    assert ranked[0].pred_label == "y"

    # Binary: true 0 with p(pos)=0.9 is worse than true 1 with p=0.2
    yb = [0, 1, 1]
    pb = np.array([0.9, 0.2, 0.95])
    lb = per_example_cross_entropy(yb, pb, 2)
    assert lb[0] == max(lb)
    assert lb[2] == min(lb)
