"""Unit tests for shared categorizer evaluation engine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from LabGym.training.evaluation import (
    compute_evaluation_metrics,
    format_taxonomy_drift_message,
    hard_labels_from_targets,
    list_evaluation_runs,
    load_evaluation_run,
    model_classnames_from_parameters,
    model_settings_from_parameters_df,
    predictions_from_model_output,
    rank_high_loss_examples,
    store_behavior_categories,
    taxonomy_drift,
    top_confused_pairs_from_matrix,
    write_evaluation_run,
)


def test_compute_metrics_perfect_multiclass():
    classnames = ["a", "b", "c"]
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    m = compute_evaluation_metrics(y_true, classnames, y_pred=y_pred)
    assert m.macro_f1 == pytest.approx(1.0)
    assert m.n_examples == 6
    assert m.n_misclassified == 0
    assert m.confusion_counts.tolist() == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert m.top_confused_pairs == []
    assert all(p.confidence == 1.0 for p in m.predictions)


def test_compute_metrics_string_labels_and_confusion():
    classnames = ["groom", "rear", "walk"]
    y_true = ["groom", "groom", "rear", "walk", "walk"]
    y_pred = ["groom", "rear", "rear", "walk", "groom"]
    m = compute_evaluation_metrics(
        y_true,
        classnames,
        y_pred=y_pred,
        example_ids=["e0", "e1", "e2", "e3", "e4"],
    )
    assert m.n_misclassified == 2
    # groom→rear once, walk→groom once
    pairs = {(a, b): c for a, b, c in m.top_confused_pairs}
    assert pairs[("groom", "rear")] == 1
    assert pairs[("walk", "groom")] == 1
    assert m.predictions[1].example_id == "e1"
    assert m.predictions[1].misclassified is True
    # worst-first: walk and groom have errors
    ranked_labels = [lab for lab, _ in m.per_class_f1_worst_first]
    assert set(ranked_labels) == set(classnames)
    # F1 values non-increasing
    f1s = [f for _, f in m.per_class_f1_worst_first]
    assert f1s == sorted(f1s)


def test_compute_metrics_from_proba_binary():
    classnames = ["neg", "pos"]
    y_true = [0, 0, 1, 1]
    y_proba = np.array([0.1, 0.6, 0.9, 0.4])  # pred: 0,1,1,0 → 2 errors
    m = compute_evaluation_metrics(y_true, classnames, y_proba=y_proba)
    assert m.n_misclassified == 2
    assert m.predictions[1].pred_label == "pos"
    assert m.predictions[1].confidence == pytest.approx(0.6)
    assert m.predictions[0].confidence == pytest.approx(0.9)  # 1-0.1


def test_compute_metrics_from_proba_multiclass():
    classnames = ["a", "b", "c"]
    y_true = [0, 1, 2]
    y_proba = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
        ]
    )
    m = compute_evaluation_metrics(y_true, classnames, y_proba=y_proba)
    assert m.n_misclassified == 0
    assert m.macro_f1 == pytest.approx(1.0)
    assert m.predictions[2].confidence == pytest.approx(0.5)


def test_row_normalized_confusion():
    classnames = ["a", "b"]
    y_true = [0, 0, 0, 1]
    y_pred = [0, 0, 1, 1]
    m = compute_evaluation_metrics(y_true, classnames, y_pred=y_pred)
    # row 0: 2 correct, 1 wrong → 2/3, 1/3
    assert m.confusion_row_norm[0, 0] == pytest.approx(2 / 3)
    assert m.confusion_row_norm[0, 1] == pytest.approx(1 / 3)
    assert m.confusion_row_norm[1, 1] == pytest.approx(1.0)


def test_top_confused_pairs_sort_order():
    counts = np.array([[5, 3, 1], [0, 4, 2], [0, 0, 6]])
    pairs = top_confused_pairs_from_matrix(counts, ["a", "b", "c"], top_k=2)
    assert pairs[0] == ("a", "b", 3)
    assert pairs[1] == ("b", "c", 2)


def test_rank_high_loss():
    ranked = rank_high_loss_examples(
        ["x", "y", "z"],
        [0.1, 2.5, 1.0],
        ["a", "b", "a"],
        pred_labels=["a", "a", "a"],
        top_k=2,
    )
    assert len(ranked) == 2
    assert ranked[0].example_id == "y"
    assert ranked[0].rank == 1
    assert ranked[0].loss == pytest.approx(2.5)
    assert ranked[1].example_id == "z"


def test_write_and_load_evaluation_run(tmp_path: Path):
    classnames = ["a", "b"]
    m = compute_evaluation_metrics(
        [0, 0, 1, 1],
        classnames,
        y_pred=[0, 1, 1, 1],
        example_ids=["p0", "p1", "p2", "p3"],
        confidences=[0.9, 0.55, 0.8, 0.7],
    )
    hl = rank_high_loss_examples(
        ["t0", "t1"], [1.2, 0.3], ["a", "b"], pred_labels=["b", "b"], top_k=1
    )
    run_dir = write_evaluation_run(
        tmp_path / "model",
        m,
        run_id="unit_run",
        source="test",
        model_settings={"time_step": 15, "network": 0},
        ground_truth_snapshot={"path": "/data/gt", "n": 4},
        high_loss=hl,
    )
    assert run_dir == tmp_path / "model" / "eval" / "unit_run"
    for name in (
        "run_meta.json",
        "confusion_counts.json",
        "confusion_row_norm.json",
        "classification_report.json",
        "metrics_summary.json",
        "predictions.csv",
        "high_loss.csv",
        "classification_report.csv",
    ):
        assert (run_dir / name).is_file(), name

    loaded = load_evaluation_run(run_dir)
    assert loaded["run_meta"]["source"] == "test"
    assert loaded["run_meta"]["model_settings"]["time_step"] == 15
    assert loaded["metrics_summary"]["n_misclassified"] == 1
    assert loaded["metrics_summary"]["macro_f1"] == pytest.approx(m.macro_f1)
    assert list(loaded["predictions"]["example_id"]) == ["p0", "p1", "p2", "p3"]
    assert loaded["high_loss"].iloc[0]["example_id"] == "t0"

    counts = json.loads((run_dir / "confusion_counts.json").read_text(encoding="utf-8"))
    assert counts["classnames"] == classnames
    assert counts["matrix"][0][1] == 1  # one a→b error


def test_hard_labels_from_targets():
    # one-hot
    y = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)
    assert hard_labels_from_targets(y, 3).tolist() == [0, 2]
    # stacked hard+soft
    hard = np.array([[1, 0], [0, 1]], dtype=np.float32)
    soft = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
    stacked = np.concatenate([hard, soft], axis=1)
    assert hard_labels_from_targets(stacked, 2).tolist() == [0, 1]
    # binary column
    yb = np.array([[0.0], [1.0], [0.2]])
    assert hard_labels_from_targets(yb, 2).tolist() == [0, 1, 0]


def test_predictions_from_model_output():
    multi = np.array([[0.1, 0.7, 0.2], [0.9, 0.05, 0.05]])
    pred, conf = predictions_from_model_output(multi, 3)
    assert pred.tolist() == [1, 0]
    assert conf[0] == pytest.approx(0.7)

    binary = np.array([[0.2], [0.8]])
    pred2, conf2 = predictions_from_model_output(binary, 2)
    assert pred2.tolist() == [0, 1]
    assert conf2[0] == pytest.approx(0.8)


def test_model_settings_from_parameters_df():
    df = pd.DataFrame(
        {
            "classnames": ["a", "b", "c"],
            "time_step": [15, np.nan, np.nan],
            "network": [2, np.nan, np.nan],
            "label_mode": ["hard_only", np.nan, np.nan],
        }
    )
    settings = model_settings_from_parameters_df(df)
    assert settings["classnames"] == ["a", "b", "c"]
    assert settings["time_step"] == 15
    assert settings["network"] == 2
    assert settings["label_mode"] == "hard_only"


def test_unknown_label_raises():
    with pytest.raises(ValueError, match="unknown label"):
        compute_evaluation_metrics(["a", "ghost"], ["a", "b"], y_pred=["a", "b"])


def test_list_evaluation_runs_newest_first(tmp_path: Path):
    model = tmp_path / "m"
    classnames = ["a", "b"]
    m = compute_evaluation_metrics(
        [0, 1], classnames, y_pred=[0, 1], example_ids=["e0", "e1"]
    )
    write_evaluation_run(
        model,
        m,
        run_id="older",
        source="test",
        extra_meta={"created_utc": "2020-01-01T00:00:00+00:00"},
    )
    # write_evaluation_run always stamps created_utc; rewrite meta for sort test
    older_meta = model / "eval" / "older" / "run_meta.json"
    data = json.loads(older_meta.read_text(encoding="utf-8"))
    data["created_utc"] = "2020-01-01T00:00:00+00:00"
    older_meta.write_text(json.dumps(data), encoding="utf-8")

    write_evaluation_run(
        model,
        m,
        run_id="newer",
        source="evaluate",
        extra_meta={"created_utc": "2025-06-01T12:00:00+00:00"},
    )
    newer_meta = model / "eval" / "newer" / "run_meta.json"
    data = json.loads(newer_meta.read_text(encoding="utf-8"))
    data["created_utc"] = "2025-06-01T12:00:00+00:00"
    newer_meta.write_text(json.dumps(data), encoding="utf-8")

    # Empty dir without meta still listed
    (model / "eval" / "orphan").mkdir()

    runs = list_evaluation_runs(model)
    ids = [r.run_id for r in runs]
    assert ids[0] == "newer"
    assert "older" in ids
    assert "orphan" in ids
    assert runs[0].source == "evaluate"
    assert runs[0].macro_f1 == pytest.approx(1.0)
    assert "F1=" in runs[0].display_label()


def test_taxonomy_drift_and_store_categories(tmp_path: Path):
    store = tmp_path / "examples"
    (store / "groom").mkdir(parents=True)
    (store / "walk").mkdir()
    (store / "rear").mkdir()
    cats = store_behavior_categories(store)
    assert cats == ["groom", "rear", "walk"]

    drift = taxonomy_drift(["groom", "walk", "climb"], cats)
    assert drift["has_drift"] is True
    assert drift["only_in_model"] == ["climb"]
    assert drift["only_in_store"] == ["rear"]
    msg = format_taxonomy_drift_message(drift)
    assert "climb" in msg and "rear" in msg
    assert format_taxonomy_drift_message(taxonomy_drift(cats, cats)) == ""


def test_model_classnames_from_parameters(tmp_path: Path):
    model = tmp_path / "cat"
    model.mkdir()
    pd.DataFrame(
        {"classnames": ["a", "b"], "network": [0, None], "time_step": [8, None]}
    ).to_csv(model / "model_parameters.txt", index=False)
    assert model_classnames_from_parameters(model) == ["a", "b"]
    assert model_classnames_from_parameters(tmp_path / "missing") == []
