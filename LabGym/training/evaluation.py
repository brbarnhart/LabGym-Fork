"""Shared categorizer evaluation metrics and durable run artifacts.

One evaluation engine for Test categorizer, Manage dataset → Evaluate, and
post-train hold-out scoring. Artifacts live under::

    <model_dir>/eval/<run_id>/

See ``docs/adr/0003-evaluation-artifacts-with-categorizer.md``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

PathLike = Union[str, Path]


@dataclass
class ExamplePrediction:
    """One example's ground truth, prediction, and confidence."""

    example_id: str
    true_label: str
    pred_label: str
    confidence: float
    misclassified: bool = False

    def __post_init__(self) -> None:
        self.misclassified = self.true_label != self.pred_label


@dataclass
class HighLossExample:
    """End-of-train high-loss row (train-for-weights partition only)."""

    example_id: str
    loss: float
    true_label: str
    pred_label: str = ""
    rank: int = 0


@dataclass
class EvaluationMetrics:
    """Full metrics package for one evaluation run."""

    classnames: List[str]
    confusion_counts: np.ndarray  # (C, C) true rows, pred cols
    confusion_row_norm: np.ndarray
    classification_report: Dict[str, Any]
    macro_f1: float
    per_class_f1_worst_first: List[Tuple[str, float]]
    top_confused_pairs: List[Tuple[str, str, int]]
    predictions: List[ExamplePrediction] = field(default_factory=list)
    n_examples: int = 0
    n_misclassified: int = 0

    def __post_init__(self) -> None:
        self.n_examples = len(self.predictions) if self.predictions else int(
            np.asarray(self.confusion_counts).sum()
        )
        if self.predictions:
            self.n_misclassified = sum(1 for p in self.predictions if p.misclassified)


def _to_int_labels(
    y: Sequence[Any],
    classnames: Sequence[str],
    *,
    name: str,
) -> np.ndarray:
    """Map string or integer labels into class index space."""
    arr = np.asarray(y)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    name_to_i = {str(c): i for i, c in enumerate(classnames)}
    if arr.dtype.kind in ("U", "S", "O"):
        out = np.empty(arr.shape[0], dtype=np.int64)
        for i, v in enumerate(arr):
            key = str(v)
            if key not in name_to_i:
                raise ValueError(f"{name} contains unknown label {key!r}; known={list(classnames)}")
            out[i] = name_to_i[key]
        return out
    return arr.astype(np.int64, copy=False)


def _pred_indices_and_confidence(
    y_pred: Optional[Sequence[Any]],
    y_proba: Optional[np.ndarray],
    classnames: Sequence[str],
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve predicted class indices and per-example confidence."""
    C = len(classnames)
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=np.float64)
        if proba.ndim == 1:
            # Binary sigmoid: P(class1)
            if C != 2:
                raise ValueError("1-D y_proba requires exactly 2 classnames")
            conf = np.where(proba >= 0.5, proba, 1.0 - proba)
            pred = (proba >= 0.5).astype(np.int64)
            return pred, conf
        if proba.ndim != 2:
            raise ValueError(f"y_proba must be 1-D or 2-D, got shape {proba.shape}")
        if proba.shape[1] == 1 and C == 2:
            p1 = proba[:, 0]
            conf = np.where(p1 >= 0.5, p1, 1.0 - p1)
            pred = (p1 >= 0.5).astype(np.int64)
            return pred, conf
        if proba.shape[1] != C:
            raise ValueError(
                f"y_proba columns ({proba.shape[1]}) must match classnames ({C})"
            )
        pred = proba.argmax(axis=1).astype(np.int64)
        conf = proba.max(axis=1)
        return pred, conf

    if y_pred is None:
        raise ValueError("Provide y_pred and/or y_proba")
    pred = _to_int_labels(y_pred, classnames, name="y_pred")
    if pred.shape[0] != n:
        raise ValueError(f"y_pred length {pred.shape[0]} != y_true length {n}")
    conf = np.ones(n, dtype=np.float64)
    return pred, conf


def top_confused_pairs_from_matrix(
    counts: np.ndarray,
    classnames: Sequence[str],
    *,
    top_k: int = 20,
) -> List[Tuple[str, str, int]]:
    """Off-diagonal confusion cells sorted by count (true → pred)."""
    C = len(classnames)
    pairs: List[Tuple[str, str, int]] = []
    mat = np.asarray(counts, dtype=np.int64)
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            c = int(mat[i, j])
            if c > 0:
                pairs.append((str(classnames[i]), str(classnames[j]), c))
    pairs.sort(key=lambda t: (-t[2], t[0], t[1]))
    if top_k is not None and top_k >= 0:
        return pairs[:top_k]
    return pairs


def compute_evaluation_metrics(
    y_true: Sequence[Any],
    classnames: Sequence[str],
    *,
    y_pred: Optional[Sequence[Any]] = None,
    y_proba: Optional[np.ndarray] = None,
    example_ids: Optional[Sequence[str]] = None,
    confidences: Optional[Sequence[float]] = None,
    top_confused_k: int = 20,
    zero_division: int = 0,
) -> EvaluationMetrics:
    """Build the standard metrics package from labels / probabilities.

    Scoring is always in **model label space** (``classnames`` order). Taxonomy
    drift is the caller's concern (banner in UI); this function does not remap.

    Args:
        y_true: Ground-truth labels (indices or names in ``classnames``).
        classnames: Ordered behavior category names for the categorizer.
        y_pred: Predicted labels (indices or names). Optional if ``y_proba`` set.
        y_proba: Model probabilities. Shape ``(N,)`` / ``(N,1)`` for binary
            sigmoid or ``(N, C)`` for multiclass.
        example_ids: Optional stable ids (paths or basenames) for review tables.
        confidences: Override confidences; otherwise max proba or 1.0.
        top_confused_k: Max off-diagonal pairs to keep (0 = all).
        zero_division: Passed to sklearn report / F1.

    Returns:
        EvaluationMetrics with matrices, report, F1 rankings, pairs, predictions.
    """
    names = [str(c) for c in classnames]
    if not names:
        raise ValueError("classnames must be non-empty")

    true_idx = _to_int_labels(y_true, names, name="y_true")
    n = int(true_idx.shape[0])
    pred_idx, conf_from_proba = _pred_indices_and_confidence(
        y_pred, y_proba, names, n
    )
    if confidences is not None:
        conf = np.asarray(confidences, dtype=np.float64)
        if conf.shape[0] != n:
            raise ValueError("confidences length must match y_true")
    else:
        conf = conf_from_proba

    labels = list(range(len(names)))
    counts = confusion_matrix(true_idx, pred_idx, labels=labels)
    row_sums = counts.sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_norm = np.divide(
            counts.astype(np.float64),
            row_sums,
            out=np.zeros_like(counts, dtype=np.float64),
            where=row_sums > 0,
        )

    report = classification_report(
        true_idx,
        pred_idx,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=zero_division,
    )
    per_f1 = f1_score(
        true_idx,
        pred_idx,
        labels=labels,
        average=None,
        zero_division=zero_division,
    )
    macro = float(
        f1_score(
            true_idx,
            pred_idx,
            labels=labels,
            average="macro",
            zero_division=zero_division,
        )
    )
    ranked = sorted(
        ((names[i], float(per_f1[i])) for i in range(len(names))),
        key=lambda t: (t[1], t[0]),
    )
    k = top_confused_k if top_confused_k > 0 else -1
    pairs = top_confused_pairs_from_matrix(counts, names, top_k=k if k > 0 else 10**9)

    if example_ids is None:
        ids = [str(i) for i in range(n)]
    else:
        ids = [str(x) for x in example_ids]
        if len(ids) != n:
            raise ValueError("example_ids length must match y_true")

    preds = [
        ExamplePrediction(
            example_id=ids[i],
            true_label=names[int(true_idx[i])],
            pred_label=names[int(pred_idx[i])],
            confidence=float(conf[i]),
        )
        for i in range(n)
    ]

    return EvaluationMetrics(
        classnames=names,
        confusion_counts=counts.astype(np.int64),
        confusion_row_norm=row_norm,
        classification_report=report,
        macro_f1=macro,
        per_class_f1_worst_first=ranked,
        top_confused_pairs=pairs,
        predictions=preds,
    )


def rank_high_loss_examples(
    example_ids: Sequence[str],
    losses: Sequence[float],
    true_labels: Sequence[str],
    *,
    pred_labels: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
) -> List[HighLossExample]:
    """Rank train-partition examples by loss (highest first).

    High loss alone does not imply a wrong label; rows are review candidates.
    """
    ids = list(example_ids)
    loss_arr = np.asarray(losses, dtype=np.float64)
    trues = [str(t) for t in true_labels]
    if not (len(ids) == len(loss_arr) == len(trues)):
        raise ValueError("example_ids, losses, and true_labels must have equal length")
    if pred_labels is None:
        preds = [""] * len(ids)
    else:
        preds = [str(p) for p in pred_labels]
        if len(preds) != len(ids):
            raise ValueError("pred_labels length must match example_ids")

    order = np.argsort(-loss_arr, kind="stable")
    if top_k is not None and top_k >= 0:
        order = order[:top_k]
    out: List[HighLossExample] = []
    for rank, idx in enumerate(order, start=1):
        out.append(
            HighLossExample(
                example_id=ids[int(idx)],
                loss=float(loss_arr[int(idx)]),
                true_label=trues[int(idx)],
                pred_label=preds[int(idx)],
                rank=rank,
            )
        )
    return out


def make_run_id(prefix: str = "") -> str:
    """UTC timestamp + short uuid for unique eval run directories."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    base = f"{stamp}_{short}"
    return f"{prefix}_{base}" if prefix else base


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def write_evaluation_run(
    model_dir: PathLike,
    metrics: EvaluationMetrics,
    *,
    run_id: Optional[str] = None,
    source: str = "test",
    model_settings: Optional[Mapping[str, Any]] = None,
    ground_truth_snapshot: Optional[Mapping[str, Any]] = None,
    high_loss: Optional[Sequence[HighLossExample]] = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Persist an evaluation run under ``<model_dir>/eval/<run_id>/``.

    Writes:

    - ``run_meta.json`` — source, timestamps, settings, GT snapshot
    - ``confusion_counts.json`` / ``confusion_row_norm.json``
    - ``classification_report.json``
    - ``metrics_summary.json`` — macro F1, worst-first F1, top pairs
    - ``predictions.csv`` — per-example review table
    - ``high_loss.csv`` — optional end-of-train ranks

    Args:
        model_dir: Trained categorizer directory.
        metrics: Output of :func:`compute_evaluation_metrics`.
        run_id: Directory name; generated if omitted.
        source: Provenance tag (``test``, ``train_holdout``, ``evaluate``, …).
        model_settings: Snapshot of training / model parameters.
        ground_truth_snapshot: Paths, split, n_examples, taxonomy notes.
        high_loss: Optional high-loss table (train partition only).
        extra_meta: Merged into ``run_meta.json``.

    Returns:
        Path to the run directory.
    """
    model_path = Path(model_dir)
    rid = run_id or make_run_id(prefix=source)
    run_dir = model_path / "eval" / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    names = metrics.classnames
    counts_payload = {
        "classnames": names,
        "matrix": _json_safe(metrics.confusion_counts),
        "axis": "rows=true, cols=pred",
    }
    row_norm_payload = {
        "classnames": names,
        "matrix": _json_safe(metrics.confusion_row_norm),
        "axis": "rows=true, cols=pred",
    }
    (run_dir / "confusion_counts.json").write_text(
        json.dumps(counts_payload, indent=2), encoding="utf-8"
    )
    (run_dir / "confusion_row_norm.json").write_text(
        json.dumps(row_norm_payload, indent=2), encoding="utf-8"
    )
    (run_dir / "classification_report.json").write_text(
        json.dumps(_json_safe(metrics.classification_report), indent=2),
        encoding="utf-8",
    )

    n_ex = int(metrics.n_examples)
    n_mis = int(metrics.n_misclassified)
    accuracy = float(n_ex - n_mis) / float(n_ex) if n_ex > 0 else None
    summary = {
        "macro_f1": metrics.macro_f1,
        "accuracy": accuracy,
        "n_examples": n_ex,
        "n_misclassified": n_mis,
        "per_class_f1_worst_first": [
            {"label": lab, "f1": f1} for lab, f1 in metrics.per_class_f1_worst_first
        ],
        "top_confused_pairs": [
            {"true": a, "pred": b, "count": c}
            for a, b, c in metrics.top_confused_pairs
        ],
        "classnames": names,
    }
    (run_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    pred_rows = [
        {
            "example_id": p.example_id,
            "true_label": p.true_label,
            "pred_label": p.pred_label,
            "confidence": p.confidence,
            "misclassified": int(p.misclassified),
        }
        for p in metrics.predictions
    ]
    pd.DataFrame(pred_rows).to_csv(run_dir / "predictions.csv", index=False)

    if high_loss is not None:
        hl_rows = [asdict(h) for h in high_loss]
        pd.DataFrame(hl_rows).to_csv(run_dir / "high_loss.csv", index=False)

    # Also write sklearn-style CSV for continuity with training_metrics.csv
    try:
        pd.DataFrame(metrics.classification_report).transpose().to_csv(
            run_dir / "classification_report.csv", float_format="%.4f"
        )
    except Exception:
        pass

    meta: Dict[str, Any] = {
        "run_id": rid,
        "source": source,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_path.resolve()),
        "classnames": names,
        "n_examples": metrics.n_examples,
        "n_misclassified": metrics.n_misclassified,
        "macro_f1": metrics.macro_f1,
        "model_settings": dict(model_settings or {}),
        "ground_truth_snapshot": dict(ground_truth_snapshot or {}),
    }
    if extra_meta:
        meta.update(dict(extra_meta))
    (run_dir / "run_meta.json").write_text(
        json.dumps(_json_safe(meta), indent=2), encoding="utf-8"
    )
    return run_dir


def load_evaluation_run(run_dir: PathLike) -> Dict[str, Any]:
    """Load a previously written evaluation run into a plain dict.

    Returns keys for ``run_meta``, ``metrics_summary``, ``predictions`` (DataFrame),
    ``confusion_counts``, and optional ``high_loss``.
    """
    path = Path(run_dir)
    out: Dict[str, Any] = {"run_dir": str(path)}
    meta_path = path / "run_meta.json"
    if meta_path.is_file():
        out["run_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    summary_path = path / "metrics_summary.json"
    if summary_path.is_file():
        out["metrics_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    counts_path = path / "confusion_counts.json"
    if counts_path.is_file():
        out["confusion_counts"] = json.loads(counts_path.read_text(encoding="utf-8"))
    row_norm_path = path / "confusion_row_norm.json"
    if row_norm_path.is_file():
        out["confusion_row_norm"] = json.loads(row_norm_path.read_text(encoding="utf-8"))
    pred_path = path / "predictions.csv"
    if pred_path.is_file():
        out["predictions"] = pd.read_csv(pred_path)
    hl_path = path / "high_loss.csv"
    if hl_path.is_file():
        out["high_loss"] = pd.read_csv(hl_path)
    report_path = path / "classification_report.json"
    if report_path.is_file():
        out["classification_report"] = json.loads(
            report_path.read_text(encoding="utf-8")
        )
    return out


@dataclass
class EvaluationRunInfo:
    """Lightweight index row for one eval run under a model directory."""

    run_id: str
    run_dir: str
    source: str = ""
    created_utc: str = ""
    macro_f1: Optional[float] = None
    n_examples: int = 0
    n_misclassified: int = 0
    ground_truth_path: str = ""
    model_settings: Dict[str, Any] = field(default_factory=dict)

    def display_label(self) -> str:
        """Short list label for UI combos / list widgets."""
        f1 = f"F1={self.macro_f1:.3f}" if self.macro_f1 is not None else "F1=?"
        src = self.source or "run"
        when = (self.created_utc or "")[:19].replace("T", " ")
        return f"{self.run_id}  [{src}]  {f1}  n={self.n_examples}  {when}".strip()


def list_evaluation_runs(model_dir: PathLike) -> List[EvaluationRunInfo]:
    """List evaluation runs under ``<model_dir>/eval/``, newest first.

    Runs without ``run_meta.json`` are still listed if the directory exists
    (using the directory name as ``run_id``).
    """
    root = Path(model_dir) / "eval"
    if not root.is_dir():
        return []
    infos: List[EvaluationRunInfo] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        meta: Dict[str, Any] = {}
        meta_path = child / "run_meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                meta = {}
        summary: Dict[str, Any] = {}
        summary_path = child / "metrics_summary.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                summary = {}
        gt = meta.get("ground_truth_snapshot") or {}
        gt_path = ""
        if isinstance(gt, dict):
            gt_path = str(gt.get("path") or gt.get("ground_truth_path") or "")
        macro = meta.get("macro_f1", summary.get("macro_f1"))
        try:
            macro_f = float(macro) if macro is not None else None
        except (TypeError, ValueError):
            macro_f = None
        n_ex = int(meta.get("n_examples") or summary.get("n_examples") or 0)
        n_mis = int(
            meta.get("n_misclassified") or summary.get("n_misclassified") or 0
        )
        settings = meta.get("model_settings") or {}
        if not isinstance(settings, dict):
            settings = {}
        infos.append(
            EvaluationRunInfo(
                run_id=str(meta.get("run_id") or child.name),
                run_dir=str(child),
                source=str(meta.get("source") or ""),
                created_utc=str(meta.get("created_utc") or ""),
                macro_f1=macro_f,
                n_examples=n_ex,
                n_misclassified=n_mis,
                ground_truth_path=gt_path,
                model_settings=dict(settings),
            )
        )

    def _sort_key(info: EvaluationRunInfo) -> Tuple[str, str]:
        # Newest first; fall back to run_id
        return (info.created_utc or "", info.run_id)

    infos.sort(key=_sort_key, reverse=True)
    return infos


def store_behavior_categories(store_path: PathLike) -> List[str]:
    """Immediate subfolder names of an example store (behavior categories)."""
    root = Path(store_path)
    if not root.is_dir():
        return []
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def model_classnames_from_parameters(model_dir: PathLike) -> List[str]:
    """Read ``classnames`` from a categorizer's ``model_parameters.txt``."""
    path = Path(model_dir) / "model_parameters.txt"
    if not path.is_file():
        return []
    try:
        params = pd.read_csv(path)
    except Exception:
        return []
    if "classnames" not in params.columns:
        return []
    return [str(v) for v in params["classnames"].tolist() if pd.notna(v)]


def taxonomy_drift(
    model_classnames: Sequence[str],
    store_categories: Sequence[str],
) -> Dict[str, Any]:
    """Compare model label space to example-store category folders.

    Scoring always uses the model label space; this report is for UI banners.
    """
    model_set = {str(c) for c in model_classnames}
    store_set = {str(c) for c in store_categories}
    only_model = sorted(model_set - store_set)
    only_store = sorted(store_set - model_set)
    shared = sorted(model_set & store_set)
    return {
        "has_drift": bool(only_model or only_store),
        "only_in_model": only_model,
        "only_in_store": only_store,
        "shared": shared,
        "model_classnames": list(model_classnames),
        "store_categories": list(store_categories),
    }


def align_store_labels_to_model(
    store_categories: Sequence[str],
    model_classnames: Sequence[str],
) -> Dict[str, Any]:
    """Plan evaluation under **model label space** when taxonomies drift.

    Store folders whose names match a model classname are scorable; their
    ground-truth index is the model's class index. Store-only folders are
    skipped (not remapped). Model-only classes remain in the report with
    zero support. Scoring is allowed when at least one category is shared.

    Returns:
        Dict with ``can_score``, ``scorable_categories``, ``only_in_store``,
        ``only_in_model``, ``shared``, ``label_to_index`` (store label → model
        index for scorable names only), and ``model_classnames``.
    """
    model = [str(c) for c in model_classnames]
    store = [str(c) for c in store_categories]
    model_set = set(model)
    store_set = set(store)
    only_model = sorted(model_set - store_set)
    only_store = sorted(store_set - model_set)
    shared = sorted(model_set & store_set)
    # Preserve store listing order for scorable, but only shared names
    scorable = [c for c in store if c in model_set]
    # Deduplicate while preserving order
    seen: set = set()
    scorable_unique: List[str] = []
    for c in scorable:
        if c not in seen:
            seen.add(c)
            scorable_unique.append(c)
    label_to_index = {name: model.index(name) for name in scorable_unique}
    return {
        "can_score": bool(scorable_unique),
        "scorable_categories": scorable_unique,
        "only_in_store": only_store,
        "only_in_model": only_model,
        "shared": shared,
        "label_to_index": label_to_index,
        "model_classnames": list(model),
        "store_categories": list(store),
        "has_drift": bool(only_model or only_store),
    }


def format_taxonomy_drift_message(drift: Mapping[str, Any]) -> str:
    """Human-readable taxonomy-drift banner text (empty if no drift)."""
    if not drift.get("has_drift"):
        return ""
    parts: List[str] = ["Taxonomy drift: scoring uses the model's class list."]
    only_m = drift.get("only_in_model") or []
    only_s = drift.get("only_in_store") or []
    if only_m:
        parts.append(f"Only in model: {', '.join(only_m)}.")
    if only_s:
        parts.append(f"Only in ground-truth folders: {', '.join(only_s)}.")
    return " ".join(parts)


def model_settings_from_parameters_df(parameters: pd.DataFrame) -> Dict[str, Any]:
    """Flatten a ``model_parameters.txt`` DataFrame into a JSON-friendly dict."""
    settings: Dict[str, Any] = {}
    if parameters is None or len(parameters.columns) == 0:
        return settings
    for col in parameters.columns:
        series = parameters[col]
        if col == "classnames":
            settings["classnames"] = [str(v) for v in series.tolist() if pd.notna(v)]
        else:
            val = series.iloc[0] if len(series) else None
            if pd.isna(val):
                continue
            # Prefer plain Python types
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    val = str(val)
            settings[str(col)] = val
    return settings


def model_settings_from_model_dir(model_dir: PathLike) -> Dict[str, Any]:
    """Load training settings from ``model_parameters.txt`` under a model folder."""
    path = Path(model_dir) / "model_parameters.txt"
    if not path.is_file():
        return {}
    try:
        params = pd.read_csv(path)
    except Exception:
        return {}
    return model_settings_from_parameters_df(params)


# Columns for side-by-side already-trained model comparison (UI + CSV export).
COMPARE_ROW_KEYS: Tuple[str, ...] = (
    "model",
    "macro_f1",
    "accuracy",
    "n_examples",
    "n_misclassified",
    "worst_class",
    "worst_f1",
    "time_step",
    "network",
    "level",
    "dim",
    "label_mode",
    "lambda_soft",
    "run_id",
    "source",
    "metrics_mode",
    "error",
)


def format_level_summary(settings: Optional[Mapping[str, Any]]) -> str:
    """Human-readable network depth(s) from model settings."""
    if not settings:
        return ""
    tconv = settings.get("level_tconv")
    conv = settings.get("level_conv", settings.get("level"))
    if tconv is not None and conv is not None and str(tconv) != str(conv):
        return f"t{tconv}/c{conv}"
    if conv is not None:
        return str(conv)
    if tconv is not None:
        return str(tconv)
    return ""


def format_dim_summary(settings: Optional[Mapping[str, Any]]) -> str:
    """Human-readable input dim(s) from model settings."""
    if not settings:
        return ""
    tconv = settings.get("dim_tconv")
    conv = settings.get("dim_conv", settings.get("dim"))
    if tconv is not None and conv is not None and str(tconv) != str(conv):
        return f"t{tconv}/c{conv}"
    if conv is not None:
        return str(conv)
    if tconv is not None:
        return str(tconv)
    return ""


def _as_optional_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def worst_class_from_summary(
    metrics_summary: Optional[Mapping[str, Any]],
) -> Tuple[str, Optional[float]]:
    """Return (worst_class_label, worst_f1) from metrics_summary, if present."""
    if not metrics_summary:
        return "", None
    ranked = metrics_summary.get("per_class_f1_worst_first") or []
    if not ranked:
        return "", None
    first = ranked[0]
    if isinstance(first, Mapping):
        lab = str(first.get("label") or first.get("class") or "")
        return lab, _as_optional_float(first.get("f1"))
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        return str(first[0]), _as_optional_float(first[1])
    return "", None


def accuracy_from_counts(
    n_examples: Any,
    n_misclassified: Any,
    *,
    explicit: Any = None,
) -> Optional[float]:
    """Accuracy from explicit value or (n - mis) / n."""
    acc = _as_optional_float(explicit)
    if acc is not None:
        return acc
    try:
        n = int(n_examples)
        mis = int(n_misclassified)
    except (TypeError, ValueError):
        return None
    if n <= 0:
        return None
    return float(n - mis) / float(n)


def build_compare_row(
    *,
    model_path: PathLike,
    run_meta: Optional[Mapping[str, Any]] = None,
    metrics_summary: Optional[Mapping[str, Any]] = None,
    model_settings: Optional[Mapping[str, Any]] = None,
    classnames: Optional[Sequence[str]] = None,
    classification_report: Optional[Mapping[str, Any]] = None,
    error: Optional[str] = None,
    metrics_mode: str = "",
    run_id: str = "",
    source: str = "",
) -> Dict[str, Any]:
    """Build one side-by-side compare table row (settings + metrics).

    Pure helper for UI and CSV export. Missing fields become empty strings /
    ``None`` rather than raising.
    """
    meta = dict(run_meta or {})
    summary = dict(metrics_summary or {})
    settings = dict(model_settings or meta.get("model_settings") or {})
    report = dict(classification_report or {})

    names: List[str] = []
    if classnames:
        names = [str(c) for c in classnames]
    elif isinstance(settings.get("classnames"), list):
        names = [str(c) for c in settings["classnames"]]
    elif isinstance(meta.get("classnames"), list):
        names = [str(c) for c in meta["classnames"]]
    elif isinstance(summary.get("classnames"), list):
        names = [str(c) for c in summary["classnames"]]

    macro = summary.get("macro_f1", meta.get("macro_f1"))
    n_ex = summary.get("n_examples", meta.get("n_examples", ""))
    n_mis = summary.get("n_misclassified", meta.get("n_misclassified", ""))
    worst_lab, worst_f1 = worst_class_from_summary(summary)
    acc_explicit = None
    if "accuracy" in summary:
        acc_explicit = summary.get("accuracy")
    elif isinstance(report.get("accuracy"), (int, float)):
        acc_explicit = report.get("accuracy")
    elif isinstance(report.get("accuracy"), Mapping):
        acc_explicit = report["accuracy"].get("f1-score")  # unlikely
    accuracy = accuracy_from_counts(n_ex, n_mis, explicit=acc_explicit)

    rid = run_id or str(meta.get("run_id") or "")
    src = source or str(meta.get("source") or "")

    row: Dict[str, Any] = {
        "model": Path(model_path).name,
        "model_path": str(model_path),
        "macro_f1": _as_optional_float(macro),
        "accuracy": accuracy,
        "n_examples": n_ex if n_ex != "" else "",
        "n_misclassified": n_mis if n_mis != "" else "",
        "worst_class": worst_lab,
        "worst_f1": worst_f1,
        "time_step": settings.get("time_step", ""),
        "network": settings.get("network", ""),
        "level": format_level_summary(settings),
        "dim": format_dim_summary(settings),
        "label_mode": settings.get("label_mode", ""),
        "lambda_soft": settings.get("lambda_soft", ""),
        "run_id": rid,
        "source": src,
        "metrics_mode": metrics_mode,
        "error": error or "",
        "classnames": names,
    }
    return row


def compare_row_from_loaded_run(
    model_path: PathLike,
    loaded: Mapping[str, Any],
    *,
    metrics_mode: str = "reeval",
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a compare row from :func:`load_evaluation_run` output."""
    meta = loaded.get("run_meta") or {}
    summary = loaded.get("metrics_summary") or {}
    settings = meta.get("model_settings") if isinstance(meta, Mapping) else {}
    if not settings:
        settings = model_settings_from_model_dir(model_path)
    return build_compare_row(
        model_path=model_path,
        run_meta=meta if isinstance(meta, Mapping) else {},
        metrics_summary=summary if isinstance(summary, Mapping) else {},
        model_settings=settings if isinstance(settings, Mapping) else {},
        classification_report=loaded.get("classification_report")
        if isinstance(loaded.get("classification_report"), Mapping)
        else None,
        error=error,
        metrics_mode=metrics_mode,
    )


def compare_row_from_stored_eval(
    model_dir: PathLike,
    *,
    prefer_sources: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build a compare row from the newest stored eval run under a model.

    Prefers runs whose ``source`` is in ``prefer_sources`` when provided
    (still newest-first within that filter). Falls back to any listed run.
    """
    model_path = Path(model_dir)
    settings = model_settings_from_model_dir(model_path)
    classnames = model_classnames_from_parameters(model_path)
    runs = list_evaluation_runs(model_path)
    if not runs:
        return build_compare_row(
            model_path=model_path,
            model_settings=settings,
            classnames=classnames,
            error="No stored evaluation runs under model/eval/",
            metrics_mode="stored",
        )

    chosen = runs
    if prefer_sources:
        pref = {str(s) for s in prefer_sources}
        filtered = [r for r in runs if r.source in pref]
        if filtered:
            chosen = filtered
    info = chosen[0]
    try:
        loaded = load_evaluation_run(info.run_dir)
    except Exception as exc:
        return build_compare_row(
            model_path=model_path,
            model_settings=settings,
            classnames=classnames,
            run_id=info.run_id,
            source=info.source,
            error=f"Failed to load run: {exc}",
            metrics_mode="stored",
        )
    row = compare_row_from_loaded_run(
        model_path, loaded, metrics_mode="stored"
    )
    # Prefer on-disk parameters when run meta omitted settings
    if not row.get("time_step") and settings:
        filled = build_compare_row(
            model_path=model_path,
            run_meta=loaded.get("run_meta") or {},
            metrics_summary=loaded.get("metrics_summary") or {},
            model_settings=settings,
            classnames=classnames or row.get("classnames"),
            classification_report=loaded.get("classification_report")
            if isinstance(loaded.get("classification_report"), Mapping)
            else None,
            metrics_mode="stored",
        )
        return filled
    if not row.get("classnames") and classnames:
        row["classnames"] = list(classnames)
    return row


def classnames_mismatch_report(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Describe classname-set differences across compare rows.

    Returns::

        {
          "has_mismatch": bool,
          "message": str,  # empty if no mismatch
          "sets": [sorted classnames frozenset as list, ...],
        }
    """
    sets: List[Tuple[str, ...]] = []
    labels: List[str] = []
    for row in rows:
        if row.get("error") and not row.get("classnames"):
            continue
        names = row.get("classnames") or []
        key = tuple(sorted(str(c) for c in names))
        if not key:
            continue
        sets.append(key)
        labels.append(str(row.get("model") or row.get("model_path") or "?"))
    unique = sorted(set(sets))
    if len(unique) <= 1:
        return {"has_mismatch": False, "message": "", "sets": [list(s) for s in unique]}
    parts = [
        "Classname sets differ across models; metrics are not strictly comparable."
    ]
    for s in unique:
        models = [
            labels[i] for i, key in enumerate(sets) if key == s
        ]
        preview = ", ".join(s[:6]) + ("…" if len(s) > 6 else "")
        parts.append(f"{', '.join(models)}: {{{preview}}} ({len(s)} classes).")
    return {
        "has_mismatch": True,
        "message": " ".join(parts),
        "sets": [list(s) for s in unique],
    }


def filter_rows_matching_classnames(
    rows: Sequence[Mapping[str, Any]],
    reference_classnames: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Keep rows whose classname set matches the reference (or first good row)."""
    ref: Optional[Tuple[str, ...]] = None
    if reference_classnames is not None:
        ref = tuple(sorted(str(c) for c in reference_classnames))
    else:
        for row in rows:
            names = row.get("classnames") or []
            if names and not row.get("error"):
                ref = tuple(sorted(str(c) for c in names))
                break
    if ref is None:
        return [dict(r) for r in rows]
    out: List[Dict[str, Any]] = []
    for row in rows:
        names = row.get("classnames") or []
        key = tuple(sorted(str(c) for c in names))
        if key == ref:
            out.append(dict(row))
    return out


def best_macro_f1_indices(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    """Indices of rows tied for highest macro_f1 (errors / missing excluded)."""
    best: Optional[float] = None
    idxs: List[int] = []
    for i, row in enumerate(rows):
        if row.get("error"):
            continue
        f1 = _as_optional_float(row.get("macro_f1"))
        if f1 is None:
            continue
        if best is None or f1 > best + 1e-12:
            best = f1
            idxs = [i]
        elif abs(f1 - best) <= 1e-12:
            idxs.append(i)
    return idxs


def export_compare_table_csv(
    rows: Sequence[Mapping[str, Any]],
    path: PathLike,
    *,
    columns: Optional[Sequence[str]] = None,
) -> Path:
    """Write a compare table to CSV. Returns the written path."""
    cols = list(columns) if columns is not None else list(COMPARE_ROW_KEYS)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for row in rows:
        rec: Dict[str, Any] = {}
        for c in cols:
            rec[c] = row.get(c, "")
        records.append(rec)
    pd.DataFrame(records, columns=cols).to_csv(out, index=False)
    return out


def hard_labels_from_targets(
    targets: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Extract hard class indices from Keras target tensors.

    Handles one-hot, stacked hard+soft (``2*C`` columns), and binary (N,) / (N,1).
    """
    y = np.asarray(targets)
    if y.ndim == 1:
        return y.astype(np.int64)
    if y.ndim != 2:
        raise ValueError(f"targets must be 1-D or 2-D, got shape {y.shape}")
    if y.shape[1] == 2 * n_classes:
        hard = y[:, :n_classes]
        return hard.argmax(axis=1).astype(np.int64)
    if n_classes == 2 and y.shape[1] == 1:
        return np.round(y[:, 0]).astype(np.int64)
    if n_classes == 2 and y.shape[1] == 2:
        # Could be one-hot binary or hard+soft stacked with C=1 weirdness;
        # prefer argmax if rows sum ~1, else first column as hard logit/prob.
        row_sum = y.sum(axis=1)
        if np.allclose(row_sum, 1.0, atol=1e-3) or (y.max() <= 1.0 and y.min() >= 0.0):
            return y.argmax(axis=1).astype(np.int64)
        return np.round(y[:, 0]).astype(np.int64)
    return y.argmax(axis=1).astype(np.int64)


def predictions_from_model_output(
    raw_predictions: np.ndarray,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map Keras ``model.predict`` output to class indices and confidences.

    Returns:
        (pred_indices, confidences) each shape ``(N,)``.
    """
    p = np.asarray(raw_predictions, dtype=np.float64)
    if p.ndim == 1 or (p.ndim == 2 and p.shape[1] == 1):
        flat = p.reshape(-1)
        if n_classes != 2:
            raise ValueError("scalar/binary predict output requires n_classes == 2")
        pred = (flat >= 0.5).astype(np.int64)
        conf = np.where(flat >= 0.5, flat, 1.0 - flat)
        return pred, conf
    pred = p.argmax(axis=1).astype(np.int64)
    conf = p.max(axis=1)
    return pred, conf


def per_example_cross_entropy(
    y_true_idx: Sequence[Any],
    y_proba: np.ndarray,
    n_classes: int,
    *,
    eps: float = 1e-7,
) -> np.ndarray:
    """Per-example cross-entropy loss from hard indices and model probabilities.

    Supports binary sigmoid outputs ``(N,)`` / ``(N, 1)`` and multiclass
    ``(N, C)``. Used for end-of-train high-loss ranking on the train partition.
    """
    true_idx = np.asarray(y_true_idx, dtype=np.int64).reshape(-1)
    p = np.asarray(y_proba, dtype=np.float64)
    n = true_idx.shape[0]
    if p.ndim == 1 or (p.ndim == 2 and p.shape[1] == 1):
        if n_classes != 2:
            raise ValueError("1-D / single-column proba requires n_classes == 2")
        p1 = p.reshape(-1)
        if p1.shape[0] != n:
            raise ValueError("y_proba length must match y_true_idx")
        p1 = np.clip(p1, eps, 1.0 - eps)
        # binary CE with true in {0,1}
        t = true_idx.astype(np.float64)
        return -(t * np.log(p1) + (1.0 - t) * np.log(1.0 - p1))
    if p.ndim != 2 or p.shape[0] != n:
        raise ValueError(f"y_proba shape {p.shape} incompatible with n={n}")
    if p.shape[1] != n_classes:
        raise ValueError(
            f"y_proba columns ({p.shape[1]}) must match n_classes ({n_classes})"
        )
    p = np.clip(p, eps, 1.0)
    rows = np.arange(n)
    return -np.log(p[rows, true_idx])


def high_loss_from_predictions(
    example_ids: Sequence[str],
    y_true_idx: Sequence[Any],
    y_proba: np.ndarray,
    classnames: Sequence[str],
    *,
    top_k: Optional[int] = 100,
) -> List[HighLossExample]:
    """Rank train-partition examples by cross-entropy loss (highest first)."""
    names = [str(c) for c in classnames]
    true_idx = np.asarray(y_true_idx, dtype=np.int64).reshape(-1)
    losses = per_example_cross_entropy(true_idx, y_proba, len(names))
    pred_idx, _ = predictions_from_model_output(np.asarray(y_proba), len(names))
    true_labels = [names[int(i)] if 0 <= int(i) < len(names) else str(i) for i in true_idx]
    pred_labels = [names[int(i)] if 0 <= int(i) < len(names) else str(i) for i in pred_idx]
    return rank_high_loss_examples(
        example_ids,
        losses,
        true_labels,
        pred_labels=pred_labels,
        top_k=top_k,
    )
