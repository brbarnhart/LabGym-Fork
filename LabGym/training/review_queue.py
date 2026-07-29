"""Build Review examples queues from evaluation runs.

Queues are the union of misclassified predictions and high-loss train
examples, tagged with source run ids. Curation actions apply to the
dataset manifest (see ``DatasetManifest``), not to eval artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

from LabGym.training.evaluation import load_evaluation_run
from LabGym.training.dataset_manifest import (
    DatasetManifest,
    example_id_from_path,
    original_label_from_flat_name,
    resolve_example_media_path,
    scan_example_store,
)

PathLike = Union[str, Path]

SOURCE_MISCLASSIFIED = "misclassified"
SOURCE_HIGH_LOSS = "high_loss"


@dataclass
class ReviewQueueItem:
    """One review candidate surfaced from an evaluation / train run."""

    example_id: str
    true_label: str
    pred_label: str = ""
    confidence: Optional[float] = None
    loss: Optional[float] = None
    source: str = SOURCE_MISCLASSIFIED
    run_id: str = ""
    run_dir: str = ""
    path_hint: Optional[str] = None
    media_path: Optional[str] = None
    # Extra tags if the same example appears under multiple sources
    sources: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.sources:
            self.sources = [self.source]
        if self.path_hint is None and ("/" in self.example_id or "\\" in self.example_id):
            # Predictions sometimes store relative path as example_id
            self.path_hint = self.example_id.replace("\\", "/")


def normalize_example_id(raw: str) -> str:
    """Map prediction/high-loss ids to manifest ``example_id`` (stem)."""
    s = str(raw).strip().replace("\\", "/")
    if not s:
        return s
    # path-like: behavior/file.avi or nested
    name = Path(s).name
    return example_id_from_path(name)


def build_review_queue(
    run_dirs: Sequence[PathLike],
    *,
    include_misclassified: bool = True,
    include_high_loss: bool = True,
    dedupe: bool = True,
    store_root: Optional[PathLike] = None,
) -> List[ReviewQueueItem]:
    """Collect review items from one or more evaluation run directories.

    Args:
        run_dirs: Paths to ``model/eval/<run_id>/`` directories.
        include_misclassified: Use ``predictions.csv`` rows with misclassified=1.
        include_high_loss: Use ``high_loss.csv`` when present.
        dedupe: Collapse duplicate example_ids (prefer high-loss rank, merge sources).
        store_root: Optional example store to resolve media paths.

    Returns:
        Ordered list of :class:`ReviewQueueItem` (high-loss first by rank,
        then misclassified by ascending confidence when available).
    """
    items: List[ReviewQueueItem] = []
    for run_dir in run_dirs:
        path = Path(run_dir)
        if not path.is_dir():
            continue
        try:
            loaded = load_evaluation_run(path)
        except Exception:
            continue
        meta = loaded.get("run_meta") or {}
        run_id = str(meta.get("run_id") or path.name)
        run_dir_s = str(path)

        if include_high_loss and "high_loss" in loaded:
            hl = loaded["high_loss"]
            try:
                for _, row in hl.iterrows():
                    raw_id = str(row.get("example_id", ""))
                    eid = normalize_example_id(raw_id)
                    if not eid:
                        continue
                    loss_v = row.get("loss")
                    try:
                        loss_f = float(loss_v) if loss_v is not None and loss_v == loss_v else None
                    except (TypeError, ValueError):
                        loss_f = None
                    items.append(
                        ReviewQueueItem(
                            example_id=eid,
                            true_label=str(row.get("true_label", "") or ""),
                            pred_label=str(row.get("pred_label", "") or ""),
                            loss=loss_f,
                            source=SOURCE_HIGH_LOSS,
                            run_id=run_id,
                            run_dir=run_dir_s,
                            path_hint=raw_id if ("/" in raw_id or "\\" in raw_id) else None,
                        )
                    )
            except Exception:
                pass

        if include_misclassified and "predictions" in loaded:
            preds = loaded["predictions"]
            try:
                for _, row in preds.iterrows():
                    mis = row.get("misclassified", 0)
                    try:
                        is_mis = int(mis) != 0
                    except (TypeError, ValueError):
                        is_mis = bool(mis)
                    if not is_mis:
                        continue
                    raw_id = str(row.get("example_id", ""))
                    eid = normalize_example_id(raw_id)
                    if not eid:
                        continue
                    conf_v = row.get("confidence")
                    try:
                        conf_f = (
                            float(conf_v) if conf_v is not None and conf_v == conf_v else None
                        )
                    except (TypeError, ValueError):
                        conf_f = None
                    items.append(
                        ReviewQueueItem(
                            example_id=eid,
                            true_label=str(row.get("true_label", "") or ""),
                            pred_label=str(row.get("pred_label", "") or ""),
                            confidence=conf_f,
                            source=SOURCE_MISCLASSIFIED,
                            run_id=run_id,
                            run_dir=run_dir_s,
                            path_hint=raw_id if ("/" in raw_id or "\\" in raw_id) else None,
                        )
                    )
            except Exception:
                pass

    if dedupe:
        items = _dedupe_items(items)

    if store_root is not None:
        attach_media_paths(items, store_root)

    # Sort: high_loss first by loss desc, then misclassified by conf asc
    def _key(it: ReviewQueueItem) -> tuple:
        is_hl = SOURCE_HIGH_LOSS in it.sources
        loss = it.loss if it.loss is not None else -1.0
        conf = it.confidence if it.confidence is not None else 2.0
        return (0 if is_hl else 1, -loss if is_hl else 0, conf, it.example_id)

    items.sort(key=_key)
    return items


def _dedupe_items(items: Sequence[ReviewQueueItem]) -> List[ReviewQueueItem]:
    by_id: Dict[str, ReviewQueueItem] = {}
    order: List[str] = []
    for it in items:
        eid = it.example_id
        if eid not in by_id:
            by_id[eid] = it
            order.append(eid)
            continue
        cur = by_id[eid]
        if it.source not in cur.sources:
            cur.sources.append(it.source)
        # Prefer filled loss / confidence / pred
        if cur.loss is None and it.loss is not None:
            cur.loss = it.loss
        if cur.confidence is None and it.confidence is not None:
            cur.confidence = it.confidence
        if not cur.pred_label and it.pred_label:
            cur.pred_label = it.pred_label
        if not cur.true_label and it.true_label:
            cur.true_label = it.true_label
        if cur.path_hint is None and it.path_hint:
            cur.path_hint = it.path_hint
        # Prefer high_loss as primary source tag when both present
        if SOURCE_HIGH_LOSS in cur.sources:
            cur.source = SOURCE_HIGH_LOSS
        elif SOURCE_MISCLASSIFIED in cur.sources:
            cur.source = SOURCE_MISCLASSIFIED
        # Keep first run_id; note others implicitly via sources only
    return [by_id[eid] for eid in order]


def attach_media_paths(
    items: Sequence[ReviewQueueItem],
    store_root: PathLike,
) -> None:
    """Fill ``media_path`` on items when files exist under the store."""
    root = Path(store_root)
    index: Dict[str, Path] = {}
    try:
        for eid, _lab, _hint, path in scan_example_store(root):
            index[eid] = path
    except Exception:
        index = {}
    for it in items:
        if it.media_path and Path(it.media_path).is_file():
            continue
        found = resolve_example_media_path(
            root,
            it.example_id,
            path_hint=it.path_hint,
            scan_index=index,
        )
        if found is not None:
            it.media_path = str(found)
            if it.path_hint is None:
                try:
                    it.path_hint = str(found.relative_to(root)).replace("\\", "/")
                except ValueError:
                    it.path_hint = found.name


def ensure_queue_in_manifest(
    manifest: DatasetManifest,
    items: Sequence[ReviewQueueItem],
) -> int:
    """Ensure each queue item has a manifest row (for keep/exclude/recategorize).

    Returns number of newly added example records.
    """
    n_new = 0
    for it in items:
        before = len(manifest.examples)
        lab = it.true_label or original_label_from_flat_name(it.example_id)
        manifest.ensure_example(it.example_id, lab, path_hint=it.path_hint)
        if len(manifest.examples) > before:
            n_new += 1
    return n_new


def available_categories(
    store_root: PathLike,
    *,
    manifest: Optional[DatasetManifest] = None,
    extra: Optional[Iterable[str]] = None,
) -> List[str]:
    """Sorted category names for recategorize combos."""
    names: Set[str] = set()
    if extra:
        names.update(str(x) for x in extra if str(x).strip())
    try:
        for _eid, lab, _hint, _path in scan_example_store(store_root):
            if lab:
                names.add(str(lab))
    except Exception:
        pass
    if manifest is not None:
        for rec in manifest.examples.values():
            if rec.original_label:
                names.add(rec.original_label)
            if rec.label_override:
                names.add(rec.label_override)
    return sorted(names)
