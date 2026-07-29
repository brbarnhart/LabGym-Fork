"""Non-destructive dataset manifest for categorizer example stores.

Lives next to an example store root as ``dataset_manifest.json``. Records
exclusions, label overrides, soft overrides, and split assignment
(train / validation / sealed test). Training and evaluation consume the
**effective training set** (store + manifest). See ADRs 0001 and 0002.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

MANIFEST_FILENAME = "dataset_manifest.json"
SCHEMA_VERSION = 1

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_SEALED_TEST = "sealed_test"
SPLIT_UNASSIGNED = "unassigned"

VALID_SPLITS = frozenset(
    {SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_SEALED_TEST, SPLIT_UNASSIGNED}
)

# Train-time partitions that may influence the run (weights or early stopping).
TRAIN_TIME_SPLITS = frozenset({SPLIT_TRAIN, SPLIT_VALIDATION})

PathLike = Union[str, Path]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def example_id_from_path(path: PathLike) -> str:
    """Basename without extension — stable key across .jpg/.avi pairs."""
    return Path(path).stem


def original_label_from_flat_name(filename: str) -> str:
    """LabGym prepared-example convention: ``..._<behavior>.ext``."""
    stem = Path(filename).stem
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[-1]


@dataclass
class ExampleRecord:
    """One example's curation and split state in the manifest."""

    example_id: str
    original_label: str
    label_override: Optional[str] = None
    excluded: bool = False
    split: str = SPLIT_UNASSIGNED
    soft_override: Optional[List[float]] = None
    soft_override_classnames: Optional[List[str]] = None
    path_hint: Optional[str] = None  # relative path under store root when known

    @property
    def active_label(self) -> str:
        if self.label_override is not None and str(self.label_override).strip() != "":
            return str(self.label_override)
        return str(self.original_label)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExampleRecord":
        split = str(data.get("split") or SPLIT_UNASSIGNED)
        if split not in VALID_SPLITS:
            split = SPLIT_UNASSIGNED
        soft = data.get("soft_override")
        soft_list = [float(x) for x in soft] if soft is not None else None
        soft_names = data.get("soft_override_classnames")
        soft_names_list = [str(x) for x in soft_names] if soft_names is not None else None
        return cls(
            example_id=str(data["example_id"]),
            original_label=str(data.get("original_label") or ""),
            label_override=(
                None
                if data.get("label_override") in (None, "")
                else str(data.get("label_override"))
            ),
            excluded=bool(data.get("excluded", False)),
            split=split,
            soft_override=soft_list,
            soft_override_classnames=soft_names_list,
            path_hint=(
                None if data.get("path_hint") in (None, "") else str(data.get("path_hint"))
            ),
        )


@dataclass
class EffectiveExample:
    """Resolved view of one example after applying the manifest."""

    example_id: str
    original_label: str
    active_label: str
    split: str
    path: Optional[Path]
    excluded: bool = False
    label_override: Optional[str] = None


@dataclass
class DatasetManifest:
    """Durable curation state for one example store."""

    store_root: Path
    examples: Dict[str, ExampleRecord] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    split_meta: Dict[str, Any] = field(default_factory=dict)
    taxonomy_ops: List[Dict[str, Any]] = field(default_factory=list)
    undo_stack: List[Dict[str, Any]] = field(default_factory=list)
    updated_utc: str = field(default_factory=_utc_now)
    # Max undo entries retained
    max_undo: int = 100

    # --- persistence -----------------------------------------------------

    @property
    def path(self) -> Path:
        return Path(self.store_root) / MANIFEST_FILENAME

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "store_root": str(self.store_root),
            "updated_utc": self.updated_utc,
            "split_meta": dict(self.split_meta),
            "taxonomy_ops": list(self.taxonomy_ops),
            "undo_stack": list(self.undo_stack),
            "examples": {k: v.to_dict() for k, v in sorted(self.examples.items())},
        }

    def save(self, path: Optional[PathLike] = None) -> Path:
        """Write JSON manifest to disk."""
        self.updated_utc = _utc_now()
        out = Path(path) if path is not None else self.path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], store_root: Optional[PathLike] = None) -> "DatasetManifest":
        root = Path(store_root) if store_root is not None else Path(str(data.get("store_root") or "."))
        examples: Dict[str, ExampleRecord] = {}
        raw_ex = data.get("examples") or {}
        if isinstance(raw_ex, dict):
            for key, val in raw_ex.items():
                if not isinstance(val, Mapping):
                    continue
                rec = ExampleRecord.from_dict({**dict(val), "example_id": val.get("example_id", key)})
                examples[rec.example_id] = rec
        return cls(
            store_root=root,
            examples=examples,
            schema_version=int(data.get("schema_version") or SCHEMA_VERSION),
            split_meta=dict(data.get("split_meta") or {}),
            taxonomy_ops=list(data.get("taxonomy_ops") or []),
            undo_stack=list(data.get("undo_stack") or []),
            updated_utc=str(data.get("updated_utc") or _utc_now()),
        )

    @classmethod
    def load(cls, store_root: PathLike) -> "DatasetManifest":
        """Load manifest from ``store_root/dataset_manifest.json``."""
        root = Path(store_root)
        path = root / MANIFEST_FILENAME
        if not path.is_file():
            raise FileNotFoundError(str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data, store_root=root)

    @classmethod
    def load_or_create(cls, store_root: PathLike) -> "DatasetManifest":
        root = Path(store_root)
        path = root / MANIFEST_FILENAME
        if path.is_file():
            return cls.load(root)
        return cls(store_root=root)

    @classmethod
    def exists(cls, store_root: PathLike) -> bool:
        return (Path(store_root) / MANIFEST_FILENAME).is_file()

    # --- undo ------------------------------------------------------------

    def _push_undo(self, reverse_ops: Sequence[Dict[str, Any]], label: str = "") -> None:
        self.undo_stack.append(
            {
                "label": label,
                "created_utc": _utc_now(),
                "ops": list(reverse_ops),
            }
        )
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack = self.undo_stack[-self.max_undo :]

    def undo(self) -> bool:
        """Apply the most recent reverse-op bundle. Returns False if empty."""
        if not self.undo_stack:
            return False
        bundle = self.undo_stack.pop()
        for op in bundle.get("ops") or []:
            self._apply_op(op, record_undo=False)
        self.updated_utc = _utc_now()
        return True

    def _apply_op(self, op: Mapping[str, Any], *, record_undo: bool = False) -> None:
        """Internal: apply a single reverse or forward op dict."""
        kind = str(op.get("op") or "")
        eid = str(op.get("example_id") or "")
        if kind == "set_fields" and eid:
            rec = self.examples.get(eid)
            if rec is None:
                # recreate if needed
                rec = ExampleRecord(
                    example_id=eid,
                    original_label=str(op.get("original_label") or ""),
                )
                self.examples[eid] = rec
            for field_name in (
                "original_label",
                "label_override",
                "excluded",
                "split",
                "soft_override",
                "soft_override_classnames",
                "path_hint",
            ):
                if field_name in op:
                    setattr(rec, field_name, op[field_name])
        elif kind == "remove_example" and eid:
            self.examples.pop(eid, None)
        elif kind == "add_example":
            rec = ExampleRecord.from_dict(op.get("record") or op)
            self.examples[rec.example_id] = rec
        elif kind == "set_taxonomy_ops":
            self.taxonomy_ops = list(op.get("taxonomy_ops") or [])
        elif kind == "set_split_meta":
            self.split_meta = dict(op.get("split_meta") or {})

    def _snapshot_fields(self, rec: ExampleRecord) -> Dict[str, Any]:
        return {
            "op": "set_fields",
            "example_id": rec.example_id,
            "original_label": rec.original_label,
            "label_override": rec.label_override,
            "excluded": rec.excluded,
            "split": rec.split,
            "soft_override": rec.soft_override,
            "soft_override_classnames": rec.soft_override_classnames,
            "path_hint": rec.path_hint,
        }

    # --- mutations (immediate apply + undo) ------------------------------

    def exclude(self, example_id: str, *, excluded: bool = True) -> None:
        rec = self._require(example_id)
        before = self._snapshot_fields(rec)
        rec.excluded = bool(excluded)
        self._push_undo([before], label="exclude" if excluded else "include")

    def keep(self, example_id: str) -> None:
        """Keep example in the effective set with its current active label.

        Clears exclusion if set. Does not change label overrides. Used by
        Review examples as an explicit "leave as-is" decision (undoable).
        """
        rec = self._require(example_id)
        before = self._snapshot_fields(rec)
        rec.excluded = False
        self._push_undo([before], label="keep")

    def ensure_example(
        self,
        example_id: str,
        original_label: str,
        *,
        path_hint: Optional[str] = None,
    ) -> ExampleRecord:
        """Return existing record or add a new unassigned one (no undo)."""
        eid = str(example_id)
        if eid in self.examples:
            rec = self.examples[eid]
            if not rec.original_label and original_label:
                rec.original_label = str(original_label)
            if path_hint and not rec.path_hint:
                rec.path_hint = path_hint
            return rec
        rec = ExampleRecord(
            example_id=eid,
            original_label=str(original_label or ""),
            path_hint=path_hint,
            split=SPLIT_UNASSIGNED,
        )
        self.examples[eid] = rec
        return rec

    def recategorize(self, example_id: str, new_label: Optional[str]) -> None:
        """Set or clear a label override (None clears)."""
        rec = self._require(example_id)
        before = self._snapshot_fields(rec)
        if new_label is None or str(new_label).strip() == "":
            rec.label_override = None
        else:
            rec.label_override = str(new_label)
        self._push_undo([before], label="recategorize")

    def set_soft_override(
        self,
        example_id: str,
        soft: Optional[Sequence[float]],
        classnames: Optional[Sequence[str]] = None,
    ) -> None:
        rec = self._require(example_id)
        before = self._snapshot_fields(rec)
        if soft is None:
            rec.soft_override = None
            rec.soft_override_classnames = None
        else:
            rec.soft_override = [float(x) for x in soft]
            rec.soft_override_classnames = (
                [str(c) for c in classnames] if classnames is not None else None
            )
        self._push_undo([before], label="soft_override")

    # --- taxonomy operations ---------------------------------------------

    def merge_categories(
        self,
        sources: Sequence[str],
        target: str,
    ) -> int:
        """Map examples whose active label is in ``sources`` onto ``target``.

        Records a taxonomy merge op for soft projection. Returns number of
        examples whose label override was set. Undo restores fields + ops.
        """
        target = str(target).strip()
        src = sorted({str(s).strip() for s in sources if str(s).strip() and str(s).strip() != target})
        if not target:
            raise ValueError("merge target must be non-empty")
        if not src:
            raise ValueError("merge requires at least one source category distinct from target")

        reverse: List[Dict[str, Any]] = [
            {"op": "set_taxonomy_ops", "taxonomy_ops": deepcopy(self.taxonomy_ops)}
        ]
        n = 0
        for rec in self.examples.values():
            if rec.active_label in src:
                reverse.append(self._snapshot_fields(rec))
                rec.label_override = target
                n += 1
        self.taxonomy_ops.append(
            {
                "op": "merge",
                "sources": src,
                "target": target,
                "n_examples": n,
                "created_utc": _utc_now(),
            }
        )
        self._push_undo(reverse, label="merge_categories")
        return n

    def exclude_category(self, category: str, *, excluded: bool = True) -> int:
        """Bulk exclude (or re-include) all examples with active label ``category``.

        Also records a taxonomy exclude/include op for soft projection.
        Returns number of examples updated.
        """
        cat = str(category).strip()
        if not cat:
            raise ValueError("category must be non-empty")
        reverse: List[Dict[str, Any]] = [
            {"op": "set_taxonomy_ops", "taxonomy_ops": deepcopy(self.taxonomy_ops)}
        ]
        n = 0
        for rec in self.examples.values():
            if rec.active_label == cat or (
                rec.label_override is None and rec.original_label == cat
            ):
                if bool(rec.excluded) == bool(excluded):
                    continue
                reverse.append(self._snapshot_fields(rec))
                rec.excluded = bool(excluded)
                n += 1
        self.taxonomy_ops.append(
            {
                "op": "exclude_category" if excluded else "include_category",
                "category": cat,
                "excluded": bool(excluded),
                "n_examples": n,
                "created_utc": _utc_now(),
            }
        )
        self._push_undo(reverse, label="exclude_category" if excluded else "include_category")
        return n

    def merge_map(self) -> Dict[str, str]:
        """Source → active category map from taxonomy merge ops."""
        from LabGym.training.soft_projection import compose_merge_map

        return compose_merge_map(self.taxonomy_ops)

    def excluded_categories(self) -> List[str]:
        """Category names excluded via taxonomy ops (sorted)."""
        from LabGym.training.soft_projection import excluded_categories_from_ops

        return sorted(excluded_categories_from_ops(self.taxonomy_ops))

    def active_categories(self, *, include_excluded_examples: bool = False) -> List[str]:
        """Sorted active labels present on examples (optionally including excluded)."""
        names: set = set()
        for rec in self.examples.values():
            if rec.excluded and not include_excluded_examples:
                continue
            lab = rec.active_label
            if lab:
                names.add(lab)
        return sorted(names)

    def category_summary(self) -> List[Dict[str, Any]]:
        """Per active_label counts (total, excluded, by split) for Categories UI."""
        rows: Dict[str, Dict[str, Any]] = {}
        for rec in self.examples.values():
            lab = rec.active_label or rec.original_label or "(empty)"
            if lab not in rows:
                rows[lab] = {
                    "category": lab,
                    "n_total": 0,
                    "n_active": 0,
                    "n_excluded": 0,
                    "n_train": 0,
                    "n_validation": 0,
                    "n_sealed_test": 0,
                    "n_unassigned": 0,
                }
            row = rows[lab]
            row["n_total"] += 1
            if rec.excluded:
                row["n_excluded"] += 1
            else:
                row["n_active"] += 1
                key = f"n_{rec.split}" if rec.split in VALID_SPLITS else "n_unassigned"
                if key in row:
                    row[key] += 1
                else:
                    row["n_unassigned"] += 1
        return [rows[k] for k in sorted(rows.keys())]

    def set_split(self, example_id: str, split: str) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split {split!r}; expected one of {sorted(VALID_SPLITS)}")
        rec = self._require(example_id)
        before = self._snapshot_fields(rec)
        rec.split = split
        self._push_undo([before], label="set_split")

    def set_splits_bulk(
        self,
        assignment: Mapping[str, str],
        *,
        undo_label: str = "set_splits_bulk",
    ) -> None:
        """Assign many splits at once (one undo bundle)."""
        reverse: List[Dict[str, Any]] = []
        for eid, split in assignment.items():
            if split not in VALID_SPLITS:
                raise ValueError(f"Invalid split {split!r} for {eid}")
            rec = self.examples.get(str(eid))
            if rec is None:
                continue
            reverse.append(self._snapshot_fields(rec))
            rec.split = split
        if reverse:
            self._push_undo(reverse, label=undo_label)

    def _require(self, example_id: str) -> ExampleRecord:
        eid = str(example_id)
        if eid not in self.examples:
            raise KeyError(f"Unknown example_id {eid!r}")
        return self.examples[eid]

    # --- scan / sync -----------------------------------------------------

    def sync_from_scan(
        self,
        scanned: Sequence[Tuple[str, str, Optional[str]]],
        *,
        drop_missing: bool = False,
    ) -> List[str]:
        """Merge scanned examples into the manifest.

        Args:
            scanned: triples ``(example_id, original_label, path_hint)``.
            drop_missing: if True, remove manifest rows not present in scan.

        Returns:
            List of newly added example_ids.
        """
        seen = set()
        added: List[str] = []
        for eid, label, hint in scanned:
            eid = str(eid)
            seen.add(eid)
            if eid in self.examples:
                rec = self.examples[eid]
                # Keep original_label if override history exists; update if empty
                if not rec.original_label:
                    rec.original_label = str(label)
                if hint and not rec.path_hint:
                    rec.path_hint = hint
            else:
                self.examples[eid] = ExampleRecord(
                    example_id=eid,
                    original_label=str(label),
                    path_hint=hint,
                    split=SPLIT_UNASSIGNED,
                )
                added.append(eid)
        if drop_missing:
            for eid in list(self.examples.keys()):
                if eid not in seen:
                    del self.examples[eid]
        return added

    # --- splits ----------------------------------------------------------

    def has_train_val(self) -> bool:
        """True if at least one non-excluded example is train or validation."""
        for rec in self.examples.values():
            if rec.excluded:
                continue
            if rec.split in (SPLIT_TRAIN, SPLIT_VALIDATION):
                return True
        return False

    def counts_by_split(self, *, include_excluded: bool = False) -> Dict[str, int]:
        out: Dict[str, int] = defaultdict(int)
        for rec in self.examples.values():
            if rec.excluded and not include_excluded:
                continue
            out[rec.split] += 1
        return dict(out)

    def ensure_train_val_split(
        self,
        *,
        val_fraction: float = 0.2,
        seed: int = 42,
        regenerate: bool = False,
        assign_new: bool = True,
    ) -> Dict[str, int]:
        """Stratified train/validation assignment excluding sealed test.

        Stable by default: existing train/validation membership is kept.
        New unassigned (non-sealed, non-excluded) examples are assigned when
        ``assign_new`` is True. ``regenerate`` reassigns all non-sealed
        non-excluded examples.

        Sealed test membership is never moved into train or validation.
        """
        val_fraction = float(val_fraction)
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")

        eligible = [
            rec
            for rec in self.examples.values()
            if not rec.excluded and rec.split != SPLIT_SEALED_TEST
        ]
        if regenerate:
            targets = eligible
        elif assign_new:
            targets = [r for r in eligible if r.split == SPLIT_UNASSIGNED]
            # If nothing assigned yet, treat all eligible as targets
            if not self.has_train_val():
                targets = eligible
        else:
            targets = []
            if not self.has_train_val():
                targets = eligible

        reverse: List[Dict[str, Any]] = []
        if targets:
            assignment = _stratified_binary_split(
                [(r.example_id, r.active_label) for r in targets],
                val_fraction=val_fraction,
                seed=seed,
            )
            for rec in targets:
                reverse.append(self._snapshot_fields(rec))
                rec.split = assignment[rec.example_id]
            self._push_undo(reverse, label="ensure_train_val_split")

        self.split_meta = {
            **dict(self.split_meta),
            "seed": int(seed),
            "val_fraction": float(val_fraction),
            "generated_utc": _utc_now(),
            "regenerate": bool(regenerate),
        }
        return self.counts_by_split()

    def assign_sealed_test(
        self,
        *,
        example_ids: Optional[Sequence[str]] = None,
        fraction: Optional[float] = None,
        seed: int = 42,
        from_splits: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Mark examples as sealed test (never used at train time).

        Provide either explicit ``example_ids`` or a ``fraction`` of eligible
        examples (from train/val/unassigned by default). Returns assigned ids.
        """
        if example_ids is None and fraction is None:
            raise ValueError("Provide example_ids or fraction")
        if example_ids is not None and fraction is not None:
            raise ValueError("Provide only one of example_ids or fraction")

        if example_ids is not None:
            chosen = [str(e) for e in example_ids]
        else:
            frac = float(fraction)
            if not 0.0 < frac < 1.0:
                raise ValueError("fraction must be in (0, 1)")
            allowed = set(from_splits) if from_splits is not None else {
                SPLIT_TRAIN,
                SPLIT_VALIDATION,
                SPLIT_UNASSIGNED,
            }
            pool = [
                rec
                for rec in self.examples.values()
                if not rec.excluded
                and rec.split in allowed
                and rec.split != SPLIT_SEALED_TEST
            ]
            # Stratify by active label for fairness
            by_label: Dict[str, List[ExampleRecord]] = defaultdict(list)
            for rec in pool:
                by_label[rec.active_label].append(rec)
            rng = np.random.RandomState(int(seed))
            chosen = []
            for label in sorted(by_label.keys()):
                group = list(by_label[label])
                rng.shuffle(group)
                n = int(round(len(group) * frac))
                n = min(max(n, 0), len(group))
                # Prefer leaving at least one for train if group is large
                if len(group) >= 2 and n >= len(group):
                    n = len(group) - 1
                chosen.extend(r.example_id for r in group[:n])

        reverse: List[Dict[str, Any]] = []
        assigned: List[str] = []
        for eid in chosen:
            rec = self.examples.get(eid)
            if rec is None or rec.excluded:
                continue
            reverse.append(self._snapshot_fields(rec))
            rec.split = SPLIT_SEALED_TEST
            assigned.append(eid)
        if reverse:
            self._push_undo(reverse, label="assign_sealed_test")
        self.split_meta = {
            **dict(self.split_meta),
            "sealed_assigned_utc": _utc_now(),
            "sealed_count": len(assigned),
            "sealed_seed": int(seed) if fraction is not None else self.split_meta.get("sealed_seed"),
            "sealed_fraction": float(fraction) if fraction is not None else self.split_meta.get("sealed_fraction"),
        }
        return assigned

    def clear_sealed_test(self, *, to_split: str = SPLIT_UNASSIGNED) -> int:
        """Move sealed test examples to ``to_split`` (default unassigned)."""
        if to_split not in VALID_SPLITS or to_split == SPLIT_SEALED_TEST:
            raise ValueError(f"Invalid to_split {to_split!r}")
        reverse: List[Dict[str, Any]] = []
        n = 0
        for rec in self.examples.values():
            if rec.split == SPLIT_SEALED_TEST:
                reverse.append(self._snapshot_fields(rec))
                rec.split = to_split
                n += 1
        if reverse:
            self._push_undo(reverse, label="clear_sealed_test")
        return n

    # --- effective view --------------------------------------------------

    def effective_examples(
        self,
        *,
        splits: Optional[Sequence[str]] = None,
        include_excluded: bool = False,
        path_lookup: Optional[Mapping[str, PathLike]] = None,
    ) -> List[EffectiveExample]:
        """Return effective examples, optionally filtered by split.

        Sealed test is included only if requested via ``splits``.
        """
        split_filter = set(splits) if splits is not None else None
        out: List[EffectiveExample] = []
        for eid, rec in sorted(self.examples.items()):
            if rec.excluded and not include_excluded:
                continue
            if split_filter is not None and rec.split not in split_filter:
                continue
            path = None
            if path_lookup and eid in path_lookup:
                path = Path(path_lookup[eid])
            elif rec.path_hint:
                cand = Path(self.store_root) / rec.path_hint
                path = cand if cand.exists() else Path(rec.path_hint)
            out.append(
                EffectiveExample(
                    example_id=eid,
                    original_label=rec.original_label,
                    active_label=rec.active_label,
                    split=rec.split,
                    path=path,
                    excluded=rec.excluded,
                    label_override=rec.label_override,
                )
            )
        return out

    def train_time_example_ids(self) -> List[str]:
        """IDs allowed during training (train + validation; never sealed)."""
        return [
            e.example_id
            for e in self.effective_examples(splits=list(TRAIN_TIME_SPLITS))
        ]

    def sealed_test_example_ids(self) -> List[str]:
        return [
            e.example_id
            for e in self.effective_examples(splits=[SPLIT_SEALED_TEST])
        ]


def _stratified_binary_split(
    items: Sequence[Tuple[str, str]],
    *,
    val_fraction: float,
    seed: int,
) -> Dict[str, str]:
    """Assign each (id, label) to train or validation stratified by label."""
    by_label: Dict[str, List[str]] = defaultdict(list)
    for eid, label in items:
        by_label[str(label)].append(str(eid))
    rng = np.random.RandomState(int(seed))
    assignment: Dict[str, str] = {}
    for label in sorted(by_label.keys()):
        ids = list(by_label[label])
        rng.shuffle(ids)
        n = len(ids)
        if n == 1:
            # Keep singleton in train so stratification can proceed
            assignment[ids[0]] = SPLIT_TRAIN
            continue
        n_val = int(round(n * val_fraction))
        n_val = min(max(n_val, 1), n - 1)
        for eid in ids[:n_val]:
            assignment[eid] = SPLIT_VALIDATION
        for eid in ids[n_val:]:
            assignment[eid] = SPLIT_TRAIN
    return assignment


def scan_flat_example_store(
    store_root: PathLike,
    *,
    extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"),
    recursive_train_val: bool = True,
) -> List[Tuple[str, str, str, Path]]:
    """Scan a prepared flat example store.

    Returns list of ``(example_id, original_label, path_hint, path)``.
    Prefers ``.jpg`` when both image and video exist for the same stem.

    If ``train/`` and ``validation/`` subfolders exist and
    ``recursive_train_val`` is True, scans those too (onfly layout).
    """
    root = Path(store_root)
    ext_set = {e if e.startswith(".") else f".{e}" for e in extensions}
    found: Dict[str, Tuple[str, str, Path]] = {}

    def _ingest(folder: Path, hint_prefix: str = "") -> None:
        if not folder.is_dir():
            return
        for name in os.listdir(folder):
            p = folder / name
            if not p.is_file():
                continue
            if p.suffix not in ext_set:
                continue
            eid = example_id_from_path(p)
            label = original_label_from_flat_name(name)
            rel = str(Path(hint_prefix) / name) if hint_prefix else name
            # Prefer jpg over others if already seen
            if eid in found:
                prev = found[eid][2]
                if prev.suffix.lower() in {".jpg", ".jpeg"} and p.suffix.lower() not in {
                    ".jpg",
                    ".jpeg",
                }:
                    continue
            found[eid] = (label, rel.replace("\\", "/"), p)

    _ingest(root)
    if recursive_train_val:
        for sub in ("train", "validation", "test"):
            _ingest(root / sub, hint_prefix=sub)

    return [(eid, lab, hint, path) for eid, (lab, hint, path) in sorted(found.items())]


def scan_behavior_folder_store(
    store_root: PathLike,
    *,
    extensions: Sequence[str] = (".jpg", ".jpeg", ".png", ".avi", ".JPG", ".JPEG", ".PNG", ".AVI"),
) -> List[Tuple[str, str, str, Path]]:
    """Scan ground-truth layout: ``store/<behavior>/<file>``."""
    root = Path(store_root)
    ext_set = {e if e.startswith(".") else f".{e}" for e in extensions}
    found: Dict[str, Tuple[str, str, Path]] = {}
    if not root.is_dir():
        return []
    for behavior in sorted(os.listdir(root)):
        bdir = root / behavior
        if not bdir.is_dir():
            continue
        # skip known non-category dirs
        if behavior in ("train", "validation", "test", "eval") or behavior.startswith("."):
            continue
        for name in os.listdir(bdir):
            p = bdir / name
            if not p.is_file() or p.suffix not in ext_set:
                continue
            eid = example_id_from_path(p)
            rel = f"{behavior}/{name}".replace("\\", "/")
            if eid in found and found[eid][2].suffix.lower() in {".jpg", ".jpeg"}:
                continue
            found[eid] = (behavior, rel, p)
    return [(eid, lab, hint, path) for eid, (lab, hint, path) in sorted(found.items())]


def scan_example_store(store_root: PathLike) -> List[Tuple[str, str, str, Path]]:
    """Auto-detect flat vs behavior-folder layout and scan."""
    root = Path(store_root)
    # Prefer behavior folders if any subdir has media files
    behavior_hits = scan_behavior_folder_store(root)
    flat_hits = scan_flat_example_store(root)
    # If root itself has media, flat layout wins; if only subdirs with media, behavior
    root_has_media = any(
        (root / n).is_file()
        and (root / n).suffix.lower() in {".jpg", ".jpeg", ".png", ".avi"}
        for n in (os.listdir(root) if root.is_dir() else [])
    )
    if root_has_media:
        return flat_hits
    if behavior_hits:
        return behavior_hits
    return flat_hits


def resolve_train_val_paths(
    store_root: PathLike,
    path_files: Sequence[PathLike],
    labels: Sequence[str],
    *,
    ignore_manifest: bool = False,
    val_fraction: float = 0.2,
    seed: int = 42,
    persist: bool = True,
    extensions_priority: bool = True,
) -> Tuple[List[str], List[str], List[str], List[str], Optional[DatasetManifest]]:
    """Resolve train/val path lists using the dataset manifest when enabled.

    Args:
        store_root: Example store root (manifest location).
        path_files: Candidate file paths (flat prepared list).
        labels: Parallel original labels from filenames (overridden by manifest).
        ignore_manifest: If True, classical random stratified split, no persist.
        val_fraction: Validation fraction for new assignments.
        seed: RNG seed for split generation.
        persist: Save manifest after ensuring split.

    Returns:
        ``(train_files, val_files, train_labels, val_labels, manifest_or_None)``.
        Sealed test paths are never returned.
    """
    paths = [str(p) for p in path_files]
    labs = [str(l) for l in labels]
    if len(paths) != len(labs):
        raise ValueError("path_files and labels length mismatch")

    if ignore_manifest:
        from sklearn.model_selection import train_test_split

        if len(paths) < 2:
            return paths, [], labs, [], None
        try:
            tr, va, ytr, yva = train_test_split(
                paths, labs, test_size=val_fraction, stratify=labs, random_state=seed
            )
        except ValueError:
            tr, va, ytr, yva = train_test_split(
                paths, labs, test_size=val_fraction, random_state=seed
            )
        return list(tr), list(va), list(ytr), list(yva), None

    root = Path(store_root)
    manifest = DatasetManifest.load_or_create(root)

    # Index paths by example_id; apply exclusions and overrides
    id_to_path: Dict[str, str] = {}
    id_to_label: Dict[str, str] = {}
    scanned: List[Tuple[str, str, Optional[str]]] = []
    for p, lab in zip(paths, labs):
        eid = example_id_from_path(p)
        id_to_path[eid] = p
        id_to_label[eid] = lab
        try:
            hint = str(Path(p).relative_to(root)).replace("\\", "/")
        except ValueError:
            hint = Path(p).name
        scanned.append((eid, lab, hint))

    manifest.sync_from_scan(scanned)
    # Drop excluded and sealed from training consideration later
    manifest.ensure_train_val_split(
        val_fraction=val_fraction, seed=seed, regenerate=False, assign_new=True
    )
    if persist:
        manifest.save()

    train_files: List[str] = []
    val_files: List[str] = []
    train_labels: List[str] = []
    val_labels: List[str] = []

    for eid, path in id_to_path.items():
        rec = manifest.examples.get(eid)
        if rec is None or rec.excluded:
            continue
        if rec.split == SPLIT_SEALED_TEST:
            continue
        active = rec.active_label
        if rec.split == SPLIT_VALIDATION:
            val_files.append(path)
            val_labels.append(active)
        elif rec.split == SPLIT_TRAIN:
            train_files.append(path)
            train_labels.append(active)
        # unassigned: not used until assigned (stable policy)

    # Safety: if split left train empty (tiny sets), fall back to all non-sealed
    if not train_files:
        for eid, path in id_to_path.items():
            rec = manifest.examples.get(eid)
            if rec is None or rec.excluded or rec.split == SPLIT_SEALED_TEST:
                continue
            train_files.append(path)
            train_labels.append(rec.active_label)

    return train_files, val_files, train_labels, val_labels, manifest


def filter_paths_by_manifest(
    path_files: Sequence[PathLike],
    labels: Sequence[str],
    manifest: DatasetManifest,
    *,
    allow_splits: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """Filter path/label lists through manifest exclusions, overrides, splits.

    Default ``allow_splits`` is train+validation (excludes sealed test).
    """
    allowed = set(allow_splits) if allow_splits is not None else set(TRAIN_TIME_SPLITS)
    out_p: List[str] = []
    out_l: List[str] = []
    for p, lab in zip(path_files, labels):
        eid = example_id_from_path(p)
        rec = manifest.examples.get(eid)
        if rec is None:
            # Not in manifest: keep only if no sealed-only policy needed
            out_p.append(str(p))
            out_l.append(str(lab))
            continue
        if rec.excluded:
            continue
        if rec.split not in allowed and rec.split != SPLIT_UNASSIGNED:
            # sealed_test blocked; unassigned handled by caller
            continue
        if rec.split == SPLIT_SEALED_TEST:
            continue
        if allow_splits is not None and rec.split not in allowed:
            continue
        out_p.append(str(p))
        out_l.append(rec.active_label)
    return out_p, out_l


def apply_manifest_to_path_list(
    path_files: Sequence[PathLike],
    manifest: DatasetManifest,
    *,
    drop_excluded: bool = True,
    drop_sealed: bool = True,
    filename_label_fn: Optional[Any] = None,
) -> Tuple[List[str], Dict[str, str], int]:
    """Filter training paths and resolve active labels from the manifest.

    Args:
        path_files: Candidate media paths (typically ``.jpg`` pattern images).
        manifest: Loaded dataset manifest for the store.
        drop_excluded: Remove excluded examples.
        drop_sealed: Remove sealed-test examples (train-time isolation).
        filename_label_fn: Optional ``path -> original label``; default uses
            LabGym flat basename convention.

    Returns:
        ``(kept_paths, path_to_active_label, n_dropped)``.
    """
    if filename_label_fn is None:
        def filename_label_fn(p: PathLike) -> str:  # type: ignore[misc]
            return original_label_from_flat_name(Path(p).name)

    kept: List[str] = []
    labels: Dict[str, str] = {}
    n_drop = 0
    for p in path_files:
        sp = str(p)
        eid = example_id_from_path(sp)
        rec = manifest.examples.get(eid)
        if rec is not None:
            if drop_excluded and rec.excluded:
                n_drop += 1
                continue
            if drop_sealed and rec.split == SPLIT_SEALED_TEST:
                n_drop += 1
                continue
            active = rec.active_label
        else:
            active = str(filename_label_fn(sp))
        kept.append(sp)
        labels[sp] = active
    return kept, labels, n_drop


def rebuild_classmapping(
    active_labels: Iterable[str],
) -> Tuple[List[str], Dict[str, Any]]:
    """Build sorted classnames and one-hot (or binary) class mapping.

    Uses sklearn ``LabelBinarizer`` the same way as onfly Sequence loaders.
    """
    from sklearn.preprocessing import LabelBinarizer

    classnames = sorted({str(c) for c in active_labels if str(c).strip() != ""})
    if not classnames:
        return [], {}
    labels = np.array(classnames)
    lb = LabelBinarizer()
    transformed = lb.fit_transform(labels)
    transformed = [list(i) for i in transformed]
    classmapping = {name: transformed[i] for i, name in enumerate(classnames)}
    return classnames, classmapping


def resolve_example_media_path(
    store_root: PathLike,
    example_id: str,
    *,
    path_hint: Optional[str] = None,
    scan_index: Optional[Mapping[str, PathLike]] = None,
) -> Optional[Path]:
    """Locate a preview media file for ``example_id`` under the store."""
    root = Path(store_root)
    eid = str(example_id)
    candidates: List[Path] = []
    if path_hint:
        candidates.append(root / path_hint)
        candidates.append(Path(path_hint))
    if scan_index and eid in scan_index:
        candidates.append(Path(scan_index[eid]))
    # Flat prepared layout
    for ext in (".jpg", ".jpeg", ".png", ".avi", ".JPG", ".JPEG", ".PNG", ".AVI"):
        candidates.append(root / f"{eid}{ext}")
        for sub in ("train", "validation", "test"):
            candidates.append(root / sub / f"{eid}{ext}")
    # Behavior-folder layout: */<eid>.*
    if root.is_dir():
        for behavior in root.iterdir():
            if not behavior.is_dir() or behavior.name.startswith("."):
                continue
            for ext in (".jpg", ".jpeg", ".png", ".avi"):
                candidates.append(behavior / f"{eid}{ext}")

    seen: set[str] = set()
    for c in candidates:
        try:
            key = str(c.resolve()) if c.exists() else str(c)
        except OSError:
            key = str(c)
        if key in seen:
            continue
        seen.add(key)
        if c.is_file():
            return c
    return None
