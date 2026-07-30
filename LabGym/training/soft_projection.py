"""Soft-label projection into the active taxonomy (merge / exclude).

Original soft vectors live in the soft label store (``soft_labels.csv``).
Per-example soft overrides live in the dataset manifest. Projection is
deterministic and never rewrites the soft label store. See ADR 0005.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from LabGym.training.dataset_manifest import DatasetManifest, example_id_from_path
from LabGym.training.soft_labels import SoftLabelTable


def compose_merge_map(taxonomy_ops: Sequence[Mapping]) -> Dict[str, str]:
    """Compose merge ops into source → final active category map."""
    mapping: Dict[str, str] = {}
    for op in taxonomy_ops:
        if str(op.get("op") or "") != "merge":
            continue
        target = str(op.get("target") or "").strip()
        sources = [str(s).strip() for s in (op.get("sources") or []) if str(s).strip()]
        if not target or not sources:
            continue
        # Remap anything currently pointing at a source
        for k, v in list(mapping.items()):
            if v in sources or k in sources:
                mapping[k] = target
        for s in sources:
            mapping[s] = target
        # Target identity (explicit)
        if target not in mapping:
            mapping[target] = target
    return mapping


def excluded_categories_from_ops(taxonomy_ops: Sequence[Mapping]) -> Set[str]:
    """Categories marked excluded by taxonomy ops (last write wins per name)."""
    state: Dict[str, bool] = {}
    for op in taxonomy_ops:
        kind = str(op.get("op") or "")
        if kind == "exclude_category":
            cat = str(op.get("category") or "").strip()
            if cat:
                state[cat] = bool(op.get("excluded", True))
        elif kind == "include_category":
            cat = str(op.get("category") or "").strip()
            if cat:
                state[cat] = False
    return {c for c, ex in state.items() if ex}


def project_soft_vector(
    soft: Sequence[float],
    source_classnames: Sequence[str],
    target_classnames: Sequence[str],
    *,
    merge_map: Optional[Mapping[str, str]] = None,
    excluded: Optional[Iterable[str]] = None,
    renormalize: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """Project one soft vector from source class space into target space.

    - Mass of merged sources is summed onto the merge target.
    - Mass of excluded categories is dropped.
    - Remaining mass is renormalized (when ``renormalize`` and sum > 0).
    """
    src = [str(c) for c in source_classnames]
    tgt = [str(c) for c in target_classnames]
    vec = np.asarray(soft, dtype=np.float64).reshape(-1)
    if vec.shape[0] != len(src):
        # Pad / truncate defensively
        aligned = np.zeros(len(src), dtype=np.float64)
        n = min(len(src), vec.shape[0])
        aligned[:n] = vec[:n]
        vec = aligned
    mmap = dict(merge_map or {})
    excl = {str(x) for x in (excluded or [])}
    mass: Dict[str, float] = {}
    for i, name in enumerate(src):
        if name in excl:
            continue
        dest = mmap.get(name, name)
        if dest in excl:
            continue
        mass[dest] = mass.get(dest, 0.0) + float(vec[i])
    out = np.zeros(len(tgt), dtype=np.float32)
    for j, t in enumerate(tgt):
        out[j] = float(mass.get(t, 0.0))
    if renormalize:
        s = float(out.sum())
        if s > eps:
            out = (out / s).astype(np.float32)
        else:
            out = np.zeros(len(tgt), dtype=np.float32)
    return out


def is_usable_soft(soft: np.ndarray, *, eps: float = 1e-8) -> bool:
    """True if projected soft has positive mass and is finite."""
    a = np.asarray(soft, dtype=np.float64)
    return bool(np.isfinite(a).all() and float(a.sum()) > eps)


def effective_soft_for_basename(
    basename: str,
    table: SoftLabelTable,
    target_classnames: Sequence[str],
    *,
    manifest: Optional[DatasetManifest] = None,
    merge_map: Optional[Mapping[str, str]] = None,
    excluded: Optional[Iterable[str]] = None,
) -> Optional[np.ndarray]:
    """Resolve effective soft vector for one example basename.

    Prefers manifest soft override when present; otherwise the soft store row.
    Applies merge/exclude projection into ``target_classnames``.
    Returns None if no source soft is available or projection is empty.
    """
    eid = example_id_from_path(basename) if basename else ""
    # Also try raw basename as key (soft_labels uses stem)
    keys = [basename, eid]
    soft: Optional[np.ndarray] = None
    source_names: List[str] = list(table.classnames)

    if manifest is not None:
        rec = manifest.examples.get(eid) or manifest.examples.get(basename)
        if rec is not None and rec.soft_override is not None:
            soft = np.asarray(rec.soft_override, dtype=np.float32)
            if rec.soft_override_classnames:
                source_names = list(rec.soft_override_classnames)

    if soft is None:
        for k in keys:
            if k in table.rows:
                _hard, soft = table.rows[k]
                source_names = list(table.classnames)
                break
    if soft is None:
        return None

    mmap = merge_map
    excl = excluded
    if manifest is not None:
        if mmap is None:
            mmap = compose_merge_map(manifest.taxonomy_ops)
        if excl is None:
            excl = excluded_categories_from_ops(manifest.taxonomy_ops)

    projected = project_soft_vector(
        soft,
        source_names,
        target_classnames,
        merge_map=mmap,
        excluded=excl,
        renormalize=True,
    )
    if not is_usable_soft(projected):
        return None
    return projected


def effective_soft_matrix(
    table: SoftLabelTable,
    basenames: Sequence[str],
    target_classnames: Sequence[str],
    *,
    manifest: Optional[DatasetManifest] = None,
) -> Tuple[np.ndarray, List[bool]]:
    """Build (N, C) soft matrix for basenames in target class order.

    Returns:
        matrix: float32 array shape (N, C); zeros where projection unavailable.
        usable: per-row flag whether a usable soft vector was produced.
    """
    C = len(target_classnames)
    N = len(basenames)
    out = np.zeros((N, C), dtype=np.float32)
    usable = [False] * N
    mmap = compose_merge_map(manifest.taxonomy_ops) if manifest is not None else {}
    excl = excluded_categories_from_ops(manifest.taxonomy_ops) if manifest is not None else set()
    for i, base in enumerate(basenames):
        vec = effective_soft_for_basename(
            str(base),
            table,
            target_classnames,
            manifest=manifest,
            merge_map=mmap,
            excluded=excl,
        )
        if vec is not None:
            out[i] = vec
            usable[i] = True
    return out, usable


def soft_matrix_with_hard_fallback(
    soft_matrix: np.ndarray,
    usable: Sequence[bool],
    hard_labels: Sequence[int],
    *,
    n_classes: Optional[int] = None,
) -> Tuple[np.ndarray, int, str]:
    """Replace unusable soft rows with hard one-hot vectors (ADR 0005).

    Does not mutate ``soft_matrix``. Unusable rows (empty projection / missing
    soft) get a one-hot of the example's hard label so soft training modes
    degrade to hard-only for those examples only.

    Args:
        soft_matrix: Array shape ``(N, C)``.
        usable: Per-row flag from :func:`effective_soft_matrix`.
        hard_labels: Per-example hard class indices (length N).
        n_classes: Soft column count; defaults to ``soft_matrix.shape[1]``.

    Returns:
        filled: Copy of soft matrix with hard one-hots on unusable rows.
        n_filled: Number of rows replaced.
        warning: Non-empty human message when any row was filled; else ``""``.
    """
    soft = np.asarray(soft_matrix, dtype=np.float32)
    if soft.ndim != 2:
        raise ValueError(f"soft_matrix must be 2-D, got shape {soft.shape}")
    n, c = soft.shape
    if n_classes is None:
        n_classes = int(c)
    if int(n_classes) != c:
        raise ValueError(f"n_classes={n_classes} does not match soft C={c}")
    if len(usable) != n:
        raise ValueError("usable length must match soft_matrix rows")
    hard = np.asarray(hard_labels)
    if hard.shape[0] != n:
        raise ValueError("hard_labels length must match soft_matrix rows")

    filled = soft.copy()
    n_filled = 0
    for i, ok in enumerate(usable):
        if ok:
            continue
        idx = int(hard[i])
        row = np.zeros(c, dtype=np.float32)
        if 0 <= idx < c:
            row[idx] = 1.0
        filled[i] = row
        n_filled += 1

    warning = ""
    if n_filled:
        warning = (
            f"Soft projection unusable for {n_filled}/{n} examples; "
            f"using hard-only soft targets for those rows."
        )
    return filled, n_filled, warning


def project_class_means(
    table: SoftLabelTable,
    target_classnames: Sequence[str],
    *,
    manifest: Optional[DatasetManifest] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Per-active-class mean soft vectors after projection (for onfly loaders).

    Buckets by **projected hard** (original hard through merge map) when possible,
    else by argmax of projected soft.
    """
    mmap = compose_merge_map(manifest.taxonomy_ops) if manifest is not None else {}
    excl = excluded_categories_from_ops(manifest.taxonomy_ops) if manifest is not None else set()
    buckets: Dict[str, List[np.ndarray]] = {c: [] for c in target_classnames}

    for hard, soft in table.rows.values():
        projected = project_soft_vector(
            soft,
            table.classnames,
            target_classnames,
            merge_map=mmap,
            excluded=excl,
        )
        if not is_usable_soft(projected):
            continue
        dest_hard = mmap.get(str(hard), str(hard))
        if dest_hard in excl:
            # use soft argmax among targets
            dest_hard = target_classnames[int(np.argmax(projected))] if target_classnames else ""
        if dest_hard not in buckets:
            # hard not in active taxonomy — bucket by soft argmax
            if not target_classnames:
                continue
            dest_hard = target_classnames[int(np.argmax(projected))]
        if dest_hard in buckets:
            buckets[dest_hard].append(projected)

    means: Dict[str, np.ndarray] = {}
    for c, vecs in buckets.items():
        if not vecs:
            continue
        m = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        s = float(m.sum())
        if s > 1e-8:
            m = m / s
        means[c] = m
    return means if means else None
