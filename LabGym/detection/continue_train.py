"""Helpers for continuing / fine-tuning an existing LabGym detector.

Continue-training (same classes, warm-start from ``model_final.pth``) is supported.
Adding new animal categories is not supported in v1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

from LabGym.detection.batch_detect import (
    load_detector_animal_kinds,
    validate_detector_folder,
)

PathLike = Union[str, Path]

# Defaults for a fresh train vs continue-from-weights run.
DEFAULT_BASE_LR = 0.001
CONTINUE_BASE_LR = 0.0001
DEFAULT_ITERATIONS = 1000
CONTINUE_DEFAULT_ITERATIONS = 500


@dataclass(frozen=True)
class ContinueTrainPlan:
    """Validated plan for warm-starting detector training."""

    base_detector: str
    weights_path: str
    animal_names: List[str]
    inference_size: Optional[int]
    base_lr: float


def annotation_animal_names(path_to_annotation: PathLike) -> List[str]:
    """Category names from a LabGym/COCO annotation file (same rule as Detector.train).

    Uses categories with ``id > 0`` in file order (background id 0 is skipped).
    """
    path = Path(path_to_annotation)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found:\n{path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid annotation JSON:\n{path}\n\n{exc}") from exc

    cats = data.get("categories") or []
    if not isinstance(cats, list):
        raise ValueError(f"Annotation JSON has no categories list:\n{path}")

    names: List[str] = []
    for c in cats:
        if not isinstance(c, dict):
            continue
        try:
            cid = int(c.get("id", 0))
        except (TypeError, ValueError):
            continue
        if cid > 0:
            names.append(str(c.get("name", "")).strip())
    names = [n for n in names if n]
    if not names:
        raise ValueError(
            f"No animal categories (id > 0) found in annotation file:\n{path}"
        )
    return names


def class_lists_compatible(
    base_names: Sequence[str], annotation_names: Sequence[str]
) -> bool:
    """True if class lists match for fine-tuning (same names, same order)."""
    b = [str(x).strip() for x in base_names]
    a = [str(x).strip() for x in annotation_names]
    return b == a and len(b) > 0


def suggest_continued_detector_name(base_detector: PathLike) -> str:
    """Suggest an output folder name for a fine-tuned detector."""
    stem = Path(base_detector).name.strip() or "detector"
    # Avoid stacking endless _ft_ft suffixes in the suggestion.
    if stem.endswith("_ft"):
        return f"{stem}2"
    return f"{stem}_ft"


def plan_continue_training(
    base_detector: PathLike,
    path_to_annotation: PathLike,
    *,
    base_lr: Optional[float] = None,
) -> ContinueTrainPlan:
    """Validate base detector + annotation classes for continue-training.

    Raises ``FileNotFoundError`` / ``ValueError`` with user-facing messages.
    """
    root = validate_detector_folder(base_detector, require_weights=True)
    weights = root / "model_final.pth"
    base_names = load_detector_animal_kinds(root)
    ann_names = annotation_animal_names(path_to_annotation)

    if not class_lists_compatible(base_names, ann_names):
        raise ValueError(
            "Cannot continue training: animal categories do not match.\n\n"
            f"Base detector classes ({len(base_names)}):\n  {base_names}\n\n"
            f"Annotation classes ({len(ann_names)}):\n  {ann_names}\n\n"
            "Continue-training requires the same class names in the same order. "
            "Adding new categories is not supported yet — train a new detector "
            "from scratch (COCO init) for a different label set."
        )

    infer_size: Optional[int] = None
    try:
        from LabGym.detection.batch_detect import _read_detector_parameters_json

        params = _read_detector_parameters_json(root)
        raw_sz = params.get("inferencing_framesize")
        if raw_sz is not None:
            infer_size = int(raw_sz)
    except Exception:
        infer_size = None

    lr = float(base_lr) if base_lr is not None else CONTINUE_BASE_LR
    if lr <= 0:
        raise ValueError(f"base_lr must be positive, got {lr}")

    return ContinueTrainPlan(
        base_detector=str(root.resolve()),
        weights_path=str(weights.resolve()),
        animal_names=list(base_names),
        inference_size=infer_size,
        base_lr=lr,
    )
