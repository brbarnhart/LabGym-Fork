"""Training helpers: soft labels, combined losses, example sorting."""

from .soft_labels import (
    LABEL_MODE_HARD_ONLY,
    LABEL_MODE_HARD_SOFT_AUX,
    LABEL_MODE_SOFT_PRIMARY,
    SoftLabelTable,
    build_soft_targets_for_window,
    dense_frame_labels_from_session,
    write_soft_labels_sidecar,
)
from .losses import compile_with_label_mode, make_label_loss
from .example_sort import (
    parse_labgym_example_basename,
    sort_examples_from_annotations,
    sort_examples_from_csv_subject_aware,
)

__all__ = [
    "LABEL_MODE_HARD_ONLY",
    "LABEL_MODE_HARD_SOFT_AUX",
    "LABEL_MODE_SOFT_PRIMARY",
    "SoftLabelTable",
    "build_soft_targets_for_window",
    "dense_frame_labels_from_session",
    "write_soft_labels_sidecar",
    "compile_with_label_mode",
    "make_label_loss",
    "parse_labgym_example_basename",
    "sort_examples_from_annotations",
    "sort_examples_from_csv_subject_aware",
]
