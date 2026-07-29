"""Training helpers: soft labels, combined losses, example sorting, evaluation."""

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
from .ethogram_examples import (
    GenerationConfig,
    collect_windows,
    generate_examples_from_ethogram,
    sample_windows_from_bout,
)
from .evaluation import (
    EvaluationMetrics,
    ExamplePrediction,
    HighLossExample,
    compute_evaluation_metrics,
    hard_labels_from_targets,
    load_evaluation_run,
    model_settings_from_parameters_df,
    predictions_from_model_output,
    rank_high_loss_examples,
    write_evaluation_run,
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
    "GenerationConfig",
    "collect_windows",
    "generate_examples_from_ethogram",
    "sample_windows_from_bout",
    "EvaluationMetrics",
    "ExamplePrediction",
    "HighLossExample",
    "compute_evaluation_metrics",
    "hard_labels_from_targets",
    "load_evaluation_run",
    "model_settings_from_parameters_df",
    "predictions_from_model_output",
    "rank_high_loss_examples",
    "write_evaluation_run",
]
