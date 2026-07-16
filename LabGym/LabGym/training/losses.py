"""Combined hard / soft losses for LabGym categorizers."""

from __future__ import annotations

from typing import Any, Optional

from .soft_labels import (
    DEFAULT_LAMBDA_SOFT,
    LABEL_MODE_HARD_ONLY,
    LABEL_MODE_HARD_SOFT_AUX,
    LABEL_MODE_SOFT_PRIMARY,
)


def make_label_loss(
    label_mode: str = LABEL_MODE_HARD_ONLY,
    lambda_soft: float = DEFAULT_LAMBDA_SOFT,
    *,
    binary: bool = False,
    n_classes: int = 2,
    mu_hard: float = 0.25,
):
    """Return a Keras-compatible loss function.

    For ``hard_soft_aux`` / ``soft_primary``, ``y_true`` is expected as
    concatenation ``[hard | soft]`` along the last axis with shape
    ``(..., 2 * n_out)`` where ``n_out`` is 1 for binary sigmoid heads and
    ``n_classes`` for softmax heads.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import backend as K
    except Exception as exc:  # pragma: no cover
        raise ImportError("TensorFlow is required for categorizer losses") from exc

    label_mode = (label_mode or LABEL_MODE_HARD_ONLY).strip().lower()
    lambda_soft = float(lambda_soft)
    mu_hard = float(mu_hard)

    if label_mode == LABEL_MODE_HARD_ONLY:
        if binary:
            return "binary_crossentropy"
        return "categorical_crossentropy"

    def _split(y_true):
        # y_true: (batch, 2*C) or (batch, 2) for binary
        last = K.int_shape(y_true)[-1]
        if last is None:
            # dynamic
            C = n_classes if not binary else 1
        else:
            C = int(last // 2)
        hard = y_true[..., :C]
        soft = y_true[..., C:]
        return hard, soft

    def combined_loss(y_true, y_pred):
        hard, soft = _split(y_true)
        # clip soft for numerical safety
        soft = K.clip(soft, K.epsilon(), 1.0)

        if binary:
            L_hard = K.mean(K.binary_crossentropy(hard, y_pred), axis=-1)
            L_soft = K.mean(K.binary_crossentropy(soft, y_pred), axis=-1)
        else:
            L_hard = K.categorical_crossentropy(hard, y_pred)
            # Soft CE: -sum q log p
            L_soft = K.sum(-soft * K.log(K.clip(y_pred, K.epsilon(), 1.0)), axis=-1)

        if label_mode == LABEL_MODE_SOFT_PRIMARY:
            return L_soft + mu_hard * L_hard
        # hard_soft_aux (default)
        return L_hard + lambda_soft * L_soft

    combined_loss.__name__ = f"label_loss_{label_mode}"
    return combined_loss


def compile_with_label_mode(
    model: Any,
    classnames,
    label_mode: str = LABEL_MODE_HARD_ONLY,
    lambda_soft: float = DEFAULT_LAMBDA_SOFT,
    *,
    optimizer=None,
    metrics=None,
    mu_hard: float = 0.25,
):
    """Compile a Keras categorizer with the selected label mode."""
    try:
        from tensorflow.keras.optimizers import SGD
    except Exception as exc:  # pragma: no cover
        raise ImportError("TensorFlow is required") from exc

    if optimizer is None:
        optimizer = SGD(learning_rate=1e-4, momentum=0.9)
    if metrics is None:
        metrics = ["accuracy"]

    n_classes = len(list(classnames))
    binary = n_classes == 2
    loss = make_label_loss(
        label_mode,
        lambda_soft,
        binary=binary,
        n_classes=n_classes,
        mu_hard=mu_hard,
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def maybe_stack_soft_targets(
    hard_labels,
    soft_matrix,
    label_mode: str,
):
    """If soft mode, return stacked targets; else return hard_labels."""
    import numpy as np

    mode = (label_mode or LABEL_MODE_HARD_ONLY).strip().lower()
    if mode == LABEL_MODE_HARD_ONLY or soft_matrix is None:
        return hard_labels
    hard = np.asarray(hard_labels, dtype=np.float32)
    soft = np.asarray(soft_matrix, dtype=np.float32)
    if hard.ndim == 1:
        # binary LabelBinarizer can give (N,) for 2-class in some versions — expand
        hard = hard.reshape(-1, 1)
    if soft.shape[0] != hard.shape[0]:
        raise ValueError("soft_matrix batch dim mismatch")
    if soft.shape[1] != hard.shape[1]:
        # Align: if hard is multi-class one-hot and soft matches class count
        if soft.shape[1] == hard.shape[1]:
            pass
        else:
            raise ValueError(
                f"soft C={soft.shape[1]} vs hard C={hard.shape[1]}"
            )
    return np.concatenate([hard, soft], axis=-1)
