"""Background workers for prepare-examples and train categorizer."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal


def auto_aug_workers() -> int:
    """Conservative default process count for export augmentation."""
    try:
        from LabGym.augment_export import default_aug_workers

        return int(default_aug_workers(export=True))
    except Exception:
        cpu = os.cpu_count() or 1
        return max(1, min(8, cpu - 1 if cpu > 1 else 1))


class PrepWorker(QObject):
    """Run Categorizers.rename_label off the UI thread."""

    finished = Signal(str)
    error = Signal(str)

    def __init__(self, src: str, dst: str):
        super().__init__()
        self.src = src
        self.dst = dst

    def run(self) -> None:
        try:
            from LabGym.categorizer import Categorizers

            CA = Categorizers()
            CA.rename_label(self.src, self.dst, resize=None)
            self.finished.emit(self.dst)
        except Exception as exc:
            self.error.emit(str(exc))


class TrainWorker(QObject):
    """Run export-augment + train_pattern_recognizer / train_combnet."""

    finished = Signal(str)
    cancelled = Signal(str)
    error = Signal(str)
    progress = Signal(str)
    progress_aug = Signal(int, int, str)  # done, total, message
    progress_train = Signal(int, dict)  # epoch (1-based), logs

    def __init__(self, params: dict, cancel_event: threading.Event):
        super().__init__()
        self.params = params
        self.cancel_event = cancel_event

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            from LabGym.categorizer import Categorizers

            p = self.params
            Path(p["model_path"]).mkdir(parents=True, exist_ok=True)
            CA = Categorizers()
            CA.label_mode = p["label_mode"]
            CA.lambda_soft = p["lambda_soft"]
            out_folder = p.get("out_folder")
            if not out_folder:
                out_folder = str(Path(p["model_path"]) / "augmented_data")
            Path(out_folder).mkdir(parents=True, exist_ok=True)
            n_workers = int(p.get("num_workers") or 1)

            def _aug_cb(done: int, total: int, msg: str) -> None:
                self.progress_aug.emit(int(done), int(total), str(msg))

            def _train_cb(epoch: int, logs: dict) -> None:
                self.progress_train.emit(int(epoch), dict(logs or {}))

            self.progress.emit(
                f"Export-augment then train onfly → {out_folder} "
                f"({n_workers} worker{'s' if n_workers != 1 else ''})…"
            )
            if not p["animation_analyzer"]:
                CA.train_pattern_recognizer(
                    p["data_path"],
                    p["model_path"],
                    out_path=p.get("out_path"),
                    dim=p["dim_conv"],
                    channel=3 if p["behavior_mode"] != 2 else p["channel"],
                    time_step=p["length"],
                    level=p["level_conv"],
                    aug_methods=p["aug_methods"],
                    augvalid=p["augvalid"],
                    include_bodyparts=p["include_bodyparts"],
                    std=p["std"],
                    background_free=p["background_free"],
                    black_background=p["black_background"],
                    behavior_mode=p["behavior_mode"],
                    social_distance=p["social_distance"],
                    out_folder=out_folder,
                    label_mode=p["label_mode"],
                    lambda_soft=p["lambda_soft"],
                    soft_labels_path=p.get("soft_labels_path"),
                    num_workers=n_workers,
                    progress_cb=_aug_cb,
                    train_progress_cb=_train_cb,
                    cancel_event=self.cancel_event,
                    skip_augment=bool(p.get("skip_augment")),
                )
            else:
                CA.train_combnet(
                    p["data_path"],
                    p["model_path"],
                    out_path=p.get("out_path"),
                    dim_tconv=p["dim_tconv"],
                    dim_conv=p["dim_conv"],
                    channel=p["channel"],
                    time_step=p["length"],
                    level_tconv=p["level_tconv"],
                    level_conv=p["level_conv"],
                    aug_methods=p["aug_methods"],
                    augvalid=p["augvalid"],
                    include_bodyparts=p["include_bodyparts"],
                    std=p["std"],
                    background_free=p["background_free"],
                    black_background=p["black_background"],
                    behavior_mode=p["behavior_mode"],
                    social_distance=p["social_distance"],
                    color_costar=p["color_costar"],
                    out_folder=out_folder,
                    label_mode=p["label_mode"],
                    lambda_soft=p["lambda_soft"],
                    soft_labels_path=p.get("soft_labels_path"),
                    num_workers=n_workers,
                    progress_cb=_aug_cb,
                    train_progress_cb=_train_cb,
                    cancel_event=self.cancel_event,
                    skip_augment=bool(p.get("skip_augment")),
                )
            if self.cancel_event.is_set():
                self.cancelled.emit("Cancelled by user.")
            else:
                self.finished.emit(p["model_path"])
        except Exception as exc:
            from LabGym.training.progress import TrainingCancelled

            if isinstance(exc, TrainingCancelled) or self.cancel_event.is_set():
                self.cancelled.emit(str(exc) or "Cancelled by user.")
            else:
                self.error.emit(str(exc))
