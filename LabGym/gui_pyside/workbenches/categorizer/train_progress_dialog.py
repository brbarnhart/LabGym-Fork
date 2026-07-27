"""Pop-out progress UI for categorizer augmentation + training."""

from __future__ import annotations

from io import BytesIO
from typing import Dict, List

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel

from LabGym.gui_pyside.widgets.progress_dialog_base import JobProgressDialogBase


class TrainProgressDialog(JobProgressDialogBase):
    """Pop-out window for augmentation + training progress (keeps the form readable)."""

    def __init__(self, parent=None):
        super().__init__(
            "Categorizer training progress",
            parent,
            min_width=560,
            min_height=640,
            cancel_text="Cancel",
            cancel_tooltip=(
                "Cooperatively stop augmentation (between source examples) or training "
                "(at the next epoch/batch boundary). May take a short time to finish."
            ),
            show_close_button=True,
            confirm_close_while_running=True,
            close_while_running_title="Training in progress",
            close_while_running_message=(
                "Training is still running.\n\n"
                "Cancel training and close?\n"
                "Choose No to keep the window open."
            ),
        )
        self._hist: Dict[str, List[float]] = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }

        self.add_phase_label("Starting…")

        self.progress_aug = self.add_progress_bar(
            format_str="Augmentation: %p%",
            tooltip=(
                "Progress while exporting augmented train/validation examples "
                "(by source example count)."
            ),
        )

        self.progress_train = self.add_progress_bar(
            format_str="Training: waiting…",
            tooltip=(
                "Training runs until early stopping. Bar is indeterminate; epoch "
                "metrics update below after each epoch."
            ),
            determinate=False,
        )

        metrics = QGroupBox("Training metrics (live)")
        metrics.setToolTip("Updated after each training epoch.")
        mform = QFormLayout(metrics)
        self.lbl_epoch = QLabel("—")
        self.lbl_loss = QLabel("—")
        self.lbl_val_loss = QLabel("—")
        self.lbl_acc = QLabel("—")
        self.lbl_val_acc = QLabel("—")
        mform.addRow("Epoch:", self.lbl_epoch)
        mform.addRow("loss:", self.lbl_loss)
        mform.addRow("val_loss:", self.lbl_val_loss)
        mform.addRow("accuracy:", self.lbl_acc)
        mform.addRow("val_accuracy:", self.lbl_val_acc)
        self.lbl_curve = QLabel()
        self.lbl_curve.setMinimumHeight(140)
        self.lbl_curve.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # White panel so matplotlib's default black tick/axis labels stay readable
        # under any app theme.
        self.lbl_curve.setStyleSheet(
            "QLabel { background: #ffffff; color: #222; border: 1px solid #ccc; border-radius: 4px; }"
        )
        self.lbl_curve.setText("Loss curves appear after the first epoch.")
        mform.addRow(self.lbl_curve)
        self.content_layout.addWidget(metrics)

        self.add_log(stretch=True)
        self.finish_building_ui()

    def begin_job(self) -> None:
        super().begin_job()
        self.progress_aug.setRange(0, 100)
        self.progress_aug.setValue(0)
        self.progress_train.setRange(0, 0)
        self.progress_train.setFormat("Training: waiting for fit…")
        self._reset_metrics()

    def mark_finished(self, *, cancelled: bool = False, failed: bool = False) -> None:
        super().mark_finished(cancelled=cancelled, failed=failed)
        if failed:
            self.progress_train.setRange(0, 1)
            self.progress_train.setValue(0)
            self.progress_train.setFormat("Training: failed")
        elif cancelled:
            self.progress_train.setRange(0, 1)
            self.progress_train.setValue(0)
            self.progress_train.setFormat("Training: cancelled")
        else:
            self.progress_aug.setValue(100)
            self.progress_train.setRange(0, 1)
            self.progress_train.setValue(1)
            self.progress_train.setFormat("Training: done")

    def _on_cancel_clicked(self) -> None:
        if self.btn_cancel is not None:
            self.btn_cancel.setEnabled(False)
        self.set_phase("Cancel requested… finishing current step")
        self.append_log(
            "Cancel requested — will stop after the current augmentation source "
            "or training epoch/batch boundary."
        )
        self.cancel_requested.emit()

    def _reset_metrics(self) -> None:
        self._hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        self.lbl_epoch.setText("—")
        self.lbl_loss.setText("—")
        self.lbl_val_loss.setText("—")
        self.lbl_acc.setText("—")
        self.lbl_val_acc.setText("—")
        self.lbl_curve.clear()
        self.lbl_curve.setText("Loss curves appear after the first epoch.")

    def on_status(self, msg: str) -> None:
        self.append_log(msg)
        self.set_phase(msg)
        if "train" in msg.lower() and "augment" not in msg.lower():
            self.progress_train.setRange(0, 0)
            self.progress_train.setFormat("Training…")

    def on_aug_progress(self, done: int, total: int, msg: str) -> None:
        if total > 0:
            self.progress_aug.setValue(int(100 * done / total))
        self.set_phase(msg)
        if total > 0 and (done == total or done % max(1, total // 20) == 0):
            self.append_log(msg)
        if total > 0 and done >= total:
            self.progress_train.setRange(0, 0)
            self.progress_train.setFormat("Training: fitting…")
            self.set_phase("Training (epochs until early stop)…")

    def on_train_progress(self, epoch: int, logs: dict) -> None:
        self.progress_train.setRange(0, 0)
        self.progress_train.setFormat(f"Training: epoch {epoch}")
        self.lbl_epoch.setText(str(epoch))

        def _fmt(key: str) -> str:
            if key not in logs:
                return "—"
            try:
                return f"{float(logs[key]):.4f}"
            except (TypeError, ValueError):
                return str(logs[key])

        self.lbl_loss.setText(_fmt("loss"))
        self.lbl_val_loss.setText(_fmt("val_loss"))
        self.lbl_acc.setText(_fmt("accuracy"))
        self.lbl_val_acc.setText(_fmt("val_accuracy"))
        for k in self._hist:
            if k in logs:
                try:
                    self._hist[k].append(float(logs[k]))
                except (TypeError, ValueError):
                    pass
        self.set_phase(
            f"Epoch {epoch}: loss={_fmt('loss')} val_loss={_fmt('val_loss')}"
        )
        self.append_log(
            f"Epoch {epoch}: loss={_fmt('loss')} val_loss={_fmt('val_loss')} "
            f"acc={_fmt('accuracy')} val_acc={_fmt('val_accuracy')}"
        )
        self._refresh_curve()

    def _refresh_curve(self) -> None:
        loss = self._hist.get("loss") or []
        if not loss:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.0, 2.0), dpi=90, facecolor="white")
            ax.set_facecolor("white")
            ax.plot(range(1, len(loss) + 1), loss, label="loss", color="#1f77b4")
            vl = self._hist.get("val_loss") or []
            if vl:
                ax.plot(range(1, len(vl) + 1), vl, label="val_loss", color="#d62728")
            ax.set_xlabel("Epoch", color="black")
            ax.set_ylabel("Loss", color="black")
            ax.tick_params(colors="black")
            for spine in ax.spines.values():
                spine.set_color("black")
            legend = ax.legend(
                loc="upper right", fontsize=8, facecolor="white", edgecolor="#ccc"
            )
            for text in legend.get_texts():
                text.set_color("black")
            ax.grid(True, alpha=0.3, color="#888")
            fig.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png", facecolor="white", edgecolor="none")
            plt.close(fig)
            pix = QPixmap()
            pix.loadFromData(buf.getvalue())
            self.lbl_curve.setPixmap(
                pix.scaledToWidth(480, Qt.TransformationMode.SmoothTransformation)
            )
        except Exception:
            pass
