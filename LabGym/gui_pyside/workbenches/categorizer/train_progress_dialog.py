"""Pop-out progress UI for categorizer augmentation + training."""

from __future__ import annotations

from io import BytesIO
from typing import Dict, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)


class TrainProgressDialog(QDialog):
    """Pop-out window for augmentation + training progress (keeps the form readable)."""

    cancel_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Categorizer training progress")
        self.setMinimumSize(560, 640)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self._running = False
        self._hist: Dict[str, List[float]] = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }

        layout = QVBoxLayout(self)

        self.lbl_phase = QLabel("Starting…")
        self.lbl_phase.setWordWrap(True)
        layout.addWidget(self.lbl_phase)

        self.progress_aug = QProgressBar()
        self.progress_aug.setRange(0, 100)
        self.progress_aug.setValue(0)
        self.progress_aug.setFormat("Augmentation: %p%")
        self.progress_aug.setToolTip(
            "Progress while exporting augmented train/validation examples "
            "(by source example count)."
        )
        layout.addWidget(self.progress_aug)

        self.progress_train = QProgressBar()
        self.progress_train.setRange(0, 0)
        self.progress_train.setFormat("Training: waiting…")
        self.progress_train.setToolTip(
            "Training runs until early stopping. Bar is indeterminate; epoch "
            "metrics update below after each epoch."
        )
        layout.addWidget(self.progress_train)

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
        layout.addWidget(metrics)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        btn_row = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setToolTip(
            "Cooperatively stop augmentation (between source examples) or training "
            "(at the next epoch/batch boundary). May take a short time to finish."
        )
        self.btn_cancel.clicked.connect(self._on_cancel_clicked)
        self.btn_close = QPushButton("Close")
        self.btn_close.setEnabled(False)
        self.btn_close.setToolTip("Close this window (enabled when the job finishes).")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

    def begin_job(self) -> None:
        self._running = True
        self.btn_cancel.setEnabled(True)
        self.btn_close.setEnabled(False)
        self.progress_aug.setValue(0)
        self.progress_train.setRange(0, 0)
        self.progress_train.setFormat("Training: waiting for fit…")
        self._reset_metrics()
        self.lbl_phase.setText("Starting…")
        self.log.clear()
        self.show()
        self.raise_()
        self.activateWindow()

    def mark_finished(self, *, cancelled: bool = False, failed: bool = False) -> None:
        self._running = False
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)
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
        self.btn_cancel.setEnabled(False)
        self.lbl_phase.setText("Cancel requested… finishing current step")
        self.log.append(
            "Cancel requested — will stop after the current augmentation source "
            "or training epoch/batch boundary."
        )
        self.cancel_requested.emit()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._running:
            r = QMessageBox.question(
                self,
                "Training in progress",
                "Training is still running.\n\n"
                "Cancel training and close?\n"
                "Choose No to keep the window open.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if r == QMessageBox.StandardButton.Yes:
                self._on_cancel_clicked()
                event.ignore()
            else:
                event.ignore()
            return
        event.accept()

    def _reset_metrics(self) -> None:
        self._hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        self.lbl_epoch.setText("—")
        self.lbl_loss.setText("—")
        self.lbl_val_loss.setText("—")
        self.lbl_acc.setText("—")
        self.lbl_val_acc.setText("—")
        self.lbl_curve.clear()
        self.lbl_curve.setText("Loss curves appear after the first epoch.")

    def append_log(self, msg: str) -> None:
        self.log.append(msg)

    def set_phase(self, msg: str) -> None:
        self.lbl_phase.setText(msg)

    def on_status(self, msg: str) -> None:
        self.log.append(msg)
        self.lbl_phase.setText(msg)
        if "train" in msg.lower() and "augment" not in msg.lower():
            self.progress_train.setRange(0, 0)
            self.progress_train.setFormat("Training…")

    def on_aug_progress(self, done: int, total: int, msg: str) -> None:
        if total > 0:
            self.progress_aug.setValue(int(100 * done / total))
        self.lbl_phase.setText(msg)
        if total > 0 and (done == total or done % max(1, total // 20) == 0):
            self.log.append(msg)
        if total > 0 and done >= total:
            self.progress_train.setRange(0, 0)
            self.progress_train.setFormat("Training: fitting…")
            self.lbl_phase.setText("Training (epochs until early stop)…")

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
        self.lbl_phase.setText(
            f"Epoch {epoch}: loss={_fmt('loss')} val_loss={_fmt('val_loss')}"
        )
        self.log.append(
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
