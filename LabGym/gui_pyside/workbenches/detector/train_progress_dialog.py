"""Pop-out progress UI for detector training (iterations + live loss curve)."""

from __future__ import annotations

from io import BytesIO
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFormLayout, QGroupBox, QLabel

from LabGym.detection.train_progress import primary_total_loss
from LabGym.gui_pyside.widgets.progress_dialog_base import JobProgressDialogBase


class TrainDetectorProgressDialog(JobProgressDialogBase):
    """Live iteration progress and total_loss curve during detector training."""

    def __init__(self, parent=None):
        super().__init__(
            "Detector training progress",
            parent,
            min_width=560,
            min_height=560,
            cancel_text="Close when idle",
            cancel_tooltip=(
                "Detector training cannot be cancelled mid-run yet. "
                "Close is available after training finishes or fails."
            ),
            show_close_button=True,
            confirm_close_while_running=True,
            close_while_running_title="Training in progress",
            close_while_running_message=(
                "Detector training is still running.\n\n"
                "Closing this window does not stop training. "
                "Leave it open to watch loss, or hide it and wait for the tab log."
            ),
        )
        self._iters: List[int] = []
        self._loss: List[float] = []
        self._max_iter = 0
        self._curve_every_n = 1
        self._updates_since_curve = 0

        self.add_phase_label("Starting…")

        self.progress = self.add_progress_bar(
            format_str="Training: waiting…",
            tooltip="Detectron2 training iterations completed vs configured MAX_ITER.",
            determinate=True,
        )
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

        metrics = QGroupBox("Training metrics (live)")
        metrics.setToolTip(
            "Smoothed Detectron2 losses from EventStorage (same window as the "
            "console printer, typically ~20 iterations)."
        )
        mform = QFormLayout(metrics)
        self.lbl_iter = QLabel("—")
        self.lbl_total_loss = QLabel("—")
        self.lbl_lr = QLabel("—")
        self.lbl_components = QLabel("—")
        self.lbl_components.setWordWrap(True)
        mform.addRow("Iteration:", self.lbl_iter)
        mform.addRow("total_loss:", self.lbl_total_loss)
        mform.addRow("learning rate:", self.lbl_lr)
        mform.addRow("other losses:", self.lbl_components)

        self.lbl_curve = QLabel()
        self.lbl_curve.setMinimumHeight(160)
        self.lbl_curve.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_curve.setStyleSheet(
            "QLabel { background: #ffffff; color: #222; border: 1px solid #ccc; "
            "border-radius: 4px; }"
        )
        self.lbl_curve.setText("Loss curve appears after the first reported iteration.")
        mform.addRow(self.lbl_curve)
        self.content_layout.addWidget(metrics)

        self.add_log(stretch=True)
        self.finish_building_ui()
        # Cancel is not wired for detector yet — disable while running.
        if self.btn_cancel is not None:
            self.btn_cancel.setVisible(False)

    def begin_job(self, *, max_iter: int = 0) -> None:
        super().begin_job()
        self._max_iter = max(0, int(max_iter))
        self._iters = []
        self._loss = []
        self._updates_since_curve = 0
        # Redraw curve at most ~100 times for long runs (keeps UI responsive).
        self._curve_every_n = max(1, (self._max_iter // 20) // 100) if self._max_iter else 1
        if self._max_iter > 0:
            self.progress.setRange(0, self._max_iter)
            self.progress.setValue(0)
            self.progress.setFormat(f"Training: 0 / {self._max_iter}")
        else:
            self.progress.setRange(0, 0)
            self.progress.setFormat("Training…")
        self.lbl_iter.setText("—")
        self.lbl_total_loss.setText("—")
        self.lbl_lr.setText("—")
        self.lbl_components.setText("—")
        self.lbl_curve.clear()
        self.lbl_curve.setText("Loss curve appears after the first reported iteration.")

    def mark_finished(self, *, cancelled: bool = False, failed: bool = False) -> None:
        super().mark_finished(cancelled=cancelled, failed=failed)
        if self._max_iter > 0:
            self.progress.setRange(0, self._max_iter)
        else:
            self.progress.setRange(0, 1)
        if failed:
            self.progress.setFormat("Training: failed")
        elif cancelled:
            self.progress.setFormat("Training: cancelled")
        else:
            if self._max_iter > 0:
                self.progress.setValue(self._max_iter)
            else:
                self.progress.setValue(1)
            self.progress.setFormat("Training: done")
        # Final curve redraw with all points
        self._refresh_curve()

    def on_status(self, msg: str) -> None:
        self.append_log(msg)
        self.set_phase(msg)

    def on_train_progress(self, iteration: int, max_iter: int, metrics: dict) -> None:
        it = int(iteration)
        mi = int(max_iter) if max_iter else self._max_iter
        if mi > 0 and mi != self._max_iter:
            self._max_iter = mi
            self.progress.setRange(0, mi)
        if mi > 0:
            self.progress.setRange(0, mi)
            self.progress.setValue(min(it, mi))
            self.progress.setFormat(f"Training: {it} / {mi}")
        else:
            self.progress.setRange(0, 0)
            self.progress.setFormat(f"Training: iter {it}")

        self.lbl_iter.setText(f"{it} / {mi}" if mi else str(it))

        total = primary_total_loss(metrics or {})
        if total is not None:
            self.lbl_total_loss.setText(f"{total:.4f}")
            self._iters.append(it)
            self._loss.append(float(total))
        else:
            self.lbl_total_loss.setText("—")

        lr = (metrics or {}).get("lr")
        if lr is not None:
            try:
                self.lbl_lr.setText(f"{float(lr):.6g}")
            except (TypeError, ValueError):
                self.lbl_lr.setText(str(lr))
        else:
            self.lbl_lr.setText("—")

        extras = []
        for k, v in sorted((metrics or {}).items()):
            if k in ("total_loss", "lr") or "loss" not in k:
                continue
            try:
                extras.append(f"{k}={float(v):.4g}")
            except (TypeError, ValueError):
                extras.append(f"{k}={v}")
        self.lbl_components.setText("; ".join(extras) if extras else "—")

        loss_txt = f"{total:.4f}" if total is not None else "—"
        phase = f"Iter {it}" + (f"/{mi}" if mi else "") + f": total_loss={loss_txt}"
        self.set_phase(phase)
        # Log every update (already throttled by Detectron2 period ~20 iters).
        extra = f"  lr={self.lbl_lr.text()}" if self.lbl_lr.text() != "—" else ""
        self.append_log(phase + extra)

        self._updates_since_curve += 1
        if (
            self._updates_since_curve >= self._curve_every_n
            or (mi > 0 and it >= mi)
            or len(self._loss) == 1
        ):
            self._updates_since_curve = 0
            self._refresh_curve()

    def _refresh_curve(self) -> None:
        if not self._loss:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5.0, 2.2), dpi=90, facecolor="white")
            ax.set_facecolor("white")
            xs = (
                self._iters
                if len(self._iters) == len(self._loss)
                else list(range(1, len(self._loss) + 1))
            )
            ax.plot(xs, self._loss, label="total_loss", color="#1f77b4", linewidth=1.5)
            ax.set_xlabel("Iteration", color="black")
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
            # Matplotlib optional for headless CI; keep numeric metrics either way.
            pass
