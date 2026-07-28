"""Detector → Train detector (PySide wrapper around Detector.train)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from LabGym.detection.continue_train import (
    CONTINUE_BASE_LR,
    CONTINUE_DEFAULT_ITERATIONS,
    DEFAULT_BASE_LR,
    DEFAULT_ITERATIONS,
    plan_continue_training,
    suggest_continued_detector_name,
)
from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.widgets.path_browse import (
    browse_existing_directory,
    path_edit_row,
    set_line_edit_directory,
)
from LabGym.gui_pyside.workbenches.detector.train_progress_dialog import (
    TrainDetectorProgressDialog,
)


class _TrainWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    progress = Signal(str)
    progress_train = Signal(int, int, dict)

    def __init__(self, kwargs: dict):
        super().__init__()
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            self.progress.emit("Loading Detectron2 / Detector…")
            from LabGym.detector import Detector

            dt = Detector()
            if self.kwargs.get("init_from_detector"):
                self.progress.emit(
                    "Continue training from existing detector "
                    f"(LR={self.kwargs.get('base_lr')})…"
                )
            else:
                self.progress.emit("Training from COCO init (this can take a long time)…")

            def _cb(iteration: int, max_iter: int, metrics: dict) -> None:
                # Qt queued connection carries plain dict of floats safely.
                self.progress_train.emit(
                    int(iteration),
                    int(max_iter),
                    dict(metrics or {}),
                )

            train_kwargs = dict(self.kwargs)
            train_kwargs["train_progress_cb"] = _cb
            dt.train(**train_kwargs)
            self.finished.emit(self.kwargs["path_to_detector"])
        except Exception as exc:
            self.error.emit(str(exc))


class TrainDetectorTab(QWidget):
    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None
        self._progress_dlg: Optional[TrainDetectorProgressDialog] = None
        self._user_set_name = False
        self._user_set_iter = False
        self._user_set_lr = False

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Train a LabGym Mask R-CNN detector from COCO instance-segmentation "
            "annotations. Optionally <b>continue training</b> from an existing "
            "detector (same animal categories) using new or expanded annotations "
            "— useful after exporting hard-case frames from Review IDs."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(intro)

        form = QFormLayout()
        self.ed_images = QLineEdit()
        self.ed_images.setToolTip(
            "Folder of all training images referenced by the COCO annotation file. "
            "When fine-tuning, include original images plus new hard-case frames "
            "when possible to limit forgetting."
        )
        b1 = QPushButton("Browse…")
        b1.clicked.connect(lambda: set_line_edit_directory(self, self.ed_images))
        form.addRow(
            self._lab(
                "Training images folder:",
                "Directory containing the still frames used for detector training.",
            ),
            path_edit_row(self.ed_images, b1),
        )

        self.ed_ann = QLineEdit()
        self.ed_ann.setToolTip(
            "COCO instance-segmentation JSON describing animal/object masks and "
            "categories for those images (LabGym / CVAT / EZannot export format)."
        )
        b2 = QPushButton("Browse…")
        b2.clicked.connect(lambda: self._browse_file(self.ed_ann, "JSON (*.json)"))
        form.addRow(
            self._lab("COCO annotation JSON:", "Must list the same images as above."),
            path_edit_row(self.ed_ann, b2),
        )

        self.ed_out = QLineEdit()
        self.ed_out.setToolTip(
            "Parent directory where a new subfolder named below will store the "
            "trained detector (weights + config + model_parameters.txt)."
        )
        b3 = QPushButton("Browse…")
        b3.clicked.connect(lambda: set_line_edit_directory(self, self.ed_out))
        form.addRow(
            self._lab("Parent folder for detector:", "Usually the project models/ folder."),
            path_edit_row(self.ed_out, b3),
        )

        self.ed_name = QLineEdit("New_detector")
        self.ed_name.setToolTip(
            "Name of the new detector subfolder. Letters, numbers, _ and - only; "
            "must not already exist under the parent folder. Prefer a new name "
            "when fine-tuning so the previous detector is kept."
        )
        self.ed_name.textEdited.connect(self._on_name_edited)
        form.addRow(
            self._lab("Detector name:", "Becomes <parent>/<name>/."),
            self.ed_name,
        )

        # --- Continue training ---
        self.chk_continue = QCheckBox("Continue training from an existing detector")
        self.chk_continue.setToolTip(
            "Warm-start from a previous LabGym detector’s model_final.pth instead "
            "of COCO ImageNet/COCO pretrain. Animal category names in the "
            "annotation must match the base detector (same names, same order). "
            "Uses a lower default learning rate."
        )
        self.chk_continue.toggled.connect(self._on_continue_toggled)
        form.addRow(self.chk_continue)

        self.ed_base = QLineEdit()
        self.ed_base.setEnabled(False)
        self.ed_base.setPlaceholderText("Select a trained detector folder…")
        self.ed_base.setToolTip(
            "Existing detector folder (model_final.pth, config.yaml, "
            "model_parameters.txt). Not a categorizer."
        )
        self.btn_base = QPushButton("Browse…")
        self.btn_base.setEnabled(False)
        self.btn_base.clicked.connect(self._browse_base)
        form.addRow(
            self._lab(
                "Base detector:",
                "Weights and class list used to initialize continue-training.",
            ),
            path_edit_row(self.ed_base, self.btn_base),
        )
        self.lbl_base_info = QLabel("")
        self.lbl_base_info.setWordWrap(True)
        self.lbl_base_info.setStyleSheet("color: #9ab;")
        form.addRow("", self.lbl_base_info)

        self.spin_size = QSpinBox()
        self.spin_size.setRange(32, 2048)
        self.spin_size.setSingleStep(32)
        self.spin_size.setValue(480)
        tip_size = (
            "Inferencing frame size (pixels on the short side, divisible by 32). "
            "Larger → more accurate, slower. Smaller → faster, less detail. "
            "480 is LabGym’s usual default; must be divisible by 32. "
            "When continuing, defaults from the base detector if available."
        )
        self.spin_size.setToolTip(tip_size)
        form.addRow(self._lab("Inference frame size:", tip_size), self.spin_size)

        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 1_000_000)
        self.spin_iter.setValue(DEFAULT_ITERATIONS)
        tip_iter = (
            "Number of Detectron2 training iterations. More → usually better accuracy "
            "but longer training. Continue-training often needs fewer iterations "
            f"(default {CONTINUE_DEFAULT_ITERATIONS}) than training from COCO "
            f"(default {DEFAULT_ITERATIONS})."
        )
        self.spin_iter.setToolTip(tip_iter)
        self.spin_iter.valueChanged.connect(self._on_iter_changed)
        form.addRow(self._lab("Training iterations:", tip_iter), self.spin_iter)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(6)
        self.spin_lr.setRange(1e-7, 1.0)
        self.spin_lr.setSingleStep(0.0001)
        self.spin_lr.setValue(DEFAULT_BASE_LR)
        tip_lr = (
            "Solver base learning rate. Continue-training defaults to a lower LR "
            f"({CONTINUE_BASE_LR}) to avoid destroying useful weights; from-scratch "
            f"default is {DEFAULT_BASE_LR}."
        )
        self.spin_lr.setToolTip(tip_lr)
        self.spin_lr.valueChanged.connect(self._on_lr_changed)
        form.addRow(self._lab("Base learning rate:", tip_lr), self.spin_lr)

        layout.addLayout(form)

        self.btn = QPushButton("Train detector")
        self.btn.clicked.connect(self._run)
        layout.addWidget(self.btn)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        self.project.project_replaced.connect(self._defaults)
        self._defaults()

    def _defaults(self) -> None:
        p = self.project.project
        if p.root_dir:
            models = p.resolve_path(p.paths.models_root or "models")
            self.ed_out.setText(str(models))
        else:
            try:
                from LabGym.mypkg_resources import resource_filename

                self.ed_out.setText(str(resource_filename("LabGym", "detectors")))
            except Exception:
                pass
        # Prefill base detector from project default when it looks like a detector path.
        det = (p.defaults.detector_name or "").strip()
        if det and Path(det).is_dir() and not self.ed_base.text().strip():
            self.ed_base.setText(det)

    @staticmethod
    def _lab(text: str, tip: str) -> QLabel:
        lab = QLabel(text)
        lab.setToolTip(tip)
        return lab

    def _browse_file(self, edit: QLineEdit, filt: str) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Select file", edit.text(), filt)
        if p:
            edit.setText(p)

    def _browse_base(self) -> None:
        start = (
            self.ed_base.text().strip()
            or self.ed_out.text().strip()
            or self.project.project.root_dir
            or ""
        )
        d = browse_existing_directory(self, start, "Select base detector folder")
        if d:
            self.ed_base.setText(d)
            self._refresh_base_info()

    def _on_name_edited(self, _text: str) -> None:
        self._user_set_name = True

    def _on_iter_changed(self, _v: int) -> None:
        # Only mark user-set after initial continue toggle has applied defaults.
        if self.sender() is self.spin_iter and self.spin_iter.hasFocus():
            self._user_set_iter = True

    def _on_lr_changed(self, _v: float) -> None:
        if self.sender() is self.spin_lr and self.spin_lr.hasFocus():
            self._user_set_lr = True

    def _on_continue_toggled(self, on: bool) -> None:
        self.ed_base.setEnabled(on)
        self.btn_base.setEnabled(on)
        if on:
            if not self._user_set_iter:
                self.spin_iter.blockSignals(True)
                self.spin_iter.setValue(CONTINUE_DEFAULT_ITERATIONS)
                self.spin_iter.blockSignals(False)
            if not self._user_set_lr:
                self.spin_lr.blockSignals(True)
                self.spin_lr.setValue(CONTINUE_BASE_LR)
                self.spin_lr.blockSignals(False)
            self._refresh_base_info()
        else:
            if not self._user_set_iter:
                self.spin_iter.blockSignals(True)
                self.spin_iter.setValue(DEFAULT_ITERATIONS)
                self.spin_iter.blockSignals(False)
            if not self._user_set_lr:
                self.spin_lr.blockSignals(True)
                self.spin_lr.setValue(DEFAULT_BASE_LR)
                self.spin_lr.blockSignals(False)
            self.lbl_base_info.setText("")

    def _refresh_base_info(self) -> None:
        if not self.chk_continue.isChecked():
            self.lbl_base_info.setText("")
            return
        base = self.ed_base.text().strip()
        ann = self.ed_ann.text().strip()
        if not base:
            self.lbl_base_info.setText("Select a base detector folder.")
            return
        try:
            if ann and Path(ann).is_file():
                plan = plan_continue_training(base, ann, base_lr=float(self.spin_lr.value()))
            else:
                # Validate detector only; full class check when annotation is set.
                from LabGym.detection.batch_detect import (
                    load_detector_animal_kinds,
                    validate_detector_folder,
                )

                validate_detector_folder(base, require_weights=True)
                names = load_detector_animal_kinds(base)
                plan = None
                self.lbl_base_info.setText(
                    f"Base OK — classes: {', '.join(names)}. "
                    "Select annotation JSON to verify class match."
                )
                if not self._user_set_name and names:
                    self.ed_name.setText(suggest_continued_detector_name(base))
                return
        except Exception as exc:
            brief = str(exc).splitlines()[0]
            self.lbl_base_info.setText(f"Base detector error: {brief}")
            self.lbl_base_info.setToolTip(str(exc))
            return

        self.lbl_base_info.setToolTip("")
        info = (
            f"Base OK — continue with classes: {', '.join(plan.animal_names)}. "
            f"Weights: {Path(plan.weights_path).name}."
        )
        if plan.inference_size:
            info += f" Suggested inference size: {plan.inference_size}."
            # Only auto-apply size if still at a common default.
            if self.spin_size.value() in (480, plan.inference_size):
                self.spin_size.setValue(int(plan.inference_size))
        self.lbl_base_info.setText(info)
        if not self._user_set_name:
            self.ed_name.setText(suggest_continued_detector_name(base))

    def _run(self) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "Busy", "Training already running.")
            return
        images = self.ed_images.text().strip()
        ann = self.ed_ann.text().strip()
        parent = self.ed_out.text().strip()
        name = self.ed_name.text().strip()
        if not images or not Path(images).is_dir():
            QMessageBox.warning(self, "Train", "Select training images folder.")
            return
        if not ann or not Path(ann).is_file():
            QMessageBox.warning(self, "Train", "Select annotation JSON.")
            return
        if not parent or not name:
            QMessageBox.warning(self, "Train", "Set output parent folder and name.")
            return
        out = str(Path(parent) / name)
        if Path(out).exists():
            QMessageBox.warning(self, "Train", f"Already exists:\n{out}")
            return

        init_from = None
        base_lr = float(self.spin_lr.value())
        if self.chk_continue.isChecked():
            init_from = self.ed_base.text().strip()
            if not init_from:
                QMessageBox.warning(
                    self,
                    "Continue training",
                    "Select the base detector folder to continue from.",
                )
                return
            try:
                plan = plan_continue_training(init_from, ann, base_lr=base_lr)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Continue training", str(exc))
                return
            init_from = plan.base_detector
            base_lr = plan.base_lr

        size = int(self.spin_size.value())
        if size % 32 != 0:
            QMessageBox.warning(
                self,
                "Train",
                "Inference frame size must be divisible by 32 "
                f"(got {size}).",
            )
            return

        max_iter = int(self.spin_iter.value())
        kwargs = dict(
            path_to_annotation=ann,
            path_to_trainingimages=images,
            path_to_detector=out,
            iteration_num=max_iter,
            inference_size=size,
            init_from_detector=init_from,
            base_lr=base_lr,
        )
        self.btn.setEnabled(False)
        mode = "continue" if init_from else "from COCO"
        summary = (
            f"Training ({mode}) → {out}\n"
            f"  iterations={max_iter}  LR={base_lr}  "
            f"size={size}"
            + (f"\n  init_from={init_from}" if init_from else "")
        )
        self.log.append(summary)

        dlg = self._ensure_progress_dialog()
        dlg.begin_job(max_iter=max_iter)
        dlg.append_log(summary)
        dlg.append_log("Live total_loss curve updates about every 20 iterations.")

        self._thread = QThread(self)
        worker = _TrainWorker(kwargs)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(self._on_status)
        worker.progress_train.connect(dlg.on_train_progress)
        worker.finished.connect(self._on_done)
        worker.error.connect(self._on_err)
        worker.finished.connect(self._thread.quit)
        worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._worker = worker
        self._thread.start()

    def _ensure_progress_dialog(self) -> TrainDetectorProgressDialog:
        if self._progress_dlg is None:
            self._progress_dlg = TrainDetectorProgressDialog(self)
        return self._progress_dlg

    def _on_status(self, msg: str) -> None:
        self.log.append(msg)
        if self._progress_dlg is not None:
            self._progress_dlg.on_status(msg)

    def _cleanup(self) -> None:
        self._thread = None
        self.btn.setEnabled(True)

    def _on_done(self, path: str) -> None:
        self.log.append(f"Done: {path}")
        if self._progress_dlg is not None:
            self._progress_dlg.append_log(f"Done: {path}")
            self._progress_dlg.mark_finished()
        self.project.project.defaults.detector_name = path
        self.project.mark_dirty()
        QMessageBox.information(self, "Train detector", f"Trained detector saved:\n{path}")

    def _on_err(self, msg: str) -> None:
        self.log.append(f"ERROR: {msg}")
        if self._progress_dlg is not None:
            self._progress_dlg.append_log(f"ERROR: {msg}")
            self._progress_dlg.mark_finished(failed=True)
        QMessageBox.critical(self, "Train failed", msg)
