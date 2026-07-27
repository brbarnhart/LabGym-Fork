"""Categorizer → Train categorizer (PySide form + workers)."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.widgets.path_browse import (
    path_edit_row,
    set_line_edit_directory,
)
from LabGym.gui_pyside.workbenches.categorizer.train_progress_dialog import (
    TrainProgressDialog,
)
from LabGym.gui_pyside.workbenches.categorizer.train_workers import (
    PrepWorker,
    TrainWorker,
    auto_aug_workers,
)


class TrainCategorizerTab(QWidget):
    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None
        self._cancel_event: Optional[threading.Event] = None
        self._progress_dlg: Optional[TrainProgressDialog] = None

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Train on <b>prepared</b> ethogram-generated example folders "
            "(behavior subfolders of .avi+.jpg pairs). Dense generate-then-sort is not offered. "
            "Progress opens in a separate window when training starts."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        prep = QGroupBox("Optional: prepare sorted folders → flat labeled examples")
        prep.setToolTip(
            "LabGym train expects all examples in one folder with behavior names "
            "embedded in filenames. Generate examples produces behavior subfolders; "
            "Prepare copies/renames them into that flat layout."
        )
        prep_l = QFormLayout(prep)
        self.ed_sorted = QLineEdit()
        self.ed_sorted.setPlaceholderText("Folder with behavior subfolders (from Generate examples)")
        self.ed_sorted.setToolTip(
            "Input: folder containing one subfolder per behavior (e.g. approach/, fight/) "
            "as produced by Generate examples."
        )
        b_s = QPushButton("Browse…")
        b_s.clicked.connect(lambda: set_line_edit_directory(self, self.ed_sorted))
        prep_l.addRow(
            self._lab("Sorted examples:", "Behavior-subfolder layout."),
            path_edit_row(self.ed_sorted, b_s),
        )
        self.ed_prepared = QLineEdit()
        self.ed_prepared.setToolTip(
            "Output folder for prepared examples (all files together, labels in names)."
        )
        b_p = QPushButton("Browse…")
        b_p.clicked.connect(lambda: set_line_edit_directory(self, self.ed_prepared))
        prep_l.addRow(
            self._lab("Prepared output:", "Empty or new folder recommended."),
            path_edit_row(self.ed_prepared, b_p),
        )
        btn_prep = QPushButton("Prepare examples (rename_label)")
        btn_prep.setToolTip("Run LabGym Categorizers.rename_label on the folders above.")
        btn_prep.clicked.connect(self._prepare)
        prep_l.addRow(btn_prep)
        layout.addWidget(prep)

        train = QGroupBox("Train")
        form = QFormLayout(train)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.ed_data = QLineEdit()
        self.ed_data.setPlaceholderText("Prepared examples folder (or ethogram examples root)")
        self.ed_data.setToolTip(
            "Folder of training examples. Prefer the prepared flat folder, or a "
            "structure LabGym’s trainer accepts for your version."
        )
        b_d = QPushButton("Browse…")
        b_d.clicked.connect(lambda: set_line_edit_directory(self, self.ed_data))
        form.addRow(
            self._lab("Training data folder:", "Must contain enough labeled examples."),
            path_edit_row(self.ed_data, b_d),
        )

        self.ed_models = QLineEdit()
        self.ed_models.setToolTip("Parent directory where the new categorizer folder is created.")
        b_m = QPushButton("Browse…")
        b_m.clicked.connect(lambda: set_line_edit_directory(self, self.ed_models))
        form.addRow(
            self._lab("Models parent folder:", "Usually project models/."),
            path_edit_row(self.ed_models, b_m),
        )

        self.ed_name = QLineEdit("New_categorizer")
        self.ed_name.setToolTip("Name of the new model subfolder under the parent.")
        form.addRow(self._lab("Categorizer name:", "Becomes <parent>/<name>/."), self.ed_name)

        self.combo_mode = QComboBox()
        for code, lab in (
            (0, "0 — Non-interactive"),
            (1, "1 — Interactive basic"),
            (2, "2 — Interactive advanced"),
        ):
            self.combo_mode.addItem(lab, code)
        tip_mode = (
            "Must match how examples were generated and how you will analyze:\n"
            "0 = per-individual; 1 = group interaction; 2 = individual + partners/costars."
        )
        self.combo_mode.setToolTip(tip_mode)
        form.addRow(self._lab("Behavior mode:", tip_mode), self.combo_mode)

        self.chk_anim = QCheckBox("Include Animation Analyzer (combnet)")
        self.chk_anim.setChecked(True)
        self.chk_anim.setToolTip(
            "If checked, train Animation Analyzer + Pattern Recognizer (slower, "
            "usually slightly better). Unchecked = Pattern Recognizer only (faster)."
        )
        form.addRow(self.chk_anim)

        self.spin_len = QSpinBox()
        self.spin_len.setRange(1, 200)
        self.spin_len.setValue(15)
        tip_len = (
            "Number of frames per example (time_step). Must match the window length "
            "used when generating examples (see Generate examples)."
        )
        self.spin_len.setToolTip(tip_len)
        form.addRow(self._lab("Time steps / length:", tip_len), self.spin_len)

        self.spin_dim = QSpinBox()
        self.spin_dim.setRange(8, 256)
        self.spin_dim.setValue(32)
        tip_dim = (
            "Input spatial size (pixels) for Pattern Recognizer images. Even integer "
            "> 8. Larger = wider network and more detail, needs more data/GPU."
        )
        self.spin_dim.setToolTip(tip_dim)
        form.addRow(self._lab("Pattern dim:", tip_dim), self.spin_dim)

        self.spin_tdim = QSpinBox()
        self.spin_tdim.setRange(8, 256)
        self.spin_tdim.setValue(32)
        tip_tdim = (
            "Input spatial size for Animation Analyzer frames (only if combnet is on)."
        )
        self.spin_tdim.setToolTip(tip_tdim)
        form.addRow(self._lab("Animation dim:", tip_tdim), self.spin_tdim)

        self.spin_level = QSpinBox()
        self.spin_level.setRange(1, 5)
        self.spin_level.setValue(2)
        tip_lvl = (
            "Network complexity level (LabGym LV1–higher). Higher = deeper/wider "
            "and slower; 2 is the common default."
        )
        self.spin_level.setToolTip(tip_lvl)
        form.addRow(self._lab("Network level:", tip_lvl), self.spin_level)

        self.combo_label = QComboBox()
        for m in ("hard_soft_aux", "hard_only", "soft_primary"):
            self.combo_label.addItem(m, m)
        tip_lab = (
            "• hard_only — folder hard labels only.\n"
            "• hard_soft_aux (recommended) — hard labels + soft targets from "
            "soft_labels.csv as an auxiliary loss.\n"
            "• soft_primary — train mainly on soft targets."
        )
        self.combo_label.setToolTip(tip_lab)
        form.addRow(self._lab("Label mode:", tip_lab), self.combo_label)

        self.spin_lambda = QDoubleSpinBox()
        self.spin_lambda.setRange(0.0, 5.0)
        self.spin_lambda.setValue(0.4)
        tip_lam = (
            "Weight of the soft-label loss when using hard_soft_aux or soft_primary. "
            "Typical ~0.3–0.5. Ignored for hard_only."
        )
        self.spin_lambda.setToolTip(tip_lam)
        form.addRow(self._lab("lambda_soft:", tip_lam), self.spin_lambda)

        self.ed_soft = QLineEdit()
        self.ed_soft.setPlaceholderText("Optional soft_labels.csv (else auto-detect)")
        self.ed_soft.setToolTip(
            "Path to soft_labels.csv from Generate examples. Leave empty to "
            "auto-detect inside the training folder."
        )
        b_soft = QPushButton("Browse…")
        b_soft.clicked.connect(lambda: self._browse_file(self.ed_soft))
        form.addRow(
            self._lab("Soft labels CSV:", "Optional; used by soft training modes."),
            path_edit_row(self.ed_soft, b_soft),
        )

        self.chk_bg = QCheckBox("Background-free")
        self.chk_bg.setChecked(True)
        self.chk_bg.setToolTip(
            "Match background-free blob examples (no scene background in animals)."
        )
        form.addRow(self.chk_bg)
        self.chk_black = QCheckBox("Black background")
        self.chk_black.setChecked(True)
        self.chk_black.setToolTip("Use black fill behind extracted animal blobs.")
        form.addRow(self.chk_black)
        self.chk_body = QCheckBox("Include body parts")
        self.chk_body.setToolTip(
            "Include body-part detail in pattern images when examples support it."
        )
        form.addRow(self.chk_body)

        self.ed_export = QLineEdit()
        self.ed_export.setPlaceholderText("Default: <model_path>/augmented_data")
        tip_export = (
            "Folder for augmented train/validation examples written before on-the-fly "
            "training. Default is <models parent>/<categorizer name>/augmented_data. "
            "Export is always used (lower RAM). Multi-worker parallelization applies "
            "when workers > 1 and there are ≥16 source examples. Use “Reuse existing "
            "export” below to skip re-augmenting when train/ and validation/ already "
            "contain .jpg examples."
        )
        self.ed_export.setToolTip(tip_export)
        b_ex = QPushButton("Browse…")
        b_ex.clicked.connect(lambda: set_line_edit_directory(self, self.ed_export))
        form.addRow(
            self._lab("Augmented export folder:", tip_export),
            path_edit_row(self.ed_export, b_ex),
        )

        self.chk_skip_aug = QCheckBox("Reuse existing augmented export (skip re-augment)")
        self.chk_skip_aug.setChecked(False)
        tip_skip = (
            "If the export folder already has train/ and validation/ subfolders with "
            "at least one .jpg each, skip augmentation and train onfly from that data. "
            "If the export is incomplete, augmentation runs anyway. Uncheck to force "
            "a full re-augment (overwrites files with the same names)."
        )
        self.chk_skip_aug.setToolTip(tip_skip)
        form.addRow(self.chk_skip_aug)

        self.chk_auto_workers = QCheckBox("Auto workers")
        self.chk_auto_workers.setChecked(True)
        tip_auto = (
            "When checked, use a conservative default: min(8, CPU−1). "
            "Uncheck to set the count manually. Ignored when reusing an existing export."
        )
        self.chk_auto_workers.setToolTip(tip_auto)
        self.spin_workers = QSpinBox()
        self.spin_workers.setRange(1, max(32, (os.cpu_count() or 8) * 2))
        self.spin_workers.setValue(auto_aug_workers())
        tip_workers = (
            "Number of CPU processes for export augmentation. More workers = faster "
            "aug but more RAM/CPU. Set to 1 if unstable or low memory. Does not "
            "speed up GPU training epochs. Jobs with fewer than 16 source examples "
            "stay sequential. Cancel is available in the progress window."
        )
        self.spin_workers.setToolTip(tip_workers)
        self.spin_workers.setEnabled(False)
        self.chk_auto_workers.toggled.connect(self._on_auto_workers)
        workers_row = QWidget()
        wh = QHBoxLayout(workers_row)
        wh.setContentsMargins(0, 0, 0, 0)
        wh.addWidget(self.chk_auto_workers)
        wh.addWidget(self.spin_workers, 1)
        form.addRow(self._lab("Augmentation workers:", tip_workers), workers_row)

        self.ed_report = QLineEdit()
        self.ed_report.setToolTip(
            "Optional folder for training history / metric reports. Leave empty to skip."
        )
        b_r = QPushButton("Browse…")
        b_r.clicked.connect(lambda: set_line_edit_directory(self, self.ed_report))
        form.addRow(
            self._lab("Training reports (optional):", "Export training curves if set."),
            path_edit_row(self.ed_report, b_r),
        )

        layout.addWidget(train)

        self.btn = QPushButton("Train categorizer")
        self.btn.setToolTip(
            "Starts training. Augmentation and epoch progress open in a separate window."
        )
        self.btn.clicked.connect(self._train)
        layout.addWidget(self.btn)

        self.lbl_status = QLabel("Idle")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #aaa;")
        layout.addWidget(self.lbl_status)
        layout.addStretch(1)

        self.project.project_replaced.connect(self._apply_project_defaults)
        self._apply_project_defaults()

    def _on_auto_workers(self, checked: bool) -> None:
        self.spin_workers.setEnabled(not checked)
        if checked:
            self.spin_workers.setValue(auto_aug_workers())

    def _apply_project_defaults(self) -> None:
        """Pull Edit Project defaults (mode, window length, paths) into this form."""
        p = self.project.project
        d = p.defaults
        if p.root_dir:
            self.ed_models.setText(str(p.resolve_path(p.paths.models_root or "models")))
            ex = p.resolve_path(p.paths.examples_root or "examples")
            self.ed_data.setPlaceholderText(str(ex))
        idx = self.combo_mode.findData(int(d.behavior_mode))
        if idx >= 0:
            self.combo_mode.setCurrentIndex(idx)
        self.spin_len.setValue(max(1, int(d.window_length or 15)))
        self.spin_workers.setValue(auto_aug_workers())

    @staticmethod
    def _lab(text: str, tip: str) -> QLabel:
        lab = QLabel(text)
        lab.setToolTip(tip)
        return lab

    def _browse_file(self, edit: QLineEdit) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "CSV", edit.text(), "CSV (*.csv)")
        if p:
            edit.setText(p)

    def _prepare(self) -> None:
        if self._thread is not None:
            return
        src = self.ed_sorted.text().strip()
        dst = self.ed_prepared.text().strip()
        if not src or not dst:
            QMessageBox.warning(self, "Prepare", "Set sorted source and prepared output folders.")
            return
        Path(dst).mkdir(parents=True, exist_ok=True)
        self.lbl_status.setText(f"Preparing {src} → {dst}…")
        self._thread = QThread(self)
        worker = PrepWorker(src, dst)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.finished.connect(self._prep_done)
        worker.error.connect(self._err)
        worker.finished.connect(self._thread.quit)
        worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._worker = worker
        self._thread.start()

    def _prep_done(self, path: str) -> None:
        self.ed_data.setText(path)
        self.lbl_status.setText(f"Prepared examples in {path}")
        QMessageBox.information(self, "Prepare", f"Prepared:\n{path}")

    def _resolved_workers(self) -> int:
        if self.chk_auto_workers.isChecked():
            return auto_aug_workers()
        return int(self.spin_workers.value())

    def _ensure_progress_dialog(self) -> TrainProgressDialog:
        if self._progress_dlg is None:
            self._progress_dlg = TrainProgressDialog(self)
            self._progress_dlg.cancel_requested.connect(self._cancel)
        return self._progress_dlg

    def _train(self) -> None:
        if self._thread is not None:
            QMessageBox.information(self, "Busy", "A job is already running.")
            return
        data = self.ed_data.text().strip()
        parent = self.ed_models.text().strip()
        name = self.ed_name.text().strip()
        if not data or not Path(data).is_dir():
            QMessageBox.warning(self, "Train", "Select training data folder.")
            return
        if not parent or not name:
            QMessageBox.warning(self, "Train", "Set models folder and categorizer name.")
            return
        model_path = str(Path(parent) / name)
        if Path(model_path).exists() and any(Path(model_path).iterdir()):
            r = QMessageBox.question(
                self,
                "Exists",
                f"{model_path} already exists. Continue and overwrite/use it?",
            )
            if r != QMessageBox.StandardButton.Yes:
                return
        soft = self.ed_soft.text().strip() or None
        report = self.ed_report.text().strip() or None
        export = self.ed_export.text().strip() or str(Path(model_path) / "augmented_data")
        mode = int(self.combo_mode.currentData())
        channel = 3 if mode == 2 else 1
        n_workers = self._resolved_workers()
        params = dict(
            data_path=data,
            model_path=model_path,
            out_path=report,
            out_folder=export,
            animation_analyzer=self.chk_anim.isChecked(),
            behavior_mode=mode,
            length=int(self.spin_len.value()),
            dim_conv=int(self.spin_dim.value()),
            dim_tconv=int(self.spin_tdim.value()),
            level_conv=int(self.spin_level.value()),
            level_tconv=int(self.spin_level.value()),
            channel=channel,
            aug_methods=[
                "random rotation",
                "horizontal flipping",
                "vertical flipping",
                "random brightening",
                "random dimming",
            ],
            augvalid=True,
            include_bodyparts=self.chk_body.isChecked(),
            std=0,
            background_free=self.chk_bg.isChecked(),
            black_background=self.chk_black.isChecked(),
            social_distance=0,
            color_costar=False,
            label_mode=str(self.combo_label.currentData()),
            lambda_soft=float(self.spin_lambda.value()),
            soft_labels_path=soft,
            num_workers=n_workers,
            skip_augment=self.chk_skip_aug.isChecked(),
        )
        self.btn.setEnabled(False)
        self.lbl_status.setText("Training in progress (see progress window)…")

        dlg = self._ensure_progress_dialog()
        dlg.begin_job()
        dlg.append_log(f"Training → {model_path}")
        dlg.append_log(f"Augmented export → {export}")
        dlg.append_log(f"Augmentation workers → {n_workers}")
        if self.chk_skip_aug.isChecked():
            dlg.append_log("Reuse existing export: ON (skip re-augment if complete)")

        self._cancel_event = threading.Event()
        self._thread = QThread(self)
        worker = TrainWorker(params, self._cancel_event)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(dlg.on_status)
        worker.progress_aug.connect(dlg.on_aug_progress)
        worker.progress_train.connect(dlg.on_train_progress)
        worker.finished.connect(self._train_done)
        worker.cancelled.connect(self._train_cancelled)
        worker.error.connect(self._err)
        worker.finished.connect(self._thread.quit)
        worker.cancelled.connect(self._thread.quit)
        worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._worker = worker
        self._thread.start()

    def _cancel(self) -> None:
        if self._cancel_event is None:
            return
        self._cancel_event.set()
        self.lbl_status.setText("Cancel requested…")

    def _cleanup(self) -> None:
        self._thread = None
        self._cancel_event = None
        self.btn.setEnabled(True)

    def _train_done(self, path: str) -> None:
        if self._progress_dlg is not None:
            self._progress_dlg.append_log(f"Done: {path}")
            self._progress_dlg.set_phase("Done")
            self._progress_dlg.mark_finished()
        self.lbl_status.setText(f"Done: {path}")
        self.project.project.defaults.categorizer_name = path
        self.project.mark_dirty()
        QMessageBox.information(self, "Train categorizer", f"Saved:\n{path}")

    def _train_cancelled(self, msg: str) -> None:
        if self._progress_dlg is not None:
            self._progress_dlg.append_log(f"Cancelled: {msg}")
            self._progress_dlg.set_phase("Cancelled")
            self._progress_dlg.mark_finished(cancelled=True)
        self.lbl_status.setText(f"Cancelled: {msg}")
        QMessageBox.information(
            self,
            "Cancelled",
            f"{msg}\n\nPartial augmented data may remain in the export folder.",
        )

    def _err(self, msg: str) -> None:
        if self._progress_dlg is not None and self._progress_dlg.isVisible():
            self._progress_dlg.append_log(f"ERROR: {msg}")
            self._progress_dlg.set_phase("Failed")
            self._progress_dlg.mark_finished(failed=True)
        self.lbl_status.setText(f"ERROR: {msg}")
        QMessageBox.critical(self, "Failed", msg)
