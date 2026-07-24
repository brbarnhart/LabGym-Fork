"""Categorizer → Train categorizer (PySide wrapper around Categorizers)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# QFormLayout used for prepare + train groups

from LabGym.gui_pyside.project.controller import ProjectController


class _PrepWorker(QObject):
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


class _TrainWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            from LabGym.categorizer import Categorizers

            p = self.params
            Path(p["model_path"]).mkdir(parents=True, exist_ok=True)
            CA = Categorizers()
            CA.label_mode = p["label_mode"]
            CA.lambda_soft = p["lambda_soft"]
            self.progress.emit("Training categorizer (long-running)…")
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
                    out_folder=None,
                    label_mode=p["label_mode"],
                    lambda_soft=p["lambda_soft"],
                    soft_labels_path=p.get("soft_labels_path"),
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
                    out_folder=None,
                    label_mode=p["label_mode"],
                    lambda_soft=p["lambda_soft"],
                    soft_labels_path=p.get("soft_labels_path"),
                )
            self.finished.emit(p["model_path"])
        except Exception as exc:
            self.error.emit(str(exc))


class TrainCategorizerTab(QWidget):
    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Train on <b>prepared</b> ethogram-generated example folders "
            "(behavior subfolders of .avi+.jpg pairs). Dense generate-then-sort is not offered."
        ))

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
        b_s.clicked.connect(lambda: self._browse_dir(self.ed_sorted))
        prep_l.addRow(
            self._lab("Sorted examples:", "Behavior-subfolder layout."),
            self._row(self.ed_sorted, b_s),
        )
        self.ed_prepared = QLineEdit()
        self.ed_prepared.setToolTip(
            "Output folder for prepared examples (all files together, labels in names)."
        )
        b_p = QPushButton("Browse…")
        b_p.clicked.connect(lambda: self._browse_dir(self.ed_prepared))
        prep_l.addRow(
            self._lab("Prepared output:", "Empty or new folder recommended."),
            self._row(self.ed_prepared, b_p),
        )
        btn_prep = QPushButton("Prepare examples (rename_label)")
        btn_prep.setToolTip("Run LabGym Categorizers.rename_label on the folders above.")
        btn_prep.clicked.connect(self._prepare)
        prep_l.addRow(btn_prep)
        layout.addWidget(prep)

        train = QGroupBox("Train")
        form = QFormLayout(train)
        self.ed_data = QLineEdit()
        self.ed_data.setPlaceholderText("Prepared examples folder (or ethogram examples root)")
        self.ed_data.setToolTip(
            "Folder of training examples. Prefer the prepared flat folder, or a "
            "structure LabGym’s trainer accepts for your version."
        )
        b_d = QPushButton("Browse…")
        b_d.clicked.connect(lambda: self._browse_dir(self.ed_data))
        form.addRow(
            self._lab("Training data folder:", "Must contain enough labeled examples."),
            self._row(self.ed_data, b_d),
        )

        self.ed_models = QLineEdit()
        self.ed_models.setToolTip("Parent directory where the new categorizer folder is created.")
        b_m = QPushButton("Browse…")
        b_m.clicked.connect(lambda: self._browse_dir(self.ed_models))
        form.addRow(
            self._lab("Models parent folder:", "Usually project models/."),
            self._row(self.ed_models, b_m),
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
        self.spin_len.setValue(int(project.project.defaults.window_length or 15))
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
            self._row(self.ed_soft, b_soft),
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

        self.ed_report = QLineEdit()
        self.ed_report.setToolTip(
            "Optional folder for training history / metric reports. Leave empty to skip."
        )
        b_r = QPushButton("Browse…")
        b_r.clicked.connect(lambda: self._browse_dir(self.ed_report))
        form.addRow(
            self._lab("Training reports (optional):", "Export training curves if set."),
            self._row(self.ed_report, b_r),
        )

        layout.addWidget(train)

        self.btn = QPushButton("Train categorizer")
        self.btn.clicked.connect(self._train)
        layout.addWidget(self.btn)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        self._defaults()

    def _defaults(self) -> None:
        p = self.project.project
        if p.root_dir:
            self.ed_models.setText(str(p.resolve_path(p.paths.models_root or "models")))
            ex = p.resolve_path(p.paths.examples_root or "examples")
            self.ed_data.setPlaceholderText(str(ex))
        idx = self.combo_mode.findData(int(p.defaults.behavior_mode))
        if idx >= 0:
            self.combo_mode.setCurrentIndex(idx)

    @staticmethod
    def _lab(text: str, tip: str) -> QLabel:
        lab = QLabel(text)
        lab.setToolTip(tip)
        return lab

    @staticmethod
    def _row(edit, btn):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        h.addWidget(btn)
        return w

    def _browse_dir(self, edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select folder", edit.text())
        if d:
            edit.setText(d)

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
        self.log.append(f"Preparing {src} → {dst}")
        self._thread = QThread(self)
        worker = _PrepWorker(src, dst)
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
        self.log.append(f"Prepared examples in {path}")
        QMessageBox.information(self, "Prepare", f"Prepared:\n{path}")

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
        mode = int(self.combo_mode.currentData())
        channel = 3 if mode == 2 else 1
        params = dict(
            data_path=data,
            model_path=model_path,
            out_path=report,
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
        )
        self.btn.setEnabled(False)
        self.log.append(f"Training → {model_path}")
        self._thread = QThread(self)
        worker = _TrainWorker(params)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(lambda m: self.log.append(m))
        worker.finished.connect(self._train_done)
        worker.error.connect(self._err)
        worker.finished.connect(self._thread.quit)
        worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._worker = worker
        self._thread.start()

    def _cleanup(self) -> None:
        self._thread = None
        self.btn.setEnabled(True)

    def _train_done(self, path: str) -> None:
        self.log.append(f"Done: {path}")
        self.project.project.defaults.categorizer_name = path
        self.project.mark_dirty()
        QMessageBox.information(self, "Train categorizer", f"Saved:\n{path}")

    def _err(self, msg: str) -> None:
        self.log.append(f"ERROR: {msg}")
        QMessageBox.critical(self, "Failed", msg)
