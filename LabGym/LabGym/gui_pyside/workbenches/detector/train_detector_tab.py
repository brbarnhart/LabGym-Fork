"""Detector → Train detector (PySide wrapper around Detector.train)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
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

from LabGym.gui_pyside.project.controller import ProjectController


class _TrainWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, kwargs: dict):
        super().__init__()
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            self.progress.emit("Loading Detectron2 / Detector…")
            from LabGym.detector import Detector

            dt = Detector()
            self.progress.emit("Training (this can take a long time)…")
            dt.train(**self.kwargs)
            self.finished.emit(self.kwargs["path_to_detector"])
        except Exception as exc:
            self.error.emit(str(exc))


class TrainDetectorTab(QWidget):
    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Train a LabGym Mask R-CNN detector from COCO instance-segmentation "
            "annotations. Runs <code>Detector.train</code> off the UI thread."
        ))

        form = QFormLayout()
        self.ed_images = QLineEdit()
        b1 = QPushButton("Browse…")
        b1.clicked.connect(lambda: self._browse_dir(self.ed_images))
        form.addRow("Training images folder:", self._row(self.ed_images, b1))

        self.ed_ann = QLineEdit()
        b2 = QPushButton("Browse…")
        b2.clicked.connect(lambda: self._browse_file(self.ed_ann, "JSON (*.json)"))
        form.addRow("COCO annotation JSON:", self._row(self.ed_ann, b2))

        self.ed_out = QLineEdit()
        b3 = QPushButton("Browse…")
        b3.clicked.connect(lambda: self._browse_dir(self.ed_out))
        form.addRow("Parent folder for detector:", self._row(self.ed_out, b3))

        self.ed_name = QLineEdit("New_detector")
        form.addRow("Detector name:", self.ed_name)

        self.spin_size = QSpinBox()
        self.spin_size.setRange(32, 2048)
        self.spin_size.setSingleStep(32)
        self.spin_size.setValue(480)
        form.addRow("Inference frame size:", self.spin_size)

        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 1_000_000)
        self.spin_iter.setValue(1000)
        form.addRow("Training iterations:", self.spin_iter)
        layout.addLayout(form)

        self.btn = QPushButton("Train detector")
        self.btn.clicked.connect(self._run)
        layout.addWidget(self.btn)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        self._defaults()

    def _defaults(self) -> None:
        p = self.project.project
        if p.root_dir:
            models = p.resolve_path(p.paths.models_root or "models")
            self.ed_out.setText(str(models))
            return
        try:
            from LabGym.mypkg_resources import resource_filename

            self.ed_out.setText(str(resource_filename("LabGym", "detectors")))
        except Exception:
            pass

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

    def _browse_file(self, edit: QLineEdit, filt: str) -> None:
        p, _ = QFileDialog.getOpenFileName(self, "Select file", edit.text(), filt)
        if p:
            edit.setText(p)

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
        kwargs = dict(
            path_to_annotation=ann,
            path_to_trainingimages=images,
            path_to_detector=out,
            iteration_num=int(self.spin_iter.value()),
            inference_size=int(self.spin_size.value()),
        )
        self.btn.setEnabled(False)
        self.log.append(f"Training → {out}")
        self._thread = QThread(self)
        worker = _TrainWorker(kwargs)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(lambda m: self.log.append(m))
        worker.finished.connect(self._on_done)
        worker.error.connect(self._on_err)
        worker.finished.connect(self._thread.quit)
        worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._worker = worker
        self._thread.start()

    def _cleanup(self) -> None:
        self._thread = None
        self.btn.setEnabled(True)

    def _on_done(self, path: str) -> None:
        self.log.append(f"Done: {path}")
        self.project.project.defaults.detector_name = path
        self.project.mark_dirty()
        QMessageBox.information(self, "Train detector", f"Trained detector saved:\n{path}")

    def _on_err(self, msg: str) -> None:
        self.log.append(f"ERROR: {msg}")
        QMessageBox.critical(self, "Train failed", msg)
