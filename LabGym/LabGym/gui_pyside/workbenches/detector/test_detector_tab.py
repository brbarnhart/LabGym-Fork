"""Detector → Test detector (PySide wrapper around Detector.test)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.project.controller import ProjectController


class _TestWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, kwargs: dict):
        super().__init__()
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            self.progress.emit("Loading Detector…")
            from LabGym.detector import Detector

            dt = Detector()
            self.progress.emit("Running evaluation…")
            dt.test(**self.kwargs)
            self.finished.emit(self.kwargs["output_path"])
        except Exception as exc:
            self.error.emit(str(exc))


class TestDetectorTab(QWidget):
    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Evaluate a trained detector on COCO-format ground-truth images."
        ))

        form = QFormLayout()
        self.ed_det = QLineEdit()
        b0 = QPushButton("Browse…")
        b0.clicked.connect(lambda: self._browse_dir(self.ed_det))
        form.addRow("Detector folder:", self._row(self.ed_det, b0))

        self.ed_images = QLineEdit()
        b1 = QPushButton("Browse…")
        b1.clicked.connect(lambda: self._browse_dir(self.ed_images))
        form.addRow("Test images folder:", self._row(self.ed_images, b1))

        self.ed_ann = QLineEdit()
        b2 = QPushButton("Browse…")
        b2.clicked.connect(lambda: self._browse_file(self.ed_ann))
        form.addRow("Annotation JSON:", self._row(self.ed_ann, b2))

        self.ed_out = QLineEdit()
        b3 = QPushButton("Browse…")
        b3.clicked.connect(lambda: self._browse_dir(self.ed_out))
        form.addRow("Output folder:", self._row(self.ed_out, b3))
        layout.addLayout(form)

        self.btn = QPushButton("Test detector")
        self.btn.clicked.connect(self._run)
        layout.addWidget(self.btn)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

        if self.project.project.defaults.detector_name:
            self.ed_det.setText(self.project.project.defaults.detector_name)

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
        p, _ = QFileDialog.getOpenFileName(self, "JSON", edit.text(), "JSON (*.json)")
        if p:
            edit.setText(p)

    def _run(self) -> None:
        if self._thread is not None:
            return
        det = self.ed_det.text().strip()
        images = self.ed_images.text().strip()
        ann = self.ed_ann.text().strip()
        out = self.ed_out.text().strip()
        if not all([det, images, ann, out]):
            QMessageBox.warning(self, "Test", "Fill all paths.")
            return
        Path(out).mkdir(parents=True, exist_ok=True)
        kwargs = dict(
            path_to_annotation=ann,
            path_to_testingimages=images,
            path_to_detector=det,
            output_path=out,
        )
        self.btn.setEnabled(False)
        self._thread = QThread(self)
        worker = _TestWorker(kwargs)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(lambda m: self.log.append(m))
        worker.finished.connect(lambda p: self._done(p))
        worker.error.connect(self._err)
        worker.finished.connect(self._thread.quit)
        worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._worker = worker
        self._thread.start()

    def _cleanup(self) -> None:
        self._thread = None
        self.btn.setEnabled(True)

    def _done(self, path: str) -> None:
        self.log.append(f"Done → {path}")
        QMessageBox.information(self, "Test detector", f"Results in:\n{path}")

    def _err(self, msg: str) -> None:
        self.log.append(f"ERROR: {msg}")
        QMessageBox.critical(self, "Test failed", msg)
