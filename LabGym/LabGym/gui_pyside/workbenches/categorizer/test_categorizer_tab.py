"""Categorizer → Test categorizer (PySide wrapper around test_categorizer)."""

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

    def __init__(self, groundtruth: str, model: str, result: Optional[str]):
        super().__init__()
        self.groundtruth = groundtruth
        self.model = model
        self.result = result

    def run(self) -> None:
        try:
            from LabGym.categorizer import Categorizers

            self.progress.emit("Testing categorizer…")
            CA = Categorizers()
            CA.test_categorizer(self.groundtruth, self.model, result_path=self.result)
            self.finished.emit(self.result or self.groundtruth)
        except Exception as exc:
            self.error.emit(str(exc))


class TestCategorizerTab(QWidget):
    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Test a trained categorizer against ground-truth behavior example folders "
            "(same layout as training examples)."
        ))

        form = QFormLayout()
        self.ed_gt = QLineEdit()
        b1 = QPushButton("Browse…")
        b1.clicked.connect(lambda: self._browse_dir(self.ed_gt))
        form.addRow("Ground-truth examples:", self._row(self.ed_gt, b1))

        self.ed_model = QLineEdit()
        b2 = QPushButton("Browse…")
        b2.clicked.connect(lambda: self._browse_dir(self.ed_model))
        form.addRow("Categorizer folder:", self._row(self.ed_model, b2))

        self.ed_out = QLineEdit()
        b3 = QPushButton("Browse…")
        b3.clicked.connect(lambda: self._browse_dir(self.ed_out))
        form.addRow("Results folder (optional):", self._row(self.ed_out, b3))
        layout.addLayout(form)

        if project.project.defaults.categorizer_name:
            self.ed_model.setText(project.project.defaults.categorizer_name)

        self.btn = QPushButton("Test categorizer")
        self.btn.clicked.connect(self._run)
        layout.addWidget(self.btn)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

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

    def _run(self) -> None:
        if self._thread is not None:
            return
        gt = self.ed_gt.text().strip()
        model = self.ed_model.text().strip()
        out = self.ed_out.text().strip() or None
        if not gt or not model:
            QMessageBox.warning(self, "Test", "Set ground-truth and categorizer paths.")
            return
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
        self.btn.setEnabled(False)
        self._thread = QThread(self)
        worker = _TestWorker(gt, model, out)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(lambda m: self.log.append(m))
        worker.finished.connect(self._done)
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
        QMessageBox.information(self, "Test categorizer", f"Finished.\n{path}")

    def _err(self, msg: str) -> None:
        self.log.append(f"ERROR: {msg}")
        QMessageBox.critical(self, "Test failed", msg)
