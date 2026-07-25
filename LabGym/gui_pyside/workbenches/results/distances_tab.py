"""Results → Calculate distances from LabGym analysis folders."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.project.controller import ProjectController


def _discover_behaviors(analysis_root: str) -> List[str]:
    names: List[str] = []
    for name in os.listdir(analysis_root):
        video_folder = os.path.join(analysis_root, name)
        if not os.path.isdir(video_folder):
            continue
        for behavior in os.listdir(video_folder):
            if os.path.isdir(os.path.join(video_folder, behavior)):
                if behavior not in names:
                    names.append(behavior)
    names.sort()
    return names


class _DistanceWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        analysis_root: str,
        out_path: str,
        behaviors: List[str],
    ):
        super().__init__()
        self.analysis_root = analysis_root
        self.out_path = out_path
        self.behaviors = behaviors

    def run(self) -> None:
        try:
            import pandas as pd
            from LabGym.tools import calculate_distances

            Path(self.out_path).mkdir(parents=True, exist_ok=True)
            for filename in os.listdir(self.analysis_root):
                filefolder = os.path.join(self.analysis_root, filename)
                if not os.path.isdir(filefolder):
                    continue
                self.progress.emit(f"Calculating distances for {filename}…")
                calculate_distances(
                    filefolder, filename, self.behaviors, self.out_path
                )

            all_data = []
            names = []
            for file in os.listdir(self.out_path):
                lower = file.lower()
                if lower.endswith("_distance_calculation.xlsx") or lower.endswith(
                    "_distance_calculation.xls"
                ):
                    path = os.path.join(self.out_path, file)
                    all_data.append(pd.read_excel(path))
                    names.append(file.split("_distance_calculation")[0])

            summary = os.path.join(self.out_path, "all_summary.xlsx")
            if all_data:
                combined = pd.concat(
                    all_data, keys=names, names=["File name", "ID/parameter"]
                )
                combined.drop(combined.columns[0], axis=1, inplace=True)
                combined.to_excel(summary, float_format="%.2f")
                self.finished.emit(summary)
            else:
                self.finished.emit(self.out_path)
        except Exception as exc:
            self.error.emit(str(exc))


class CalculateDistancesTab(QWidget):
    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "From LabGym analysis results, calculate (1) shortest path among first "
                "occurrences of selected behaviors and (2) total route travel distance."
            )
        )

        form = QFormLayout()
        self.ed_in = QLineEdit()
        self.ed_in.setToolTip(
            "Folder that stores per-video analysis subfolders (same as Process videos output)."
        )
        b_in = QPushButton("Browse…")
        b_in.clicked.connect(self._browse_in)
        form.addRow("Analysis results folder:", self._row(self.ed_in, b_in))

        self.ed_out = QLineEdit()
        self.ed_out.setToolTip("Folder for per-video distance spreadsheets and all_summary.xlsx.")
        b_out = QPushButton("Browse…")
        b_out.clicked.connect(self._browse_out)
        form.addRow("Output folder:", self._row(self.ed_out, b_out))
        layout.addLayout(form)

        layout.addWidget(QLabel("Behaviors to include (multi-select):"))
        self.list_beh = QListWidget()
        self.list_beh.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(self.list_beh)

        row = QHBoxLayout()
        btn_refresh = QPushButton("Refresh behaviors")
        btn_refresh.clicked.connect(self._refresh_behaviors)
        self.btn_run = QPushButton("Start calculating distances")
        self.btn_run.clicked.connect(self._run)
        row.addWidget(btn_refresh)
        row.addWidget(self.btn_run)
        row.addStretch(1)
        layout.addLayout(row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)

    def _row(self, line: QLineEdit, button: QPushButton) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(line, 1)
        h.addWidget(button)
        return w

    def _browse_in(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select analysis results folder")
        if d:
            self.ed_in.setText(d)
            self._refresh_behaviors()

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.ed_out.setText(d)

    def _refresh_behaviors(self) -> None:
        root = self.ed_in.text().strip()
        self.list_beh.clear()
        if not root or not os.path.isdir(root):
            return
        for b in _discover_behaviors(root):
            self.list_beh.addItem(b)
        self.list_beh.selectAll()

    def _selected_behaviors(self) -> List[str]:
        items = self.list_beh.selectedItems()
        if items:
            return [i.text() for i in items]
        return [self.list_beh.item(i).text() for i in range(self.list_beh.count())]

    def _run(self) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "Busy", "Distance calculation is already running.")
            return
        analysis_root = self.ed_in.text().strip()
        out_path = self.ed_out.text().strip()
        if not analysis_root or not out_path:
            QMessageBox.warning(self, "Missing paths", "Select input and output folders.")
            return
        behaviors = self._selected_behaviors()
        if not behaviors:
            QMessageBox.warning(
                self,
                "No behaviors",
                "No behaviors found. Refresh after selecting a valid analysis folder.",
            )
            return

        worker = _DistanceWorker(analysis_root, out_path, behaviors)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.log.append)
        worker.finished.connect(self._on_done)
        worker.error.connect(self._on_err)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._clear_thread)
        self._thread = thread
        self.btn_run.setEnabled(False)
        self.log.append("Starting distance calculation…")
        thread.start()

    def _clear_thread(self) -> None:
        self._thread = None
        self.btn_run.setEnabled(True)

    def _on_done(self, out: str) -> None:
        self.log.append(f"Finished: {out}")
        QMessageBox.information(self, "Distances complete", f"Results written to:\n{out}")

    def _on_err(self, err: str) -> None:
        self.log.append(f"Error: {err}")
        QMessageBox.critical(self, "Distance calculation failed", err)
