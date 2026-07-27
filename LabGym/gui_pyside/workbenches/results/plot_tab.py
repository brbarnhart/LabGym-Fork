"""Results → Behavior plot from all_events.xlsx."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.widgets.path_browse import (
    browse_existing_directory,
    path_edit_row,
    set_line_edit_directory,
)


class _PlotWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        results_folder: str,
        events_probability,
        time_points,
        names_and_colors: dict,
        behaviors: List[str],
    ):
        super().__init__()
        self.results_folder = results_folder
        self.events_probability = events_probability
        self.time_points = time_points
        self.names_and_colors = names_and_colors
        self.behaviors = behaviors

    def run(self) -> None:
        try:
            from LabGym.tools import plot_events

            self.progress.emit("Generating behavior raster plot…")
            Path(self.results_folder).mkdir(parents=True, exist_ok=True)
            plot_events(
                self.results_folder,
                self.events_probability,
                self.time_points,
                self.names_and_colors,
                self.behaviors,
            )
            out = str(Path(self.results_folder) / "behaviors_plot.png")
            self.finished.emit(out)
        except Exception as exc:
            self.error.emit(str(exc))


class PlotBehaviorsTab(QWidget):
    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None
        self.events_probability = None
        self.time_points = None
        self.names_and_colors: Dict[str, Tuple[str, str]] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Generate a LabGym behavior raster plot from an "
                "<code>all_events.xlsx</code> file produced by Process videos / analysis."
            )
        )

        form = QFormLayout()
        self.ed_file = QLineEdit()
        self.ed_file.setReadOnly(True)
        self.ed_file.setToolTip("LabGym all_events.xlsx from an analysis batch.")
        b_file = QPushButton("Browse…")
        b_file.clicked.connect(self._browse_file)
        form.addRow("all_events.xlsx:", path_edit_row(self.ed_file, b_file))

        self.ed_out = QLineEdit()
        self.ed_out.setToolTip("Folder for behaviors_plot.png and colorbar images.")
        b_out = QPushButton("Browse…")
        b_out.clicked.connect(self._browse_out)
        form.addRow("Output folder:", path_edit_row(self.ed_out, b_out))
        layout.addLayout(form)

        layout.addWidget(QLabel("Behavior colors (double-click to change):"))
        self.list_colors = QListWidget()
        self.list_colors.itemDoubleClicked.connect(self._pick_color)
        layout.addWidget(self.list_colors)

        row = QHBoxLayout()
        self.btn_run = QPushButton("Generate behavior plot")
        self.btn_run.clicked.connect(self._run)
        row.addWidget(self.btn_run)
        row.addStretch(1)
        layout.addLayout(row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)


    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select all_events.xlsx",
            "",
            "Excel (*.xlsx);;All files (*.*)",
        )
        if not path:
            return
        try:
            from LabGym.tools import parse_all_events_file

            events, times, behaviors = parse_all_events_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to read file", str(exc))
            return

        self.events_probability = events
        self.time_points = times
        self.ed_file.setText(path)
        if not self.ed_out.text().strip():
            self.ed_out.setText(str(Path(path).parent))

        hex_colors = list(mcolors.cnames.values())
        self.names_and_colors = {}
        self.list_colors.clear()
        for i, behavior in enumerate(behaviors):
            color = hex_colors[i % len(hex_colors)]
            self.names_and_colors[behavior] = ("#ffffff", color)
            item = QListWidgetItem(f"{behavior}  ({color})")
            item.setData(Qt.ItemDataRole.UserRole, behavior)
            item.setBackground(QColor(color))
            self.list_colors.addItem(item)

        self.log.append(f"Loaded {len(behaviors)} behavior(s) from {path}")

    def _browse_out(self) -> None:
        set_line_edit_directory(
            self, self.ed_out, caption="Select folder for behavior plot"
        )

    def _pick_color(self, item: QListWidgetItem) -> None:
        behavior = item.data(Qt.ItemDataRole.UserRole)
        if not behavior or behavior not in self.names_and_colors:
            return
        _, current = self.names_and_colors[behavior]
        color = QColorDialog.getColor(QColor(current), self, f"Color for {behavior}")
        if not color.isValid():
            return
        hex_code = color.name()
        self.names_and_colors[behavior] = ("#ffffff", hex_code)
        item.setText(f"{behavior}  ({hex_code})")
        item.setBackground(color)

    def _run(self) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "Busy", "Plot generation is already running.")
            return
        if (
            self.events_probability is None
            or self.time_points is None
            or not self.names_and_colors
        ):
            QMessageBox.warning(self, "Missing input", "Select an all_events.xlsx file first.")
            return
        out = self.ed_out.text().strip()
        if not out:
            QMessageBox.warning(self, "Missing output", "Select an output folder.")
            return

        behaviors = list(self.names_and_colors.keys())
        worker = _PlotWorker(
            results_folder=out,
            events_probability=self.events_probability,
            time_points=self.time_points,
            names_and_colors=dict(self.names_and_colors),
            behaviors=behaviors,
        )
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
        thread.start()

    def _clear_thread(self) -> None:
        self._thread = None
        self.btn_run.setEnabled(True)

    def _on_done(self, out: str) -> None:
        self.log.append(f"Plot saved: {out}")
        QMessageBox.information(self, "Plot complete", f"Behavior plot written to:\n{out}")

    def _on_err(self, err: str) -> None:
        self.log.append(f"Error: {err}")
        QMessageBox.critical(self, "Plot failed", err)
