"""Results → Mine results (statistical comparison of LabGym summary folders)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
from LabGym.gui_pyside.widgets.path_browse import (
    browse_existing_directory,
    path_edit_row,
    set_line_edit_directory,
)


def _read_summary_folder(folder: str) -> Dict[str, Any]:
    """Load *_summary.xlsx files in a group folder (same logic as classic LabGym)."""
    import pandas as pd

    folder = folder.replace("\\", "/")
    filelist: Dict[str, str] = {}
    df: Dict[str, Any] = {}
    for name in os.listdir(folder):
        lower = name.lower()
        if lower.endswith("_summary.xlsx") or lower.endswith("_summary.xls"):
            behavior_name = name.split("_")[-2]
            filelist[behavior_name] = os.path.join(folder, name)
    if not filelist:
        return {}
    for behavior_name, path in filelist.items():
        dataset = pd.read_excel(path)
        if "ID/parameter" in dataset.columns:
            dataset = dataset.drop(columns=["ID/parameter"])
        df[behavior_name] = dataset
    return df


def _read_all_group_folders(file_path: str):
    data = []
    filenames = []
    for name in os.listdir(file_path):
        subfolder = os.path.join(file_path, name)
        if not os.path.isdir(subfolder):
            continue
        folder_data = _read_summary_folder(subfolder)
        if folder_data:
            data.append(folder_data)
            filenames.append(os.path.basename(subfolder))
    return data, filenames


class _MineWorker(QObject):
    finished = Signal(str)
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        file_path: str,
        result_path: str,
        paired: bool,
        control_name: Optional[str],
        pval: float,
    ):
        super().__init__()
        self.file_path = file_path
        self.result_path = result_path
        self.paired = paired
        self.control_name = control_name
        self.pval = pval

    def run(self) -> None:
        try:
            from LabGym.minedata import data_mining

            self.progress.emit("Reading summary folders…")
            dataset, file_names = _read_all_group_folders(self.file_path)
            if not dataset:
                raise RuntimeError(
                    "No group folders with *_summary.xlsx files found under the input path."
                )

            control = None
            if self.control_name:
                if self.control_name not in file_names:
                    raise RuntimeError(f"Control group not found: {self.control_name}")
                control_path = os.path.join(self.file_path, self.control_name)
                control = _read_summary_folder(control_path)
                del_idx = file_names.index(self.control_name)
                dataset.pop(del_idx)
                file_names.insert(0, file_names.pop(del_idx))

            Path(self.result_path).mkdir(parents=True, exist_ok=True)
            self.progress.emit("Running statistical analysis…")
            dm = data_mining(
                dataset,
                control,
                self.paired,
                self.result_path,
                self.pval,
                file_names,
            )
            dm.statistical_analysis()
            out = os.path.join(self.result_path, "data_mining_results.xlsx")
            self.finished.emit(out)
        except Exception as exc:
            self.error.emit(str(exc))


class MineResultsTab(QWidget):
    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Compare LabGym analysis batches statistically. Put each batch’s "
                "analysis output folder (with <code>*_summary.xlsx</code> files) as a "
                "subfolder of the input directory."
            )
        )

        form = QFormLayout()
        self.ed_in = QLineEdit()
        self.ed_in.setToolTip(
            "Parent folder containing one subfolder per experimental group/batch."
        )
        b_in = QPushButton("Browse…")
        b_in.clicked.connect(self._browse_in)
        form.addRow("Groups parent folder:", path_edit_row(self.ed_in, b_in))

        self.chk_paired = QCheckBox("Paired data")
        self.chk_paired.setToolTip(
            "Use paired statistical tests when the same subjects appear across groups."
        )
        form.addRow("", self.chk_paired)

        self.cmb_control = QComboBox()
        self.cmb_control.setToolTip(
            "Optional control group for post-hoc comparison. Leave empty to compare all pairs."
        )
        self.cmb_control.addItem("(none)", None)
        form.addRow("Control group:", self.cmb_control)

        self.ed_out = QLineEdit()
        self.ed_out.setToolTip("Folder for data_mining_results.xlsx.")
        b_out = QPushButton("Browse…")
        b_out.clicked.connect(self._browse_out)
        form.addRow("Output folder:", path_edit_row(self.ed_out, b_out))

        self.spin_p = QDoubleSpinBox()
        self.spin_p.setRange(1e-6, 1.0)
        self.spin_p.setDecimals(4)
        self.spin_p.setSingleStep(0.01)
        self.spin_p.setValue(0.05)
        self.spin_p.setToolTip("Significance threshold for tests.")
        form.addRow("p-value threshold:", self.spin_p)

        layout.addLayout(form)

        row = QHBoxLayout()
        self.btn_run = QPushButton("Start mining")
        self.btn_run.clicked.connect(self._run)
        row.addWidget(self.btn_run)
        row.addStretch(1)
        layout.addLayout(row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        layout.addWidget(self.log)
        layout.addStretch(1)


    def _browse_in(self) -> None:
        d = browse_existing_directory(
            self, self.ed_in.text(), "Select parent folder of analysis groups"
        )
        if not d:
            return
        self.ed_in.setText(d)
        self._refresh_controls(d)

    def _refresh_controls(self, root: str) -> None:
        self.cmb_control.clear()
        self.cmb_control.addItem("(none)", None)
        try:
            names = sorted(
                n
                for n in os.listdir(root)
                if os.path.isdir(os.path.join(root, n))
            )
        except OSError:
            names = []
        for n in names:
            self.cmb_control.addItem(n, n)

    def _browse_out(self) -> None:
        set_line_edit_directory(
            self, self.ed_out, caption="Select folder for mining results"
        )

    def _run(self) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "Busy", "Mining is already running.")
            return
        file_path = self.ed_in.text().strip()
        result_path = self.ed_out.text().strip()
        if not file_path or not result_path:
            QMessageBox.warning(self, "Missing paths", "Select input and output folders.")
            return

        control = self.cmb_control.currentData()
        worker = _MineWorker(
            file_path=file_path,
            result_path=result_path,
            paired=self.chk_paired.isChecked(),
            control_name=control,
            pval=float(self.spin_p.value()),
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
        self.log.append("Starting data mining…")
        thread.start()

    def _clear_thread(self) -> None:
        self._thread = None
        self.btn_run.setEnabled(True)

    def _on_done(self, out: str) -> None:
        self.log.append(f"Finished: {out}")
        QMessageBox.information(self, "Mining complete", f"Results written to:\n{out}")

    def _on_err(self, err: str) -> None:
        self.log.append(f"Error: {err}")
        QMessageBox.critical(self, "Mining failed", err)
