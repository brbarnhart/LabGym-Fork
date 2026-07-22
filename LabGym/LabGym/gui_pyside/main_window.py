"""Lightweight PySide6 workflow shell for the LabGym merge roadmap.

Steps that are not yet ported launch the legacy wx LabGym app (or the
standalone annotator) in a subprocess so the two GUI toolkits never share
an event loop.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

WORKFLOW_STEPS = [
    (
        "1. Preprocess videos",
        "Trim/crop/enhance videos in the legacy LabGym Preprocessing module.",
        "legacy",
    ),
    (
        "2. Detect & track",
        "Run the detector so each subject gets an initial ID. Export id_review tracklets.",
        "legacy",
    ),
    (
        "3. Fix ID swaps",
        "Scrub contact events in ID review; apply remaps and save corrected tracklets "
        "(frozen identities for later steps).",
        "legacy",
    ),
    (
        "4. Annotate ethogram",
        "Open the Behavior Annotator with the video + corrected tracklets. Label behaviors "
        "per subject (or group/roles). Save video.annotations.json — this is durable ground truth.",
        "annotator",
    ),
    (
        "5. Generate examples FROM ethogram",
        "In the annotator: Tools → Generate LabGym training pairs from ethogram. "
        "Creates sorted animation+pattern pairs from bouts (change length later without re-labeling). "
        "Legacy alternative: dense Generate Examples + sort by annotation.",
        "annotator",
    ),
    (
        "6. Train categorizer (hard / soft)",
        "Train Categorizers on the sorted folders. Optional hard_soft_aux with soft_labels.csv.",
        "legacy",
    ),
    (
        "7. Analyze videos",
        "Run Analysis with the trained categorizer.",
        "legacy",
    ),
]


class WorkflowMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LabGym Workflow (PySide6 shell)")
        self.resize(900, 560)

        central = QWidget()
        layout = QHBoxLayout(central)

        self.list = QListWidget()
        for title, _desc, _kind in WORKFLOW_STEPS:
            self.list.addItem(QListWidgetItem(title))
        self.list.currentRowChanged.connect(self._on_select)
        layout.addWidget(self.list, 1)

        right = QVBoxLayout()
        self.title = QLabel("Select a step")
        self.title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.desc = QTextEdit()
        self.desc.setReadOnly(True)
        self.btn = QPushButton("Open")
        self.btn.clicked.connect(self._open_step)
        right.addWidget(self.title)
        right.addWidget(self.desc, 1)
        right.addWidget(self.btn)
        layout.addLayout(right, 2)

        self.setCentralWidget(central)
        self.statusBar().showMessage(
            "Legacy steps launch wx LabGym in a separate process."
        )
        self.list.setCurrentRow(0)

    def _on_select(self, row: int):
        if row < 0 or row >= len(WORKFLOW_STEPS):
            return
        title, desc, kind = WORKFLOW_STEPS[row]
        self.title.setText(title)
        self.desc.setPlainText(desc + f"\n\nLauncher: {kind}")
        if kind == "annotator":
            self.btn.setText("Open Behavior Annotator")
        else:
            self.btn.setText("Open LabGym (legacy wx)")

    def _open_step(self):
        row = self.list.currentRow()
        if row < 0:
            return
        _title, _desc, kind = WORKFLOW_STEPS[row]
        try:
            if kind == "annotator":
                cmd = [sys.executable, "-m", "LabGym.annotator"]
            else:
                cmd = [sys.executable, "-m", "LabGym"]
            subprocess.Popen(cmd, close_fds=True)
            self.statusBar().showMessage("Launched: " + " ".join(cmd))
        except Exception as exc:
            QMessageBox.critical(self, "Launch failed", str(exc))


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("LabGym Workflow")
    app.setStyle("Fusion")
    w = WorkflowMainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
