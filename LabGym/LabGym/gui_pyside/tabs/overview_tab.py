"""Overview: ethogram-first pipeline checklist."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.project_state import ProjectController

STEPS = [
    (
        "Project settings",
        "Set video, post–ID-review tracklets, annotations path, behavior mode, "
        "and default example-generation parameters.",
        "project",
    ),
    (
        "Detect & track (legacy)",
        "Run LabGym detector so animals get initial IDs and id_review tracklets exist.",
        "legacy",
    ),
    (
        "Fix ID swaps (legacy)",
        "Use ID review to correct identity swaps; save remapped tracklets "
        "(frozen IDs for annotation and training).",
        "legacy",
    ),
    (
        "Annotate ethogram",
        "Open the Annotate tab (or a separate annotator window). Label behaviors "
        "per subject / group / partners. Save video.annotations.json — durable ground truth.",
        "annotate",
    ),
    (
        "Generate examples from ethogram",
        "Build sorted LabGym animation+pattern pairs from bouts + fixed tracklets. "
        "Re-run with a new window length without re-annotating.",
        "generate",
    ),
    (
        "Train categorizer (legacy)",
        "Train Categorizers on the sorted example folders. Optional hard_soft_aux + soft_labels.csv.",
        "legacy",
    ),
    (
        "Analyze videos (legacy)",
        "Run Analysis with the trained categorizer.",
        "legacy",
    ),
]


class OverviewTab(QWidget):
    """Pipeline checklist with navigation to other tabs / legacy tools."""

    go_to_tab = Signal(str)  # project | annotate | generate | pipeline
    launch_legacy = Signal()

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project

        layout = QHBoxLayout(self)

        self.list = QListWidget()
        for title, _desc, _kind in STEPS:
            self.list.addItem(QListWidgetItem(title))
        self.list.currentRowChanged.connect(self._on_select)
        layout.addWidget(self.list, 1)

        right = QVBoxLayout()
        title = QLabel("Ethogram-first workflow")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        right.addWidget(title)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(
            "QLabel { background: #2a2a2a; color: #eee; padding: 8px; border-radius: 4px; }"
        )
        right.addWidget(self.summary)

        self.desc = QTextEdit()
        self.desc.setReadOnly(True)
        right.addWidget(self.desc, 1)

        btn_row = QHBoxLayout()
        self.btn_go = QPushButton("Open this step")
        self.btn_go.clicked.connect(self._open_step)
        btn_row.addWidget(self.btn_go)
        btn_row.addStretch(1)
        right.addLayout(btn_row)

        layout.addLayout(right, 2)

        self.project.changed.connect(self._refresh_summary)
        self._refresh_summary()
        self.list.setCurrentRow(0)

    def _refresh_summary(self) -> None:
        self.summary.setText(self.project.state.status_summary())

    def _on_select(self, row: int) -> None:
        if row < 0 or row >= len(STEPS):
            return
        title, desc, kind = STEPS[row]
        self.desc.setPlainText(f"{title}\n\n{desc}\n\nHandler: {kind}")
        if kind == "legacy":
            self.btn_go.setText("Open LabGym (legacy wx)")
        elif kind == "project":
            self.btn_go.setText("Open Project settings")
        elif kind == "annotate":
            self.btn_go.setText("Open Annotate tab")
        elif kind == "generate":
            self.btn_go.setText("Open Generate tab")
        else:
            self.btn_go.setText("Open this step")

    def _open_step(self) -> None:
        row = self.list.currentRow()
        if row < 0:
            return
        _title, _desc, kind = STEPS[row]
        if kind == "legacy":
            self.launch_legacy.emit()
        elif kind == "project":
            self.go_to_tab.emit("project")
        elif kind == "annotate":
            self.go_to_tab.emit("annotate")
        elif kind == "generate":
            self.go_to_tab.emit("generate")
