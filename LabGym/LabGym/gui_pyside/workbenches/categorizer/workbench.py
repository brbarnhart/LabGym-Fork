"""Categorizer workbench — Generate training data + Train/Test (Phases 2 & 6)."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.workbenches.base import Workbench
from LabGym.gui_pyside.workbenches.categorizer.annotate_ethogram_tab import (
    AnnotateEthogramTab,
)
from LabGym.gui_pyside.workbenches.categorizer.generate_examples_tab import (
    GenerateExamplesTab,
)
from LabGym.gui_pyside.workbenches.categorizer.test_categorizer_tab import (
    TestCategorizerTab,
)
from LabGym.gui_pyside.workbenches.categorizer.train_categorizer_tab import (
    TrainCategorizerTab,
)
from LabGym.gui_pyside.workbenches.placeholder import PlaceholderTab


class GenerateTrainingHost(QWidget):
    """Two subtabs: Annotate ethogram | Generate examples."""

    request_edit_project = Signal()

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        intro = QLabel(
            "Generate training data (ethogram-first): annotate durable ethograms, "
            "then build sorted LabGym training pairs. Dense clip-sort is not offered."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.inner = QTabWidget()
        self.annotate_tab = AnnotateEthogramTab(project)
        self.generate_tab = GenerateExamplesTab(project)
        self.inner.addTab(self.annotate_tab, "Annotate ethogram")
        self.inner.addTab(self.generate_tab, "Generate examples")
        layout.addWidget(self.inner, 1)

        self.annotate_tab.request_edit_project.connect(self.request_edit_project.emit)
        self.generate_tab.request_edit_project.connect(self.request_edit_project.emit)
        self.generate_tab.request_annotate.connect(self.show_annotate)

    def show_annotate(self) -> None:
        self.inner.setCurrentWidget(self.annotate_tab)

    def show_generate(self) -> None:
        self.inner.setCurrentWidget(self.generate_tab)


class CategorizerWorkbench(Workbench):
    workbench_id = "categorizer"
    title = "Categorizer"

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(project, parent)

        self.generate_host = GenerateTrainingHost(project)
        self.train_tab = TrainCategorizerTab(project)
        self.test_tab = TestCategorizerTab(project)
        process = PlaceholderTab(
            "Process videos",
            "Batch-run the categorizer on project videos that already have "
            "detection/tracking (and preferably reviewed IDs).",
            phase_note="Port planned in Phase 7.",
        )

        self.add_subtab("generate_training", "Generate training data", self.generate_host)
        self.add_subtab("train", "Train categorizer", self.train_tab)
        self.add_subtab("test", "Test categorizer", self.test_tab)
        self.add_subtab("process", "Process videos", process)
        self._placeholder_tabs = [process]

    def connect_legacy(self, slot) -> None:
        for t in self._placeholder_tabs:
            t.launch_legacy.connect(slot)

    def connect_edit_project(self, slot) -> None:
        self.generate_host.request_edit_project.connect(slot)
