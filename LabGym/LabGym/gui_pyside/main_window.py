"""PySide6 ethogram-first workflow shell for LabGym.

Tabs:
  Overview  – pipeline checklist
  Project   – paths, modes, generation defaults
  Annotate  – embedded Behavior Annotator (manual ethogram labeling)
  Generate  – ethogram → LabGym training pairs
  Detect / ID / Train / Analyze – launch legacy wx until fully ported
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
)

from LabGym.gui_pyside.legacy_launch import launch_legacy_labgym
from LabGym.gui_pyside.project_state import ProjectController
from LabGym.gui_pyside.tabs.annotate_tab import AnnotateTab
from LabGym.gui_pyside.tabs.generate_tab import GenerateTab
from LabGym.gui_pyside.tabs.overview_tab import OverviewTab
from LabGym.gui_pyside.tabs.pipeline_tab import PipelineTab
from LabGym.gui_pyside.tabs.project_tab import ProjectTab

# Tab indices (keep in sync with addTab order)
TAB_OVERVIEW = 0
TAB_PROJECT = 1
TAB_ANNOTATE = 2
TAB_GENERATE = 3
TAB_PIPELINE = 4


class WorkflowMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LabGym Workflow — ethogram-first (PySide6)")
        self.resize(1280, 800)

        self.project = ProjectController(self)
        self.project.load_settings()

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        self.overview = OverviewTab(self.project)
        self.project_tab = ProjectTab(self.project)
        self.annotate_tab = AnnotateTab(self.project)
        self.generate_tab = GenerateTab(self.project)
        self.pipeline_tab = PipelineTab()

        self.tabs.addTab(self.overview, "Overview")
        self.tabs.addTab(self.project_tab, "Project")
        self.tabs.addTab(self.annotate_tab, "Annotate")
        self.tabs.addTab(self.generate_tab, "Generate")
        self.tabs.addTab(self.pipeline_tab, "Detect / ID / Train / Analyze")

        self.setCentralWidget(self.tabs)
        self.setStatusBar(QStatusBar())
        self._update_status()
        self.project.changed.connect(self._update_status)

        # Navigation wiring
        self.overview.go_to_tab.connect(self._go_named_tab)
        self.overview.launch_legacy.connect(self._launch_legacy)
        self.project_tab.apply_to_annotate.connect(self._apply_project_to_annotate)
        self.project_tab.open_annotate.connect(self._open_annotate_with_project)

        self._build_menu()

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        act_save = QAction("Save &project settings", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        tools_menu = menubar.addMenu("&Tools")
        act_legacy = QAction("Open LabGym (legacy wx)…", self)
        act_legacy.triggered.connect(self._launch_legacy)
        tools_menu.addAction(act_legacy)

        act_load_ann = QAction("Load project into Annotate tab", self)
        act_load_ann.triggered.connect(self._apply_project_to_annotate)
        tools_menu.addAction(act_load_ann)

        help_menu = menubar.addMenu("&Help")
        act_about = QAction("About workflow…", self)
        act_about.triggered.connect(self._about)
        help_menu.addAction(act_about)

    def _update_status(self) -> None:
        self.statusBar().showMessage(self.project.state.status_summary())

    def _go_named_tab(self, name: str) -> None:
        mapping = {
            "project": TAB_PROJECT,
            "annotate": TAB_ANNOTATE,
            "generate": TAB_GENERATE,
            "pipeline": TAB_PIPELINE,
            "overview": TAB_OVERVIEW,
        }
        idx = mapping.get(name)
        if idx is not None:
            self.tabs.setCurrentIndex(idx)

    def _save_project(self) -> None:
        # Pull latest edits from Project tab fields
        self.project_tab._commit_from_ui()
        self.project.save_settings()
        self.statusBar().showMessage("Project settings saved.  " + self.project.state.status_summary())

    def _apply_project_to_annotate(self) -> None:
        self.project_tab._commit_from_ui()
        self.tabs.setCurrentIndex(TAB_ANNOTATE)
        ok = self.annotate_tab.load_project()
        if ok:
            self.statusBar().showMessage(
                "Project loaded into Annotate tab.  " + self.project.state.status_summary()
            )

    def _open_annotate_with_project(self) -> None:
        self._apply_project_to_annotate()

    def _launch_legacy(self) -> None:
        try:
            launch_legacy_labgym()
            self.statusBar().showMessage("Launched legacy LabGym (wx) in a separate process.")
        except Exception as exc:
            QMessageBox.critical(self, "Launch failed", str(exc))

    def _about(self) -> None:
        QMessageBox.information(
            self,
            "LabGym Workflow",
            "Ethogram-first training UI (PySide6).\n\n"
            "1. Project — set video, tracklets, annotations, modes\n"
            "2. Annotate — label ethograms (embedded annotator)\n"
            "3. Generate — build sorted LabGym pairs from ethograms\n"
            "4. Detect / ID / Train / Analyze — legacy LabGym until ported\n\n"
            "Ground truth = video.annotations.json (not dense unsorted clips).",
        )

    def closeEvent(self, event) -> None:
        try:
            self.project_tab._commit_from_ui()
            self.project.save_settings()
        except Exception:
            pass
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("LabGym Workflow")
    app.setOrganizationName("LabGym")
    app.setStyle("Fusion")
    window = WorkflowMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
