"""Preprocessing workbench (Phase 5)."""

from __future__ import annotations

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.workbenches.base import Workbench
from LabGym.gui_pyside.workbenches.preprocessing.draw_markers_tab import DrawMarkersTab
from LabGym.gui_pyside.workbenches.preprocessing.preprocess_tab import PreprocessTab


class PreprocessingWorkbench(Workbench):
    workbench_id = "preprocessing"
    title = "Preprocessing"

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(project, parent)
        self.preprocess_tab = PreprocessTab(project)
        self.markers_tab = DrawMarkersTab(project)
        self.add_subtab("preprocess", "Preprocess videos", self.preprocess_tab)
        self.add_subtab("markers", "Draw markers", self.markers_tab)

    def connect_legacy(self, slot) -> None:
        return

    def connect_edit_project(self, slot) -> None:
        self.preprocess_tab.request_edit_project.connect(slot)
        self.markers_tab.request_edit_project.connect(slot)
