"""Base workbench: tab strip + content for one major task group."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from LabGym.gui_pyside.project.controller import ProjectController


class Workbench(QWidget):
    """A FreeCAD-style workbench: owns a QTabWidget of subtask panels."""

    workbench_id: str = ""
    title: str = ""

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self._tab_ids: List[str] = []
        self._id_to_index: Dict[str, int] = {}

    def add_subtab(self, tab_id: str, title: str, widget: QWidget) -> None:
        idx = self.tabs.addTab(widget, title)
        self._tab_ids.append(tab_id)
        self._id_to_index[tab_id] = idx

    def set_current_tab(self, tab_id: str) -> bool:
        idx = self._id_to_index.get(tab_id)
        if idx is None:
            return False
        self.tabs.setCurrentIndex(idx)
        return True

    def current_tab_id(self) -> str:
        i = self.tabs.currentIndex()
        if 0 <= i < len(self._tab_ids):
            return self._tab_ids[i]
        return ""
