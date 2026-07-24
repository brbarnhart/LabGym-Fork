"""Hosts workbench widgets and swaps the active one."""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtWidgets import QStackedWidget, QVBoxLayout, QWidget

from LabGym.gui_pyside.workbenches.base import Workbench


class WorkbenchHost(QWidget):
    """Stack of workbenches; each workbench owns its own tab strip."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stack = QStackedWidget(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)
        self._by_id: Dict[str, Workbench] = {}

    def add_workbench(self, workbench: Workbench) -> None:
        self._by_id[workbench.workbench_id] = workbench
        self._stack.addWidget(workbench)

    def show_workbench(self, workbench_id: str) -> None:
        w = self._by_id.get(workbench_id)
        if w is not None:
            self._stack.setCurrentWidget(w)

    def get(self, workbench_id: str) -> Optional[Workbench]:
        return self._by_id.get(workbench_id)

    def current_id(self) -> str:
        w = self._stack.currentWidget()
        if isinstance(w, Workbench):
            return w.workbench_id
        return ""

    def goto_tab(self, workbench_id: str, tab_id: str) -> bool:
        w = self._by_id.get(workbench_id)
        if w is None:
            return False
        self._stack.setCurrentWidget(w)
        return w.set_current_tab(tab_id)
