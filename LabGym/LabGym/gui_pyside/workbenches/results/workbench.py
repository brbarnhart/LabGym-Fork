"""Results workbench — placeholder only for first usable release."""

from __future__ import annotations

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.workbenches.base import Workbench
from LabGym.gui_pyside.workbenches.placeholder import PlaceholderTab


class ResultsWorkbench(Workbench):
    workbench_id = "results"
    title = "Results"

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(project, parent)
        tab = PlaceholderTab(
            "Results / data export",
            "Future home for ethogram figures, tidy tables for R, and other analyses.",
            phase_note="Placeholder for the first usable release (spec §3.4).",
            show_legacy=False,
        )
        self.add_subtab("export", "Coming soon", tab)

    def connect_legacy(self, slot) -> None:
        return
