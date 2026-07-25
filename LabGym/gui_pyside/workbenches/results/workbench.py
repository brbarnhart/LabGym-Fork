"""Results workbench — mine / plot / distances post-analysis tools."""

from __future__ import annotations

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.workbenches.base import Workbench
from LabGym.gui_pyside.workbenches.results.distances_tab import CalculateDistancesTab
from LabGym.gui_pyside.workbenches.results.mine_tab import MineResultsTab
from LabGym.gui_pyside.workbenches.results.plot_tab import PlotBehaviorsTab


class ResultsWorkbench(Workbench):
    workbench_id = "results"
    title = "Results"

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(project, parent)
        self.mine_tab = MineResultsTab(project)
        self.plot_tab = PlotBehaviorsTab(project)
        self.distances_tab = CalculateDistancesTab(project)

        self.add_subtab("mine", "Mine results", self.mine_tab)
        self.add_subtab("plot", "Behavior plot", self.plot_tab)
        self.add_subtab("distances", "Calculate distances", self.distances_tab)

