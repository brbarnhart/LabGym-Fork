"""Detector workbench — Detect+track, Review IDs, Train/Test (Phases 3–6)."""

from __future__ import annotations

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.workbenches.base import Workbench
from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab
from LabGym.gui_pyside.workbenches.detector.review_ids_tab import ReviewIdsTab
from LabGym.gui_pyside.workbenches.detector.test_detector_tab import TestDetectorTab
from LabGym.gui_pyside.workbenches.detector.train_detector_tab import TrainDetectorTab
from LabGym.gui_pyside.workbenches.placeholder import PlaceholderTab


class DetectorWorkbench(Workbench):
    workbench_id = "detector"
    title = "Detector"

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(project, parent)

        future = PlaceholderTab(
            "Create & annotate image examples",
            "Build detector training images inside this app.",
            phase_note="Future feature — not in MVP.",
            show_legacy=False,
        )
        self.train_tab = TrainDetectorTab(project)
        self.test_tab = TestDetectorTab(project)
        self.detect_tab = DetectTrackTab(project)
        self.review_tab = ReviewIdsTab(project)
        self.detect_tab.request_review_ids.connect(
            lambda: self.set_current_tab("review_ids")
        )

        self.add_subtab("image_examples", "Image examples (future)", future)
        self.add_subtab("train", "Train detector", self.train_tab)
        self.add_subtab("test", "Test detector", self.test_tab)
        self.add_subtab("detect_track", "Detect + track", self.detect_tab)
        self.add_subtab("review_ids", "Review IDs", self.review_tab)

    def connect_legacy(self, slot) -> None:
        return

    def connect_edit_project(self, slot) -> None:
        self.review_tab.request_edit_project.connect(slot)
        self.detect_tab.request_edit_project.connect(slot)
