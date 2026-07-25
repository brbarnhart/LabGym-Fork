"""Detector workbench — Generate training data, Train/Test, Detect+track, Review IDs."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.workbenches.base import Workbench
from LabGym.gui_pyside.workbenches.detector.annotate_images_tab import AnnotateImagesTab
from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab
from LabGym.gui_pyside.workbenches.detector.generate_images_tab import GenerateImagesTab
from LabGym.gui_pyside.workbenches.detector.review_ids_tab import ReviewIdsTab
from LabGym.gui_pyside.workbenches.detector.test_detector_tab import TestDetectorTab
from LabGym.gui_pyside.workbenches.detector.train_detector_tab import TrainDetectorTab


class DetectorGenerateTrainingHost(QWidget):
    """Subtabs: Generate images | Annotate images (future / EZannot)."""

    request_train = Signal()

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        intro = QLabel(
            "Generate training data for detectors: extract still frames from videos, "
            "annotate outlines externally (EZannot recommended until in-app annotation "
            "ships), then train under the <b>Train detector</b> tab."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.inner = QTabWidget()
        self.generate_tab = GenerateImagesTab(project)
        self.annotate_tab = AnnotateImagesTab()
        self.inner.addTab(self.generate_tab, "Generate images")
        self.inner.addTab(self.annotate_tab, "Annotate images")
        layout.addWidget(self.inner, 1)

        self.generate_tab.request_annotate.connect(self.show_annotate)
        self.generate_tab.request_train.connect(self.request_train.emit)

    def show_annotate(self) -> None:
        self.inner.setCurrentWidget(self.annotate_tab)

    def show_generate(self) -> None:
        self.inner.setCurrentWidget(self.generate_tab)


class DetectorWorkbench(Workbench):
    workbench_id = "detector"
    title = "Detector"

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(project, parent)

        self.generate_host = DetectorGenerateTrainingHost(project)
        self.train_tab = TrainDetectorTab(project)
        self.test_tab = TestDetectorTab(project)
        self.detect_tab = DetectTrackTab(project)
        self.review_tab = ReviewIdsTab(project)
        self.detect_tab.request_review_ids.connect(
            lambda: self.set_current_tab("review_ids")
        )
        self.generate_host.request_train.connect(lambda: self.set_current_tab("train"))

        self.add_subtab(
            "generate_training", "Generate training data", self.generate_host
        )
        self.add_subtab("train", "Train detector", self.train_tab)
        self.add_subtab("test", "Test detector", self.test_tab)
        self.add_subtab("detect_track", "Detect + track", self.detect_tab)
        self.add_subtab("review_ids", "Review IDs", self.review_tab)

    def connect_edit_project(self, slot) -> None:
        self.review_tab.request_edit_project.connect(slot)
        self.detect_tab.request_edit_project.connect(slot)
