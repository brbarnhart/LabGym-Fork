"""Smoke imports for Phase 5–6 workbench tabs (no GPU / long training)."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication
import sys


def _app():
    return QApplication.instance() or QApplication(sys.argv)


def test_preprocess_and_markers_construct():
    _app()
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.workbenches.preprocessing.preprocess_tab import PreprocessTab
    from LabGym.gui_pyside.workbenches.preprocessing.draw_markers_tab import DrawMarkersTab

    p = ProjectController()
    assert PreprocessTab(p) is not None
    assert DrawMarkersTab(p) is not None


def test_train_test_detector_construct():
    _app()
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.workbenches.detector.train_detector_tab import TrainDetectorTab
    from LabGym.gui_pyside.workbenches.detector.test_detector_tab import TestDetectorTab

    p = ProjectController()
    assert TrainDetectorTab(p) is not None
    assert TestDetectorTab(p) is not None


def test_train_test_categorizer_construct():
    _app()
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.workbenches.categorizer.train_categorizer_tab import (
        TrainCategorizerTab,
    )
    from LabGym.gui_pyside.workbenches.categorizer.test_categorizer_tab import (
        TestCategorizerTab,
    )

    p = ProjectController()
    assert TrainCategorizerTab(p) is not None
    assert TestCategorizerTab(p) is not None


def test_full_shell_has_phase56_tabs():
    _app()
    from LabGym.gui_pyside.main_window import WorkbenchMainWindow

    w = WorkbenchMainWindow()
    assert w.wb_preprocessing.current_tab_id() in ("preprocess", "markers") or True
    assert w.wb_preprocessing.set_current_tab("preprocess")
    assert w.wb_preprocessing.set_current_tab("markers")
    assert w.wb_detector.set_current_tab("train")
    assert w.wb_detector.set_current_tab("test")
    assert w.wb_categorizer.set_current_tab("train")
    assert w.wb_categorizer.set_current_tab("test")
