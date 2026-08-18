"""Project Edit defaults must flow into workbench forms on project_replaced."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication


def _app():
    return QApplication.instance() or QApplication(sys.argv)


def _ctrl(tmp_path: Path, *, mode: int = 0, window_len: int = 15):
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.project.model import Project

    ctrl = ProjectController()
    proj = Project.new(name="defs", root_dir=str(tmp_path))
    proj.defaults.behavior_mode = mode
    proj.defaults.window_length = window_len
    proj.defaults.detector_name = str(tmp_path / "MyDet")
    proj.defaults.categorizer_name = str(tmp_path / "MyCat")
    ctrl.replace(proj, dirty=False)
    return ctrl


def test_train_categorizer_picks_up_edit_project_defaults(tmp_path: Path):
    _app()
    from LabGym.gui_pyside.project.model import Project
    from LabGym.gui_pyside.workbenches.categorizer.train_categorizer_tab import (
        TrainCategorizerTab,
    )

    ctrl = _ctrl(tmp_path, mode=0, window_len=15)
    tab = TrainCategorizerTab(ctrl)
    assert tab.combo_mode.currentData() == 0
    assert tab.spin_len.value() == 15

    # Simulate Edit Project OK → replace with new defaults
    proj = Project.new(name="defs2", root_dir=str(tmp_path))
    proj.defaults.behavior_mode = 2
    proj.defaults.window_length = 21
    proj.paths.models_root = "models"
    ctrl.replace(proj, dirty=True)

    assert tab.combo_mode.currentData() == 2
    assert tab.spin_len.value() == 21
    assert "models" in tab.ed_models.text().replace("\\", "/")


def test_detect_track_picks_up_mode_and_length(tmp_path: Path):
    _app()
    from LabGym.gui_pyside.project.model import Project
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab

    ctrl = _ctrl(tmp_path, mode=0, window_len=15)
    tab = DetectTrackTab(ctrl)
    assert tab.combo_mode.currentData() == 0
    assert tab.spin_length.value() == 15

    proj = Project.new(name="d2", root_dir=str(tmp_path))
    proj.defaults.behavior_mode = 2
    proj.defaults.window_length = 30
    proj.defaults.detector_name = str(tmp_path / "DetB")
    ctrl.replace(proj, dirty=True)

    assert tab.combo_mode.currentData() == 2
    assert tab.spin_length.value() == 30
    assert tab.ed_detector.currentText() == str(tmp_path / "DetB")


def test_detect_track_maps_mode_1_to_0(tmp_path: Path):
    _app()
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab

    ctrl = _ctrl(tmp_path, mode=1, window_len=12)
    tab = DetectTrackTab(ctrl)
    # Tab has no mode 1 item → map to 0
    assert tab.combo_mode.currentData() == 0
    assert tab.spin_length.value() == 12


def test_process_and_test_tabs_model_defaults(tmp_path: Path):
    _app()
    from LabGym.gui_pyside.project.model import Project
    from LabGym.gui_pyside.workbenches.categorizer.process_videos_tab import (
        ProcessVideosTab,
    )
    from LabGym.gui_pyside.workbenches.categorizer.test_categorizer_tab import (
        TestCategorizerTab,
    )
    from LabGym.gui_pyside.workbenches.detector.test_detector_tab import TestDetectorTab

    ctrl = _ctrl(tmp_path)
    proc = ProcessVideosTab(ctrl)
    assert proc.ed_categorizer.text() == str(tmp_path / "MyCat")

    tdet = TestDetectorTab(ctrl)
    assert tdet.ed_det.text() == str(tmp_path / "MyDet")

    tcat = TestCategorizerTab(ctrl)
    assert tcat.ed_model.text() == str(tmp_path / "MyCat")

    proj = Project.new(name="x", root_dir=str(tmp_path))
    proj.defaults.detector_name = str(tmp_path / "NewDet")
    proj.defaults.categorizer_name = str(tmp_path / "NewCat")
    ctrl.replace(proj, dirty=True)

    assert proc.ed_categorizer.text() == str(tmp_path / "NewCat")
    assert tdet.ed_det.text() == str(tmp_path / "NewDet")
    assert tcat.ed_model.text() == str(tmp_path / "NewCat")
