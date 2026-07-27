"""AUD-TEST-001: workbench behavioral tests (no GPU / no long jobs).

Covers construct smoke for more tabs, batch-active table refresh policy,
sticky status across project.changed, and shared progress-dialog base.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


def _app():
    return QApplication.instance() or QApplication(sys.argv)


def _controller_with_videos(tmp_path: Path, names=("a.avi", "b.avi")):
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.project.model import Project

    ctrl = ProjectController()
    proj = Project.new(name="test", root_dir=str(tmp_path))
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"fake")
        proj.add_video(str(p))
    ctrl.replace(proj, dirty=False)
    return ctrl


# --- construct smoke (beyond phase 5–6 train/test only) ---


def test_construct_detect_process_review_results():
    _app()
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab
    from LabGym.gui_pyside.workbenches.detector.review_ids_tab import ReviewIdsTab
    from LabGym.gui_pyside.workbenches.categorizer.process_videos_tab import (
        ProcessVideosTab,
    )
    from LabGym.gui_pyside.workbenches.results.mine_tab import MineResultsTab
    from LabGym.gui_pyside.workbenches.results.plot_tab import PlotBehaviorsTab
    from LabGym.gui_pyside.workbenches.results.distances_tab import CalculateDistancesTab

    p = ProjectController()
    assert DetectTrackTab(p) is not None
    assert ReviewIdsTab(p) is not None
    assert ProcessVideosTab(p) is not None
    assert MineResultsTab(p) is not None
    assert PlotBehaviorsTab(p) is not None
    assert CalculateDistancesTab(p) is not None


def test_shell_has_detect_process_review_tabs():
    _app()
    from LabGym.gui_pyside.main_window import WorkbenchMainWindow

    w = WorkbenchMainWindow()
    assert w.wb_detector.set_current_tab("detect_track")
    assert w.wb_detector.set_current_tab("review_ids")
    assert w.wb_categorizer.set_current_tab("process")
    assert w.wb_results.set_current_tab("mine")


# --- batch-active / sticky status ---


def test_detect_track_skips_refresh_while_batch_active(tmp_path: Path):
    _app()
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab

    ctrl = _controller_with_videos(tmp_path)
    tab = DetectTrackTab(ctrl)
    assert tab.table.rowCount() == 2

    # Simulate mid-batch: status set and rebuild blocked
    tab._batch_active = True
    path0 = str(tab.table.item(0, 0).data(Qt.ItemDataRole.UserRole))
    tab._set_status(0, path0, "done", "id_review/foo")
    assert tab.table.item(0, 2).text() == "done"

    # project.changed would normally rebuild and wipe — must not while batch active
    ctrl.mark_dirty()
    assert tab.table.item(0, 2).text() == "done"
    assert tab._status_by_path[path0][0] == "done"

    # After batch ends, refresh restores sticky status
    tab._batch_active = False
    tab.refresh_videos()
    assert tab.table.rowCount() == 2
    # Find path0 row after rebuild
    found = False
    for r in range(tab.table.rowCount()):
        p = str(tab.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
        if p == path0:
            assert tab.table.item(r, 2).text() == "done"
            assert "id_review" in tab.table.item(r, 3).text()
            found = True
    assert found


def test_process_videos_skips_refresh_while_batch_active(tmp_path: Path):
    _app()
    from LabGym.gui_pyside.workbenches.categorizer.process_videos_tab import (
        ProcessVideosTab,
    )

    ctrl = _controller_with_videos(tmp_path)
    tab = ProcessVideosTab(ctrl)
    assert tab.table.rowCount() == 2

    tab._batch_active = True
    path0 = str(tab.table.item(0, 0).data(Qt.ItemDataRole.UserRole))
    tab._set_status(0, path0, "running", "frames 10/100")
    ctrl.mark_dirty()
    assert tab.table.item(0, 2).text() == "running"
    assert tab.table.item(0, 3).text() == "frames 10/100"

    tab._batch_active = False
    tab.refresh_videos()
    for r in range(tab.table.rowCount()):
        p = str(tab.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
        if p == path0:
            assert tab.table.item(r, 2).text() == "running"
            return
    pytest.fail("path0 missing after refresh")


def test_preprocess_skips_refresh_while_batch_active(tmp_path: Path):
    _app()
    from LabGym.gui_pyside.workbenches.preprocessing.preprocess_tab import PreprocessTab

    ctrl = _controller_with_videos(tmp_path)
    tab = PreprocessTab(ctrl)
    assert tab.table.rowCount() == 2

    path0 = str(tab.table.item(0, 0).data(Qt.ItemDataRole.UserRole))
    tab._batch_active = True
    tab._status_by_path[path0] = "done"
    tab.table.item(0, 2).setText("done")

    # Uncheck second row before a blocked refresh would lose it
    tab.table.item(1, 0).setCheckState(Qt.CheckState.Unchecked)
    ctrl.mark_dirty()
    assert tab.table.item(0, 2).text() == "done"
    assert tab.table.item(1, 0).checkState() == Qt.CheckState.Unchecked

    tab._batch_active = False
    tab.refresh_videos()
    # sticky status restored
    for r in range(tab.table.rowCount()):
        p = str(tab.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
        if p == path0:
            assert tab.table.item(r, 2).text() == "done"
            return
    pytest.fail("path0 missing")


def test_detect_track_preserves_check_state_on_refresh(tmp_path: Path):
    _app()
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab

    ctrl = _controller_with_videos(tmp_path)
    tab = DetectTrackTab(ctrl)
    tab.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    tab.table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    path_unchecked = str(tab.table.item(0, 0).data(Qt.ItemDataRole.UserRole))
    tab.refresh_videos()
    for r in range(tab.table.rowCount()):
        p = str(tab.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
        if p == path_unchecked:
            assert tab.table.item(r, 0).checkState() == Qt.CheckState.Unchecked
            return
    pytest.fail("unchecked path missing")


# --- shared progress dialog base ---


def test_job_progress_dialog_base_show_and_running():
    _app()
    from LabGym.gui_pyside.widgets.progress_dialog_base import JobProgressDialogBase

    dlg = JobProgressDialogBase("Test job")
    assert dlg.is_job_running is False
    dlg.set_job_running(True)
    assert dlg.is_job_running is True
    dlg.show_as_window()
    assert dlg.isVisible()
    dlg.set_job_running(False)
    dlg.close()


def test_detect_track_progress_dialog_begin_finish():
    _app()
    from LabGym.gui_pyside.workbenches.detector.detect_track_progress import (
        DetectTrackProgressDialog,
    )

    dlg = DetectTrackProgressDialog()
    dlg.begin_batch(3)
    assert dlg.is_job_running
    assert dlg.bar_files.maximum() == 3
    dlg.set_current_video("clip.avi", 0)
    dlg.set_frame_progress(5, 20)
    assert dlg.bar_frames.value() == 5
    dlg.mark_file_finished()
    assert dlg.bar_files.value() == 1
    dlg.finish_batch()
    assert not dlg.is_job_running
    dlg.close()


# --- review package helpers (behavioral IO without GUI load of full package) ---


def test_review_resolve_video_path(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
        resolve_video_path,
    )

    vid = tmp_path / "clip.avi"
    vid.write_bytes(b"x")
    review = tmp_path / "id_review"
    review.mkdir()
    # relative name next to review parent
    assert resolve_video_path(str(review), {"video": "clip.avi"}, [], None) == str(vid)
    # project override
    other = tmp_path / "other.avi"
    other.write_bytes(b"y")
    assert (
        resolve_video_path(str(review), {"video": "missing.avi"}, [], str(other))
        == str(other)
    )


def test_review_clone_markers_and_events_for_kind():
    from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
        clone_markers,
        events_for_kind,
    )
    from LabGym.id_review.types import ContactEvent, SwitchMarker

    m = SwitchMarker(
        marker_id="s1",
        frame=3,
        animal_kind="mouse",
        involved_ids=[0, 1],
        time_sec=0.3,
    )
    cloned = clone_markers([m])
    assert len(cloned) == 1
    assert cloned[0].marker_id == "s1"
    assert cloned[0] is not m

    events = [
        ContactEvent(
            event_id="e1",
            animal_kind="mouse",
            involved_ids=[0, 1],
            start_frame=0,
            end_frame=5,
            pre_frame=0,
            post_frame=5,
            risk_score=0.9,
            fps=10.0,
            video="x.avi",
        ),
        ContactEvent(
            event_id="e2",
            animal_kind="rat",
            involved_ids=[0, 1],
            start_frame=0,
            end_frame=5,
            pre_frame=0,
            post_frame=5,
            risk_score=0.5,
            fps=10.0,
            video="x.avi",
        ),
    ]
    only_mouse = events_for_kind(events, "mouse")
    assert len(only_mouse) == 1
    assert only_mouse[0].event_id == "e1"


def test_process_videos_preserves_check_state_on_refresh(tmp_path: Path):
    _app()
    from LabGym.gui_pyside.workbenches.categorizer.process_videos_tab import (
        ProcessVideosTab,
    )

    ctrl = _controller_with_videos(tmp_path)
    tab = ProcessVideosTab(ctrl)
    tab.table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    path_unchecked = str(tab.table.item(0, 0).data(Qt.ItemDataRole.UserRole))
    tab.refresh_videos()
    for r in range(tab.table.rowCount()):
        p = str(tab.table.item(r, 0).data(Qt.ItemDataRole.UserRole))
        if p == path_unchecked:
            assert tab.table.item(r, 0).checkState() == Qt.CheckState.Unchecked
            return
    pytest.fail("unchecked path missing")


def test_detect_track_start_uses_path_job_ids(tmp_path: Path, monkeypatch):
    """Queue is started with path job_ids without running real detection."""
    _app()
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab
    from LabGym.gui_pyside.jobs.sequential_queue import SequentialJobQueue

    ctrl = _controller_with_videos(tmp_path)
    tab = DetectTrackTab(ctrl)
    det = tmp_path / "det"
    det.mkdir()
    (det / "model_parameters.txt").write_text(
        '{"animal_names": ["mouse"]}', encoding="utf-8"
    )
    tab.ed_detector.setEditText(str(det))

    captured = {}

    def fake_start(self, items, runner):
        captured["items"] = list(items)
        captured["runner"] = runner

    monkeypatch.setattr(SequentialJobQueue, "start", fake_start)
    # Avoid QMessageBox / real batch side effects after start
    tab._start_batch()
    assert "items" in captured
    assert len(captured["items"]) == 2
    for it in captured["items"]:
        assert it.job_id == it.payload
        assert Path(it.job_id).suffix.lower() in {".avi", ".mp4"}
        assert tab._job_rows[it.job_id] is not None
    assert tab._batch_active is True
    # reset so teardown doesn't leave flag
    tab._batch_active = False


def test_markers_table_selection_and_frame():
    _app()
    from LabGym.gui_pyside.workbenches.detector.review_ids_markers import MarkersTable
    from LabGym.id_review.types import SwitchMarker

    panel = MarkersTable()
    markers = [
        SwitchMarker(
            marker_id="s10",
            frame=10,
            animal_kind="mouse",
            involved_ids=[0, 1],
            time_sec=1.0,
        ),
        SwitchMarker(
            marker_id="s3",
            frame=3,
            animal_kind="mouse",
            involved_ids=[0, 1],
            time_sec=0.3,
        ),
    ]
    panel.set_markers(markers)
    # Sorted by frame: s3 then s10
    assert panel.table.rowCount() == 2
    assert panel.table.item(0, 0).text() == "s3"
    assert panel.frame_at_row(0) == 3
    assert panel.selected_marker_id() is None
    panel.table.selectRow(1)
    assert panel.selected_marker_id() == "s10"
    assert panel.frame_at_row(1) == 10


def test_construct_generate_and_annotate_placeholders():
    _app()
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.workbenches.detector.generate_images_tab import (
        GenerateImagesTab,
    )
    from LabGym.gui_pyside.workbenches.detector.annotate_images_tab import (
        AnnotateImagesTab,
    )
    from LabGym.gui_pyside.workbenches.categorizer.generate_examples_tab import (
        GenerateExamplesTab,
    )

    p = ProjectController()
    assert GenerateImagesTab(p) is not None
    assert AnnotateImagesTab() is not None
    assert GenerateExamplesTab(p) is not None
