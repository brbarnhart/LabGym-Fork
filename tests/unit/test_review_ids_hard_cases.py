"""Unit tests for Review IDs hard-case training image extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from LabGym.gui_pyside.workbenches.detector.review_ids_hard_cases import (
    AnalysisFrameRange,
    default_output_dir,
    extract_hard_case_frames,
    frames_to_extract,
    normalize_ranges,
    output_filename,
)


def test_analysis_frame_range_swaps_and_clamps():
    r = AnalysisFrameRange(10, 5)
    assert r.start_frame == 5
    assert r.end_frame == 10
    assert r.n_frames == 6


def test_normalize_ranges_merges_overlap():
    ranges = [
        AnalysisFrameRange(0, 5),
        AnalysisFrameRange(4, 10, note="a"),
        AnalysisFrameRange(20, 22),
    ]
    out = normalize_ranges(ranges, n_frames=100)
    assert len(out) == 2
    assert out[0].start_frame == 0 and out[0].end_frame == 10
    assert out[1].start_frame == 20 and out[1].end_frame == 22


def test_normalize_ranges_clamps_to_n_frames():
    out = normalize_ranges([AnalysisFrameRange(0, 500)], n_frames=10)
    assert out[0].end_frame == 9


def test_frames_to_extract_includes_ends():
    frames = frames_to_extract(
        [AnalysisFrameRange(0, 25)],
        skip=10,
    )
    assert frames == [0, 10, 20, 25]


def test_frames_to_extract_dedupes_across_ranges():
    frames = frames_to_extract(
        [AnalysisFrameRange(0, 5), AnalysisFrameRange(5, 10)],
        skip=5,
    )
    assert frames == [0, 5, 10]


def test_output_filename_and_default_dir(tmp_path: Path):
    assert output_filename("my video!", 42) == "my_video__af000042.jpg"
    assert default_output_dir(str(tmp_path)).endswith("detector_training_images")
    assert "detector_training_images" in default_output_dir(None)


def _write_tiny_video(path: Path, n: int = 30, fps: float = 10.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 64, 48
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    assert writer.isOpened(), "VideoWriter failed to open"
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (i * 7 % 255, 40, 80)
        cv2.putText(
            frame,
            str(i),
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        writer.write(frame)
    writer.release()


def test_extract_hard_case_frames_writes_jpgs(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    _write_tiny_video(video, n=40, fps=10.0)
    out = tmp_path / "imgs"
    result = extract_hard_case_frames(
        str(video),
        str(out),
        [AnalysisFrameRange(0, 20), AnalysisFrameRange(30, 35)],
        store_meta={"fps": 10.0, "start_t": 0.0},
        fps=10.0,
        skip=10,
        framewidth=32,
        n_frames=40,
    )
    assert result.error == ""
    assert result.n_written >= 4  # 0,10,20 + ends + second range
    assert result.n_failed == 0
    paths = list(out.glob("*.jpg"))
    assert len(paths) == result.n_written
    # Names use analysis-frame index
    assert (out / "clip_af000000.jpg").is_file()
    assert (out / "clip_af000020.jpg").is_file()
    img = cv2.imread(str(out / "clip_af000000.jpg"))
    assert img is not None
    assert img.shape[1] == 32  # resized width


def test_extract_missing_video(tmp_path: Path):
    result = extract_hard_case_frames(
        str(tmp_path / "nope.mp4"),
        str(tmp_path / "out"),
        [AnalysisFrameRange(0, 5)],
    )
    assert result.n_written == 0
    assert "not found" in result.error.lower() or "Video" in result.error


def test_extract_empty_ranges(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    _write_tiny_video(video, n=5)
    result = extract_hard_case_frames(
        str(video),
        str(tmp_path / "out"),
        [],
    )
    assert result.n_written == 0
    assert "No frames" in result.error


def test_extract_cancel_cooperative(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    _write_tiny_video(video, n=40, fps=10.0)
    out = tmp_path / "imgs"
    calls = {"n": 0}

    def cancel_after_two() -> bool:
        # Allow two progress ticks then cancel before third frame write completes.
        return calls["n"] >= 2

    def progress(current, total, msg):
        calls["n"] = current

    result = extract_hard_case_frames(
        str(video),
        str(out),
        [AnalysisFrameRange(0, 30)],
        store_meta={"fps": 10.0, "start_t": 0.0},
        fps=10.0,
        skip=1,
        progress_callback=progress,
        cancel_check=cancel_after_two,
    )
    assert result.cancelled
    assert result.n_written >= 1
    assert result.n_written < 30


def test_hard_case_progress_dialog_constructs():
    import sys

    from PySide6.QtWidgets import QApplication

    from LabGym.gui_pyside.workbenches.detector.hard_case_progress import (
        HardCaseProgressDialog,
    )

    app = QApplication.instance() or QApplication(sys.argv)
    dlg = HardCaseProgressDialog()
    dlg.begin_export(10, out_path="/tmp/out", video_label="clip.mp4")
    dlg.set_frame_progress(3, 10, "[3/10] analysis frame 12")
    dlg.finish_export(status="Wrote 10 image(s)")
    assert dlg.bar_frames.value() == 10
    assert not dlg.is_job_running
    dlg.close()
    del app


@pytest.mark.gui
def test_hard_case_extract_lives_on_dialog_not_right_column():
    """Extract form is hosted in a modeless popup, not the Review IDs column."""
    import sys

    from PySide6.QtGui import QKeySequence, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QLineEdit,
        QListWidget,
        QPushButton,
        QSpinBox,
        QWidget,
    )

    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.workbenches.detector.review_ids_tab import ReviewIdsTab

    app = QApplication.instance() or QApplication(sys.argv)
    tab = ReviewIdsTab(ProjectController())

    opener = tab.findChild(QPushButton, "btn_hard_extract")
    assert opener is not None
    assert "Hard cases" in opener.text()
    assert tab.markers_panel is not None
    assert tab.subjects_table is not None
    assert tab.btn_save is not None

    right = tab.findChild(QWidget, "review_ids_right")
    assert right is not None
    assert right.findChild(QLineEdit, "ed_hard_out") is None
    assert right.findChild(QSpinBox, "spin_hard_skip") is None
    assert right.findChild(QPushButton, "btn_gen_hard") is None
    assert right.findChild(QListWidget, "list_ranges") is None
    assert tab._hard_extract_dlg is None

    tab._training_ranges.append(AnalysisFrameRange(2, 8, note="pre"))
    tab._refresh_range_list()
    tab._open_hard_extract_dialog()

    dlg = tab._hard_extract_dlg
    assert dlg is not None
    assert dlg.isVisible()
    assert not dlg.isModal()
    assert dlg.findChild(QListWidget, "list_ranges") is not None
    assert dlg.findChild(QSpinBox, "spin_hard_skip") is not None
    assert dlg.spin_hard_skip.value() == 10
    assert dlg.findChild(QLineEdit, "ed_hard_out") is not None
    assert dlg.findChild(QPushButton, "btn_gen_hard") is not None
    assert dlg.list_ranges.count() == 1
    assert "f2" in dlg.list_ranges.item(0).text()
    dlg_keys = {s.key().toString() for s in dlg.findChildren(QShortcut)}
    assert QKeySequence("[").toString() in dlg_keys
    assert QKeySequence("]").toString() in dlg_keys
    dlg.set_extract_enabled(False)
    assert not dlg.btn_hard_out.isEnabled()
    dlg.set_extract_enabled(True)
    assert dlg.btn_hard_out.isEnabled()

    # Still not in the main right column after the popup is shown.
    assert right.findChild(QLineEdit, "ed_hard_out") is None
    assert right.findChild(QListWidget, "list_ranges") is None
    first = dlg
    tab._open_hard_extract_dialog()
    assert tab._hard_extract_dlg is first

    tab.close()
    dlg.close()
    del app
