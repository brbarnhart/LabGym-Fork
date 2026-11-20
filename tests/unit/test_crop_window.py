"""Crop window geometry, editor gestures, and preprocess-tab wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from LabGym.gui_pyside.workbenches.preprocessing.crop_geometry import (
    CropEdges,
    CropWindowState,
    FrameSize,
    HitZone,
    RememberedFrame,
    drag_moved,
    edges_from_points,
    fit_image,
    hit_test,
    image_to_widget,
    intersect_edges,
    mismatch_lines,
    nudge_edges,
    resized_frame_size,
    resize_edges,
    scale_edges,
    widget_to_image,
)


def _app():
    return QApplication.instance() or QApplication(sys.argv)


def test_resized_frame_size_matches_preprocess_truncation():
    assert resized_frame_size(1920, 1080, None) == FrameSize(1920, 1080)
    assert resized_frame_size(1920, 1080, 0) == FrameSize(1920, 1080)
    assert resized_frame_size(1920, 1080, 960) == FrameSize(960, 540)
    assert resized_frame_size(100, 80, 50) == FrameSize(50, 40)
    assert resized_frame_size(0, 10, 20) == FrameSize(0, 0)


def test_scale_edges_keeps_the_same_fraction():
    edges = CropEdges(100, 20, 500, 200)
    scaled = scale_edges(edges, FrameSize(1000, 400), FrameSize(500, 200))
    assert scaled == CropEdges(50, 10, 250, 100)


def test_intersect_and_nudge_adopt_the_on_frame_part():
    frame = FrameSize(100, 80)
    hanging = CropEdges(-10, 0, 50, 40)
    assert intersect_edges(hanging, frame) == CropEdges(0, 0, 50, 40)
    assert nudge_edges(hanging, 5, 0, frame) == CropEdges(5, 0, 55, 40)
    assert intersect_edges(CropEdges(90, 0, 95, 10), FrameSize(80, 80)) is None
    assert nudge_edges(CropEdges(90, 0, 95, 10), 1, 0, FrameSize(80, 80)) is None


def test_resize_edges_stays_inside_and_keeps_one_pixel():
    frame = FrameSize(100, 80)
    edges = CropEdges(10, 10, 40, 30)
    shrunk = resize_edges(edges, HitZone.RIGHT, 10, 10, frame)
    assert shrunk == CropEdges(10, 10, 11, 30)
    full_right = resize_edges(edges, HitZone.BOTTOM_RIGHT, 1000, 1000, frame)
    assert full_right == CropEdges(10, 10, 100, 80)


def test_hit_test_corners_win_and_letterbox_is_outside_the_image():
    fit = fit_image(100, 50, 100, 50)
    edges = CropEdges(10, 10, 40, 30)
    assert hit_test(10, 10, edges, fit) == HitZone.TOP_LEFT
    assert hit_test(25, 10, edges, fit) == HitZone.TOP
    assert hit_test(25, 20, edges, fit) == HitZone.INSIDE
    assert hit_test(0, 0, edges, fit) == HitZone.OUTSIDE
    letterboxed = fit_image(100, 50, 100, 100)
    assert widget_to_image(50, 5, letterboxed, 100, 50) is None
    assert widget_to_image(50, 25, letterboxed, 100, 50) == (50, 0)


def test_drag_threshold_and_point_edges():
    assert drag_moved((0, 0), (3, 0)) is False
    assert drag_moved((0, 0), (4, 0)) is True
    assert edges_from_points(30, 10, 5, 25) == CropEdges(5, 10, 30, 25)
    assert edges_from_points(4, 4, 4, 9).valid() is False


def test_state_remembers_preview_on_open_and_not_on_the_next_video():
    state = CropWindowState()
    state.edges = CropEdges(0, 0, 10, 8)
    native = FrameSize(80, 60)
    resized = FrameSize(40, 30)
    assert state.note_preview(native, resized) is True
    assert state.remembered == RememberedFrame(native, resized)
    assert state.note_preview(FrameSize(200, 100), FrameSize(200, 100)) is False
    assert state.remembered.native == native


def test_typing_remembers_the_preview_and_does_not_check_or_undo():
    state = CropWindowState()
    preview = (FrameSize(80, 60), FrameSize(80, 60))
    state.type_edges(CropEdges(1, 2, 30, 20), preview)
    assert state.crop_enabled is False
    assert state.remembered.resized == FrameSize(80, 60)
    assert state.undo() is False


def test_gesture_checks_crop_and_undo_restores_one_step():
    state = CropWindowState()
    native, resized = FrameSize(80, 60), FrameSize(80, 60)
    state.commit_gesture(CropEdges(5, 5, 20, 15), native, resized)
    assert state.crop_enabled is True
    state.type_edges(CropEdges(6, 5, 20, 15))
    assert state.undo() is True
    assert state.edges == CropEdges(0, 0, 0, 0)
    assert state.crop_enabled is False
    assert state.remembered is None
    assert state.undo() is False


def test_clear_forgets_the_frame_and_undo_brings_it_back():
    state = CropWindowState()
    native, resized = FrameSize(80, 60), FrameSize(80, 60)
    state.commit_gesture(CropEdges(1, 1, 9, 8), native, resized)
    state.clear()
    assert state.edges == CropEdges(0, 0, 0, 0)
    assert state.crop_enabled is False
    assert state.remembered is None
    assert state.undo() is True
    assert state.edges == CropEdges(1, 1, 9, 8)
    assert state.crop_enabled is True
    assert state.remembered == RememberedFrame(native, resized)


def test_frame_width_scales_the_remembered_video_only():
    state = CropWindowState()
    state.edges = CropEdges(0, 0, 100, 50)
    state.crop_enabled = True
    state.remembered = RememberedFrame(FrameSize(200, 100), FrameSize(100, 50))
    assert state.apply_framewidth(50) is True
    assert state.edges == CropEdges(0, 0, 50, 25)
    assert state.remembered == RememberedFrame(FrameSize(200, 100), FrameSize(50, 25))
    assert state.crop_enabled is True
    assert state.undo() is False

    fresh = CropWindowState()
    fresh.edges = CropEdges(1, 1, 4, 4)
    assert fresh.apply_framewidth(20) is False
    assert fresh.edges == CropEdges(1, 1, 4, 4)


def test_mismatch_lines_keep_order():
    remembered = FrameSize(80, 60)
    rows = [
        ("a.avi", FrameSize(80, 60)),
        ("b.avi", FrameSize(40, 30)),
        ("c.avi", FrameSize(80, 60)),
        ("d.avi", FrameSize(100, 80)),
    ]
    assert mismatch_lines(remembered, rows) == (
        ("b.avi", FrameSize(40, 30)),
        ("d.avi", FrameSize(100, 80)),
    )


def test_scale_edges_noop_on_empty_frame():
    edges = CropEdges(1, 1, 2, 2)
    assert scale_edges(edges, FrameSize(0, 10), FrameSize(10, 10)) == edges


def _mouse(widget, kind: QEvent.Type, x: float, y: float) -> None:
    local = QPointF(x, y)
    buttons = (
        Qt.MouseButton.NoButton
        if kind == QEvent.Type.MouseButtonRelease
        else Qt.MouseButton.LeftButton
    )
    event = QMouseEvent(
        kind,
        local,
        local,
        Qt.MouseButton.LeftButton,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )
    if kind == QEvent.Type.MouseButtonPress:
        widget.mousePressEvent(event)
    elif kind == QEvent.Type.MouseMove:
        widget.mouseMoveEvent(event)
    else:
        widget.mouseReleaseEvent(event)


@pytest.mark.gui
def test_canvas_drag_sets_edges_and_a_click_does_not():
    _app()
    from LabGym.gui_pyside.workbenches.preprocessing.crop_window import CropWindow

    window = CropWindow()
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    window.canvas.setMinimumSize(0, 0)
    window.canvas.resize(640, 360)
    window.canvas.set_frame(frame)
    window._native = FrameSize(320, 180)
    window._resized = FrameSize(320, 180)
    fit = fit_image(320, 180, window.canvas.width(), window.canvas.height())
    start = image_to_widget(10, 5, fit)
    end = image_to_widget(40, 20, fit)

    _mouse(window.canvas, QEvent.Type.MouseButtonPress, *start)
    _mouse(window.canvas, QEvent.Type.MouseButtonRelease, *start)
    assert window.state.edges == CropEdges(0, 0, 0, 0)
    assert window.state.crop_enabled is False

    _mouse(window.canvas, QEvent.Type.MouseButtonPress, *start)
    _mouse(window.canvas, QEvent.Type.MouseMove, *end)
    _mouse(window.canvas, QEvent.Type.MouseButtonRelease, *end)
    assert window.state.edges == CropEdges(10, 5, 40, 20)
    assert window.state.crop_enabled is True
    assert window.state.remembered.resized == FrameSize(320, 180)

    window.canvas.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    )
    assert window.state.edges == CropEdges(20, 5, 50, 20)
    window._on_undo()
    assert window.state.edges == CropEdges(10, 5, 40, 20)

    window._on_clear()
    assert window.state.edges == CropEdges(0, 0, 0, 0)
    assert window.state.crop_enabled is False
    assert window.state.remembered is None
    window._on_undo()
    assert window.state.edges == CropEdges(10, 5, 40, 20)
    assert window.state.crop_enabled is True
    window.close()


def _write_video(path: Path, width: int, height: int, frames: int = 3) -> None:
    import cv2

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        (width, height),
    )
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((height, width, 3), index * 20, dtype=np.uint8)
        writer.write(image)
    writer.release()


@pytest.mark.gui
def test_show_video_remembers_edges_and_frame_width_scales_them(tmp_path: Path):
    _app()
    from LabGym.gui_pyside.workbenches.preprocessing.crop_window import CropWindow

    video = tmp_path / "clip.avi"
    _write_video(video, 80, 60)
    window = CropWindow()
    window.state.edges = CropEdges(0, 0, 10, 8)
    window.show_video(str(video), None, reset_frame=True)
    assert window.canvas.frame_size() == FrameSize(80, 60)
    assert window.remembered == RememberedFrame(FrameSize(80, 60), FrameSize(80, 60))
    assert window.state.edges == CropEdges(0, 0, 10, 8)
    window.apply_frame_width(40)
    assert window.state.edges == CropEdges(0, 0, 5, 4)
    assert window.remembered.resized == FrameSize(40, 30)
    window.show_video(str(video) + ".missing", None, reset_frame=True)
    assert window.state.edges == CropEdges(0, 0, 5, 4)
    assert window.remembered.resized == FrameSize(40, 30)
    window.close()


def test_crop_kwargs_clamps_to_the_resized_frame(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.preprocessing.crop_preview import (
        crop_kwargs_for_video,
    )

    video = tmp_path / "clip.avi"
    _write_video(video, 100, 80)
    kwargs = dict(framewidth=50, crop_frame=True, left=0, right=80, top=0, bottom=10)
    clamped = crop_kwargs_for_video(str(video), kwargs)
    assert clamped["left"] == 0
    assert clamped["right"] == 50
    assert clamped["top"] == 0
    assert clamped["bottom"] == 10
    assert kwargs["right"] == 80

    missing = dict(framewidth=50, crop_frame=True, left=90, right=100, top=0, bottom=10)
    with pytest.raises(ValueError, match="does not overlap"):
        crop_kwargs_for_video(str(video), missing)


def test_worker_reports_a_crop_that_misses_the_frame(tmp_path: Path, monkeypatch):
    _app()
    from LabGym.gui_pyside.workbenches.preprocessing.preprocess_tab import (
        _PreprocessWorker,
    )

    video = tmp_path / "clip.avi"
    _write_video(video, 100, 80)
    called = []
    monkeypatch.setattr(
        "LabGym.gui_pyside.workbenches.preprocessing.preprocess_tab.preprocess_video",
        lambda *args, **kwargs: called.append(kwargs),
    )
    worker = _PreprocessWorker(
        [str(video)],
        str(tmp_path),
        dict(framewidth=50, crop_frame=True, left=90, right=100, top=0, bottom=10),
    )
    lines = []
    worker.progress.connect(lambda _path, status, line: lines.append((status, line)))
    worker.run()
    assert called == []
    assert lines[-1][0] == "error"
    assert "does not overlap" in lines[-1][1]


@pytest.mark.gui
def test_preprocess_tab_refuses_an_invalid_crop_and_confirms_a_size_mismatch(tmp_path, monkeypatch):
    _app()
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.workbenches.preprocessing.preprocess_tab import PreprocessTab

    wide = tmp_path / "wide.avi"
    tall = tmp_path / "tall.avi"
    _write_video(wide, 80, 40)
    _write_video(tall, 40, 80)
    tab = PreprocessTab(ProjectController())
    assert tab._crop_editor.parent() is tab
    assert tab._crop_editor.current_path() is None

    tab._extra.extend([str(wide), str(tall)])
    tab.refresh_videos()
    assert tab._crop_editor.current_path() == str(wide)
    assert tab._highlighted_path() == str(wide)
    assert tab.table.parent() is not tab._crop_editor
    tab.ed_out.setText(str(tmp_path))
    tab.chk_crop.setChecked(True)
    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args[2]),
    )
    started = []
    monkeypatch.setattr(tab, "_start_batch", lambda *args: started.append(args))
    tab._run()
    assert started == []
    assert any("positive width" in text for text in warnings)

    tab.spin_left.setValue(0)
    tab.spin_top.setValue(0)
    tab.spin_right.setValue(20)
    tab.spin_bottom.setValue(10)
    editor = tab._ensure_crop_editor()
    editor.state.remembered = RememberedFrame(FrameSize(80, 40), FrameSize(80, 40))
    editor.state.edges = tab._crop_edges()
    editor.state.crop_enabled = True

    answers = []

    def _cancel(remembered, rows):
        answers.append((remembered, tuple(path for path, _size in rows)))
        return False

    monkeypatch.setattr(tab, "_confirm_size_mismatch", _cancel)
    tab._run()
    assert started == []
    assert answers[0][0] == FrameSize(80, 40)
    assert Path(answers[0][1][0]).name == "tall.avi"

    monkeypatch.setattr(tab, "_confirm_size_mismatch", lambda *_args: True)
    tab._run()
    assert len(started) == 1
    assert started[0][2]["crop_frame"] is True
    assert started[0][2]["right"] == 20
    tab.close()
