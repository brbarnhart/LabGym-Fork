"""Unit tests for Review IDs render helpers (no real video decode required)."""

from __future__ import annotations

import sys

import numpy as np
from PySide6.QtWidgets import QApplication

from LabGym.gui_pyside.workbenches.detector.review_ids_render import (
    VideoCaptureCache,
    bgr_to_qpixmap,
    compose_preview_frame,
    format_frame_status,
    resize_if_needed,
)
from LabGym.id_review.types import SwitchMarker


def _app():
    return QApplication.instance() or QApplication(sys.argv)


def test_format_frame_status():
    s = format_frame_status(
        analysis_frame=3,
        n_frames=100,
        t_sec=0.3,
        video_idx=12,
        n_markers=2,
        in_risk=True,
        n_preview_markers=1,
    )
    assert "Analysis frame 3/99" in s
    assert "video f=12" in s
    assert "switches=2" in s
    assert "in risk=True" in s
    assert "preview markers=1" in s


def test_format_frame_status_missing_video_idx():
    s = format_frame_status(
        analysis_frame=0,
        n_frames=1,
        t_sec=0.0,
        video_idx=None,
        n_markers=0,
        in_risk=False,
        n_preview_markers=0,
    )
    assert "video f=—" in s


def test_resize_if_needed():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert resize_if_needed(frame, None).shape == (100, 200, 3)
    out = resize_if_needed(frame, 100)
    assert out.shape[1] == 100
    assert out.shape[0] == 50


def test_compose_preview_no_store():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    markers = [
        SwitchMarker(
            marker_id="s1",
            frame=5,
            animal_kind="mouse",
            involved_ids=[0, 1],
        )
    ]
    out, n = compose_preview_frame(
        frame,
        store=None,
        markers=markers,
        animal_kind="mouse",
        analysis_frame=10,
        already_corrected=False,
        highlight_ids=[0, 1],
    )
    assert n == 1
    assert out is frame


def test_bgr_to_qpixmap():
    _app()
    arr = np.zeros((60, 80, 3), dtype=np.uint8)
    arr[:, :] = (0, 0, 255)  # red in BGR
    pix = bgr_to_qpixmap(arr, max_w=40, max_h=40)
    assert not pix.isNull()
    assert pix.width() <= 40
    assert pix.height() <= 40


def test_video_capture_cache_missing_path():
    cache = VideoCaptureCache()
    assert cache.ensure("Z:/no/such/video.avi") is False
    assert cache.path is None
    cache.release()
