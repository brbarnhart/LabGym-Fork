"""Video capture + overlay rendering for Review IDs preview."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap

from LabGym.id_review.samples import (
    analysis_frame_to_video_frame,
    detections_at_frame_after_markers,
    draw_detections_overlay,
)
from LabGym.id_review.types import SwitchMarker


class VideoCaptureCache:
    """Own a single ``cv2.VideoCapture`` and reopen when the path changes."""

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._path: Optional[str] = None

    @property
    def path(self) -> Optional[str]:
        return self._path

    def ensure(self, path: str) -> bool:
        if self._cap is not None and self._path == path:
            return True
        self.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        self._cap = cap
        self._path = path
        return True

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._path = None

    def read_at_video_index(self, v_idx: int) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, int(v_idx))
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame


def resize_if_needed(frame: np.ndarray, framewidth: Optional[int]) -> np.ndarray:
    if framewidth is None:
        return frame
    try:
        fw = int(framewidth)
        h, w = frame.shape[:2]
        if w != fw and fw > 0:
            return cv2.resize(
                frame, (fw, int(h * fw / w)), interpolation=cv2.INTER_AREA
            )
    except Exception:
        pass
    return frame


def compose_preview_frame(
    frame: np.ndarray,
    *,
    store: Any,
    markers: Sequence[SwitchMarker],
    animal_kind: str,
    analysis_frame: int,
    already_corrected: bool,
    highlight_ids: Sequence[int],
) -> Tuple[np.ndarray, int]:
    """Draw tracklet overlays; return (bgr_frame, n_markers_applied_for_preview)."""
    applied = [
        m
        for m in markers
        if m.frame <= analysis_frame and m.animal_kind == animal_kind
    ]
    if store is None:
        return frame, len(applied)
    if already_corrected:
        dets = detections_at_frame_after_markers(store, analysis_frame, [])
        n_prev = 0
    else:
        dets = detections_at_frame_after_markers(store, analysis_frame, applied)
        n_prev = len(applied)
    out = draw_detections_overlay(
        frame,
        dets,
        highlight_ids=list(highlight_ids),
        frame_idx=analysis_frame,
        n_markers_applied=n_prev,
    )
    return out, n_prev


def read_preview_frame(
    cache: VideoCaptureCache,
    *,
    video_path: str,
    store_meta: dict,
    analysis_frame: int,
    fps: float,
) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """Return ``(bgr, video_frame_index, error_message)``."""
    if not video_path or not cache.ensure(str(video_path)):
        return None, None, f"video unavailable: {video_path}"
    v_idx = analysis_frame_to_video_frame(store_meta, analysis_frame, fps)
    frame = cache.read_at_video_index(v_idx)
    if frame is None:
        return None, v_idx, f"Failed to read video frame {v_idx}"
    fw = store_meta.get("framewidth")
    frame = resize_if_needed(frame, int(fw) if fw is not None else None)
    return frame, v_idx, None


def bgr_to_qpixmap(arr: np.ndarray, max_w: int = 900, max_h: int = 540) -> QPixmap:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    rgb = np.ascontiguousarray(
        cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    )
    qimg = QImage(rgb.data, nw, nh, nw * 3, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def format_frame_status(
    *,
    analysis_frame: int,
    n_frames: int,
    t_sec: float,
    video_idx: Optional[int],
    n_markers: int,
    in_risk: bool,
    n_preview_markers: int,
) -> str:
    vtxt = f"video f={video_idx}" if video_idx is not None else "video f=—"
    return (
        f"Analysis frame {analysis_frame}/{max(0, n_frames - 1)}  |  "
        f"t={t_sec:.2f}s  |  {vtxt}  |  switches={n_markers}  |  "
        f"in risk={in_risk}  |  preview markers={n_preview_markers}"
    )
