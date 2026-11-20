"""Read and resize a source frame for the crop-window editor.

The resize matches ``preprocess_video``: width is the requested frame width
and height is ``int(framewidth * native_height / native_width)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from LabGym.gui_pyside.workbenches.preprocessing.crop_geometry import (
    CropEdges,
    FrameSize,
    intersect_edges,
    resized_frame_size,
)


@dataclass(frozen=True)
class ProbedVideo:
    """Size and timing reported by opening a video."""

    width: int
    height: int
    frame_count: int
    fps: float


def probe_video(path: str) -> Optional[ProbedVideo]:
    """Open ``path`` and return its native size, frame count, and FPS.

    Args:
        path: Video file to probe.

    Returns:
        Probe data, or None when the file cannot be opened.
    """
    capture = cv2.VideoCapture(path)
    try:
        if not capture.isOpened():
            return None
        return ProbedVideo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        )
    finally:
        capture.release()


def crop_kwargs_for_video(path: str, kwargs: dict) -> dict:
    """Return ``kwargs`` with the crop window clamped to ``path``.

    When cropping is off, the copy is unchanged. When it is on, Left / Right /
    Top / Bottom become the overlap with that video's resized frame.

    Args:
        path: Video about to be preprocessed.
        kwargs: Arguments for ``preprocess_video``.

    Returns:
        A new dict safe to pass to ``preprocess_video``.

    Raises:
        ValueError: Cropping is on and the crop window misses this frame.
    """
    params = dict(kwargs)
    if not params.get("crop_frame"):
        return params
    info = probe_video(path)
    if info is None:
        return params
    frame = resized_frame_size(info.width, info.height, params.get("framewidth"))
    edges = CropEdges(
        int(params.get("left", 0)),
        int(params.get("top", 0)),
        int(params.get("right", 0)),
        int(params.get("bottom", 0)),
    )
    clamped = intersect_edges(edges, frame)
    if clamped is None:
        raise ValueError("crop window does not overlap this frame")
    params["left"] = clamped.left
    params["top"] = clamped.top
    params["right"] = clamped.right
    params["bottom"] = clamped.bottom
    return params


def resize_frame(frame: np.ndarray, framewidth: Optional[int]) -> np.ndarray:
    """Resize ``frame`` the way preprocess resizes before cropping.

    Args:
        frame: BGR frame.
        framewidth: Target width, or None / 0 to keep the native size.

    Returns:
        The resized frame. The input is returned when no resize is needed.
    """
    height, width = frame.shape[:2]
    size = resized_frame_size(width, height, framewidth)
    if size.width == width and size.height == height:
        return frame
    if size.width <= 0 or size.height <= 0:
        return frame
    return cv2.resize(
        frame,
        (size.width, size.height),
        interpolation=cv2.INTER_AREA,
    )


class PreviewSession:
    """One open video used to scrub frames in the crop window."""

    def __init__(self) -> None:
        self._capture: Optional[cv2.VideoCapture] = None
        self.path: Optional[str] = None
        self.info: Optional[ProbedVideo] = None
        self.error: Optional[str] = None

    def close(self) -> None:
        """Release the open video."""
        if self._capture is not None:
            self._capture.release()
        self._capture = None
        self.path = None
        self.info = None

    def open(self, path: str) -> bool:
        """Open ``path`` when it is not already the current video.

        Args:
            path: Video file to show.

        Returns:
            True when a frame can be read.
        """
        if self._capture is not None and self.path == path and self.info is not None:
            self.error = None
            return True
        self.close()
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            self.error = f"Could not read {path}"
            return False
        self._capture = capture
        self.path = path
        self.error = None
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.info = ProbedVideo(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=max(0, count),
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        )
        return True

    def read(self, index: int) -> Optional[np.ndarray]:
        """Return the BGR frame at ``index``, or None when the read fails."""
        if self._capture is None:
            return None
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return None
        return frame

    def read_resized(
        self,
        index: int,
        framewidth: Optional[int],
    ) -> tuple[Optional[np.ndarray], Optional[FrameSize], Optional[FrameSize]]:
        """Return a resized frame plus its native and resized sizes.

        Args:
            index: Zero-based source frame index.
            framewidth: Target width, or None / 0 to keep the native size.

        Returns:
            ``(frame, native, resized)``. The frame is None when the read fails.
            Sizes are None in that case too.
        """
        raw = self.read(index)
        if raw is None:
            return None, None, None
        native_h, native_w = raw.shape[:2]
        shown = resize_frame(raw, framewidth)
        shown_h, shown_w = shown.shape[:2]
        return shown, FrameSize(native_w, native_h), FrameSize(shown_w, shown_h)
