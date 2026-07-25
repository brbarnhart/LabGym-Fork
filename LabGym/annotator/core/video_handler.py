"""VideoHandler: abstraction over video backends for frame-accurate access.

Primary recommendation: decord (fast + accurate random access).
Fallback: opencv-python (cv2).

Important: OpenCV set(CAP_PROP_POS_FRAMES) is unreliable on many codecs/containers.
We therefore prefer index-based reading or decord.VideoReader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import decord
    from decord import VideoReader, cpu
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


@dataclass
class VideoMetadata:
    path: str
    fps: float
    total_frames: int
    width: int
    height: int
    duration_sec: float


class VideoHandler:
    """Thread-safe-ish frame provider. Not thread-safe internally; use from one thread or protect access."""

    def __init__(self):
        self._path: Optional[str] = None
        self._meta: Optional[VideoMetadata] = None
        self._vr: Optional["VideoReader"] = None   # decord
        self._cap: Optional["cv2.VideoCapture"] = None  # cv2
        self._backend: str = "none"

    @property
    def is_loaded(self) -> bool:
        return self._meta is not None

    @property
    def metadata(self) -> VideoMetadata:
        if self._meta is None:
            raise RuntimeError("No video loaded")
        return self._meta

    @property
    def fps(self) -> float:
        return self.metadata.fps

    @property
    def total_frames(self) -> int:
        return self.metadata.total_frames

    def load(self, path: str | Path) -> VideoMetadata:
        """Load video and return metadata. Chooses best available backend."""
        path = str(path)
        if not Path(path).exists():
            raise FileNotFoundError(f"Video not found: {path}")

        if _HAS_DECORD:
            try:
                vr = VideoReader(path, ctx=cpu(0))
                # decord gives accurate frame count via len(vr)
                nframes = len(vr)
                # fps can be in vr.get_avg_fps() or metadata
                fps = float(vr.get_avg_fps()) if hasattr(vr, "get_avg_fps") else 30.0
                # Get first frame for shape
                first = vr[0].asnumpy()
                h, w = first.shape[:2]
                meta = VideoMetadata(
                    path=path,
                    fps=fps,
                    total_frames=int(nframes),
                    width=w,
                    height=h,
                    duration_sec=float(nframes) / max(fps, 1e-6),
                )
                self._vr = vr
                self._cap = None
                self._backend = "decord"
                self._path = path
                self._meta = meta
                return meta
            except Exception as e:
                # Fall through to cv2
                print(f"[VideoHandler] decord failed ({e}), falling back to OpenCV...")

        if _HAS_CV2:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"OpenCV could not open video: {path}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

            # Some videos report 0 or incorrect metadata; try to read first frame to validate dimensions
            if w <= 0 or h <= 0 or nframes <= 0:
                ok, test_frame = cap.read()
                if ok and test_frame is not None:
                    h, w = test_frame.shape[:2]
                    if nframes <= 0:
                        nframes = 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

            # Some videos report 0 or incorrect frame count; we can count manually if needed
            if nframes <= 0:
                # Best effort: seek to end or iterate (slow). For now use a safe large number and warn.
                print("[VideoHandler] Warning: frame count unreliable. Some features may be limited.")
                nframes = 1  # will be updated on first successful read if possible

            meta = VideoMetadata(
                path=path,
                fps=float(fps),
                total_frames=max(1, nframes),
                width=w or 640,
                height=h or 480,
                duration_sec=float(max(1, nframes)) / max(fps, 1e-6),
            )
            self._cap = cap
            self._vr = None
            self._backend = "opencv"
            self._path = path
            self._meta = meta
            return meta

        raise RuntimeError("No video backend available. Install opencv-python (and preferably decord).")

    def get_frame(self, frame_num: int) -> np.ndarray:
        """Return frame as RGB uint8 ndarray (H, W, 3)."""
        if not self.is_loaded:
            raise RuntimeError("No video loaded")

        frame_num = max(0, min(frame_num, self.total_frames - 1))

        if self._backend == "decord" and self._vr is not None:
            # decord supports direct indexing
            arr = self._vr[frame_num].asnumpy()
            # decord returns RGB already in many cases
            if arr.shape[2] == 4:  # RGBA?
                arr = arr[:, :, :3]
            return arr

        if self._backend == "opencv" and self._cap is not None:
            # Seeking strategy: set + read. Known to be flaky; we accept it for fallback.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                # Fallback brute force from current (rarely needed)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                for _ in range(frame_num + 1):
                    ok, frame = self._cap.read()
                    if not ok:
                        break
            if frame is None:
                # Return black frame as last resort
                h, w = self._meta.height or 480, self._meta.width or 640
                return np.zeros((h, w, 3), dtype=np.uint8)
            # OpenCV gives BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame

        raise RuntimeError("No active backend for frame reading")

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._vr = None
        self._meta = None
        self._backend = "none"
        self._path = None

    def frame_to_seconds(self, frame: int) -> float:
        return frame / max(self.fps, 1e-6)

    def seconds_to_frame(self, seconds: float) -> int:
        return int(round(seconds * self.fps))

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
