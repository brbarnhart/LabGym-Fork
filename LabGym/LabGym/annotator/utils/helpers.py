"""Small shared helpers (color conversion, frame formatting, QImage, etc.)."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def format_timecode(frame: int, fps: float, show_frames: bool = True) -> str:
    total_sec = frame / max(fps, 1e-6)
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    f = int(frame)
    if h > 0:
        base = f"{h:02d}:{m:02d}:{s:02d}"
    else:
        base = f"{m:02d}:{s:02d}"
    if show_frames:
        return f"{base}.{f:05d}f"
    return base


def numpy_to_qpixmap(frame: np.ndarray) -> QPixmap:
    """Convert HxWx3 RGB uint8 numpy array to QPixmap.
    Returns a valid (possibly null) QPixmap; never raises for bad input in a way that leaks to setPixmap.
    """
    if frame is None:
        return QPixmap()

    # Ensure we have a proper RGB image
    if frame.ndim != 3:
        # Try to handle grayscale etc. gracefully
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        else:
            return QPixmap()

    if frame.shape[2] == 4:
        frame = frame[:, :, :3]  # drop alpha
    elif frame.shape[2] != 3:
        return QPixmap()

    # Ensure contiguous uint8
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8, copy=False)
    frame = np.ascontiguousarray(frame)

    h, w, _ = frame.shape
    bytes_per_line = 3 * w
    qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    pix = QPixmap.fromImage(qimg.copy())
    return pix if not pix.isNull() else QPixmap()
