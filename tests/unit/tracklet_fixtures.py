"""Shared TrackletStore builders for packed / session-load unit tests."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from LabGym.id_review.types import SCHEMA_VERSION, TrackletStore


def contour_square(x: int, y: int, size: int = 4) -> np.ndarray:
    """Axis-aligned square contour at ``(x, y)``."""
    return np.array(
        [
            [[x, y]],
            [[x + size, y]],
            [[x + size, y + size]],
            [[x, y + size]],
        ],
        dtype=np.int32,
    )


def make_tracklet_store(
    *,
    n_frames: int = 8,
    kind: str = "mouse",
    ids: Iterable[int] = (0, 1),
    size: int = 4,
    x_scale: float = 40.0,
    y_scale: float = 1.0,
    video: str = "clip.avi",
    fps: float = 10.0,
    meta: Optional[dict] = None,
) -> TrackletStore:
    """Build an in-memory store with one square contour per id/frame.

    Args:
        n_frames: Analysis-frame count.
        kind: Animal kind stored on the object.
        ids: Track ids (row order).
        size: Square contour side in pixels.
        x_scale: ``center_x = tid * x_scale + frame``.
        y_scale: ``center_y = tid * y_scale``.
        video: ``meta['video']`` when *meta* is omitted.
        fps: ``meta['fps']`` when *meta* is omitted.
        meta: Optional store metadata override.

    Returns:
        A store with expanded contour lists (save will pack them).
    """
    id_list = [int(i) for i in ids]
    n_ids = len(id_list)
    centers = np.zeros((n_ids, n_frames, 2), dtype=np.float64)
    valid = np.ones((n_ids, n_frames), dtype=bool)
    heights = np.full((n_ids, n_frames), 10.0)
    contours = []
    for row, tid in enumerate(id_list):
        row_c = []
        for frame in range(n_frames):
            centers[row, frame] = [tid * x_scale + frame, tid * y_scale]
            x, y = int(centers[row, frame, 0]), int(centers[row, frame, 1])
            row_c.append(contour_square(x, y, size=size))
        contours.append(row_c)
    return TrackletStore(
        schema_version=SCHEMA_VERSION,
        animal_kind=kind,
        ids=id_list,
        n_frames=n_frames,
        centers=centers,
        valid=valid,
        heights=heights,
        contours=contours,
        meta=meta if meta is not None else {"video": video, "fps": fps},
    )
