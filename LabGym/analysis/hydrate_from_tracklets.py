"""Fill AnalyzeAnimalDetector state from remapped tracklets (no re-detect)."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from LabGym.id_review.samples import analysis_frame_to_video_frame
from LabGym.id_review.raw_store import load_kind_stores
from LabGym.tools import (
    extract_blob_background,
    generate_patternimage,
    generate_patternimage_interact,
    get_inner,
)

ProgressCb = Optional[Callable[[str], None]]
FrameProgressCb = Optional[Callable[[int, int], None]]


def load_remapped_stores(directory: str | Path) -> Dict[str, Any]:
    """Load published remapped tracklets from an identity package root.

    Args:
        directory: Identity package directory containing remapped ``*_tracklets.npz``.

    Returns:
        Mapping of animal kind to ``TrackletStore``.
    """
    return load_kind_stores(directory)


def kinds_and_counts(stores: Dict[str, Any]) -> Dict[str, int]:
    """Map each remapped kind to a prepare_analysis slot count.

    Empty kinds (no track IDs) still get one slot so the kind stays in the
    analyzer kind set.

    Args:
        stores: Remapped stores keyed by animal kind.

    Returns:
        Animal kind to slot count (at least 1 per kind).
    """
    return {kind: max(1, len(store.ids)) for kind, store in stores.items()}


def resolve_package_kinds(
    stores: Dict[str, Any],
    requested: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return remapped-package kinds; error if a requested kind is missing.

    The remapped package is the kind set. An empty kind stays. A missing
    kind (asked for, not in the package) raises. Package kinds are never
    dropped.

    Args:
        stores: Remapped stores keyed by animal kind.
        requested: Optional kinds a caller asked to analyze.

    Returns:
        Package kinds in store order.

    Raises:
        ValueError: A requested kind is not in the remapped package.
    """
    package = list(stores.keys())
    missing = [kind for kind in (requested or ()) if kind not in stores]
    if missing:
        raise ValueError(f"missing kind(s) not in remapped tracklets: {missing}")
    return package


def fill_geometry_from_stores(analyzer: Any, stores: Dict[str, Any]) -> None:
    """Copy remapped centers/contours/heights into analyzer per-ID series.

    Args:
        analyzer: Prepared ``AnalyzeAnimalDetector`` (or a compatible stub).
        stores: Remapped stores keyed by animal kind.

    Raises:
        ValueError: A remapped kind has no slot on the analyzer.
    """
    centers = getattr(analyzer, "animal_centers", None) or {}
    for kind, store in stores.items():
        if kind not in centers:
            raise ValueError(
                f"Remapped kind {kind!r} cannot be loaded into the analyzer"
            )
        n = min(int(store.n_frames), int(analyzer.total_analysis_framecount))
        for tid in store.ids:
            tid = int(tid)
            if tid not in analyzer.animal_centers[kind]:
                _allocate_id_slot(analyzer, kind, tid)
            row = store.id_index(tid)
            last = (-10000, -10000)
            for f in range(n):
                if store.valid[row, f]:
                    c = store.centers[row, f]
                    center = (float(c[0]), float(c[1]))
                    analyzer.animal_centers[kind][tid][f] = center
                    last = center
                    cnt = store.contours[row][f]
                    analyzer.animal_contours[kind][tid][f] = cnt
                    h = store.heights[row, f]
                    analyzer.animal_heights[kind][tid][f] = (
                        None if h is None or np.isnan(h) else float(h)
                    )
                    if getattr(analyzer, "register_counts", None) is not None:
                        analyzer.register_counts[kind][tid] = f
            if hasattr(analyzer, "animal_existingcenters"):
                analyzer.animal_existingcenters[kind][tid] = last


def _allocate_id_slot(analyzer: Any, kind: str, tid: int) -> None:
    n = int(analyzer.total_analysis_framecount)
    analyzer.animal_centers[kind][tid] = [None] * n
    analyzer.animal_contours[kind][tid] = [None] * n
    analyzer.animal_heights[kind][tid] = [None] * n
    if hasattr(analyzer, "animal_existingcenters"):
        analyzer.animal_existingcenters[kind][tid] = (-10000, -10000)
    if hasattr(analyzer, "to_deregister"):
        analyzer.to_deregister[kind][tid] = 0
    if hasattr(analyzer, "register_counts"):
        analyzer.register_counts[kind][tid] = None
    if getattr(analyzer, "animation_analyzer", False):
        analyzer.animations[kind][tid] = [
            np.zeros(
                (analyzer.length, analyzer.dim_tconv, analyzer.dim_tconv, analyzer.channel),
                dtype="uint8",
            )
        ] * n
    if getattr(analyzer, "include_bodyparts", False):
        inners = getattr(analyzer, "animal_inners", None)
        if inners is not None:
            inners.setdefault(kind, {})[tid] = []
        if int(getattr(analyzer, "behavior_mode", 0) or 0) == 2:
            other = getattr(analyzer, "animal_other_inners", None)
            if other is not None:
                other.setdefault(kind, {})[tid] = []
    analyzer.pattern_images[kind][tid] = [
        np.zeros((analyzer.dim_conv, analyzer.dim_conv, 3), dtype="uint8")
    ] * n


def rebuild_categorizer_inputs(
    analyzer: Any,
    *,
    video_path: str,
    store_meta: Optional[dict] = None,
    background_free: bool = True,
    black_background: bool = True,
    progress: ProgressCb = None,
    frame_progress: FrameProgressCb = None,
) -> None:
    """Rebuild pattern images and rolling animation clips from remapped outlines.

    Animations at each analysis frame are the last ``length`` blobs (zeros
    when that ID is absent). Body-part inners are recomputed when
    ``analyzer.include_bodyparts`` is true.

    Args:
        analyzer: Prepared analyzer already filled with remapped geometry.
        video_path: Source video used to crop blobs and inners.
        store_meta: Tracklet meta for analysis-frame to video-frame mapping.
        background_free: Whether blob extraction strips background pixels.
        black_background: Whether stripped background is black (else white).
        progress: Optional text progress callback.
        frame_progress: Optional ``(current, total)`` frame callback.

    Raises:
        FileNotFoundError: The video cannot be opened.
        RuntimeError: A required video frame cannot be read.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video for hydrate: {video_path}")
    n = int(analyzer.total_analysis_framecount)
    fps = float(getattr(analyzer, "fps", 0) or 0)
    meta = dict(store_meta or {})
    length = int(getattr(analyzer, "length", 15) or 15)
    frame_roll: Deque[np.ndarray] = deque(maxlen=length)
    inner_rolls = _empty_inner_rolls(analyzer, length)
    try:
        for f in range(n):
            if progress and f % 50 == 0:
                progress(f"Rebuilding categorizer inputs {f}/{n}…")
            v_idx = analysis_frame_to_video_frame(meta, f, fps or 1.0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(v_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(
                    f"Cannot read video frame {int(v_idx)} while hydrating {video_path}"
                )
            if analyzer.framewidth is not None:
                h, w = frame.shape[:2]
                fw = int(analyzer.framewidth)
                if w != fw and fw > 0:
                    frame = cv2.resize(
                        frame,
                        (fw, int(h * fw / w)),
                        interpolation=cv2.INTER_AREA,
                    )
            frame_roll.append(frame)
            _fill_frame_features(
                analyzer,
                frame,
                f,
                background_free=background_free,
                black_background=black_background,
                frame_roll=frame_roll,
                inner_rolls=inner_rolls,
            )
            if frame_progress:
                frame_progress(f + 1, n)
    finally:
        cap.release()


def _empty_inner_rolls(
    analyzer: Any, length: int
) -> Dict[Tuple[str, int], Deque[Any]]:
    rolls: Dict[Tuple[str, int], Deque[Any]] = {}
    if not getattr(analyzer, "include_bodyparts", False):
        return rolls
    for kind in getattr(analyzer, "animal_kinds", []) or []:
        for tid in analyzer.animal_centers.get(kind, {}):
            rolls[(kind, int(tid))] = deque(maxlen=length)
    return rolls


def _inner_from_outline(frame: np.ndarray, contour: Any) -> Any:
    """Recompute body-part inners from a remapped outline and video frame."""
    if contour is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [np.asarray(contour, dtype=np.int32)], 0, 255, -1)
    masked = (gray * (mask > 0)).astype(np.uint8)
    return get_inner(masked, contour)


def _normalized_blob(
    blob: np.ndarray,
    *,
    dim: int,
    channel: int,
) -> np.ndarray:
    blob = np.asarray(blob, dtype="uint8")
    if blob.ndim == 2:
        blob = blob[:, :, None]
    if channel == 1 and blob.shape[-1] != 1:
        blob = cv2.cvtColor(blob, cv2.COLOR_BGR2GRAY)
        blob = blob[:, :, None]
    if blob.shape[0] != dim or blob.shape[1] != dim:
        blob = cv2.resize(blob, (dim, dim), interpolation=cv2.INTER_AREA)
        if blob.ndim == 2:
            blob = blob[:, :, None]
    return blob


def _fill_frame_features(
    analyzer: Any,
    frame: np.ndarray,
    frame_idx: int,
    *,
    background_free: bool,
    black_background: bool,
    frame_roll: Deque[np.ndarray],
    inner_rolls: Dict[Tuple[str, int], Deque[Any]],
) -> None:
    length = int(getattr(analyzer, "length", 15) or 15)
    mode = int(getattr(analyzer, "behavior_mode", 0) or 0)
    dt = int(getattr(analyzer, "dim_tconv", 32) or 32)
    ch = int(getattr(analyzer, "channel", 1) or 1)
    include_bodyparts = bool(getattr(analyzer, "include_bodyparts", False))
    zeros_blob = np.zeros((dt, dt, ch), dtype="uint8")
    for kind in analyzer.animal_kinds:
        ids = list(analyzer.animal_centers.get(kind, {}))
        if include_bodyparts:
            for tid in ids:
                contours = analyzer.animal_contours[kind][tid]
                contour = contours[frame_idx] if frame_idx < len(contours) else None
                inner = _inner_from_outline(frame, contour)
                roll = inner_rolls.get((kind, int(tid)))
                if roll is not None:
                    roll.append(inner)
                stored = getattr(analyzer, "animal_inners", {}).get(kind, {})
                if tid in stored:
                    stored[tid].append(inner)
        for tid in ids:
            contours = analyzer.animal_contours[kind][tid]
            window = contours[max(0, frame_idx - length + 1) : frame_idx + 1]
            if any(c is not None for c in window):
                inners = (
                    list(inner_rolls[(kind, int(tid))])
                    if include_bodyparts and (kind, int(tid)) in inner_rolls
                    else None
                )
                if mode == 2:
                    other_seq = []
                    other_inners = [] if include_bodyparts else None
                    for w_i, _c in enumerate(window):
                        src_f = max(0, frame_idx - length + 1) + w_i
                        row = []
                        inner_row = []
                        for oid in ids:
                            if oid == tid:
                                continue
                            oc = analyzer.animal_contours[kind][oid][src_f]
                            if oc is not None:
                                row.append(oc)
                            if include_bodyparts:
                                oid_roll = inner_rolls.get((kind, int(oid)))
                                if oid_roll is not None and w_i < len(oid_roll):
                                    oi = list(oid_roll)[w_i]
                                    if oi is not None:
                                        inner_row.append(oi)
                        other_seq.append(row)
                        if include_bodyparts:
                            other_inners.append(inner_row or [None])
                    if include_bodyparts:
                        stored_o = getattr(analyzer, "animal_other_inners", {}).get(
                            kind, {}
                        )
                        if tid in stored_o:
                            stored_o[tid].append(
                                other_inners[-1] if other_inners else [None]
                            )
                    pattern = generate_patternimage_interact(
                        frame,
                        window,
                        other_seq,
                        inners=inners,
                        other_inners=other_inners,
                        std=int(getattr(analyzer, "std", 0) or 0),
                    )
                else:
                    pattern = generate_patternimage(
                        frame,
                        window,
                        inners=inners,
                        std=int(getattr(analyzer, "std", 0) or 0),
                    )
                dim = int(analyzer.dim_conv)
                if pattern is not None and pattern.size:
                    if pattern.shape[0] != dim or pattern.shape[1] != dim:
                        pattern = cv2.resize(
                            pattern, (dim, dim), interpolation=cv2.INTER_AREA
                        )
                    analyzer.pattern_images[kind][tid][frame_idx] = np.asarray(
                        pattern, dtype="uint8"
                    )
            if not getattr(analyzer, "animation_analyzer", False):
                continue
            hist = list(frame_roll)
            pad = length - len(hist)
            clip = []
            for i in range(length):
                if i < pad:
                    clip.append(zeros_blob.copy())
                    continue
                hist_i = i - pad
                src_f = max(0, frame_idx - length + 1) + hist_i
                cnt = contours[src_f] if src_f < len(contours) else None
                if cnt is None:
                    clip.append(zeros_blob.copy())
                    continue
                blob = extract_blob_background(
                    hist[hist_i],
                    window,
                    contour=cnt,
                    channel=int(analyzer.channel),
                    background_free=background_free,
                    black_background=black_background,
                )
                clip.append(_normalized_blob(blob, dim=dt, channel=ch))
            analyzer.animations[kind][tid][frame_idx] = np.stack(clip, axis=0)
