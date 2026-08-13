"""Fill AnalyzeAnimalDetector state from remapped tracklets (no re-detect)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

from LabGym.id_review.samples import analysis_frame_to_video_frame
from LabGym.id_review.tracklets import load_tracklets
from LabGym.annotator.core.tracklets_bridge import discover_tracklet_kinds
from LabGym.tools import (
    extract_blob_background,
    generate_patternimage,
    generate_patternimage_interact,
)

ProgressCb = Optional[Callable[[str], None]]


def load_remapped_stores(directory: str | Path) -> Dict[str, Any]:
    """Load published remapped tracklets from an identity package root."""
    directory = Path(directory)
    kinds = discover_tracklet_kinds(directory)
    return {kind: load_tracklets(str(directory), kind) for kind in kinds}


def kinds_and_counts(stores: Dict[str, Any]) -> Dict[str, int]:
    """Animal kind → number of track IDs (for prepare_analysis slots)."""
    return {kind: max(1, len(store.ids)) for kind, store in stores.items()}


def fill_geometry_from_stores(analyzer, stores: Dict[str, Any]) -> None:
    """Copy remapped centers/contours/heights into analyzer per-ID series."""
    for kind, store in stores.items():
        if kind not in getattr(analyzer, "animal_centers", {}):
            continue
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


def _allocate_id_slot(analyzer, kind: str, tid: int) -> None:
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
    analyzer.pattern_images[kind][tid] = [
        np.zeros((analyzer.dim_conv, analyzer.dim_conv, 3), dtype="uint8")
    ] * n


def rebuild_categorizer_inputs(
    analyzer,
    *,
    video_path: str,
    store_meta: Optional[dict] = None,
    background_free: bool = True,
    black_background: bool = True,
    progress: ProgressCb = None,
) -> None:
    """Rebuild pattern images (and animations) from remapped outlines + video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video for hydrate: {video_path}")
    n = int(analyzer.total_analysis_framecount)
    fps = float(getattr(analyzer, "fps", 0) or 0)
    meta = dict(store_meta or {})
    try:
        for f in range(n):
            if progress and f % 50 == 0:
                progress(f"Rebuilding categorizer inputs {f}/{n}…")
            v_idx = analysis_frame_to_video_frame(meta, f, fps or 1.0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(v_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if analyzer.framewidth is not None:
                h, w = frame.shape[:2]
                fw = int(analyzer.framewidth)
                if w != fw and fw > 0:
                    frame = cv2.resize(
                        frame,
                        (fw, int(h * fw / w)),
                        interpolation=cv2.INTER_AREA,
                    )
            _fill_frame_features(
                analyzer,
                frame,
                f,
                background_free=background_free,
                black_background=black_background,
            )
    finally:
        cap.release()


def _fill_frame_features(
    analyzer,
    frame: np.ndarray,
    frame_idx: int,
    *,
    background_free: bool,
    black_background: bool,
) -> None:
    length = int(getattr(analyzer, "length", 15) or 15)
    mode = int(getattr(analyzer, "behavior_mode", 0) or 0)
    for kind in analyzer.animal_kinds:
        ids = list(analyzer.animal_centers.get(kind, {}))
        for tid in ids:
            contours = analyzer.animal_contours[kind][tid]
            window = contours[max(0, frame_idx - length + 1) : frame_idx + 1]
            if not any(c is not None for c in window):
                continue
            if mode == 2:
                others = []
                for oid in ids:
                    if oid == tid:
                        continue
                    oc = analyzer.animal_contours[kind][oid][frame_idx]
                    if oc is not None:
                        others.append(oc)
                other_seq = []
                for w_i, _c in enumerate(window):
                    src_f = max(0, frame_idx - length + 1) + w_i
                    row = []
                    for oid in ids:
                        if oid == tid:
                            continue
                        oc = analyzer.animal_contours[kind][oid][src_f]
                        if oc is not None:
                            row.append(oc)
                    other_seq.append(row)
                try:
                    pattern = generate_patternimage_interact(
                        frame,
                        window,
                        other_seq,
                        inners=None,
                        other_inners=None,
                        std=int(getattr(analyzer, "std", 0) or 0),
                    )
                except Exception:
                    pattern = generate_patternimage(
                        frame, window, inners=None, std=0
                    )
            else:
                pattern = generate_patternimage(
                    frame,
                    window,
                    inners=None,
                    std=int(getattr(analyzer, "std", 0) or 0),
                )
            dim = int(analyzer.dim_conv)
            if pattern is not None and pattern.size:
                if pattern.shape[0] != dim or pattern.shape[1] != dim:
                    pattern = cv2.resize(pattern, (dim, dim), interpolation=cv2.INTER_AREA)
                analyzer.pattern_images[kind][tid][frame_idx] = np.asarray(
                    pattern, dtype="uint8"
                )
            if not getattr(analyzer, "animation_analyzer", False):
                continue
            contour = contours[frame_idx]
            if contour is None:
                continue
            try:
                blob = extract_blob_background(
                    frame,
                    window,
                    contour=contour,
                    channel=int(analyzer.channel),
                    background_free=background_free,
                    black_background=black_background,
                )
                blob = np.asarray(blob, dtype="uint8")
                dt = int(analyzer.dim_tconv)
                if blob.ndim == 2:
                    blob = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
                if blob.shape[0] != dt or blob.shape[1] != dt:
                    blob = cv2.resize(blob, (dt, dt), interpolation=cv2.INTER_AREA)
                ch = int(analyzer.channel)
                if ch == 1 and blob.ndim == 3:
                    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2GRAY)
                    blob = blob.reshape(dt, dt, 1)
                # Stack last ``length`` blobs; fall back to repeating this blob.
                stack = np.stack([blob] * length, axis=0)
                analyzer.animations[kind][tid][frame_idx] = stack
            except Exception:
                continue
