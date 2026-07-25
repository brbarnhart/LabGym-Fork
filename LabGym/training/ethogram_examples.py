"""Generate LabGym training pairs (animation .avi + pattern .jpg) FROM ethograms.

Ethogram-first workflow:
  fixed tracklets + annotations.json → sorted behavior folders of examples

Does not re-detect. Uses corrected TrackletStore contours and video frames.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from LabGym.annotator.core.data_models import (
    BEHAVIOR_MODE_INTERACTIVE_ADVANCED,
    BEHAVIOR_MODE_INTERACTIVE_BASIC,
    BEHAVIOR_MODE_NON_INTERACTIVE,
    AnnotationSession,
    Bout,
)
from LabGym.annotator.core.tracklets_bridge import (
    LoadedTracklets,
    load_tracklets_for_annotator,
    video_to_analysis_frame,
)
from LabGym.id_review.types import TrackletStore
from LabGym.tools import (
    crop_frame,
    extract_blob_all,
    extract_blob_background,
    generate_patternimage,
    generate_patternimage_all,
    generate_patternimage_interact,
)
from LabGym.training.soft_labels import (
    SoftLabelTable,
    build_soft_targets_for_window,
    dense_frame_labels_from_session,
)

ProgressCb = Optional[Callable[[int, int, str], None]]


@dataclass
class GenerationConfig:
    """Parameters for ethogram-driven example generation (persisted to JSON)."""

    video_path: str
    annotations_path: str
    tracklets_dir: str
    output_dir: str
    length: int = 15
    behavior_mode: int = 0
    sampling: str = "dense_in_bout"  # bout_end | bout_center | dense_in_bout | coverage
    stride: int = 0  # 0 → max(1, length // 3)
    min_bout_frames: int = 1
    max_invalid_fraction: float = 0.5
    background_free: bool = True
    black_background: bool = True
    channel: int = 3
    social_distance: float = 0.0  # folds of animal size; 0 = all others (mode 2)
    animal_size: float = 50.0  # px, used with social_distance
    color_costar: bool = False
    analysis_start_frame: Optional[int] = None
    write_soft_labels: bool = True
    fps_write: Optional[float] = None  # None → video fps / 5 like LabGym

    def resolved_stride(self) -> int:
        if self.stride and self.stride > 0:
            return int(self.stride)
        return max(1, int(self.length) // 3)


@dataclass
class WindowSpec:
    """One training window to materialize."""

    behavior: str
    subject_id: Optional[int]  # None for group mode
    center_frame: int  # video frame index (end of window for end-aligned)
    start_frame: int  # inclusive video frame
    end_frame: int  # inclusive video frame (== center for end-aligned length L)
    partner_ids: List[int] = field(default_factory=list)


def sample_windows_from_bout(
    bout: Bout,
    behavior: str,
    subject_id: Optional[int],
    length: int,
    sampling: str,
    stride: int,
    min_bout_frames: int,
    total_frames: int,
) -> List[WindowSpec]:
    """Sample end-aligned (or centered) windows of fixed length from one bout."""
    if length < 1:
        length = 1
    b0 = max(0, int(bout.start_frame))
    b1 = min(int(total_frames) - 1, int(bout.end_frame))
    if b1 < b0:
        return []
    bout_len = b1 - b0 + 1
    if bout_len < min_bout_frames:
        return []

    partners = list(bout.partner_ids or [])
    specs: List[WindowSpec] = []

    def _end_aligned(center: int) -> Optional[WindowSpec]:
        end = int(center)
        start = end - length + 1
        if start < 0 or end >= total_frames:
            return None
        # window must intersect bout substantially: require center in bout
        if end < b0 or end > b1:
            return None
        return WindowSpec(
            behavior=behavior,
            subject_id=subject_id,
            center_frame=end,
            start_frame=start,
            end_frame=end,
            partner_ids=partners,
        )

    def _centered(mid: int) -> Optional[WindowSpec]:
        half = length // 2
        start = mid - half
        end = start + length - 1
        if start < 0:
            start = 0
            end = length - 1
        if end >= total_frames:
            end = total_frames - 1
            start = max(0, end - length + 1)
        if end - start + 1 < length:
            return None
        return WindowSpec(
            behavior=behavior,
            subject_id=subject_id,
            center_frame=end,  # LabGym names use end frame convention
            start_frame=start,
            end_frame=end,
            partner_ids=partners,
        )

    sampling = (sampling or "dense_in_bout").lower()
    if sampling == "bout_end":
        # Need full length ending at bout end; start may precede bout
        if b1 >= length - 1:
            w = _end_aligned(b1)
            if w:
                specs.append(w)
    elif sampling == "bout_center":
        mid = (b0 + b1) // 2
        w = _centered(mid)
        if w:
            specs.append(w)
    elif sampling == "coverage":
        # Non-overlapping end-aligned windows covering bout
        # First center where window ends inside bout and start <= b1
        pos = b0 + length - 1
        if pos < b0:
            pos = b0
        while pos <= b1:
            w = _end_aligned(pos)
            if w:
                specs.append(w)
            pos += length
        # ensure last bout end covered
        if not specs or specs[-1].end_frame < b1:
            w = _end_aligned(b1)
            if w and (not specs or specs[-1].center_frame != w.center_frame):
                specs.append(w)
    else:  # dense_in_bout
        # Centers from first frame where end-aligned window fully exists
        # and center in bout: center in [max(b0, length-1), b1]
        c0 = max(b0, length - 1)
        c1 = b1
        if c0 > c1:
            # bout shorter than length: still emit centered if min_bout allows
            w = _centered((b0 + b1) // 2)
            if w:
                specs.append(w)
        else:
            for c in range(c0, c1 + 1, max(1, stride)):
                w = _end_aligned(c)
                if w:
                    specs.append(w)
            if specs and specs[-1].center_frame != c1:
                w = _end_aligned(c1)
                if w:
                    specs.append(w)

    return specs


def collect_windows(
    session: AnnotationSession,
    length: int,
    sampling: str,
    stride: int,
    min_bout_frames: int,
) -> List[WindowSpec]:
    """Collect all windows from ethogram for the session behavior mode."""
    total = int(session.total_frames)
    mode = int(session.behavior_mode)
    out: List[WindowSpec] = []

    if mode == BEHAVIOR_MODE_INTERACTIVE_BASIC:
        bmap = session.interaction_bouts.get("group", {})
        for beh, blist in bmap.items():
            for bout in blist:
                out.extend(
                    sample_windows_from_bout(
                        bout,
                        beh,
                        None,
                        length,
                        sampling,
                        stride,
                        min_bout_frames,
                        total,
                    )
                )
    else:
        for subj in session.subjects:
            bmap = session.bouts_for_subject(subj.subject_id)
            for beh, blist in bmap.items():
                for bout in blist:
                    out.extend(
                        sample_windows_from_bout(
                            bout,
                            beh,
                            int(subj.subject_id),
                            length,
                            sampling,
                            stride,
                            min_bout_frames,
                            total,
                        )
                    )
    # stable order
    out.sort(key=lambda w: (w.behavior, w.subject_id if w.subject_id is not None else -1, w.center_frame))
    return out


def _contour_at(
    store: TrackletStore, track_id: int, analysis_frame: int
) -> Optional[np.ndarray]:
    if analysis_frame < 0 or analysis_frame >= store.n_frames:
        return None
    try:
        row = store.id_index(int(track_id))
    except ValueError:
        return None
    if not bool(store.valid[row, analysis_frame]):
        return None
    cnt = store.contours[row][analysis_frame]
    if cnt is None:
        return None
    return np.asarray(cnt, dtype=np.int32)


def _center_at(
    store: TrackletStore, track_id: int, analysis_frame: int
) -> Optional[Tuple[float, float]]:
    if analysis_frame < 0 or analysis_frame >= store.n_frames:
        return None
    try:
        row = store.id_index(int(track_id))
    except ValueError:
        return None
    if not bool(store.valid[row, analysis_frame]):
        return None
    c = store.centers[row, analysis_frame]
    return float(c[0]), float(c[1])


def _subject_track(
    loaded: LoadedTracklets, subject_id: int
) -> Tuple[str, int, TrackletStore]:
    kind, tid = loaded.subject_to_track[int(subject_id)]
    store = loaded.stores[kind]
    return kind, int(tid), store


def _window_contours_for_subject(
    loaded: LoadedTracklets,
    subject_id: int,
    start_v: int,
    end_v: int,
    analysis_start: int,
) -> List[Optional[np.ndarray]]:
    kind, tid, store = _subject_track(loaded, subject_id)
    out: List[Optional[np.ndarray]] = []
    for vf in range(start_v, end_v + 1):
        af = video_to_analysis_frame(vf, analysis_start)
        out.append(_contour_at(store, tid, af))
    return out


def _invalid_fraction(contours: Sequence[Optional[np.ndarray]]) -> float:
    if not contours:
        return 1.0
    bad = sum(1 for c in contours if c is None)
    return bad / float(len(contours))


def _read_frames_bgr(
    cap: cv2.VideoCapture, start: int, end: int
) -> List[Optional[np.ndarray]]:
    frames: List[Optional[np.ndarray]] = []
    # Seek once then sequential read
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for f in range(start, end + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            frames.append(None)
        else:
            frames.append(frame)
    return frames


def _blank_like(frame: np.ndarray) -> np.ndarray:
    return np.zeros_like(frame)


def _write_pair(
    out_dir: Path,
    basename: str,
    blobs: List[np.ndarray],
    pattern: np.ndarray,
    fps: float,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path_avi = out_dir / f"{basename}.avi"
    path_jpg = out_dir / f"{basename}.jpg"
    h, w = blobs[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path_avi),
        cv2.VideoWriter_fourcc(*"MJPG"),
        max(fps, 1.0),
        (w, h),
        True,
    )
    for blob in blobs:
        if blob.ndim == 2:
            bgr = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
        elif blob.shape[2] == 1:
            bgr = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
        else:
            bgr = blob
        if bgr.shape[0] != h or bgr.shape[1] != w:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(np.uint8(bgr))
    writer.release()
    cv2.imwrite(str(path_jpg), pattern)
    return path_avi, path_jpg


def _build_mode0_pair(
    frames: List[np.ndarray],
    contours: List[Optional[np.ndarray]],
    *,
    background_free: bool,
    black_background: bool,
    channel: int,
) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
    valid = [c for c in contours if c is not None]
    if not valid:
        return None
    # Use first valid frame as reference shape for pattern background
    ref = frames[0]
    for i, fr in enumerate(frames):
        if fr is not None and contours[i] is not None:
            ref = fr
            break
    blobs: List[np.ndarray] = []
    for fr, cnt in zip(frames, contours):
        if fr is None:
            fr = _blank_like(ref)
        if cnt is None:
            blob = np.zeros(
                (max(8, ref.shape[0] // 8), max(8, ref.shape[1] // 8), 3),
                dtype=np.uint8,
            )
        else:
            # crop using full window contours for stable ROI
            blob = extract_blob_background(
                fr,
                [c for c in contours if c is not None],
                contour=cnt,
                channel=3,
                background_free=background_free,
                black_background=black_background,
            )
            if channel == 1 and blob.ndim == 3:
                blob = cv2.cvtColor(blob, cv2.COLOR_BGR2GRAY)
                blob = cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR)
        blobs.append(np.uint8(blob))
    # normalize blob sizes to max in window
    mh = max(b.shape[0] for b in blobs)
    mw = max(b.shape[1] for b in blobs)
    norm = []
    for b in blobs:
        if b.shape[0] != mh or b.shape[1] != mw:
            b = cv2.resize(b, (mw, mh), interpolation=cv2.INTER_AREA)
        norm.append(b)
    pattern = generate_patternimage(ref, contours, inners=None, std=0)
    return norm, pattern


def _build_mode1_pair(
    frames: List[np.ndarray],
    contours_per_frame: List[List[np.ndarray]],
    *,
    background_free: bool,
    black_background: bool,
) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
    ref = None
    for fr in frames:
        if fr is not None:
            ref = fr
            break
    if ref is None:
        return None
    # Flatten all contours for global crop over window
    all_cnts = []
    for clist in contours_per_frame:
        all_cnts.extend(clist)
    if not all_cnts:
        return None
    y_bt, y_tp, x_lf, x_rt = crop_frame(ref, all_cnts)
    blobs = []
    outlines_list = []
    for fr, clist in zip(frames, contours_per_frame):
        if fr is None:
            fr = _blank_like(ref)
        cnts = clist if clist else None
        blob = extract_blob_all(
            fr,
            y_bt,
            y_tp,
            x_lf,
            x_rt,
            contours=cnts,
            channel=3,
            background_free=background_free,
            black_background=black_background,
        )
        blobs.append(np.uint8(blob))
        outlines_list.append(clist if clist else [])
    pattern = generate_patternimage_all(
        ref, y_bt, y_tp, x_lf, x_rt, outlines_list, None, std=0
    )
    return blobs, pattern


def _others_in_range(
    loaded: LoadedTracklets,
    main_sid: int,
    analysis_frame: int,
    analysis_start: int,
    social_distance: float,
    animal_size: float,
    partner_ids: Sequence[int],
) -> List[Tuple[int, np.ndarray]]:
    """Return (subject_id, contour) for costars at analysis_frame."""
    try:
        mkind, mtid, mstore = _subject_track(loaded, main_sid)
    except KeyError:
        return []
    main_c = _center_at(mstore, mtid, analysis_frame)
    others: List[Tuple[int, np.ndarray]] = []
    prefer = set(int(p) for p in partner_ids)
    for subj in loaded.subjects:
        if subj.subject_id == main_sid:
            continue
        kind, tid, store = _subject_track(loaded, subj.subject_id)
        cnt = _contour_at(store, tid, analysis_frame)
        if cnt is None:
            continue
        if social_distance and social_distance > 0 and main_c is not None:
            oc = _center_at(store, tid, analysis_frame)
            if oc is None:
                continue
            dist = float(np.hypot(main_c[0] - oc[0], main_c[1] - oc[1]))
            thresh = float(social_distance) * float(animal_size)
            if subj.subject_id not in prefer and dist > thresh:
                continue
        others.append((subj.subject_id, cnt))
    # Prefer partners first
    others.sort(key=lambda x: (0 if x[0] in prefer else 1, x[0]))
    return others


def _build_mode2_pair(
    frames: List[np.ndarray],
    main_contours: List[Optional[np.ndarray]],
    other_contours_seq: List[List[np.ndarray]],
    *,
    background_free: bool,
    black_background: bool,
    color_costar: bool,
) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
    ref = None
    for i, fr in enumerate(frames):
        if fr is not None and main_contours[i] is not None:
            ref = fr
            break
    if ref is None:
        return None
    # Crop using main + others over window
    all_c = [c for c in main_contours if c is not None]
    for olist in other_contours_seq:
        all_c.extend(olist)
    if not all_c:
        return None
    y_bt, y_tp, x_lf, x_rt = crop_frame(ref, all_c)
    blobs = []
    for fr, mc, olist in zip(frames, main_contours, other_contours_seq):
        if fr is None:
            fr = _blank_like(ref)
        # Draw main + costars: use extract_blob_all with all contours
        cnts = []
        if mc is not None:
            cnts.append(mc)
        cnts.extend(olist)
        if not cnts:
            blob = np.zeros((y_tp - y_bt, x_rt - x_lf, 3), dtype=np.uint8)
        else:
            blob = extract_blob_all(
                fr,
                y_bt,
                y_tp,
                x_lf,
                x_rt,
                contours=cnts,
                channel=3,
                background_free=background_free,
                black_background=black_background,
            )
            # Optionally gray costars: re-extract main in color is complex; skip for MVP
            if not color_costar and olist and mc is not None:
                # gray entire blob then re-paint main region in color — simplified: leave RGB
                pass
        blobs.append(np.uint8(blob))
    # Pattern interact expects outlines + other_outlines as lists aligned to frames
    other_outlines = other_contours_seq
    pattern = generate_patternimage_interact(
        ref,
        main_contours,
        other_outlines,
        inners=None,
        other_inners=None,
        std=0,
    )
    return blobs, pattern


def generate_examples_from_ethogram(
    config: GenerationConfig,
    *,
    session: Optional[AnnotationSession] = None,
    loaded_tracklets: Optional[LoadedTracklets] = None,
    progress: ProgressCb = None,
) -> Dict[str, Any]:
    """Main API: ethogram + tracklets + video → sorted LabGym example pairs.

    Returns dict with counts, written paths summary, config path.
    """
    out_root = Path(config.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if session is None:
        data = json.loads(Path(config.annotations_path).read_text(encoding="utf-8"))
        session = AnnotationSession.from_dict(data)

    if loaded_tracklets is None:
        loaded_tracklets = load_tracklets_for_annotator(
            config.tracklets_dir,
            analysis_start_frame=config.analysis_start_frame,
            video_total_frames=session.total_frames,
        )
    analysis_start = int(
        config.analysis_start_frame
        if config.analysis_start_frame is not None
        else loaded_tracklets.analysis_start_frame
    )
    # Prefer session mode
    mode = int(session.behavior_mode)
    config.behavior_mode = mode

    length = int(config.length)
    stride = config.resolved_stride()
    windows = collect_windows(
        session,
        length=length,
        sampling=config.sampling,
        stride=stride,
        min_bout_frames=int(config.min_bout_frames),
    )

    video_path = config.video_path or session.video_path
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or session.fps or 30.0)
    write_fps = float(config.fps_write) if config.fps_write else max(fps / 5.0, 1.0)
    stem = Path(video_path).stem.split("_", 1)[0] if Path(video_path).stem else "video"

    # Soft label tables
    classnames = [b.name for b in session.behaviors]
    soft_rows: Dict[str, Tuple[str, np.ndarray]] = {}
    dense_by_subject: Dict[Optional[int], np.ndarray] = {}
    if config.write_soft_labels:
        if mode == BEHAVIOR_MODE_INTERACTIVE_BASIC:
            _, arr = dense_frame_labels_from_session(session, use_group=True)
            dense_by_subject[None] = arr
        else:
            for subj in session.subjects:
                _, arr = dense_frame_labels_from_session(
                    session, subject_id=subj.subject_id, use_group=False
                )
                dense_by_subject[int(subj.subject_id)] = arr

    counts: Dict[str, int] = {b: 0 for b in classnames}
    written = 0
    skipped = 0
    total = len(windows)

    for i, win in enumerate(windows):
        if progress:
            progress(i, total, f"{win.behavior} f={win.center_frame}")

        frames_raw = _read_frames_bgr(cap, win.start_frame, win.end_frame)
        # Replace None frames with black if we have any shape
        shape = None
        for fr in frames_raw:
            if fr is not None:
                shape = fr.shape
                break
        if shape is None:
            skipped += 1
            continue
        frames = [fr if fr is not None else np.zeros(shape, dtype=np.uint8) for fr in frames_raw]

        pair = None
        kind_token = "animal"
        id_token = 0

        if mode == BEHAVIOR_MODE_INTERACTIVE_BASIC:
            # all subjects contours each frame
            cpf: List[List[np.ndarray]] = []
            for vf in range(win.start_frame, win.end_frame + 1):
                af = video_to_analysis_frame(vf, analysis_start)
                clist: List[np.ndarray] = []
                for subj in loaded_tracklets.subjects:
                    try:
                        k, tid, store = _subject_track(loaded_tracklets, subj.subject_id)
                        c = _contour_at(store, tid, af)
                        if c is not None:
                            clist.append(c)
                    except KeyError:
                        continue
                cpf.append(clist)
            if _invalid_fraction([c[0] if c else None for c in cpf]) > config.max_invalid_fraction:
                # softer: require at least one animal most frames
                valid_frac = sum(1 for c in cpf if c) / max(len(cpf), 1)
                if valid_frac < (1.0 - config.max_invalid_fraction):
                    skipped += 1
                    continue
            pair = _build_mode1_pair(
                frames,
                cpf,
                background_free=config.background_free,
                black_background=config.black_background,
            )
            kind_token = "group"
            id_token = 0
            suffix = f"_len{length}_itbs"
        elif mode == BEHAVIOR_MODE_INTERACTIVE_ADVANCED:
            if win.subject_id is None:
                skipped += 1
                continue
            main_c = _window_contours_for_subject(
                loaded_tracklets,
                win.subject_id,
                win.start_frame,
                win.end_frame,
                analysis_start,
            )
            if _invalid_fraction(main_c) > config.max_invalid_fraction:
                skipped += 1
                continue
            other_seq: List[List[np.ndarray]] = []
            for vf in range(win.start_frame, win.end_frame + 1):
                af = video_to_analysis_frame(vf, analysis_start)
                others = _others_in_range(
                    loaded_tracklets,
                    win.subject_id,
                    af,
                    analysis_start,
                    config.social_distance,
                    config.animal_size,
                    win.partner_ids,
                )
                other_seq.append([c for _, c in others])
            pair = _build_mode2_pair(
                frames,
                main_c,
                other_seq,
                background_free=config.background_free,
                black_background=config.black_background,
                color_costar=config.color_costar,
            )
            try:
                kind_token, id_token, _ = _subject_track(loaded_tracklets, win.subject_id)
            except KeyError:
                kind_token, id_token = "animal", int(win.subject_id)
            suffix = f"_len{length}_itadv"
        else:
            # mode 0
            if win.subject_id is None:
                skipped += 1
                continue
            main_c = _window_contours_for_subject(
                loaded_tracklets,
                win.subject_id,
                win.start_frame,
                win.end_frame,
                analysis_start,
            )
            if _invalid_fraction(main_c) > config.max_invalid_fraction:
                skipped += 1
                continue
            pair = _build_mode0_pair(
                frames,
                main_c,
                background_free=config.background_free,
                black_background=config.black_background,
                channel=config.channel,
            )
            try:
                kind_token, id_token, _ = _subject_track(loaded_tracklets, win.subject_id)
            except KeyError:
                kind_token, id_token = "animal", int(win.subject_id)
            suffix = f"_len{length}"

        if pair is None:
            skipped += 1
            continue
        blobs, pattern = pair
        basename = (
            f"{stem}_{kind_token}_{id_token}_{win.center_frame}{suffix}"
        )
        beh_dir = out_root / win.behavior
        try:
            _write_pair(beh_dir, basename, blobs, pattern, write_fps)
        except Exception:
            skipped += 1
            continue
        counts[win.behavior] = counts.get(win.behavior, 0) + 1
        written += 1

        if config.write_soft_labels and classnames:
            sid_key: Optional[int]
            if mode == BEHAVIOR_MODE_INTERACTIVE_BASIC:
                sid_key = None
            else:
                sid_key = win.subject_id
            fl = dense_by_subject.get(sid_key)
            if fl is not None:
                hard, soft = build_soft_targets_for_window(
                    fl,
                    win.center_frame,
                    length,
                    classnames=classnames,
                    exclusive=True,
                    edge_smooth=2,
                    end_aligned=True,
                )
                soft_rows[basename] = (win.behavior if not hard else hard, soft)

    cap.release()
    if progress:
        progress(total, total, "done")

    # Sidecars
    cfg_path = out_root / "generation_config.json"
    cfg_dict = asdict(config)
    cfg_dict["analysis_start_frame_used"] = analysis_start
    cfg_dict["written"] = written
    cfg_dict["skipped"] = skipped
    cfg_dict["counts"] = counts
    cfg_path.write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

    soft_path = None
    if config.write_soft_labels and soft_rows:
        table = SoftLabelTable(classnames=classnames, rows=soft_rows)
        soft_path = str(table.save_csv(out_root / "soft_labels.csv"))

    return {
        "written": written,
        "skipped": skipped,
        "counts": counts,
        "output_dir": str(out_root),
        "generation_config": str(cfg_path),
        "soft_labels": soft_path,
        "n_windows": total,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Generate LabGym training pairs from ethogram + tracklets"
    )
    p.add_argument("--annotations", required=True, help="Path to .annotations.json")
    p.add_argument("--tracklets", required=True, help="id_review / tracklets directory")
    p.add_argument("--video", required=True, help="Video path")
    p.add_argument("--out", required=True, help="Output root for sorted examples")
    p.add_argument("--length", type=int, default=15)
    p.add_argument(
        "--sampling",
        default="dense_in_bout",
        choices=["bout_end", "bout_center", "dense_in_bout", "coverage"],
    )
    p.add_argument("--stride", type=int, default=0)
    p.add_argument("--min-bout-frames", type=int, default=1)
    p.add_argument("--analysis-start-frame", type=int, default=None)
    p.add_argument("--social-distance", type=float, default=0.0)
    p.add_argument("--no-soft-labels", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    cfg = GenerationConfig(
        video_path=args.video,
        annotations_path=args.annotations,
        tracklets_dir=args.tracklets,
        output_dir=args.out,
        length=args.length,
        sampling=args.sampling,
        stride=args.stride,
        min_bout_frames=args.min_bout_frames,
        analysis_start_frame=args.analysis_start_frame,
        social_distance=args.social_distance,
        write_soft_labels=not args.no_soft_labels,
    )

    def _prog(done, tot, msg):
        if tot:
            print(f"\r[{done}/{tot}] {msg}", end="", flush=True)

    result = generate_examples_from_ethogram(cfg, progress=_prog)
    print()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
