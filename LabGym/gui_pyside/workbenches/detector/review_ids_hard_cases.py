"""Extract detector training frames from Review IDs hard-case time ranges.

Pure helpers (no Qt). Analysis-frame ranges are mapped to absolute video frames
via ``analysis_frame_to_video_frame`` so extraction matches the review preview.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2

from LabGym.id_review.samples import analysis_frame_to_video_frame
from LabGym.gui_pyside.workbenches.detector.review_ids_render import resize_if_needed


@dataclass(frozen=True)
class AnalysisFrameRange:
    """Inclusive analysis-frame window selected during ID review."""

    start_frame: int
    end_frame: int
    note: str = ""

    def __post_init__(self) -> None:
        s = int(self.start_frame)
        e = int(self.end_frame)
        if e < s:
            s, e = e, s
        object.__setattr__(self, "start_frame", max(0, s))
        object.__setattr__(self, "end_frame", max(0, e))

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def label(self, fps: float = 0.0) -> str:
        if fps and fps > 0:
            t0 = self.start_frame / fps
            t1 = self.end_frame / fps
            base = f"f{self.start_frame}–{self.end_frame}  ({t0:.2f}–{t1:.2f}s)"
        else:
            base = f"f{self.start_frame}–{self.end_frame}"
        if self.note:
            return f"{base}  [{self.note}]"
        return base


def normalize_ranges(
    ranges: Sequence[AnalysisFrameRange],
    *,
    n_frames: Optional[int] = None,
) -> List[AnalysisFrameRange]:
    """Clamp, drop empties, merge overlapping / adjacent ranges (sorted)."""
    cleaned: List[AnalysisFrameRange] = []
    for r in ranges:
        s, e = int(r.start_frame), int(r.end_frame)
        if e < s:
            s, e = e, s
        s = max(0, s)
        e = max(0, e)
        if n_frames is not None and n_frames > 0:
            s = min(s, n_frames - 1)
            e = min(e, n_frames - 1)
        if e < s:
            continue
        cleaned.append(AnalysisFrameRange(s, e, note=r.note or ""))

    if not cleaned:
        return []

    cleaned.sort(key=lambda r: (r.start_frame, r.end_frame))
    merged: List[AnalysisFrameRange] = [cleaned[0]]
    for r in cleaned[1:]:
        prev = merged[-1]
        # Merge if overlapping or adjacent (share an endpoint).
        if r.start_frame <= prev.end_frame + 1:
            note = prev.note
            if r.note and r.note not in note:
                note = f"{note}+{r.note}" if note else r.note
            merged[-1] = AnalysisFrameRange(
                prev.start_frame, max(prev.end_frame, r.end_frame), note=note
            )
        else:
            merged.append(r)
    return merged


def frames_to_extract(
    ranges: Sequence[AnalysisFrameRange],
    *,
    skip: int = 1,
    n_frames: Optional[int] = None,
) -> List[int]:
    """Unique sorted analysis frames to write, sampling every *skip* within each range.

    Always includes the start and end of each range so short failure spots are kept.
    """
    skip = max(1, int(skip))
    out: List[int] = []
    seen = set()
    for r in normalize_ranges(ranges, n_frames=n_frames):
        candidates = list(range(r.start_frame, r.end_frame + 1, skip))
        if r.end_frame not in candidates:
            candidates.append(r.end_frame)
        for f in candidates:
            if f not in seen:
                seen.add(f)
                out.append(f)
    out.sort()
    return out


@dataclass
class ExtractHardCaseResult:
    n_written: int
    n_failed: int
    paths: List[str]
    error: str = ""
    cancelled: bool = False


def default_output_dir(project_root: Optional[str]) -> str:
    if project_root:
        return str(Path(project_root) / "detector_training_images")
    return str(Path.cwd() / "detector_training_images")


def output_filename(video_stem: str, analysis_frame: int) -> str:
    """Stable, collision-friendly name shared with Generate images folder usage."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_stem) or "video"
    return f"{safe}_af{int(analysis_frame):06d}.jpg"


def extract_hard_case_frames(
    video_path: str,
    out_path: str,
    ranges: Sequence[AnalysisFrameRange],
    *,
    store_meta: Optional[Dict] = None,
    fps: float = 10.0,
    skip: int = 10,
    framewidth: Optional[int] = None,
    n_frames: Optional[int] = None,
    progress_callback=None,
    cancel_check=None,
) -> ExtractHardCaseResult:
    """Write JPG frames for the given analysis-frame ranges.

    Files are written as ``{video_stem}_af{analysis:06d}.jpg`` into *out_path*.
    Existing files with the same name are overwritten.

    *progress_callback*, if set, is called as ``(current, total, message)``.
    *cancel_check*, if set, is a zero-arg callable returning True to stop early
    (cooperative; already-written images are kept).
    """
    video_path = str(video_path or "").strip()
    if not video_path or not Path(video_path).is_file():
        return ExtractHardCaseResult(
            0, 0, [], error=f"Video not found:\n{video_path or '(empty)'}"
        )

    frames = frames_to_extract(ranges, skip=skip, n_frames=n_frames)
    if not frames:
        return ExtractHardCaseResult(0, 0, [], error="No frames in the selected ranges.")

    Path(out_path).mkdir(parents=True, exist_ok=True)
    meta = dict(store_meta or {})
    stem = Path(video_path).stem
    fps = float(fps or meta.get("fps") or 10.0) or 10.0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ExtractHardCaseResult(0, 0, [], error=f"Could not open video:\n{video_path}")

    written: List[str] = []
    failed = 0
    cancelled = False
    try:
        total = len(frames)
        for i, af in enumerate(frames, start=1):
            if cancel_check is not None and cancel_check():
                cancelled = True
                break
            msg = f"[{i}/{total}] analysis frame {af}"
            if progress_callback is not None:
                progress_callback(i, total, msg)
            v_idx = analysis_frame_to_video_frame(meta, af, fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(v_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                failed += 1
                continue
            if framewidth is not None:
                frame = resize_if_needed(frame, int(framewidth))
            # Prefer meta framewidth only when user did not request a resize.
            elif meta.get("framewidth") is not None:
                try:
                    frame = resize_if_needed(frame, int(meta["framewidth"]))
                except (TypeError, ValueError):
                    pass
            name = output_filename(stem, af)
            dest = os.path.join(out_path, name)
            if cv2.imwrite(dest, frame):
                written.append(dest)
            else:
                failed += 1
    finally:
        cap.release()

    if cancelled:
        err = ""
        if not written:
            err = "Cancelled before any frames were written."
        return ExtractHardCaseResult(
            n_written=len(written),
            n_failed=failed,
            paths=written,
            error=err,
            cancelled=True,
        )

    return ExtractHardCaseResult(
        n_written=len(written),
        n_failed=failed,
        paths=written,
        error=""
        if written
        else (
            f"Wrote 0 images ({failed} read/write failures)." if failed else ""
        ),
        cancelled=False,
    )


def ranges_from_risk_event(start_frame: int, end_frame: int, note: str = "risk") -> AnalysisFrameRange:
    return AnalysisFrameRange(int(start_frame), int(end_frame), note=note)
