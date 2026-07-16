"""
ExampleGenerator: produces curated short video clips from annotated bouts
and the frame_labels.csv for LabGym compatibility.

Designed to be used from UI (in QThread) or headless/scripted.

Key outputs:
- Subfolders per behavior containing short .mp4 clips
- frame_labels.csv (frame, behavior1, behavior2, ... with 0/1)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError:
    cv2 = None

from core.annotation_manager import AnnotationManager
from core.data_models import AnnotationSession, Bout
from core.video_handler import VideoHandler


@dataclass
class ClipSpec:
    behavior: str
    start_frame: int
    end_frame: int
    bout_index: int   # original bout index for naming


class ExampleGenerator:
    def __init__(
        self,
        session: AnnotationSession,
        video_path: Optional[str] = None,
        video_handler: Optional[VideoHandler] = None,
    ):
        self.session = session
        self.video_path = video_path or session.video_path
        self._video_handler = video_handler

    @property
    def video_handler(self) -> VideoHandler:
        if self._video_handler is None:
            self._video_handler = VideoHandler()
            self._video_handler.load(self.video_path)
        return self._video_handler

    def parent_video_id(self) -> str:
        """Short id from the parent video filename: stem up to the first underscore.

        Examples:
            J5904-M-DCZ_processed.avi  -> J5904-M-DCZ
            sample_behavior_video.avi  -> sample
            noUnderscore.mp4           -> noUnderscore
        """
        stem = Path(self.video_path).stem if self.video_path else "video"
        if not stem:
            return "video"
        return stem.split("_", 1)[0]

    # ------------------------------------------------------------------
    # Frame labels (the high-value LabGym integration artifact)
    # ------------------------------------------------------------------

    def build_frame_labels_df(self, open_starts: Optional[dict[str, int]] = None) -> pd.DataFrame:
        """
        Returns a DataFrame with:
            frame, <behavior1>, <behavior2>, ...
        One row per frame, values are 0 or 1.
        Compatible with LabGym's frame-wise label sorter.

        open_starts: optional dict of currently open bouts {behavior: start_frame}.
                     These will be marked active from their start to the end of the video.
        """
        total = self.session.total_frames
        behaviors = [b.name for b in self.session.behaviors]

        data = {"frame": list(range(total))}
        for name in behaviors:
            data[name] = [0] * total

        # Completed bouts
        for name in behaviors:
            for bout in self.session.bouts.get(name, []):
                for f in range(max(0, bout.start_frame), min(total, bout.end_frame + 1)):
                    data[name][f] = 1

        # Any still-open bouts at time of export: treat as going to end of video
        if open_starts:
            for name, start in open_starts.items():
                if name in data:
                    for f in range(max(0, start), total):
                        data[name][f] = 1

        return pd.DataFrame(data)

    def export_frame_labels_csv(self, output_path: str | Path, open_starts: Optional[dict[str, int]] = None) -> Path:
        df = self.build_frame_labels_df(open_starts=open_starts)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        return out

    # ------------------------------------------------------------------
    # Clip sampling
    # ------------------------------------------------------------------

    def _centered_window(
        self,
        bout_start: int,
        bout_end: int,
        window_len: int,
    ) -> Tuple[int, int]:
        """Return [start, end] of length window_len centered on the bout, clamped to the video."""
        total = self.session.total_frames
        if window_len < 1:
            window_len = 1
        center = (bout_start + bout_end) / 2.0
        cstart = int(round(center - (window_len - 1) / 2.0))
        cend = cstart + window_len - 1
        if cstart < 0:
            cstart = 0
            cend = window_len - 1
        if total > 0 and cend > total - 1:
            cend = total - 1
            cstart = max(0, cend - window_len + 1)
        if total > 0:
            cstart = max(0, min(cstart, total - 1))
            cend = max(0, min(cend, total - 1))
        if cend < cstart:
            cend = cstart
        return cstart, cend

    def sample_clips(
        self,
        clip_length: int,
        mode: str = "centered",   # "centered", "random", "all", "full_bout", "full_bout_coverage"
        behaviors: Optional[List[str]] = None,
        n_random: int = 3,
        random_seed: Optional[int] = None,
        min_bout_duration: int = 1,
    ) -> List[ClipSpec]:
        """
        Sample clip ranges from the annotated bouts.

        modes:
          - "centered": one clip per bout, centered on the bout when possible
          - "random": up to n_random clips per behavior (uniform inside bouts)
          - "all": every non-overlapping clip of `clip_length` inside bouts
          - "full_bout": one video per bout with optional min length rules:
              - bout duration < min_bout_duration: skip
              - min_bout_duration <= bout duration < clip_length (min clip length):
                one clip of length clip_length centered on the bout
              - bout duration >= clip_length: one clip spanning the entire bout
          - "full_bout_coverage":
              - if bout length < clip_length/2: skip
              - if bout length <= clip_length (but > half): one clip_length clip centered on the bout
                (the clip may extend before/after the bout)
              - if bout length > clip_length: generate non-overlapping clip_length clips starting
                from the beginning of the bout, plus one final clip that reaches the very end of
                the bout (the final clip may overlap the previous one). This ensures the entire
                bout is covered by the extracted clips.
        """
        if clip_length < 1:
            raise ValueError("clip_length must be >= 1")
        if min_bout_duration < 1:
            raise ValueError("min_bout_duration must be >= 1")

        if behaviors is None:
            behaviors = [b.name for b in self.session.behaviors]

        rng = np.random.default_rng(random_seed)
        specs: List[ClipSpec] = []
        total = self.session.total_frames

        for bname in behaviors:
            bouts = self.session.bouts.get(bname, [])
            for bout_idx, bout in enumerate(bouts):
                b_len = bout.duration_frames()

                if mode == "full_bout":
                    # Skip bouts shorter than the minimum duration threshold
                    if b_len < min_bout_duration:
                        continue
                    if b_len < clip_length:
                        # Short but eligible: pad to min clip length, centered on bout
                        cstart, cend = self._centered_window(
                            bout.start_frame, bout.end_frame, clip_length
                        )
                        if cend >= cstart:
                            specs.append(ClipSpec(bname, cstart, cend, bout_idx))
                    else:
                        # Long enough: export the full bout
                        start = max(0, bout.start_frame)
                        end = max(start, bout.end_frame)
                        if total > 0:
                            end = min(end, total - 1)
                            start = min(start, total - 1)
                        if end >= start:
                            specs.append(ClipSpec(bname, start, end, bout_idx))
                    continue

                if mode == "full_bout_coverage":
                    half = clip_length / 2.0
                    if b_len < half:
                        continue
                    if b_len <= clip_length:
                        # single clip of full clip_length, centered on the bout
                        # (will extend outside the annotated bout on the sides if bout is short)
                        center = (bout.start_frame + bout.end_frame) / 2.0
                        cstart = int(center - clip_length / 2.0)
                        cend = cstart + clip_length - 1
                        total = self.session.total_frames
                        cstart = max(0, min(cstart, total - 1))
                        cend = max(0, min(cend, total - 1))
                        if cend >= cstart:
                            specs.append(ClipSpec(bname, cstart, cend, bout_idx))
                    else:
                        # L > clip_length: non-overlapping from start + final clip reaching bout end
                        L = b_len
                        pos = 0
                        while pos + clip_length <= L:
                            s = bout.start_frame + pos
                            e = s + clip_length - 1
                            specs.append(ClipSpec(bname, s, e, bout_idx))
                            pos += clip_length
                        # add/adjust last clip to reach the end of the bout
                        final_s = max(0, L - clip_length)
                        fs = bout.start_frame + final_s
                        fe = bout.start_frame + L - 1
                        if not specs or specs[-1].end_frame < fe:
                            specs.append(ClipSpec(bname, fs, fe, bout_idx))
                    continue

                if b_len < clip_length:
                    # still allow a single (shorter) clip? For now we take what fits.
                    start = bout.start_frame
                    end = bout.end_frame
                    specs.append(ClipSpec(bname, start, end, bout_idx))
                    continue

                if mode == "centered":
                    # center the clip on the bout
                    mid = (bout.start_frame + bout.end_frame) // 2
                    half = clip_length // 2
                    start = max(bout.start_frame, mid - half)
                    end = min(bout.end_frame, start + clip_length - 1)
                    # adjust if we went past end
                    if end - start + 1 < clip_length:
                        start = max(bout.start_frame, end - clip_length + 1)
                    specs.append(ClipSpec(bname, start, end, bout_idx))

                elif mode == "random":
                    for _ in range(min(n_random, max(1, b_len - clip_length + 1))):
                        possible_start = bout.start_frame
                        max_start = bout.end_frame - clip_length + 1
                        if max_start <= possible_start:
                            start = possible_start
                        else:
                            start = rng.integers(possible_start, max_start + 1)
                        end = start + clip_length - 1
                        specs.append(ClipSpec(bname, int(start), int(end), bout_idx))

                elif mode == "all":
                    step = clip_length
                    pos = bout.start_frame
                    clip_count = 0
                    while pos + clip_length - 1 <= bout.end_frame:
                        specs.append(ClipSpec(bname, pos, pos + clip_length - 1, bout_idx))
                        pos += step
                        clip_count += 1
                        if clip_count > 10000:  # safety
                            break
                else:
                    raise ValueError(f"Unknown sampling mode: {mode}")

        return specs

    def sample_clips_from_ranges(
        self,
        ranges: List[Tuple[int, int]],
        behaviors: Optional[List[str]] = None,
        min_bout_duration: int = 1,
        clip_length: Optional[int] = None,
        min_duration: Optional[int] = None,  # alias for min_bout_duration
    ) -> List[ClipSpec]:
        """Build clip specs from annotated bouts that intersect selection ranges.

        For each bout B and each range S, take the intersection I = B ∩ S, then
        apply the same full-bout rules as mode="full_bout":

          - |I| < min_bout_duration → skip
          - min_bout_duration ≤ |I| < clip_length → clip of length clip_length,
            centered on I (may extend outside I / the selection)
          - |I| ≥ clip_length → one clip spanning all of I

        If clip_length is None, intersections are exported as-is (only the
        min_bout_duration filter is applied).
        """
        if min_duration is not None:
            min_bout_duration = min_duration
        if min_bout_duration < 1:
            raise ValueError("min_bout_duration must be >= 1")
        if clip_length is not None and clip_length < 1:
            raise ValueError("clip_length must be >= 1")
        if not ranges:
            return []

        if behaviors is None:
            behaviors = [b.name for b in self.session.behaviors]

        total = self.session.total_frames
        norm_ranges: List[Tuple[int, int]] = []
        for a, b in ranges:
            lo, hi = (a, b) if a <= b else (b, a)
            if total > 0:
                lo = max(0, min(int(lo), total - 1))
                hi = max(0, min(int(hi), total - 1))
            else:
                lo, hi = max(0, int(lo)), max(0, int(hi))
            if hi < lo:
                hi = lo
            norm_ranges.append((lo, hi))

        specs: List[ClipSpec] = []
        for bname in behaviors:
            bouts = self.session.bouts.get(bname, [])
            for bout_idx, bout in enumerate(bouts):
                for rs, re in norm_ranges:
                    is_start = max(bout.start_frame, rs)
                    is_end = min(bout.end_frame, re)
                    if is_end < is_start:
                        continue
                    b_len = is_end - is_start + 1
                    if b_len < min_bout_duration:
                        continue

                    if clip_length is not None and b_len < clip_length:
                        cstart, cend = self._centered_window(
                            is_start, is_end, clip_length
                        )
                        if cend >= cstart:
                            specs.append(ClipSpec(bname, cstart, cend, bout_idx))
                    else:
                        # Full intersection (or full_bout long case)
                        specs.append(ClipSpec(bname, is_start, is_end, bout_idx))

        specs.sort(key=lambda s: (s.start_frame, s.behavior, s.bout_index))
        return specs

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _write_clip_specs(
        self,
        specs: List[ClipSpec],
        output_dir: str | Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        filename_template: str = "{video}_{behavior}_bout{bout:03d}_f{start:05d}-{end:05d}.mp4",
        fourcc: str = "mp4v",
    ) -> List[Path]:
        """Write ClipSpec list to behavior subfolders under output_dir."""
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        by_behavior: dict[str, list[ClipSpec]] = {}
        for spec in specs:
            by_behavior.setdefault(spec.behavior, []).append(spec)

        written: List[Path] = []
        total = len(specs)
        done = 0

        vh = self.video_handler
        video_id = self.parent_video_id()

        for behavior, blist in by_behavior.items():
            beh_dir = out_root / behavior
            beh_dir.mkdir(exist_ok=True)

            for spec in blist:
                clip_frames = []
                for f in range(spec.start_frame, spec.end_frame + 1):
                    try:
                        frame = vh.get_frame(f)
                        if frame is not None:
                            bgr = frame[:, :, ::-1] if frame.shape[2] == 3 else frame
                            clip_frames.append(bgr)
                    except Exception:
                        continue

                if not clip_frames:
                    continue

                h, w = clip_frames[0].shape[:2]
                fname = filename_template.format(
                    video=video_id,
                    behavior=behavior,
                    bout=spec.bout_index,
                    start=spec.start_frame,
                    end=spec.end_frame,
                )
                out_path = beh_dir / fname

                if cv2 is None:
                    raise RuntimeError("OpenCV (cv2) is required for clip export")

                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*fourcc), vh.fps, (w, h)
                )
                for frm in clip_frames:
                    writer.write(frm)
                writer.release()

                written.append(out_path)
                done += 1
                if progress_callback:
                    progress_callback(done, total)

        return written

    def export_clips(
        self,
        output_dir: str | Path,
        clip_length: int = 45,
        mode: str = "centered",
        behaviors: Optional[List[str]] = None,
        n_random: int = 3,
        random_seed: Optional[int] = 42,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        filename_template: str = "{video}_{behavior}_bout{bout:03d}_f{start:05d}-{end:05d}.mp4",
        fourcc: str = "mp4v",
        open_starts: Optional[dict[str, int]] = None,
        write_frame_labels: bool = False,
        min_bout_duration: int = 1,
    ) -> List[Path]:
        """
        Generate and write video clips (from completed bouts), sorted into
        subfolders named by behavior type.

        For mode="full_bout":
          - skip bouts shorter than min_bout_duration
          - bouts shorter than clip_length (min clip length) get a centered
            clip of length clip_length
          - longer bouts are exported at full bout duration

        Filename placeholders: {video} (parent stem up to first underscore),
        {behavior}, {bout}, {start}, {end}.

        open_starts (if provided) are used when writing the labels file
        (treated as extending to video end).
        """
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        specs = self.sample_clips(
            clip_length=clip_length,
            mode=mode,
            behaviors=behaviors,
            n_random=n_random,
            random_seed=random_seed,
            min_bout_duration=min_bout_duration,
        )

        written = self._write_clip_specs(
            specs,
            out_root,
            progress_callback=progress_callback,
            filename_template=filename_template,
            fourcc=fourcc,
        )

        if write_frame_labels:
            self.export_frame_labels_csv(out_root / "frame_labels.csv", open_starts=open_starts)

        return written

    def export_range_clips(
        self,
        output_dir: str | Path,
        ranges: List[Tuple[int, int]],
        behaviors: Optional[List[str]] = None,
        min_bout_duration: int = 1,
        clip_length: Optional[int] = None,
        min_duration: Optional[int] = None,  # alias
        progress_callback: Optional[Callable[[int, int], None]] = None,
        filename_template: str = "{video}_{behavior}_bout{bout:03d}_f{start:05d}-{end:05d}.mp4",
        fourcc: str = "mp4v",
    ) -> List[Path]:
        """Export clips for bout∩selection-range intersections into behavior folders.

        Uses the same min_bout_duration / clip_length (min clip length) rules as
        full-bout export when clip_length is provided.
        """
        specs = self.sample_clips_from_ranges(
            ranges,
            behaviors=behaviors,
            min_bout_duration=min_bout_duration,
            clip_length=clip_length,
            min_duration=min_duration,
        )
        return self._write_clip_specs(
            specs,
            output_dir,
            progress_callback=progress_callback,
            filename_template=filename_template,
            fourcc=fourcc,
        )

    def close(self):
        if self._video_handler is not None:
            self._video_handler.release()
            self._video_handler = None
