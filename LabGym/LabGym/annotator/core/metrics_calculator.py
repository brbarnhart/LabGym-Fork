"""Basic metrics calculator for Phase 1/3.

Computes the core numbers listed in project_plan.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from LabGym.annotator.core.data_models import AnnotationSession, Bout


@dataclass
class BehaviorMetrics:
    name: str
    num_bouts: int
    total_duration_frames: int
    total_duration_sec: float
    mean_duration_frames: float
    median_duration_frames: float
    min_duration_frames: int
    max_duration_frames: int
    latency_frames: int | None   # None = never observed
    latency_sec: float | None
    frequency_per_min: float


class MetricsCalculator:
    def __init__(self, session: AnnotationSession, subject_id: int | None = None):
        self.session = session
        self.subject_id = subject_id

    def _resolved_subject_id(self) -> int:
        if self.subject_id is not None:
            return int(self.subject_id)
        return int(self.session.active_subject_id)

    def compute(self) -> Dict[str, BehaviorMetrics]:
        results: Dict[str, BehaviorMetrics] = {}
        fps = max(self.session.fps, 1e-6)
        total_sec = self.session.total_frames / fps
        bmap = self.session.bouts_for_subject(self._resolved_subject_id())

        for beh in self.session.behaviors:
            name = beh.name
            bouts: List[Bout] = bmap.get(name, [])
            durations = [b.duration_frames() for b in bouts]

            num = len(durations)
            if num == 0:
                results[name] = BehaviorMetrics(
                    name=name, num_bouts=0, total_duration_frames=0,
                    total_duration_sec=0.0, mean_duration_frames=0.0,
                    median_duration_frames=0.0, min_duration_frames=0,
                    max_duration_frames=0, latency_frames=None, latency_sec=None,
                    frequency_per_min=0.0
                )
                continue

            total_f = sum(durations)
            total_s = total_f / fps
            mean_f = total_f / num
            # median
            sorted_d = sorted(durations)
            mid = num // 2
            med = sorted_d[mid] if num % 2 == 1 else (sorted_d[mid-1] + sorted_d[mid]) / 2

            # Latency = first start frame (or None)
            first_start = min(b.start_frame for b in bouts)
            latency_f = first_start
            latency_s = latency_f / fps

            freq = (num / total_sec) * 60.0 if total_sec > 0 else 0.0

            results[name] = BehaviorMetrics(
                name=name,
                num_bouts=num,
                total_duration_frames=total_f,
                total_duration_sec=round(total_s, 3),
                mean_duration_frames=round(mean_f, 2),
                median_duration_frames=round(med, 2),
                min_duration_frames=min(durations),
                max_duration_frames=max(durations),
                latency_frames=latency_f,
                latency_sec=round(latency_s, 3),
                frequency_per_min=round(freq, 3),
            )
        return results

    def to_dataframe(self) -> pd.DataFrame:
        metrics = self.compute()
        rows = []
        for m in metrics.values():
            rows.append({
                "behavior": m.name,
                "bouts": m.num_bouts,
                "total_frames": m.total_duration_frames,
                "total_sec": m.total_duration_sec,
                "mean_frames": m.mean_duration_frames,
                "median_frames": m.median_duration_frames,
                "min_frames": m.min_duration_frames,
                "max_frames": m.max_duration_frames,
                "latency_frames": m.latency_frames if m.latency_frames is not None else "never",
                "latency_sec": m.latency_sec if m.latency_sec is not None else "never",
                "bouts_per_min": m.frequency_per_min,
            })
        return pd.DataFrame(rows)
