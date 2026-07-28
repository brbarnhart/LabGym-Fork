"""Wall-clock breakdown for detect+track frame acquisition loops."""

from __future__ import annotations


class AcquisitionTiming:
    """Accumulate decode / GPU inference / track-post wall times for one video.

    Used to see whether detector batch size is GPU-bound or limited by CPU
    post-processing (contours, tracking assignment, etc.).
    """

    def __init__(self) -> None:
        self.decode_s = 0.0
        self.infer_s = 0.0
        self.post_s = 0.0
        self.n_batches = 0
        self.n_frames = 0
        self._last_report_at = 0

    @property
    def total_s(self) -> float:
        return float(self.decode_s + self.infer_s + self.post_s)

    def add_decode(self, seconds: float) -> None:
        self.decode_s += max(0.0, float(seconds))

    def add_batch(self, n_frames: int, infer_s: float, post_s: float) -> None:
        n = max(0, int(n_frames))
        self.n_frames += n
        self.n_batches += 1
        self.infer_s += max(0.0, float(infer_s))
        self.post_s += max(0.0, float(post_s))

    def format_line(self, *, final: bool = False, batch_size: int | None = None) -> str:
        tot = self.total_s
        if tot <= 0 or self.n_frames <= 0:
            return "timing: (no frames yet)"
        fps = self.n_frames / tot
        prefix = "timing total" if final else "timing"
        parts = [
            f"{prefix}: {self.n_frames} frames in {tot:.1f}s ({fps:.1f} f/s)",
            f"decode {self.decode_s:.1f}s ({100.0 * self.decode_s / tot:.0f}%)",
            f"GPU infer {self.infer_s:.1f}s ({100.0 * self.infer_s / tot:.0f}%)",
            f"track/post {self.post_s:.1f}s ({100.0 * self.post_s / tot:.0f}%)",
            f"batches {self.n_batches}",
        ]
        if batch_size is not None:
            parts.append(f"batch_size={int(batch_size)}")
        return " | ".join(parts)

    def maybe_report(
        self, every_frames: int = 300, batch_size: int | None = None
    ) -> str | None:
        """Return a summary string every *every_frames* analyzed frames, else None."""
        every = max(1, int(every_frames))
        if self.n_frames - self._last_report_at < every:
            return None
        self._last_report_at = self.n_frames
        return self.format_line(final=False, batch_size=batch_size)


def emit_status(status_progress, analyzer_log, msg: str) -> None:
    """Print, append analyzer log, and optionally notify UI status callback."""
    if not msg:
        return
    print(msg, flush=True)
    if analyzer_log is not None:
        analyzer_log.append(msg)
    if status_progress is not None:
        try:
            status_progress(msg)
        except Exception:
            pass
