"""Unit tests for detect+track acquisition timing helper."""

from __future__ import annotations

from LabGym.detection.acquisition_timing import AcquisitionTiming, emit_status


def test_acquisition_timing_percentages_and_fps():
    t = AcquisitionTiming()
    t.add_decode(2.0)
    t.add_batch(10, infer_s=5.0, post_s=3.0)
    t.add_batch(10, infer_s=5.0, post_s=3.0)
    assert t.n_frames == 20
    assert t.n_batches == 2
    assert abs(t.total_s - 18.0) < 1e-9
    line = t.format_line(final=True, batch_size=8)
    assert "timing total" in line
    assert "20 frames" in line
    assert "decode 2.0s (11%)" in line or "decode 2.0s (11" in line
    assert "GPU infer" in line
    assert "track/post" in line
    assert "batch_size=8" in line
    # 20 frames / 18s ≈ 1.1 f/s
    assert "1.1 f/s" in line


def test_maybe_report_throttles():
    t = AcquisitionTiming()
    t.add_batch(100, 1.0, 1.0)
    assert t.maybe_report(every_frames=300) is None
    t.add_batch(200, 1.0, 1.0)
    msg = t.maybe_report(every_frames=300)
    assert msg is not None
    assert "300 frames" in msg
    assert t.maybe_report(every_frames=300) is None  # not yet another 300


def test_emit_status_callback():
    seen = []
    emit_status(seen.append, [], "hello timing")
    assert seen == ["hello timing"]
