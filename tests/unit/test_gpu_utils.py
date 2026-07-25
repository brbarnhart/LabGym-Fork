"""Tests for GPU release helpers (no real CUDA required)."""

from __future__ import annotations

from LabGym.gpu_utils import release_analyzer_gpu, release_cuda_cache


class _FakeDetector:
    def __init__(self):
        self.current_detector = object()
        self.released = False

    def release(self):
        self.current_detector = None
        self.released = True


def test_release_analyzer_gpu_calls_detector_release():
    class A:
        pass

    a = A()
    a.detector = _FakeDetector()
    a.animations = {"x": 1}
    release_analyzer_gpu(a)
    assert a.detector is None or a.detector.released
    # heavy attrs cleared when detector was present
    assert a.animations is None


def test_release_cuda_cache_noop_safe():
    release_cuda_cache()  # must not raise without CUDA
