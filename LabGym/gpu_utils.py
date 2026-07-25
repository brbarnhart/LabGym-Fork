"""Helpers to release GPU memory held by LabGym analyzers / detectors."""

from __future__ import annotations

import gc
from typing import Any, Optional


def release_cuda_cache() -> None:
    """Run GC and empty the PyTorch CUDA caching allocator if available."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def release_analyzer_gpu(analyzer: Any) -> None:
    """Detach detector (and TF graphs if present) from an analyzer instance."""
    if analyzer is None:
        return
    det = getattr(analyzer, "detector", None)
    if det is not None:
        release_fn = getattr(det, "release", None)
        if callable(release_fn):
            try:
                release_fn()
            except Exception:
                pass
        else:
            try:
                if getattr(det, "current_detector", None) is not None:
                    del det.current_detector
                    det.current_detector = None
            except Exception:
                pass
        try:
            analyzer.detector = None
        except Exception:
            pass
    # Large CPU track buffers also pin RAM; drop common heavy attrs
    for name in (
        "animations",
        "pattern_images",
        "animal_blobs",
        "animal_contours",
        "animal_centers",
        "background",
        "temp_frames",
    ):
        if hasattr(analyzer, name):
            try:
                setattr(analyzer, name, None)
            except Exception:
                pass
    release_cuda_cache()
