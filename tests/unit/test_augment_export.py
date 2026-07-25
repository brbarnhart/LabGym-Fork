"""Unit tests for categorizer example augmentation (export path, cancel, skip)."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from LabGym.augment_export import (
    default_aug_workers,
    resolve_aug_methods,
    augment_one_example,
)
from LabGym.categorizer import Categorizers
from LabGym.training.progress import (
    TrainingCancelled,
    is_cancelled,
    make_cancel_callback,
    make_epoch_progress_callback,
    sanitize_logs,
)


def _write_jpg(path: Path, value: int = 120) -> None:
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[4:16, 4:16] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def test_resolve_aug_methods_empty_is_orig():
    assert resolve_aug_methods([]) == ["orig"]
    assert resolve_aug_methods(None) == ["orig"]


def test_resolve_aug_methods_horizontal_flip_includes_flph():
    methods = resolve_aug_methods(["horizontal flipping"])
    assert "orig" in methods
    assert any("flph" in m for m in methods)
    assert not any("flpv" in m and "flph" not in m for m in methods if m == "flpv")


def test_default_aug_workers_positive():
    n = default_aug_workers(export=True)
    assert isinstance(n, int)
    assert 1 <= n <= 8


def test_has_exported_aug_data(tmp_path: Path):
    assert not Categorizers.has_exported_aug_data(None)
    assert not Categorizers.has_exported_aug_data(str(tmp_path))
    train = tmp_path / "train"
    val = tmp_path / "validation"
    train.mkdir()
    val.mkdir()
    assert not Categorizers.has_exported_aug_data(str(tmp_path))
    _write_jpg(train / "0_orig_walk.jpg")
    assert not Categorizers.has_exported_aug_data(str(tmp_path))
    _write_jpg(val / "0_orig_walk.jpg")
    assert Categorizers.has_exported_aug_data(str(tmp_path))


def test_build_data_export_sequential_and_empty(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for i, lab in enumerate(("walk", "run", "walk")):
        p = src / f"ex{i}_{lab}.jpg"
        _write_jpg(p, 100 + i * 10)
        paths.append(str(p))

    out = tmp_path / "export_seq"
    out.mkdir()
    CA = Categorizers()
    prog = []

    def cb(d, t, m):
        prog.append((d, t))

    CA.build_data(
        paths,
        dim_tconv=0,
        dim_conv=8,
        channel=3,
        time_step=5,
        aug_methods=["horizontal flipping"],
        out_path=str(out),
        num_workers=1,
        progress_cb=cb,
    )
    jpgs = sorted(p.name for p in out.glob("*.jpg"))
    methods = resolve_aug_methods(["horizontal flipping"])
    assert len(jpgs) == len(paths) * len(methods)
    assert prog[-1][0] == len(paths)

    # empty sources
    empty_out = tmp_path / "export_empty"
    empty_out.mkdir()
    anims, pats, labs = CA.build_data(
        [],
        dim_tconv=0,
        dim_conv=8,
        out_path=str(empty_out),
        num_workers=1,
    )
    assert list(empty_out.glob("*.jpg")) == []


def test_build_data_parallel_export_matches_count(tmp_path: Path):
    """With ≥16 sources, 2 workers should write the same number of files as sequential."""
    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for i in range(16):
        lab = "walk" if i % 2 == 0 else "run"
        p = src / f"ex{i}_{lab}.jpg"
        _write_jpg(p)
        paths.append(str(p))

    out_seq = tmp_path / "seq"
    out_par = tmp_path / "par"
    out_seq.mkdir()
    out_par.mkdir()
    CA = Categorizers()
    methods_user = ["horizontal flipping"]
    kwargs = dict(
        dim_tconv=0,
        dim_conv=8,
        channel=3,
        time_step=5,
        aug_methods=methods_user,
    )
    CA.build_data(paths, out_path=str(out_seq), num_workers=1, **kwargs)
    CA.build_data(paths, out_path=str(out_par), num_workers=2, **kwargs)
    n_seq = len(list(out_seq.glob("*.jpg")))
    n_par = len(list(out_par.glob("*.jpg")))
    assert n_seq == n_par
    assert n_seq == 16 * len(resolve_aug_methods(methods_user))


def test_build_data_cancel_sequential(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for i in range(6):
        p = src / f"ex{i}_walk.jpg"
        _write_jpg(p)
        paths.append(str(p))
    out = tmp_path / "out"
    out.mkdir()
    cancel = threading.Event()
    CA = Categorizers()

    def prog(d, t, m):
        if d >= 2:
            cancel.set()

    with pytest.raises(TrainingCancelled):
        CA.build_data(
            paths,
            dim_tconv=0,
            dim_conv=8,
            aug_methods=[],
            out_path=str(out),
            num_workers=1,
            progress_cb=prog,
            cancel_event=cancel,
        )
    assert len(list(out.glob("*.jpg"))) < 6


def test_num_workers_clamped(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    p = src / "ex0_walk.jpg"
    _write_jpg(p)
    out = tmp_path / "out"
    out.mkdir()
    CA = Categorizers()
    # Should not raise for invalid worker counts
    CA.build_data(
        [str(p)],
        dim_tconv=0,
        dim_conv=8,
        aug_methods=[],
        out_path=str(out),
        num_workers=0,
    )
    CA.build_data(
        [str(p)],
        dim_tconv=0,
        dim_conv=8,
        aug_methods=[],
        out_path=str(out),
        num_workers=-3,
    )
    assert list(out.glob("*.jpg"))


def test_in_memory_orig_labels(tmp_path: Path):
    p = tmp_path / "ex0_approach.jpg"
    _write_jpg(p)
    CA = Categorizers()
    _, patterns, labels = CA.build_data(
        [str(p)],
        dim_tconv=0,
        dim_conv=8,
        aug_methods=[],
        out_path=None,
        num_workers=4,  # ignored for in-memory
    )
    assert patterns.shape[0] == 1
    assert list(labels) == ["approach"]


def test_sanitize_logs_and_epoch_callback():
    assert sanitize_logs({"loss": 0.5})["loss"] == 0.5
    seen = []
    cb = make_epoch_progress_callback(lambda e, logs: seen.append((e, logs)))
    assert cb is not None
    cb.on_epoch_end(0, {"loss": 1.0, "val_loss": 1.1})
    assert seen[0][0] == 1
    assert seen[0][1]["loss"] == 1.0


def test_cancel_callback_sets_stop_training():
    ev = threading.Event()
    cb = make_cancel_callback(ev)
    assert cb is not None

    class _M:
        stop_training = False

    cb.model = _M()
    cb.on_epoch_begin(0)
    assert cb.model.stop_training is False
    ev.set()
    cb.on_epoch_end(0)
    assert cb.model.stop_training is True
    assert is_cancelled(ev)
