"""Downstream gate: annotate / generate / process share one predicate."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from LabGym.identity.downstream import (
    apply_context_to_annotator,
    may_open_video_for_annotation,
    may_use_downstream,
)


@pytest.mark.parametrize("mode", [0, 2])
def test_per_animal_modes_refuse_without_accepted_identities(mode: int) -> None:
    assert may_use_downstream(mode, False) is False


def test_interactive_basic_allowed_without_accepted_identities() -> None:
    assert may_use_downstream(1, False) is True


def test_identity_package_requires_accepted_even_if_mode_is_interactive_basic() -> None:
    assert may_use_downstream(1, False, identity_package=True) is False
    assert may_use_downstream(1, True, identity_package=True) is True


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_accepted_identities_allow_every_mode(mode: int) -> None:
    assert may_use_downstream(mode, True) is True


def test_annotate_apply_refuses_without_opening_video() -> None:
    """Per-animal annotate must not call load when the gate fails."""

    class _Window:
        def __init__(self) -> None:
            self.loaded = False

        def load_video_from_path(self, *args, **kwargs):
            self.loaded = True
            return True

    window = _Window()
    ctx = SimpleNamespace(
        video_path="clip.avi",
        annotations_path="",
        tracklets_dir="/tmp/id_review",
        accepted_identities=False,
        behavior_mode=0,
        exclusive_mode=False,
    )
    ok = apply_context_to_annotator(window, ctx)
    assert ok is False
    assert window.loaded is False


def test_annotate_apply_allows_interactive_basic_without_accepted() -> None:
    class _Window:
        def __init__(self) -> None:
            self.loaded = False
            self._loaded_tracklets = None

        def load_video_from_path(self, *args, **kwargs):
            self.loaded = True
            return True

    window = _Window()
    ctx = SimpleNamespace(
        video_path="clip.avi",
        annotations_path="",
        tracklets_dir="",
        accepted_identities=False,
        behavior_mode=1,
        exclusive_mode=False,
    )
    assert apply_context_to_annotator(window, ctx) is True
    assert window.loaded is True


def test_file_open_refuses_unaccepted_sibling_package(tmp_path: Path) -> None:
    from LabGym.id_review.apply import write_tracklets_identity_status
    from LabGym.identity.package import discover_identity_package_for_video

    video = tmp_path / "clip.avi"
    video.write_bytes(b"x")
    review = tmp_path / "id_review"
    review.mkdir()
    write_tracklets_identity_status(
        str(review), corrected=False, accepted=False, has_raw=True
    )
    assert discover_identity_package_for_video(video) == str(review)
    assert may_open_video_for_annotation(video) is False


def test_file_open_allows_accepted_sibling_package(tmp_path: Path) -> None:
    from LabGym.id_review.apply import write_tracklets_identity_status
    from LabGym.id_review.types import SCHEMA_VERSION, TrackletStore
    from LabGym.id_review.tracklets import save_tracklets
    import numpy as np

    video = tmp_path / "clip.avi"
    video.write_bytes(b"x")
    review = tmp_path / "id_review"
    review.mkdir()
    n = 4
    store = TrackletStore(
        schema_version=SCHEMA_VERSION,
        animal_kind="mouse",
        ids=[0],
        n_frames=n,
        centers=np.zeros((1, n, 2)),
        valid=np.ones((1, n), dtype=bool),
        heights=np.ones((1, n)),
        contours=[[None] * n],
        meta={},
    )
    save_tracklets(store, str(review))
    write_tracklets_identity_status(
        str(review), corrected=True, accepted=True, has_raw=True
    )
    assert may_open_video_for_annotation(video) is True


def test_file_open_without_package_allowed_standalone(tmp_path: Path) -> None:
    video = tmp_path / "clip.avi"
    video.write_bytes(b"x")
    assert may_open_video_for_annotation(video) is True
    assert (
        may_open_video_for_annotation(
            video, behavior_mode=0, allow_without_package=False
        )
        is False
    )
