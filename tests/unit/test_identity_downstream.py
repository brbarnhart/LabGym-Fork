"""Downstream gate: annotate / generate / process share one predicate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from LabGym.identity.downstream import apply_context_to_annotator, may_use_downstream


@pytest.mark.parametrize("mode", [0, 2])
def test_per_animal_modes_refuse_without_accepted_identities(mode: int) -> None:
    assert may_use_downstream(mode, False) is False


def test_interactive_basic_allowed_without_accepted_identities() -> None:
    assert may_use_downstream(1, False) is True


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
