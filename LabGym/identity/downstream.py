"""One gate for annotate, generate examples, and Process videos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from LabGym.identity.package import writes_identity_package


def may_use_downstream(behavior_mode: int, accepted_identities: bool) -> bool:
    """Return whether this video may be annotated, used for examples, or processed.

    Per-animal modes (non-interactive and interactive advanced) require
    accepted identities. Interactive basic has no identity package and is
    always allowed.

    Args:
        behavior_mode: LabGym behavior mode code (0, 1, 2, …).
        accepted_identities: True when Review IDs has published remapped
            tracklets for this video.

    Returns:
        True if the video may proceed; False if Annotate / generate /
        Process videos must refuse.
    """
    if not writes_identity_package(int(behavior_mode)):
        return True
    return bool(accepted_identities)


def apply_context_to_annotator(window: Any, ctx: Any) -> bool:
    """Load a project video into the annotator, or refuse without opening it.

    Does not call ``window.load_video_from_path`` when the downstream gate
    fails. Callers own any user-facing warning.

    Args:
        window: Annotator window with ``load_video_from_path`` (and optional
            ``load_tracklets_from_path`` / ``statusBar``).
        ctx: Resolved video context (``behavior_mode``, ``accepted_identities``,
            paths).

    Returns:
        True if the video was opened; False if the gate refused or load failed.
    """
    if not may_use_downstream(int(ctx.behavior_mode), bool(ctx.accepted_identities)):
        return False

    video = str(getattr(ctx, "video_path", "") or "")
    ann = getattr(ctx, "annotations_path", None) or None
    ann_arg = ann if (ann and Path(ann).is_file()) else None
    tracks = str(getattr(ctx, "tracklets_dir", "") or "")
    tracklets_dir = tracks if (tracks and bool(ctx.accepted_identities)) else None

    ok = window.load_video_from_path(
        video,
        annotations_path=ann_arg,
        tracklets_dir=tracklets_dir,
        behavior_mode=int(ctx.behavior_mode),
        exclusive_mode=bool(getattr(ctx, "exclusive_mode", False)),
        prefer_sidecar=ann_arg is None,
    )
    if not ok:
        return False

    loaded = getattr(window, "_loaded_tracklets", None)
    if tracklets_dir and loaded is None and hasattr(window, "load_tracklets_from_path"):
        window.load_tracklets_from_path(tracklets_dir)

    status = getattr(window, "statusBar", None)
    if callable(status):
        bar = status()
        if bar is not None and hasattr(bar, "showMessage"):
            bar.showMessage(f"Loaded project video  ·  mode={ctx.behavior_mode}")
    return True
