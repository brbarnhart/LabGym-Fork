"""One gate for annotate, generate examples, and Process videos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from LabGym.identity.package import (
    behavior_mode_from_package,
    discover_identity_package_for_video,
    has_identity_package,
    writes_identity_package,
)


def may_use_downstream(
    behavior_mode: int,
    accepted_identities: bool,
    *,
    identity_package: bool = False,
) -> bool:
    """Return whether this video may be annotated, used for examples, or processed.

    A present identity package is a per-animal detect world and always
    requires accepted identities. Interactive basic (no package) is exempt.
    Project default and categorizer ``behavior_kind`` must not be used to
    invent that exemption when a package exists.

    Args:
        behavior_mode: Detect-world behavior mode for this video (0, 1, 2, …).
        accepted_identities: True when Review IDs has published remapped
            tracklets for this video.
        identity_package: True when this video has a per-animal identity
            package (raw, remapped, or detect job). Overrides *behavior_mode*
            so a mode-1 project default cannot skip Review IDs.

    Returns:
        True if the video may proceed; False if Annotate / generate /
        Process videos must refuse.
    """
    if identity_package or writes_identity_package(int(behavior_mode)):
        return bool(accepted_identities)
    return True


def review_ids_required_message(package_dir: str = "") -> str:
    """User-facing reason Annotate / File → Open refused this video."""
    return (
        "Save Review IDs (Detector workbench) before annotating.\n"
        "That step publishes accepted identities (you may accept "
        "detector IDs with no swaps).\n\n"
        f"Package:\n{package_dir or '(none found)'}"
    )


def may_open_video_for_annotation(
    video_path: str | Path,
    *,
    tracklets_dir: str = "",
    behavior_mode: int = 0,
    allow_without_package: bool = True,
) -> bool:
    """Whether File → Open may load this video into the annotator.

    If an identity package is found (explicit dir or sibling ``id_review``),
    accepted identities are required. With no package, standalone File → Open
    is allowed (``allow_without_package=True``); the workbench Load path
    should pass False and use the project default / detect world instead.

    Args:
        video_path: Video the user chose.
        tracklets_dir: Known identity package, if the caller already resolved one.
        behavior_mode: Fallback mode when no package records one.
        allow_without_package: When True, no package means allow (standalone).

    Returns:
        True if the annotator may open the video.
    """
    directory = str(tracklets_dir or "") or discover_identity_package_for_video(
        video_path
    )
    package = bool(directory) and has_identity_package(directory)
    if not package:
        if allow_without_package:
            return True
        return may_use_downstream(int(behavior_mode), False)
    from LabGym.id_review.raw_store import has_accepted_identities

    accepted = has_accepted_identities(directory)
    mode = behavior_mode_from_package(directory, behavior_mode)
    return may_use_downstream(mode, accepted, identity_package=True)


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
    tracks = str(getattr(ctx, "tracklets_dir", "") or "")
    package = bool(tracks) and has_identity_package(tracks)
    if not may_use_downstream(
        int(ctx.behavior_mode),
        bool(ctx.accepted_identities),
        identity_package=package,
    ):
        return False

    video = str(getattr(ctx, "video_path", "") or "")
    ann = getattr(ctx, "annotations_path", None) or None
    ann_arg = ann if (ann and Path(ann).is_file()) else None
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
