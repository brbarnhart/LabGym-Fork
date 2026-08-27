"""Session-dirty gate for **project swap** (Open / Open Recent / Edit Project).

Unsaved **switch markers** and an unsaved ethogram block swap until Save or
Discard. This is independent of the ``*.labproj.json`` dirty flag.

Clean sessions are not prompted. The session stays loaded until a **project
swap** is committed (``load_from_path`` / ``new_project`` / editor OK);
``project_replaced`` on the owning tabs is the drop.

Proposed switch markers are not switch markers and must not be reported here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, Sequence

CHOICE_SAVE = "save"
CHOICE_DISCARD = "discard"
CHOICE_CANCEL = "cancel"

KIND_SWITCH_MARKERS = "switch_markers"
KIND_ETHOGRAM = "ethogram"


class ReviewIdsSession(Protocol):
    """Review IDs session consulted on **project swap**."""

    def has_unsaved_switch_markers(self) -> bool:
        """True when accepted switch markers differ from the last save/load."""

    def save_switch_markers(self) -> bool:
        """Persist switch markers (Review IDs save). Return False on failure."""

    def discard_unsaved_switch_markers(self) -> None:
        """Revert in-memory switch markers; do not write."""

    def drop_session(self) -> None:
        """Release capture and tracklet arrays; clear the viewer."""


class EthogramSession(Protocol):
    """Annotate ethogram session consulted on **project swap**."""

    def has_unsaved_ethogram(self) -> bool:
        """True when the in-memory ethogram differs from the last save/load."""

    def save_ethogram(self) -> bool:
        """Persist the ethogram. Return False on failure."""

    def discard_unsaved_ethogram(self) -> None:
        """Revert in-memory ethogram edits; do not write."""

    def drop_session(self) -> None:
        """Release capture and tracklet arrays; clear the viewer."""


@dataclass(frozen=True)
class SessionSwapReport:
    """Which session kinds are dirty (not the project file)."""

    unsaved_switch_markers: bool
    unsaved_ethogram: bool

    @property
    def is_dirty(self) -> bool:
        """True when Save or Discard is required before **project swap**."""
        return self.unsaved_switch_markers or self.unsaved_ethogram

    def kinds(self) -> tuple[str, ...]:
        """Stable identifiers for the dirty session kinds."""
        out: list[str] = []
        if self.unsaved_switch_markers:
            out.append(KIND_SWITCH_MARKERS)
        if self.unsaved_ethogram:
            out.append(KIND_ETHOGRAM)
        return tuple(out)


def switch_markers_signature(markers: Sequence[Any]) -> str:
    """Canonical JSON for comparing **switch marker** lists.

    Args:
        markers: SwitchMarker-like objects with ``to_dict``, or mappings.

    Returns:
        Stable JSON string. Empty input is ``[]``.
    """
    rows = []
    for m in markers:
        if hasattr(m, "to_dict"):
            rows.append(m.to_dict())
        elif isinstance(m, dict):
            rows.append(m)
        else:
            rows.append(repr(m))
    return json.dumps(rows, sort_keys=True, default=str)


def inspect_sessions(
    *,
    review_ids: Optional[ReviewIdsSession] = None,
    ethogram: Optional[EthogramSession] = None,
) -> SessionSwapReport:
    """Read unsaved switch markers / ethogram without touching the project flag.

    Args:
        review_ids: Review IDs session, or None if the tab is absent.
        ethogram: Annotate ethogram session, or None if the tab is absent.

    Returns:
        Report used to decide whether to prompt Save or Discard.
    """
    return SessionSwapReport(
        unsaved_switch_markers=bool(
            review_ids is not None and review_ids.has_unsaved_switch_markers()
        ),
        unsaved_ethogram=bool(
            ethogram is not None and ethogram.has_unsaved_ethogram()
        ),
    )


def resolve_unsaved_session(
    *,
    review_ids: Optional[ReviewIdsSession] = None,
    ethogram: Optional[EthogramSession] = None,
    prompt: Optional[Callable[[SessionSwapReport], str]] = None,
) -> bool:
    """Prompt Save or Discard when the session is dirty; do not drop yet.

    A clean session returns True without calling *prompt*. Cancel or a failed
    save returns False and leaves session memory in place. Discard reverts
    unsaved switch markers / ethogram in memory and does not write.

    Save is not atomic across kinds: Review IDs is written first, then the
    ethogram. If the second write fails, the first stays on disk, the swap
    is blocked, and the session is not dropped.

    Args:
        review_ids: Review IDs session host.
        ethogram: Annotate ethogram session host.
        prompt: Called with the dirty report; must return ``CHOICE_SAVE``,
            ``CHOICE_DISCARD``, or ``CHOICE_CANCEL``. Required when dirty.

    Returns:
        True if **project swap** may continue (session is clean or resolved).
    """
    report = inspect_sessions(review_ids=review_ids, ethogram=ethogram)
    if not report.is_dirty:
        return True
    if prompt is None:
        return False
    choice = prompt(report)
    if choice == CHOICE_CANCEL:
        return False
    if choice == CHOICE_SAVE:
        if report.unsaved_switch_markers and review_ids is not None:
            if not review_ids.save_switch_markers():
                return False
        if report.unsaved_ethogram and ethogram is not None:
            if not ethogram.save_ethogram():
                return False
        return True
    if choice == CHOICE_DISCARD:
        if report.unsaved_switch_markers and review_ids is not None:
            review_ids.discard_unsaved_switch_markers()
        if report.unsaved_ethogram and ethogram is not None:
            ethogram.discard_unsaved_ethogram()
        return True
    return False


def drop_sessions(
    *,
    review_ids: Optional[ReviewIdsSession] = None,
    ethogram: Optional[EthogramSession] = None,
) -> None:
    """Release capture and tracklet arrays for loaded sessions.

    Idempotent. Does not mark the project dirty or start a **session load**.

    Args:
        review_ids: Review IDs session host.
        ethogram: Annotate ethogram session host.
    """
    if review_ids is not None:
        review_ids.drop_session()
    if ethogram is not None:
        ethogram.drop_session()
