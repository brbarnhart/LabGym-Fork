"""Live identity association (greedy last-center-of-mass default)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from scipy.spatial import distance

# Far dummy last COM used when an unmatched slot times out (current teleport).
DUMMY_COM = (-10000, -10000)


@dataclass
class FrameDetections:
    """Per-frame detections for one animal kind.

    Args:
        centers: Detection centers of mass. Required for live association.
        contours: Optional outlines; unused by the greedy default, passed
            through for later association tickets.
        areas: Optional detection areas; unused by the greedy default.
    """

    centers: Sequence[Sequence[float]]
    contours: Optional[Sequence[Any]] = None
    areas: Optional[Sequence[float]] = None


@dataclass
class IdentitySlotState:
    """Prior / updated live state for identity slots of one animal kind.

    Args:
        last_centers: Last bound COM per slot index.
        unused_counts: Consecutive unmatched-frame counts (``to_deregister``).
    """

    last_centers: dict[int, tuple[float, float]]
    unused_counts: dict[int, int]

    @classmethod
    def initial(cls, animals_per_kind: int) -> IdentitySlotState:
        """Build dummy-COM slots for a kind that has not been bound yet.

        Args:
            animals_per_kind: Maximum identity slots for this kind.

        Returns:
            Slot state with every slot at ``DUMMY_COM`` and unused count 0.
        """
        n = int(animals_per_kind)
        return cls(
            last_centers={i: DUMMY_COM for i in range(n)},
            unused_counts={i: 0 for i in range(n)},
        )


@dataclass
class SlotAssignment:
    """Assignment of this frame's detections to identity slots.

    Args:
        slot_to_detection: Slot index to detection index for bound slots.
        unmatched_slots: Slot indices with no detection this frame.
        extra_detections: Detection indices left unused (dropped; no new slot).
        slot_state: Updated last COMs and unused counts (same object if the
            caller passed a mutable state).
    """

    slot_to_detection: dict[int, int] = field(default_factory=dict)
    unmatched_slots: list[int] = field(default_factory=list)
    extra_detections: list[int] = field(default_factory=list)
    slot_state: Optional[IdentitySlotState] = None


def associate_identity_slots(
    detections: FrameDetections,
    slot_state: IdentitySlotState,
    *,
    animals_per_kind: int,
    count_to_deregister: int,
) -> SlotAssignment:
    """Assign this frame's detections to identity slots for one animal kind.

    Default live behavior is greedy nearest last-center-of-mass matching
    (numpy flatten / argsort index arithmetic, including equal-distance ties).
    Extra detections are dropped. Unmatched slots increment an unused count
    while ``unused <= count_to_deregister`` and then teleport last COM to
    ``DUMMY_COM``. Empty ``centers`` skips the greedy loop and treats every
    slot as unmatched. First-frame binding is all slots at the dummy COM.

    Association is independent per animal kind; callers slice detections by
    kind before calling.

    Args:
        detections: Per-frame detections for this kind (centers required).
        slot_state: Prior last COMs and unused-frame counts. Updated in place.
        animals_per_kind: Maximum identity slots for this kind. Used to
            initialize dummy slots when ``slot_state`` has none.
        count_to_deregister: Unused-frame timeout (engine uses ``fps * 2``).

    Returns:
        Slot assignment plus the updated ``slot_state``.
    """
    if not slot_state.last_centers:
        initialized = IdentitySlotState.initial(animals_per_kind)
        slot_state.last_centers.update(initialized.last_centers)
        slot_state.unused_counts.update(initialized.unused_counts)

    centers = list(detections.centers)
    unused_existing_indices = list(slot_state.last_centers)
    unused_new_indices = list(range(len(centers)))
    slot_to_detection: dict[int, int] = {}
    length = len(centers)

    # length == 0: skip greedy (cdist on an empty new set is undefined) and
    # treat every slot as unmatched — same as the engine's empty loop.
    if length != 0:
        existing_centers = list(slot_state.last_centers.values())
        dt_flattened = distance.cdist(existing_centers, centers).flatten()
        dt_sort_index = dt_flattened.argsort()
        for idx in dt_sort_index:
            index_in_existing = int(idx / length)
            index_in_new = int(idx % length)
            if index_in_existing in unused_existing_indices:
                if index_in_new in unused_new_indices:
                    unused_existing_indices.remove(index_in_existing)
                    unused_new_indices.remove(index_in_new)
                    slot_to_detection[index_in_existing] = index_in_new
                    slot_state.unused_counts[index_in_existing] = 0
                    slot_state.last_centers[index_in_existing] = centers[index_in_new]

    if len(unused_existing_indices) > 0:
        for i in unused_existing_indices:
            if slot_state.unused_counts[i] <= count_to_deregister:
                slot_state.unused_counts[i] += 1
            else:
                slot_state.last_centers[i] = DUMMY_COM

    return SlotAssignment(
        slot_to_detection=slot_to_detection,
        unmatched_slots=list(unused_existing_indices),
        extra_detections=list(unused_new_indices),
        slot_state=slot_state,
    )
