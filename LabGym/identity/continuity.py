"""Live identity association (Hungarian matching to predicted COM)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

# Unbound slots start here so they are not confused with a real last COM.
DUMMY_COM = (-10000, -10000)


@dataclass
class FrameDetections:
    """Per-frame detections for one animal kind.

    Args:
        centers: Detection centers of mass.
    """

    centers: Sequence[Sequence[float]]


@dataclass
class IdentitySlotState:
    """Prior / updated live state for identity slots of one animal kind.

    Args:
        last_centers: Last bound COM per slot index.
        unused_counts: Consecutive unmatched-frame counts.
        last_steps: Last bound displacement (new COM minus previous COM).
    """

    last_centers: dict[int, tuple[float, float]]
    unused_counts: dict[int, int]
    last_steps: dict[int, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def initial(cls, animals_per_kind: int) -> IdentitySlotState:
        """Build unbound dummy-COM slots for a kind that has not been bound yet.

        Args:
            animals_per_kind: Maximum identity slots for this kind.

        Returns:
            Slot state with every slot at ``DUMMY_COM``, zero velocity, and
            unused count 0.
        """
        n = int(animals_per_kind)
        return cls(
            last_centers={i: DUMMY_COM for i in range(n)},
            unused_counts={i: 0 for i in range(n)},
            last_steps={i: (0.0, 0.0) for i in range(n)},
        )


@dataclass
class SlotAssignment:
    """Assignment of this frame's detections to identity slots.

    Args:
        slot_to_detection: Slot index to detection index for bound slots.
        unmatched_slots: Slot indices with no detection this frame.
        extra_detections: Detection indices left unused (dropped; no new slot).
        slot_state: Updated last COMs, velocities, and unused counts (same
            object if the caller passed a mutable state).
    """

    slot_to_detection: dict[int, int] = field(default_factory=dict)
    unmatched_slots: list[int] = field(default_factory=list)
    extra_detections: list[int] = field(default_factory=list)
    slot_state: Optional[IdentitySlotState] = None


def _xy(point: Sequence[float]) -> tuple[float, float]:
    return (float(point[0]), float(point[1]))


def _is_unbound(center: Sequence[float]) -> bool:
    return _xy(center) == (float(DUMMY_COM[0]), float(DUMMY_COM[1]))


def _predicted_center(state: IdentitySlotState, slot: int) -> tuple[float, float]:
    cx, cy = _xy(state.last_centers[slot])
    dx, dy = _xy(state.last_steps.get(slot, (0.0, 0.0)))
    if _is_unbound(state.last_centers[slot]):
        return (cx, cy)
    return (cx + dx, cy + dy)


def _hungarian(
    slot_ids: Sequence[int],
    detections: Sequence[tuple[float, float]],
    predictions: dict[int, tuple[float, float]],
) -> dict[int, int]:
    """Min-cost assignment of slots to detections by predicted-COM distance."""
    if not slot_ids or not detections:
        return {}
    pred_rows = [predictions[s] for s in slot_ids]
    cost = distance.cdist(pred_rows, detections)
    row_ind, col_ind = linear_sum_assignment(cost)
    return {int(slot_ids[int(r)]): int(c) for r, c in zip(row_ind, col_ind)}


def _bind_slot(
    slot_state: IdentitySlotState,
    slot: int,
    center: Sequence[float],
) -> None:
    prev = slot_state.last_centers[slot]
    if _is_unbound(prev):
        slot_state.last_steps[slot] = (0.0, 0.0)
    else:
        px, py = _xy(prev)
        nx, ny = _xy(center)
        slot_state.last_steps[slot] = (nx - px, ny - py)
    slot_state.last_centers[slot] = (center[0], center[1])
    slot_state.unused_counts[slot] = 0


def associate_identity_slots(
    detections: FrameDetections,
    slot_state: IdentitySlotState,
    *,
    animals_per_kind: int,
    count_to_deregister: int,
) -> SlotAssignment:
    """Assign this frame's detections to identity slots for one animal kind.

    Live association is Hungarian matching to constant-velocity predicted
    centers of mass (last center plus last step) over bound slots (active and
    parked). Extra detections are dropped. Unmatched slots park: last COM and
    velocity stay put (no dummy teleport). A parked slot keeps an assignment
    only if that detection is closer to the parked prediction than to any
    active prediction. Unbound (never-bound) slots may take leftovers after
    bound slots are matched. Association is independent per animal kind;
    callers slice detections by kind before calling.

    Args:
        detections: Per-frame detections for this kind (centers required).
        slot_state: Prior last COMs, last steps, and unused-frame counts.
            Updated in place.
        animals_per_kind: Maximum identity slots for this kind. Used to
            initialize dummy slots when ``slot_state`` has none.
        count_to_deregister: Unused-frame timeout retained for the Detect +
            track adapter. Parked slots no longer teleport after this count.

    Returns:
        Slot assignment plus the updated ``slot_state``.
    """
    del count_to_deregister
    if not slot_state.last_centers:
        initialized = IdentitySlotState.initial(animals_per_kind)
        slot_state.last_centers.update(initialized.last_centers)
        slot_state.unused_counts.update(initialized.unused_counts)
        slot_state.last_steps.update(initialized.last_steps)

    for slot in slot_state.last_centers:
        slot_state.last_steps.setdefault(slot, (0.0, 0.0))
        if _is_unbound(slot_state.last_centers[slot]):
            slot_state.last_steps[slot] = (0.0, 0.0)

    raw_centers = list(detections.centers)
    centers = [_xy(c) for c in raw_centers]
    slot_ids = list(slot_state.last_centers)
    predictions = {s: _predicted_center(slot_state, s) for s in slot_ids}

    active: list[int] = []
    parked: list[int] = []
    unbound: list[int] = []
    for slot in slot_ids:
        if _is_unbound(slot_state.last_centers[slot]):
            unbound.append(slot)
        elif int(slot_state.unused_counts.get(slot, 0)) > 0:
            parked.append(slot)
        else:
            active.append(slot)

    slot_to_detection: dict[int, int] = {}
    used_detections: set[int] = set()
    parked_set = set(parked)

    bound_matches = _hungarian(active + parked, centers, predictions)
    for slot, det_i in bound_matches.items():
        if slot in parked_set and active:
            parked_cost = distance.euclidean(predictions[slot], centers[det_i])
            active_cost = min(
                distance.euclidean(predictions[s], centers[det_i]) for s in active
            )
            if parked_cost >= active_cost:
                continue
        slot_to_detection[slot] = det_i
        used_detections.add(det_i)
        _bind_slot(slot_state, slot, raw_centers[det_i])

    leftover = [i for i in range(len(centers)) if i not in used_detections]
    still_unbound = [s for s in unbound if s not in slot_to_detection]
    for slot, det_i in zip(still_unbound, leftover):
        slot_to_detection[slot] = det_i
        used_detections.add(det_i)
        _bind_slot(slot_state, slot, raw_centers[det_i])

    unmatched_slots = [s for s in slot_ids if s not in slot_to_detection]
    for slot in unmatched_slots:
        slot_state.unused_counts[slot] = int(slot_state.unused_counts.get(slot, 0)) + 1

    extra_detections = [i for i in range(len(centers)) if i not in used_detections]
    return SlotAssignment(
        slot_to_detection=slot_to_detection,
        unmatched_slots=unmatched_slots,
        extra_detections=extra_detections,
        slot_state=slot_state,
    )
