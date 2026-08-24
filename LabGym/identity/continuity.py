"""Live identity association (Hungarian matching, split detection, occlusion freeze)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

# Unbound slots start here so they are not confused with a real last COM.
DUMMY_COM = (-10000, -10000)

# Occlusion / split geometry. Module constants, not user-facing controls.
OUTLINE_DILATION_PX = 3
CONTOUR_GAP_PX = 4.0
AREA_COLLAPSE_RATIO = 0.5
UNMATCHED_NEAR_SIZE_FACTOR = 1.5
# Overlapping outlines are a split only if their COMs sit on one body.
# Two mice that almost touch have COMs ~one animal-size apart.
SPLIT_SAME_BODY_COM_FACTOR = 0.5
# If every stolen-to slot still has its last COM near the overlapping
# cluster, the animals are present in a pile — do not park one of them.
SPLIT_PILE_LAST_COM_FACTOR = 2.0


@dataclass
class FrameDetections:
    """Per-frame detections for one animal kind.

    Args:
        centers: Detection centers of mass.
        outlines: Optional raw outlines (OpenCV contours or point sequences),
            aligned with ``centers``. Used for split detection and occlusion.
        areas: Optional contour areas, aligned with ``centers``.
    """

    centers: Sequence[Sequence[float]]
    outlines: Sequence = ()
    areas: Sequence = ()


@dataclass
class IdentitySlotState:
    """Prior / updated live state for identity slots of one animal kind.

    Args:
        last_centers: Last bound COM per slot index.
        unused_counts: Consecutive unmatched-frame counts.
        last_steps: Last bound displacement (new COM minus previous COM).
        typical_area: Running typical contour area for this kind (collapse test).
    """

    last_centers: dict[int, tuple[float, float]]
    unused_counts: dict[int, int]
    last_steps: dict[int, tuple[float, float]] = field(default_factory=dict)
    typical_area: Optional[float] = None

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
        split_detection: True when overlapping outlines refused a steal.
        occlusion_bout: True when composite occlusion tests fired this frame.
        frozen_slots: Identity slots whose last COM and velocity were not
            updated because they are in an occlusion bout.
    """

    slot_to_detection: dict[int, int] = field(default_factory=dict)
    unmatched_slots: list[int] = field(default_factory=list)
    extra_detections: list[int] = field(default_factory=list)
    slot_state: Optional[IdentitySlotState] = None
    split_detection: bool = False
    occlusion_bout: bool = False
    frozen_slots: list[int] = field(default_factory=list)


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


def _outline_points(outline) -> Optional[np.ndarray]:
    """Nx2 float points, or None if the outline is missing."""
    if outline is None:
        return None
    arr = np.asarray(outline, dtype=np.float64)
    if arr.size == 0:
        return None
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr.reshape(-1, 2)
    elif arr.ndim == 1:
        return None
    else:
        arr = arr.reshape(-1, 2)
    if arr.shape[0] < 3:
        return None
    return arr


def _as_cv_contour(outline) -> Optional[np.ndarray]:
    points = _outline_points(outline)
    if points is None:
        return None
    return np.round(points).astype(np.int32).reshape(-1, 1, 2)


def _contour_area(outline, provided: Optional[float] = None) -> Optional[float]:
    if provided is not None:
        return float(provided)
    contour = _as_cv_contour(outline)
    if contour is None:
        return None
    return float(cv2.contourArea(contour))


def _pair_masks(outline_a, outline_b, dilation: int = 0) -> Optional[tuple]:
    ca = _as_cv_contour(outline_a)
    cb = _as_cv_contour(outline_b)
    if ca is None or cb is None:
        return None
    xa, ya, wa, ha = cv2.boundingRect(ca)
    xb, yb, wb, hb = cv2.boundingRect(cb)
    pad = max(1, int(dilation) + 1)
    x0 = min(xa, xb) - pad
    y0 = min(ya, yb) - pad
    x1 = max(xa + wa, xb + wb) + pad
    y1 = max(ya + ha, yb + hb) + pad
    if x1 <= x0 or y1 <= y0:
        return None
    shape = (y1 - y0, x1 - x0)
    ma = np.zeros(shape, dtype=np.uint8)
    mb = np.zeros(shape, dtype=np.uint8)
    origin = np.array([[[x0, y0]]])
    cv2.drawContours(ma, [ca - origin], -1, 1, thickness=-1)
    cv2.drawContours(mb, [cb - origin], -1, 1, thickness=-1)
    if dilation > 0:
        k = 2 * int(dilation) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        ma = cv2.dilate(ma, kernel)
        mb = cv2.dilate(mb, kernel)
    return ma, mb


def _raw_outlines_overlap(outline_a, outline_b) -> bool:
    """True when two raw outlines intersect (split-detection geometry)."""
    masks = _pair_masks(outline_a, outline_b, dilation=0)
    if masks is None:
        return False
    return bool(np.any(masks[0] & masks[1]))


def _fattened_outlines_intersect(outline_a, outline_b) -> bool:
    masks = _pair_masks(outline_a, outline_b, dilation=OUTLINE_DILATION_PX)
    if masks is None:
        return False
    return bool(np.any(masks[0] & masks[1]))


def _min_contour_gap(outline_a, outline_b) -> Optional[float]:
    pa = _outline_points(outline_a)
    pb = _outline_points(outline_b)
    if pa is None or pb is None:
        return None
    if _raw_outlines_overlap(outline_a, outline_b):
        return 0.0
    return float(distance.cdist(pa, pb).min())


def _animal_size(slot_state: IdentitySlotState, detections: FrameDetections) -> float:
    if slot_state.typical_area is not None and slot_state.typical_area > 0:
        return float(math.sqrt(slot_state.typical_area))
    areas: list[float] = []
    n = len(list(detections.centers))
    for i in range(n):
        area = _detection_area(detections, i)
        if area is not None and area > 0:
            areas.append(float(area))
    if areas:
        return float(math.sqrt(float(np.median(np.array(areas)))))
    return 10.0


def _detection_area(detections: FrameDetections, index: int) -> Optional[float]:
    areas = list(detections.areas) if detections.areas is not None else []
    outlines = list(detections.outlines) if detections.outlines is not None else []
    provided = areas[index] if index < len(areas) else None
    outline = outlines[index] if index < len(outlines) else None
    return _contour_area(outline, provided)


def _apply_split_detection(
    slot_to_detection: dict[int, int],
    used_detections: set[int],
    outlines: Sequence,
    centers: Sequence[tuple[float, float]],
    predictions: dict[int, tuple[float, float]],
    animal_size: float,
    last_centers: dict[int, tuple[float, float]],
) -> bool:
    """Unassign stolen slots when overlapping outlines belong to one animal.

    If two or more assigned detections have overlapping raw outlines *and*
    their centers of mass sit on one body, keep the slot whose predicted COM
    is nearest the overlapping cluster and park the others. Two animals that
    almost touch (COMs about one body apart) keep both assignments; occlusion
    freeze handles that. Returns True when a steal was refused.
    """
    assigned = list(slot_to_detection.items())
    if len(assigned) < 2:
        return False

    parent = {i: i for i in range(len(assigned))}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(assigned)):
        for j in range(i + 1, len(assigned)):
            d1 = assigned[i][1]
            d2 = assigned[j][1]
            o1 = outlines[d1] if d1 < len(outlines) else None
            o2 = outlines[d2] if d2 < len(outlines) else None
            if _raw_outlines_overlap(o1, o2):
                union(i, j)

    clusters: dict[int, list[tuple[int, int]]] = {}
    for i, pair in enumerate(assigned):
        clusters.setdefault(find(i), []).append(pair)

    stole = False
    for members in clusters.values():
        if len(members) < 2:
            continue
        det_idxs = [d for _, d in members]
        max_com = 0.0
        for i in range(len(det_idxs)):
            for j in range(i + 1, len(det_idxs)):
                max_com = max(
                    max_com,
                    float(distance.euclidean(centers[det_idxs[i]], centers[det_idxs[j]])),
                )
        if max_com >= SPLIT_SAME_BODY_COM_FACTOR * max(float(animal_size), 1.0):
            continue
        cx = sum(centers[d][0] for d in det_idxs) / len(det_idxs)
        cy = sum(centers[d][1] for d in det_idxs) / len(det_idxs)
        size = max(float(animal_size), 1.0)
        if all(
            not _is_unbound(last_centers[s])
            and distance.euclidean(_xy(last_centers[s]), (cx, cy))
            <= SPLIT_PILE_LAST_COM_FACTOR * size
            for s, _ in members
        ):
            continue
        slots = [s for s, _ in members]
        votes: dict[int, int] = {s: 0 for s in slots}
        for _, det_i in members:
            nearest = min(
                slots,
                key=lambda s: distance.euclidean(predictions[s], centers[det_i]),
            )
            votes[nearest] += 1
        # Split: every overlapping blob prefers one slot (the extra outline is
        # a steal). A pile splits votes across slots — leave both assigned.
        if max(votes.values()) < len(members):
            continue
        owner_slot, _ = min(
            members,
            key=lambda sd: distance.euclidean(predictions[sd[0]], (cx, cy)),
        )
        for slot, det_i in members:
            if slot == owner_slot:
                continue
            slot_to_detection.pop(slot, None)
            used_detections.discard(det_i)
            stole = True
    return stole


def _occlusion_involved_slots(
    detections: FrameDetections,
    slot_state: IdentitySlotState,
    slot_to_detection: dict[int, int],
    unmatched_slots: Sequence[int],
    predictions: dict[int, tuple[float, float]],
    centers: Sequence[tuple[float, float]],
    outlines: Sequence,
) -> list[int]:
    """Slots in an occlusion bout this frame, or empty if tests do not fire.

    Composite test: fattened-outline intersection, small raw-contour gap,
    collapsed detection area next to another detection, or an unmatched slot
    with a leftover/remaining detection nearby. Mere center-of-mass closeness
    is not enough. Split leftovers are not an occlusion bout.
    """
    involved: set[int] = set()
    assigned = list(slot_to_detection.items())
    size = _animal_size(slot_state, detections)
    near = UNMATCHED_NEAR_SIZE_FACTOR * size

    for i in range(len(assigned)):
        for j in range(i + 1, len(assigned)):
            s1, d1 = assigned[i]
            s2, d2 = assigned[j]
            o1 = outlines[d1] if d1 < len(outlines) else None
            o2 = outlines[d2] if d2 < len(outlines) else None
            pair = False
            if _fattened_outlines_intersect(o1, o2):
                pair = True
            gap = _min_contour_gap(o1, o2)
            if gap is not None and gap <= CONTOUR_GAP_PX:
                pair = True
            a1 = _detection_area(detections, d1)
            a2 = _detection_area(detections, d2)
            typical = slot_state.typical_area
            if typical is None or typical <= 0:
                typical = size * size if size > 0 else None
            if typical is not None and typical > 0:
                com_dist = distance.euclidean(centers[d1], centers[d2])
                collapsed = (
                    (a1 is not None and a1 < AREA_COLLAPSE_RATIO * typical)
                    or (a2 is not None and a2 < AREA_COLLAPSE_RATIO * typical)
                )
                if collapsed and com_dist <= 2.0 * size:
                    pair = True
            if pair:
                involved.add(s1)
                involved.add(s2)

    extras = [i for i in range(len(centers)) if i not in set(slot_to_detection.values())]
    leftover_or_assigned = extras + [d for _, d in assigned]
    for slot in unmatched_slots:
        pred = predictions.get(slot, _xy(slot_state.last_centers[slot]))
        if _is_unbound(slot_state.last_centers[slot]):
            continue
        for det_i in leftover_or_assigned:
            if distance.euclidean(pred, centers[det_i]) <= near:
                involved.add(slot)
                for other_slot, other_det in assigned:
                    if other_det == det_i:
                        involved.add(other_slot)
                break

    return sorted(involved)


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
    enable_split_detection: bool = True,
    enable_occlusion_freeze: bool = True,
) -> SlotAssignment:
    """Assign this frame's detections to identity slots for one animal kind.

    Live association is Hungarian matching to constant-velocity predicted
    centers of mass (last center plus last step) over bound slots (active and
    parked). Extra detections are dropped. Unmatched slots park: last COM and
    velocity stay put (no dummy teleport). A parked slot keeps an assignment
    only if that detection is closer to the parked prediction than to any
    active prediction. Unbound (never-bound) slots may take leftovers after
    bound slots are matched. Split detection (raw outline overlap plus an
    unmatched slot) refuses the steal. During an occlusion bout, last COM and
    velocity do not update off the pile; rematch at freeze-lift uses the last
    pre-freeze identities. Association is independent per animal kind;
    callers slice detections by kind before calling.

    Args:
        detections: Per-frame detections for this kind (centers required).
        slot_state: Prior last COMs, last steps, and unused-frame counts.
            Updated in place.
        animals_per_kind: Maximum identity slots for this kind. Used to
            initialize dummy slots when ``slot_state`` has none.
        count_to_deregister: Unused-frame timeout retained for the Detect +
            track adapter. Parked slots no longer teleport after this count.
        enable_split_detection: When False, overlapping outlines may bind two
            slots (Hungarian-only). Troubleshooting toggle.
        enable_occlusion_freeze: When False, last COM and velocity still update
            during almost-touch / pile frames. Troubleshooting toggle.

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

    outlines = list(detections.outlines) if detections.outlines is not None else []

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

    animal_size = _animal_size(slot_state, detections)
    split_detection = False
    if enable_split_detection:
        split_detection = _apply_split_detection(
            slot_to_detection,
            used_detections,
            outlines,
            centers,
            predictions,
            animal_size,
            slot_state.last_centers,
        )

    leftover = [i for i in range(len(centers)) if i not in used_detections]
    still_unbound = [s for s in unbound if s not in slot_to_detection]
    assigned_outlines = [
        outlines[d] if d < len(outlines) else None
        for d in slot_to_detection.values()
    ]
    claimable: list[int] = []
    for det_i in leftover:
        leftover_outline = outlines[det_i] if det_i < len(outlines) else None
        if (
            enable_split_detection
            and leftover_outline is not None
            and any(
                _raw_outlines_overlap(leftover_outline, other)
                for other in assigned_outlines
                if other is not None
            )
        ):
            split_detection = True
            continue
        claimable.append(det_i)
    for slot, det_i in zip(still_unbound, claimable):
        slot_to_detection[slot] = det_i
        used_detections.add(det_i)

    unmatched_slots = [s for s in slot_ids if s not in slot_to_detection]
    frozen_slots: list[int] = []
    occlusion_bout = False
    if enable_occlusion_freeze and not split_detection:
        frozen_slots = _occlusion_involved_slots(
            detections,
            slot_state,
            slot_to_detection,
            unmatched_slots,
            predictions,
            centers,
            outlines,
        )
        occlusion_bout = bool(frozen_slots)
    frozen_set = set(frozen_slots)

    for slot, det_i in slot_to_detection.items():
        if slot in frozen_set and not _is_unbound(slot_state.last_centers[slot]):
            slot_state.unused_counts[slot] = 0
            continue
        _bind_slot(slot_state, slot, raw_centers[det_i])

    for slot in unmatched_slots:
        slot_state.unused_counts[slot] = int(slot_state.unused_counts.get(slot, 0)) + 1

    extra_detections = [i for i in range(len(centers)) if i not in used_detections]
    return SlotAssignment(
        slot_to_detection=slot_to_detection,
        unmatched_slots=unmatched_slots,
        extra_detections=extra_detections,
        slot_state=slot_state,
        split_detection=split_detection,
        occlusion_bout=occlusion_bout,
        frozen_slots=frozen_slots,
    )
