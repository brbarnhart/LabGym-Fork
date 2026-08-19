"""Tests for live Hungarian predicted-COM identity association."""

from __future__ import annotations

from LabGym.identity.continuity import (
    DUMMY_COM,
    FrameDetections,
    IdentitySlotState,
    associate_identity_slots,
)


def _associate(centers, state, animals_per_kind, count_to_deregister=1000):
    return associate_identity_slots(
        FrameDetections(centers=centers),
        state,
        animals_per_kind=animals_per_kind,
        count_to_deregister=count_to_deregister,
    )


def test_one_to_one_assignment_follows_predicted_com():
    """Two well-separated detections stay on the slots they last bound."""
    state = IdentitySlotState.initial(2)
    first = _associate([(10, 10), (80, 10)], state, 2)
    assert first.slot_to_detection == {0: 0, 1: 1}
    assert state.last_centers[0] == (10, 10)
    assert state.last_centers[1] == (80, 10)

    second = _associate([(12, 12), (78, 11)], state, 2)
    assert second.slot_to_detection == {0: 0, 1: 1}
    assert second.unmatched_slots == []
    assert second.extra_detections == []
    assert state.last_centers[0] == (12, 12)
    assert state.last_centers[1] == (78, 11)

    third = _associate([(79, 10), (11, 11)], state, 2)
    assert third.slot_to_detection == {0: 1, 1: 0}
    assert state.last_centers[0] == (11, 11)
    assert state.last_centers[1] == (79, 10)


def test_extra_detections_dropped_no_third_slot():
    """Animals per kind is 2: a third detection is unused and creates no slot."""
    state = IdentitySlotState.initial(2)
    assignment = _associate([(0, 0), (10, 0), (100, 0)], state, 2)
    assert assignment.slot_to_detection == {0: 0, 1: 1}
    assert assignment.extra_detections == [2]
    assert assignment.unmatched_slots == []
    assert list(state.last_centers) == [0, 1]
    assert 2 not in state.last_centers


def test_unmatched_slot_parks_instead_of_teleporting():
    """A missed animal keeps its last COM; dummy teleport never fires."""
    fps = 3
    count = fps * 2
    state = IdentitySlotState.initial(1)
    bound = _associate([(50, 50)], state, 1, count_to_deregister=count)
    assert bound.slot_to_detection == {0: 0}
    assert state.last_centers[0] == (50, 50)
    assert state.unused_counts[0] == 0

    for step in range(count + 3):
        assignment = _associate([], state, 1, count_to_deregister=count)
        assert assignment.slot_to_detection == {}
        assert assignment.unmatched_slots == [0]
        assert state.last_centers[0] == (50, 50)
        assert state.last_centers[0] != DUMMY_COM
        assert state.unused_counts[0] == step + 1


def test_two_animal_chase_keeps_identity_slots():
    """Close pursuit keeps slots via predicted COM, not greedy last COM.

    Leader walks slowly; pursuer closes from behind. At the last frame the
    pursuer is nearer the leader's last center than the leader is, so greedy
    last-COM would swap. Constant-velocity prediction keeps each slot.
    """
    state = IdentitySlotState.initial(2)
    frames = [
        [(0, 0), (25, 0)],
        [(10, 0), (28, 0)],
        [(20, 0), (31, 0)],
        [(30, 0), (34, 0)],
    ]
    maps = []
    for centers in frames:
        maps.append(_associate(centers, state, 2).slot_to_detection)
    assert maps == [{0: 0, 1: 1}] * len(frames)
    assert state.last_centers[0] == (30, 0)
    assert state.last_centers[1] == (34, 0)


def test_two_animal_crossing_keeps_identity_slots():
    """Animals that cross keep identity via predicted motion."""
    state = IdentitySlotState.initial(2)
    frames = [
        [(0, 0), (20, 0)],
        [(4, 0), (16, 0)],
        [(8, 0), (12, 0)],
        [(12, 0), (8, 0)],
    ]
    maps = []
    for centers in frames:
        maps.append(_associate(centers, state, 2).slot_to_detection)
    assert maps == [{0: 0, 1: 1}] * len(frames)
    assert state.last_centers[0] == (12, 0)
    assert state.last_centers[1] == (8, 0)


def test_parked_slot_does_not_bind_detection_better_matched_to_active():
    """Parked animal does not snap onto a leftover blob next to someone present."""
    state = IdentitySlotState.initial(2)
    _associate([(10, 10), (80, 10)], state, 2)
    _associate([(10, 10), (80, 10)], state, 2)

    only_right = _associate([(82, 10)], state, 2)
    assert only_right.slot_to_detection == {1: 0}
    assert only_right.unmatched_slots == [0]
    assert state.last_centers[0] == (10, 10)

    extras_on_right = _associate([(78, 10), (84, 10)], state, 2)
    assert 0 not in extras_on_right.slot_to_detection
    assert extras_on_right.unmatched_slots == [0]
    assert len(extras_on_right.extra_detections) == 1
    assert extras_on_right.slot_to_detection.keys() == {1}
    assert state.last_centers[0] == (10, 10)
    assert state.last_centers[1] in ((78, 10), (84, 10))


def test_parked_slot_does_not_snap_across_the_arena():
    """A leftover far from the parked animal stays extra, not a teleport."""
    state = IdentitySlotState.initial(2)
    _associate([(10, 10), (80, 10)], state, 2)
    _associate([(10, 10), (80, 10)], state, 2)
    _associate([(82, 10)], state, 2)

    far = _associate([(80, 10), (200, 200)], state, 2)
    assert 0 not in far.slot_to_detection
    assert far.slot_to_detection == {1: 0}
    assert far.unmatched_slots == [0]
    assert far.extra_detections == [1]
    assert state.last_centers[0] == (10, 10)


def test_returning_parked_animal_keeps_slot_when_the_other_is_missed():
    """A blob at the parked animal is not given to the still-active slot."""
    state = IdentitySlotState.initial(2)
    _associate([(10, 10), (80, 10)], state, 2)
    _associate([(10, 10), (80, 10)], state, 2)
    _associate([(82, 10)], state, 2)

    returned = _associate([(12, 10)], state, 2)
    assert returned.slot_to_detection == {0: 0}
    assert returned.unmatched_slots == [1]
    assert returned.extra_detections == []
    assert state.last_centers[0] == (12, 10)
    assert state.last_centers[1] == (82, 10)


def test_leftover_detection_may_claim_parked_slot_when_closer_to_it():
    """A leftover nearer the parked animal than anyone active reclaims that slot."""
    state = IdentitySlotState.initial(2)
    _associate([(10, 10), (80, 10)], state, 2)
    _associate([(10, 10), (80, 10)], state, 2)
    parked = _associate([(82, 10)], state, 2)
    assert parked.unmatched_slots == [0]

    returned = _associate([(12, 10), (80, 10)], state, 2)
    assert returned.slot_to_detection == {0: 0, 1: 1}
    assert returned.unmatched_slots == []
    assert returned.extra_detections == []
    assert state.last_centers[0] == (12, 10)
    assert state.last_centers[1] == (80, 10)


def test_last_steps_persist_when_state_is_rebuilt_around_same_dicts():
    """Detect + track wraps engine dicts in a new state object every frame."""
    last_centers = {}
    unused_counts = {}
    last_steps = {}
    frames = [
        [(0, 0), (25, 0)],
        [(10, 0), (28, 0)],
        [(20, 0), (31, 0)],
        [(30, 0), (34, 0)],
    ]
    maps = []
    for centers in frames:
        state = IdentitySlotState(
            last_centers=last_centers,
            unused_counts=unused_counts,
            last_steps=last_steps,
        )
        maps.append(_associate(centers, state, 2).slot_to_detection)
    assert maps == [{0: 0, 1: 1}] * len(frames)
    assert last_centers[0] == (30, 0)
    assert last_centers[1] == (34, 0)


def test_association_is_independent_per_animal_kind():
    """Each kind has its own slots; a mouse never takes an object slot."""
    mice = IdentitySlotState.initial(2)
    objects = IdentitySlotState.initial(1)
    mouse_asg = _associate([(5, 5), (50, 5)], mice, 2)
    object_asg = _associate([(200, 200)], objects, 1)
    assert mouse_asg.slot_to_detection == {0: 0, 1: 1}
    assert object_asg.slot_to_detection == {0: 0}
    assert list(mice.last_centers) == [0, 1]
    assert list(objects.last_centers) == [0]
    assert mice.last_centers[0] == (5, 5)
    assert objects.last_centers[0] == (200, 200)
