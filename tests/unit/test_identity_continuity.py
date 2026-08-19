"""Characterization tests for live greedy last-COM identity association."""

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


def test_one_to_one_nearest_assignment_follows_last_com():
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

    # Detection order reversed; last COM still maps each slot to its neighbor.
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


def test_unmatched_slot_teleports_after_unused_timeout():
    """Pin ``<= count`` increment vs ``> count`` dummy-COM teleport timing.

    Engine timeout is ``count_to_deregister = fps * 2``. Unused count starts
    at 0; increment while ``unused <= count``; teleport on the next unused
    frame without incrementing further.
    """
    fps = 3
    count = fps * 2  # 6
    state = IdentitySlotState.initial(1)
    bound = _associate([(50, 50)], state, 1, count_to_deregister=count)
    assert bound.slot_to_detection == {0: 0}
    assert state.last_centers[0] == (50, 50)
    assert state.unused_counts[0] == 0

    for step in range(count + 1):
        assignment = _associate([], state, 1, count_to_deregister=count)
        assert assignment.slot_to_detection == {}
        assert assignment.unmatched_slots == [0]
        assert state.last_centers[0] == (50, 50)
        assert state.unused_counts[0] == step + 1

    assert state.unused_counts[0] == count + 1
    teleported = _associate([], state, 1, count_to_deregister=count)
    assert teleported.unmatched_slots == [0]
    assert state.last_centers[0] == DUMMY_COM
    assert state.unused_counts[0] == count + 1


def test_two_animal_crossing_swaps_under_greedy_last_com():
    """Known greedy last-COM bug: IDs swap when animals cross.

    Lock the swap; do not correct it in this ticket.
    """
    state = IdentitySlotState.initial(2)
    frames = [
        [(0, 0), (20, 0)],
        [(4, 0), (16, 0)],
        [(8, 0), (12, 0)],
        [(12, 0), (8, 0)],
    ]
    expected = [
        {0: 0, 1: 1},
        {0: 0, 1: 1},
        {0: 0, 1: 1},
        {0: 1, 1: 0},
    ]
    assignments = []
    for centers in frames:
        assignments.append(_associate(centers, state, 2).slot_to_detection)
    assert assignments == expected
    assert state.last_centers[0] == (8, 0)
    assert state.last_centers[1] == (12, 0)


def test_detect_track_sequence_matches_known_greedy_assignments():
    """Synthetic Detect + track sequence: same slot map as pre-extract greedy."""
    state = IdentitySlotState.initial(2)
    frames = [
        [(10, 10), (80, 10)],
        [(12, 10), (78, 10)],
        [(14, 10), (76, 10), (200, 200)],
        [(16, 10)],
        [(18, 10)],
        [(20, 10), (70, 10)],
        [(30, 10), (60, 10)],
        [(40, 10), (50, 10)],
        [(50, 10), (40, 10)],
    ]
    expected_maps = [
        {0: 0, 1: 1},
        {0: 0, 1: 1},
        {0: 0, 1: 1},
        {0: 0},
        {0: 0},
        {0: 0, 1: 1},
        {0: 0, 1: 1},
        {0: 0, 1: 1},
        {0: 1, 1: 0},
    ]
    expected_extras = [[], [], [2], [], [], [], [], [], []]
    expected_unmatched = [[], [], [], [1], [1], [], [], [], []]

    maps = []
    extras = []
    unmatched = []
    for centers in frames:
        assignment = _associate(centers, state, 2)
        maps.append(assignment.slot_to_detection)
        extras.append(assignment.extra_detections)
        unmatched.append(assignment.unmatched_slots)

    assert maps == expected_maps
    assert extras == expected_extras
    assert unmatched == expected_unmatched
    assert state.last_centers[0] == (40, 10)
    assert state.last_centers[1] == (50, 10)
