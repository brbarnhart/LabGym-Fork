"""Tests for live Hungarian predicted-COM identity association."""

from __future__ import annotations

import numpy as np

from LabGym.identity.continuity import (
    DUMMY_COM,
    FrameDetections,
    IdentitySlotState,
    associate_identity_slots,
)


def _box(cx, cy, w=10, h=10):
    """Axis-aligned rectangle outline centered at (cx, cy), OpenCV contour shape."""
    x0 = int(round(cx - w / 2.0))
    y0 = int(round(cy - h / 2.0))
    x1 = int(round(cx + w / 2.0))
    y1 = int(round(cy + h / 2.0))
    return np.array(
        [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]],
        dtype=np.int32,
    )


def _associate(
    centers,
    state,
    animals_per_kind,
    count_to_deregister=1000,
    outlines=None,
    areas=None,
):
    return associate_identity_slots(
        FrameDetections(
            centers=centers,
            outlines=() if outlines is None else outlines,
            areas=() if areas is None else areas,
        ),
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


def _bind_left_right(state):
    """Two well-separated animals: slot 0 left, slot 1 right, with velocity."""
    _associate([(10, 50), (80, 50)], state, 2)
    _associate([(12, 50), (82, 50)], state, 2)


def test_split_detection_parks_unmatched_slot_instead_of_stealing():
    """Overlapping outlines on one animal do not bind both slots.

    Two overlapping boxes sit on the left animal; the right animal is missed.
    Hungarian would give the extra outline to the right slot. Split detection
    parks that slot and leaves the extra detection unused.
    """
    state = IdentitySlotState.initial(2)
    _bind_left_right(state)

    split = _associate(
        [(12, 50), (14, 50)],
        state,
        2,
        outlines=[_box(12, 50), _box(14, 50)],
    )
    assert split.split_detection is True
    assert split.occlusion_bout is False
    assert 0 in split.slot_to_detection
    assert 1 not in split.slot_to_detection
    assert split.unmatched_slots == [1]
    assert len(split.extra_detections) == 1
    assert split.extra_detections[0] not in split.slot_to_detection.values()
    assert state.last_centers[1] == (82, 50)


def test_split_detection_parks_when_missed_animal_is_nearby():
    """Overlapping outlines on one animal still park a close unmatched slot."""
    state = IdentitySlotState.initial(2)
    _associate(
        [(20, 50), (50, 50)],
        state,
        2,
        outlines=[_box(20, 50), _box(50, 50)],
    )
    _associate(
        [(28, 50), (52, 50)],
        state,
        2,
        outlines=[_box(28, 50), _box(52, 50)],
    )
    split = _associate(
        [(28, 50), (30, 50)],
        state,
        2,
        outlines=[_box(28, 50), _box(30, 50)],
    )
    assert split.split_detection is True
    assert 0 in split.slot_to_detection
    assert 1 not in split.slot_to_detection
    assert split.unmatched_slots == [1]
    assert state.last_centers[1] == (52, 50)


def test_one_or_two_frame_split_does_not_leave_lasting_swap():
    """After a short split on the right animal, identities recover.

    Two overlapping outlines on the moving right animal would steal the left
    slot onto that body. Once both animals are detected again, the parked left
    slot still sits on the left and rematches there.
    """
    state = IdentitySlotState.initial(2)
    _associate([(10, 50), (70, 50)], state, 2)
    _associate([(10, 50), (78, 50)], state, 2)

    for _ in range(2):
        split = _associate(
            [(78, 50), (80, 50)],
            state,
            2,
            outlines=[_box(78, 50), _box(80, 50)],
        )
        assert 0 not in split.slot_to_detection
        assert split.unmatched_slots == [0]
        assert state.last_centers[0] == (10, 50)

    recovered = _associate(
        [(10, 50), (86, 50)],
        state,
        2,
        outlines=[_box(10, 50, w=10), _box(86, 50, w=10)],
    )
    assert recovered.slot_to_detection == {0: 0, 1: 1}
    assert recovered.unmatched_slots == []
    assert state.last_centers[0] == (10, 50)
    assert state.last_centers[1] == (86, 50)


def test_first_frame_pile_still_binds_unbound_slots():
    """Unbound slots still claim detections even if the first frame is a pile."""
    state = IdentitySlotState.initial(2)
    first = _associate(
        [(44, 50), (56, 50)],
        state,
        2,
        outlines=[_box(44, 50), _box(56, 50)],
    )
    assert first.slot_to_detection.keys() == {0, 1}
    assert state.last_centers[0] != DUMMY_COM
    assert state.last_centers[1] != DUMMY_COM


def test_intersecting_outlines_of_two_animals_are_occlusion_not_split():
    """Raw overlap of two piled animals freezes; it does not park one slot."""
    state = IdentitySlotState.initial(2)
    _associate(
        [(20, 50), (80, 50)],
        state,
        2,
        outlines=[_box(20, 50), _box(80, 50)],
    )
    _associate(
        [(30, 50), (70, 50)],
        state,
        2,
        outlines=[_box(30, 50), _box(70, 50)],
    )
    pre_centers = dict(state.last_centers)
    piled = _associate(
        [(48, 50), (52, 50)],
        state,
        2,
        outlines=[_box(48, 50), _box(52, 50)],
    )
    assert piled.split_detection is False
    assert piled.occlusion_bout is True
    assert piled.slot_to_detection.keys() == {0, 1}
    assert state.last_centers == pre_centers


def test_almost_touching_outlines_freeze_occlusion_bout():
    """Almost-touching, non-intersecting outlines freeze last COM and velocity."""
    state = IdentitySlotState.initial(2)
    _associate(
        [(20, 50), (80, 50)],
        state,
        2,
        outlines=[_box(20, 50), _box(80, 50)],
    )
    _associate(
        [(30, 50), (70, 50)],
        state,
        2,
        outlines=[_box(30, 50), _box(70, 50)],
    )
    pre_centers = dict(state.last_centers)
    pre_steps = dict(state.last_steps)

    # 10x10 boxes centered 12 px apart: gap 2 px, raw outlines do not intersect.
    piled = _associate(
        [(44, 50), (56, 50)],
        state,
        2,
        outlines=[_box(44, 50), _box(56, 50)],
    )
    assert piled.occlusion_bout is True
    assert piled.split_detection is False
    assert sorted(piled.frozen_slots) == [0, 1]
    assert state.last_centers == pre_centers
    assert state.last_steps == pre_steps


def test_hidden_animal_one_blob_is_occlusion_not_proximity():
    """One leftover detection with a nearby unmatched slot freezes as occlusion."""
    state = IdentitySlotState.initial(2)
    _associate(
        [(20, 50), (80, 50)],
        state,
        2,
        outlines=[_box(20, 50), _box(80, 50)],
    )
    _associate(
        [(36, 50), (64, 50)],
        state,
        2,
        outlines=[_box(36, 50), _box(64, 50)],
    )
    pre_centers = dict(state.last_centers)
    assert pre_centers[0] == (36, 50)
    assert pre_centers[1] == (64, 50)

    hidden = _associate(
        [(50, 50)],
        state,
        2,
        outlines=[_box(50, 50)],
    )
    assert hidden.occlusion_bout is True
    assert hidden.split_detection is False
    assert len(hidden.slot_to_detection) == 1
    assert hidden.unmatched_slots == [s for s in (0, 1) if s not in hidden.slot_to_detection]
    assert state.last_centers == pre_centers


def test_chase_close_centers_without_occlusion_does_not_freeze():
    """Close centers of mass with a clear outline gap are chase, not freeze."""
    state = IdentitySlotState.initial(2)
    state.typical_area = 100.0
    frames = [
        ([(20, 50), (50, 50)], [_box(20, 50), _box(50, 50)]),
        ([(28, 50), (52, 50)], [_box(28, 50), _box(52, 50)]),
        ([(36, 50), (54, 50)], [_box(36, 50), _box(54, 50)]),
        ([(40, 50), (58, 50)], [_box(40, 50), _box(58, 50)]),
    ]
    maps = []
    for centers, outlines in frames:
        asg = _associate(centers, state, 2, outlines=outlines)
        maps.append(asg.slot_to_detection)
        assert asg.occlusion_bout is False
        assert asg.split_detection is False
        assert asg.frozen_slots == []
    assert maps == [{0: 0, 1: 1}] * len(frames)
    assert state.last_centers[0] == (40, 50)
    assert state.last_centers[1] == (58, 50)


def test_collapsed_area_near_another_detection_freezes():
    """A shrunken fragment next to a full detection is an occlusion bout."""
    state = IdentitySlotState.initial(2)
    state.typical_area = 100.0
    _associate(
        [(20, 50), (80, 50)],
        state,
        2,
        outlines=[_box(20, 50), _box(80, 50)],
        areas=[100.0, 100.0],
    )
    _associate(
        [(30, 50), (70, 50)],
        state,
        2,
        outlines=[_box(30, 50), _box(70, 50)],
        areas=[100.0, 100.0],
    )
    pre_centers = dict(state.last_centers)
    # Gap is 8 px (no fattened intersect at 3 px, gap above 4 px) but the
    # right blob has collapsed versus typical area.
    collapsed = _associate(
        [(40, 50), (58, 50)],
        state,
        2,
        outlines=[_box(40, 50, w=10, h=10), _box(58, 50, w=4, h=4)],
        areas=[100.0, 16.0],
    )
    assert collapsed.occlusion_bout is True
    assert state.last_centers == pre_centers


def test_freeze_lift_rematch_uses_pre_freeze_identities_not_pile_center():
    """At freeze-lift, slots rematch from last pre-freeze COMs.

    Detection order at lift is reversed so a rematch off the pile center
    would swap; pre-freeze identities keep slot 0 on the left.
    """
    state = IdentitySlotState.initial(2)
    _associate(
        [(20, 50), (80, 50)],
        state,
        2,
        outlines=[_box(20, 50), _box(80, 50)],
    )
    _associate(
        [(30, 50), (70, 50)],
        state,
        2,
        outlines=[_box(30, 50), _box(70, 50)],
    )
    piled = _associate(
        [(44, 50), (56, 50)],
        state,
        2,
        outlines=[_box(44, 50), _box(56, 50)],
    )
    assert piled.occlusion_bout is True
    assert state.last_centers[0] == (30, 50)
    assert state.last_centers[1] == (70, 50)

    # Right body first, left body second — pile-center rematch would swap.
    lift = _associate(
        [(68, 50), (32, 50)],
        state,
        2,
        outlines=[_box(68, 50), _box(32, 50)],
    )
    assert lift.occlusion_bout is False
    assert lift.slot_to_detection == {0: 1, 1: 0}
    assert state.last_centers[0] == (32, 50)
    assert state.last_centers[1] == (68, 50)


def test_several_freeze_lifts_in_one_wrestle_rematch_each_time():
    """A mid-bout separation rematches immediately, not only at the final exit."""
    state = IdentitySlotState.initial(2)
    _associate(
        [(20, 50), (80, 50)],
        state,
        2,
        outlines=[_box(20, 50), _box(80, 50)],
    )
    _associate(
        [(30, 50), (70, 50)],
        state,
        2,
        outlines=[_box(30, 50), _box(70, 50)],
    )

    first_pile = _associate(
        [(44, 50), (56, 50)],
        state,
        2,
        outlines=[_box(44, 50), _box(56, 50)],
    )
    assert first_pile.occlusion_bout is True
    assert state.last_centers[0] == (30, 50)

    mid_lift = _associate(
        [(66, 50), (34, 50)],
        state,
        2,
        outlines=[_box(66, 50), _box(34, 50)],
    )
    assert mid_lift.occlusion_bout is False
    assert mid_lift.slot_to_detection == {0: 1, 1: 0}
    assert state.last_centers[0] == (34, 50)
    assert state.last_centers[1] == (66, 50)

    second_pile = _associate(
        [(46, 50), (58, 50)],
        state,
        2,
        outlines=[_box(46, 50), _box(58, 50)],
    )
    assert second_pile.occlusion_bout is True
    assert state.last_centers[0] == (34, 50)
    assert state.last_centers[1] == (66, 50)

    final_lift = _associate(
        [(70, 50), (30, 50)],
        state,
        2,
        outlines=[_box(70, 50), _box(30, 50)],
    )
    assert final_lift.occlusion_bout is False
    assert final_lift.slot_to_detection == {0: 1, 1: 0}
    assert state.last_centers[0] == (30, 50)
    assert state.last_centers[1] == (70, 50)
