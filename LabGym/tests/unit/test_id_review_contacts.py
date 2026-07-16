'''Tests for contact event detection.'''

import numpy as np

from LabGym.id_review.contacts import detect_contact_events_for_kind
from LabGym.id_review.types import ContactDetectorConfig


def _two_animals_approach_and_separate(n_frames=100, contact_start=40, contact_end=55):
	'''Two trajectories that approach, stay close, then separate without spatial swap.'''
	ids = [0, 1]
	centers = np.zeros((2, n_frames, 2), dtype=np.float64)
	valid = np.ones((2, n_frames), dtype=bool)
	heights = np.full((2, n_frames), 20.0)

	for f in range(n_frames):
		# animal 0 moves right from x=0; animal 1 left from x=100
		if f < contact_start:
			centers[0, f] = [f * 1.0, 50]
			centers[1, f] = [100 - f * 1.0, 50]
		elif f <= contact_end:
			centers[0, f] = [40, 50]
			centers[1, f] = [45, 50]  # ~5 px apart, size 20 -> contact if factor 1.5
		else:
			# separate again without crossing
			centers[0, f] = [30 - (f - contact_end), 50]
			centers[1, f] = [60 + (f - contact_end), 50]
	return ids, centers, valid, heights


def _two_animals_with_apparent_swap(n_frames=80, contact_start=30, contact_end=45):
	'''After contact, centers continue on swapped spatial paths (tracker switch pattern).'''
	ids = [0, 1]
	centers = np.zeros((2, n_frames, 2), dtype=np.float64)
	valid = np.ones((2, n_frames), dtype=bool)
	heights = np.full((2, n_frames), 20.0)

	for f in range(n_frames):
		if f < contact_start:
			centers[0, f] = [10, 10 + f * 0.1]
			centers[1, f] = [90, 10 + f * 0.1]
		elif f <= contact_end:
			centers[0, f] = [48, 20]
			centers[1, f] = [52, 20]
		else:
			# tracker labels follow wrong bodies: id0 goes to where id1 was headed
			t = f - contact_end
			centers[0, f] = [90, 20 + t]
			centers[1, f] = [10, 20 + t]
	return ids, centers, valid, heights


def test_detects_contact_bout():
	ids, centers, valid, heights = _two_animals_approach_and_separate()
	cfg = ContactDetectorConfig(contact_distance_factor=1.5, min_contact_frames=3)
	events = detect_contact_events_for_kind(ids, centers, valid, heights, 'mouse', config=cfg)
	assert len(events) >= 1
	ev = events[0]
	assert ev.involved_ids == [0, 1]
	assert ev.start_frame <= 40
	assert ev.end_frame >= 55
	assert ev.pre_frame < ev.start_frame
	assert ev.post_frame > ev.end_frame


def test_min_contact_frames_filters_brief():
	ids = [0, 1]
	n = 30
	centers = np.zeros((2, n, 2))
	valid = np.ones((2, n), dtype=bool)
	heights = np.full((2, n), 20.0)
	for f in range(n):
		centers[0, f] = [0, 0]
		centers[1, f] = [100, 0]
	# 2-frame contact only
	centers[0, 10] = [50, 0]
	centers[1, 10] = [55, 0]
	centers[0, 11] = [50, 0]
	centers[1, 11] = [55, 0]
	cfg = ContactDetectorConfig(min_contact_frames=3, contact_distance_factor=1.5)
	events = detect_contact_events_for_kind(ids, centers, valid, heights, 'mouse', config=cfg)
	assert events == []


def test_gap_bridging():
	ids = [0, 1]
	n = 40
	centers = np.zeros((2, n, 2))
	valid = np.ones((2, n), dtype=bool)
	heights = np.full((2, n), 20.0)
	for f in range(n):
		centers[0, f] = [0, 0]
		centers[1, f] = [100, 0]
	# contact 10-14, gap 15, contact 16-20
	for f in list(range(10, 15)) + list(range(16, 21)):
		centers[0, f] = [50, 0]
		centers[1, f] = [55, 0]
	cfg = ContactDetectorConfig(min_contact_frames=3, gap_bridge_frames=2, contact_distance_factor=1.5)
	events = detect_contact_events_for_kind(ids, centers, valid, heights, 'mouse', config=cfg)
	assert len(events) == 1
	assert events[0].start_frame == 10
	assert events[0].end_frame == 20


def test_possible_swap_flag_when_trajectories_cross_labels():
	ids, centers, valid, heights = _two_animals_with_apparent_swap()
	cfg = ContactDetectorConfig(contact_distance_factor=1.5, min_contact_frames=3)
	events = detect_contact_events_for_kind(ids, centers, valid, heights, 'mouse', config=cfg)
	assert len(events) >= 1
	assert 'possible_swap' in events[0].risk_flags or events[0].risk_score > 0.3
