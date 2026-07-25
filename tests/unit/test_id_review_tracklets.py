'''Tests for tracklet I/O and remapping.'''

import os
import tempfile

import numpy as np

from LabGym.id_review.tracklets import apply_mapping_to_store, load_tracklets, save_tracklets
from LabGym.id_review.types import TrackletStore, SCHEMA_VERSION, swap_mapping, identity_mapping
from LabGym.id_review.apply import apply_decision_to_analyzer
from LabGym.id_review.types import ReviewDecision
from LabGym.id_review.dataset import make_decision_for_event
from LabGym.id_review.types import ContactEvent


def _store(n_frames=20):
	ids = [0, 1]
	centers = np.zeros((2, n_frames, 2), dtype=np.float64)
	valid = np.ones((2, n_frames), dtype=bool)
	heights = np.full((2, n_frames), 10.0)
	contours = []
	for row, tid in enumerate(ids):
		row_c = []
		for f in range(n_frames):
			centers[row, f] = [tid * 100 + f, tid * 10]
			# simple square contour
			x, y = centers[row, f]
			cnt = np.array([[[int(x), int(y)]], [[int(x) + 5, int(y)]], [[int(x) + 5, int(y) + 5]], [[int(x), int(y) + 5]]], dtype=np.int32)
			row_c.append(cnt)
		contours.append(row_c)
	return TrackletStore(
		schema_version=SCHEMA_VERSION,
		animal_kind='mouse',
		ids=ids,
		n_frames=n_frames,
		centers=centers,
		valid=valid,
		heights=heights,
		contours=contours,
		meta={'video': 'x.mp4', 'fps': 30},
	)


def test_save_load_roundtrip():
	store = _store()
	with tempfile.TemporaryDirectory() as td:
		save_tracklets(store, td)
		loaded = load_tracklets(td, 'mouse')
		assert loaded.ids == store.ids
		assert loaded.n_frames == store.n_frames
		assert np.allclose(loaded.centers, store.centers)
		assert np.array_equal(loaded.valid, store.valid)
		assert loaded.contours[0][5] is not None
		assert loaded.contours[0][5].shape[-1] == 2


def test_apply_mapping_swap_from_frame():
	store = _store(n_frames=15)
	before_c0 = store.centers[0].copy()
	before_c1 = store.centers[1].copy()
	f0 = 8
	apply_mapping_to_store(store, swap_mapping([0, 1]), f0)
	# before unchanged
	assert np.allclose(store.centers[0, :f0], before_c0[:f0])
	assert np.allclose(store.centers[1, :f0], before_c1[:f0])
	# after swapped
	assert np.allclose(store.centers[0, f0:], before_c1[f0:])
	assert np.allclose(store.centers[1, f0:], before_c0[f0:])
	# contours swapped too
	assert np.array_equal(store.contours[0][f0], _store(15).contours[1][f0])


def test_identity_mapping_noop():
	store = _store()
	orig = store.centers.copy()
	apply_mapping_to_store(store, identity_mapping([0, 1]), 5)
	assert np.allclose(store.centers, orig)


class _FakeAnalyzer:
	def __init__(self):
		self.animal_kinds = ['mouse']
		self.animal_centers = {'mouse': {0: [(0, 0)] * 10, 1: [(1, 1)] * 10}}
		self.animal_contours = {'mouse': {0: [None] * 10, 1: [None] * 10}}
		self.animal_heights = {'mouse': {0: [1] * 10, 1: [2] * 10}}
		self.pattern_images = {'mouse': {0: ['p0'] * 10, 1: ['p1'] * 10}}
		self.animal_existingcenters = {'mouse': {0: (0, 0), 1: (1, 1)}}
		# make values distinct per frame for centers
		for f in range(10):
			self.animal_centers['mouse'][0][f] = (f, 0)
			self.animal_centers['mouse'][1][f] = (f, 100)


def test_apply_decision_to_analyzer_swap():
	az = _FakeAnalyzer()
	d = ReviewDecision(
		event_id='e000001',
		decision='swap',
		mapping={0: 1, 1: 0},
		remap_from_frame=5,
	)
	apply_decision_to_analyzer(az, d, animal_kind='mouse')
	assert az.animal_centers['mouse'][0][4] == (4, 0)
	assert az.animal_centers['mouse'][0][5] == (5, 100)
	assert az.animal_centers['mouse'][1][5] == (5, 0)
	assert az.pattern_images['mouse'][0][5] == 'p1'
	assert az.pattern_images['mouse'][1][5] == 'p0'


def test_make_decision_keep_swap():
	ev = ContactEvent(
		event_id='e000001',
		animal_kind='mouse',
		involved_ids=[0, 1],
		start_frame=10,
		end_frame=20,
		pre_frame=8,
		post_frame=22,
	)
	k = make_decision_for_event(ev, 'keep')
	assert not k.applies_remap
	s = make_decision_for_event(ev, 'swap')
	assert s.applies_remap
	assert s.mapping == {0: 1, 1: 0}
	# Default remap is end_frame+1 (not post_frame)
	assert s.remap_from_frame == 21
	s2 = make_decision_for_event(ev, 'swap', remap_from_frame=15)
	assert s2.remap_from_frame == 15
