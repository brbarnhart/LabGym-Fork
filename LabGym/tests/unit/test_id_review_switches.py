'''Tests for switch markers and sequential apply.'''

import os
import tempfile

from LabGym.id_review.types import SwitchMarker, swap_mapping
from LabGym.id_review.dataset import (
	finalize_switch_annotations,
	load_switches,
	make_swap_marker,
	switches_to_decisions,
	write_risk_labels,
	load_events,
)
from LabGym.id_review.types import ContactEvent, dumps_jsonl_line
from LabGym.id_review.apply import apply_decisions_to_analyzer
from LabGym.id_review.samples import detections_at_frame_after_markers
from LabGym.id_review.types import TrackletStore, SCHEMA_VERSION
import numpy as np


def test_make_swap_marker_and_decision():
	m = make_swap_marker(100, 'mouse', [0, 1], fps=10.0)
	assert m.frame == 100
	assert m.time_sec == 10.0
	assert m.mapping == {0: 1, 1: 0}
	d = m.to_review_decision()
	assert d.remap_from_frame == 100
	assert d.applies_remap


def test_switches_jsonl_roundtrip():
	m1 = make_swap_marker(50, 'mouse', [0, 1], fps=10)
	m2 = make_swap_marker(200, 'mouse', [0, 1], fps=10)
	with tempfile.TemporaryDirectory() as td:
		finalize_switch_annotations(td, [m2, m1], events=[], export_samples=False)
		loaded = load_switches(td)
		assert [m.frame for m in loaded] == [50, 200]
		assert os.path.isfile(os.path.join(td, 'decisions.jsonl'))
		assert os.path.isfile(os.path.join(td, 'risk_labels.jsonl'))


def test_risk_labels_has_switch():
	ev = ContactEvent(
		event_id='e000001',
		animal_kind='mouse',
		involved_ids=[0, 1],
		start_frame=40,
		end_frame=60,
		pre_frame=35,
		post_frame=65,
		risk_score=0.8,
	)
	m = make_swap_marker(55, 'mouse', [0, 1])
	with tempfile.TemporaryDirectory() as td:
		write_risk_labels(td, [ev], [m])
		with open(os.path.join(td, 'risk_labels.jsonl'), encoding='utf-8') as f:
			line = f.read().strip()
		assert '"has_switch":true' in line.replace(' ', '')


def test_sequential_swap_apply():
	class Az:
		pass

	az = Az()
	az.animal_kinds = ['mouse']
	az.animal_centers = {
		'mouse': {
			0: [(0, 0)] * 20,
			1: [(1, 1)] * 20,
		}
	}
	az.animal_contours = {'mouse': {0: [None] * 20, 1: [None] * 20}}
	az.animal_heights = {'mouse': {0: [1] * 20, 1: [2] * 20}}
	for f in range(20):
		az.animal_centers['mouse'][0][f] = (f, 0)
		az.animal_centers['mouse'][1][f] = (f, 100)

	m1 = make_swap_marker(5, 'mouse', [0, 1])
	m2 = make_swap_marker(10, 'mouse', [0, 1])
	decs = switches_to_decisions([m1, m2])
	apply_decisions_to_analyzer(az, decs, event_kind_lookup={m1.marker_id: 'mouse', m2.marker_id: 'mouse'})
	# after two swaps, back to original assignment for frames >= 10
	assert az.animal_centers['mouse'][0][10] == (10, 0)
	assert az.animal_centers['mouse'][1][10] == (10, 100)
	# between 5 and 9: swapped
	assert az.animal_centers['mouse'][0][7] == (7, 100)
	assert az.animal_centers['mouse'][1][7] == (7, 0)


def test_preview_detections_after_markers():
	n = 15
	centers = np.zeros((2, n, 2))
	valid = np.ones((2, n), dtype=bool)
	heights = np.full((2, n), 10.0)
	contours = [[None] * n, [None] * n]
	for f in range(n):
		centers[0, f] = [f, 0]
		centers[1, f] = [f, 50]
	store = TrackletStore(
		schema_version=SCHEMA_VERSION,
		animal_kind='mouse',
		ids=[0, 1],
		n_frames=n,
		centers=centers,
		valid=valid,
		heights=heights,
		contours=contours,
		meta={},
	)
	m = make_swap_marker(5, 'mouse', [0, 1])
	before = detections_at_frame_after_markers(store, 3, [m])
	assert before[0][1] == (3.0, 0.0)
	after = detections_at_frame_after_markers(store, 8, [m])
	assert after[0][1] == (8.0, 50.0)
	assert after[1][1] == (8.0, 0.0)
