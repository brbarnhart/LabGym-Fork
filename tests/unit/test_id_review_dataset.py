'''Tests for dataset / decision schema I/O.'''

import json
import os
import tempfile

from LabGym.id_review.dataset import (
	append_decision,
	load_events,
	make_decision_for_event,
	write_pair_label,
)
from LabGym.id_review.types import ContactEvent, SCHEMA_VERSION, dumps_jsonl_line


def test_events_and_decisions_roundtrip():
	ev = ContactEvent(
		event_id='e000007',
		animal_kind='mouse',
		involved_ids=[0, 1],
		start_frame=1,
		end_frame=5,
		pre_frame=0,
		post_frame=6,
		risk_score=0.5,
		risk_flags=['close_contact'],
		sample_dir='samples/e000007',
	)
	with tempfile.TemporaryDirectory() as td:
		with open(os.path.join(td, 'events.jsonl'), 'w', encoding='utf-8') as f:
			f.write(dumps_jsonl_line(ev.to_dict()) + '\n')
		events = load_events(td)
		assert len(events) == 1
		assert events[0].event_id == 'e000007'
		assert events[0].to_dict()['schema_version'] == SCHEMA_VERSION

		write_pair_label(td, ev, decision=None, status='unlabeled')
		pair_path = os.path.join(td, 'pairs', 'e000007.json')
		with open(pair_path, 'r', encoding='utf-8') as f:
			pair = json.load(f)
		assert pair['status'] == 'unlabeled'
		assert pair['label_permutation'] is None

		d = make_decision_for_event(ev, 'swap')
		append_decision(td, d)
		with open(os.path.join(td, 'decisions.jsonl'), 'r', encoding='utf-8') as f:
			lines = [ln for ln in f.read().splitlines() if ln.strip()]
		assert len(lines) == 1
		payload = json.loads(lines[0])
		assert payload['schema_version'] == SCHEMA_VERSION
		assert payload['decision'] == 'swap'
		assert payload['mapping'] == {'0': 1, '1': 0}

		with open(pair_path, 'r', encoding='utf-8') as f:
			pair = json.load(f)
		assert pair['status'] == 'labeled'
		assert pair['label_permutation'] == [1, 0]
		assert pair['label_source'] == 'human'


def test_default_remap_from_frame_is_end_plus_one():
	from LabGym.id_review.dataset import default_remap_from_frame
	ev = ContactEvent(
		event_id='e000009',
		animal_kind='mouse',
		involved_ids=[0, 1],
		start_frame=10,
		end_frame=20,
		pre_frame=8,
		post_frame=30,
	)
	assert default_remap_from_frame(ev) == 21


def test_uncertain_stays_unlabeled_pair():
	ev = ContactEvent(
		event_id='e000008',
		animal_kind='mouse',
		involved_ids=[0, 1],
		start_frame=1,
		end_frame=2,
		pre_frame=0,
		post_frame=3,
	)
	with tempfile.TemporaryDirectory() as td:
		with open(os.path.join(td, 'events.jsonl'), 'w', encoding='utf-8') as f:
			f.write(dumps_jsonl_line(ev.to_dict()) + '\n')
		d = make_decision_for_event(ev, 'uncertain')
		append_decision(td, d)
		with open(os.path.join(td, 'pairs', 'e000008.json'), 'r', encoding='utf-8') as f:
			pair = json.load(f)
		assert pair['status'] == 'unlabeled'
		assert not d.applies_remap
