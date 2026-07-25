'''Apply review decisions to in-memory AnalyzeAnimalDetector state.'''

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from .types import ReviewDecision


def _swap_frame_values(series_by_id: Dict[Any, list], mapping: Dict[int, int], frame: int) -> None:
	'''new[mapping[i]][frame] = old[i][frame] for involved ids present in series_by_id.'''
	involved = [i for i in mapping.keys() if i in series_by_id]
	if not involved:
		return
	old = {}
	for i in involved:
		seq = series_by_id[i]
		if frame < len(seq):
			old[i] = seq[frame]
		else:
			old[i] = None
	for src in involved:
		dst = mapping[src]
		seq = series_by_id[dst]
		if frame < len(seq):
			seq[frame] = old[src]


def _series_dicts_for_kind(analyzer, animal_kind: str) -> List[Dict[Any, list]]:
	'''Collect ID-keyed per-frame lists that must stay aligned.'''
	series = []
	for attr in (
		'animal_centers',
		'animal_contours',
		'animal_heights',
		'pattern_images',
		'animations',
	):
		container = getattr(analyzer, attr, None)
		if container and animal_kind in container:
			series.append(container[animal_kind])

	# event_probability: dict id -> list of [name, prob]
	ep = getattr(analyzer, 'event_probability', None)
	if ep and animal_kind in ep:
		series.append(ep[animal_kind])

	# behavior parameter probability matrices if present
	abp = getattr(analyzer, 'all_behavior_parameters', None)
	if abp and animal_kind in abp:
		for behavior_name, params in abp[animal_kind].items():
			if isinstance(params, dict) and 'probability' in params:
				# probability may be id -> list
				prob = params['probability']
				if isinstance(prob, dict):
					series.append(prob)

	return series


def apply_decision_to_analyzer(analyzer, decision: ReviewDecision, animal_kind: Optional[str] = None) -> None:
	'''
	Apply one decision's mapping to analyzer state for frames >= remap_from_frame.

	If animal_kind is None, uses the kind that contains the involved ids
	(first match among analyzer.animal_kinds).
	'''
	if not decision.applies_remap:
		return

	mapping = {int(k): int(v) for k, v in decision.mapping.items()}
	if animal_kind is None:
		animal_kind = _infer_kind(analyzer, list(mapping.keys()))
	if animal_kind is None:
		return

	f0 = int(decision.remap_from_frame)
	# determine frame span from centers
	ids = list(analyzer.animal_centers.get(animal_kind, {}))
	if not ids:
		return
	n_frames = len(analyzer.animal_centers[animal_kind][ids[0]])
	if f0 >= n_frames:
		return

	for series in _series_dicts_for_kind(analyzer, animal_kind):
		for f in range(f0, n_frames):
			_swap_frame_values(series, mapping, f)

	# Also fix animal_existingcenters to last known center if available
	if hasattr(analyzer, 'animal_existingcenters') and animal_kind in analyzer.animal_existingcenters:
		for tid in mapping.keys():
			if tid not in analyzer.animal_centers[animal_kind]:
				continue
			# last non-None center
			last = (-10000, -10000)
			for c in analyzer.animal_centers[animal_kind][tid]:
				if c is not None:
					last = c
			analyzer.animal_existingcenters[animal_kind][tid] = last


def apply_decisions_to_analyzer(
	analyzer,
	decisions: Sequence[ReviewDecision],
	event_kind_lookup: Optional[Dict[str, str]] = None,
) -> List[ReviewDecision]:
	'''
	Apply decisions sorted by remap_from_frame ascending.

	Each mapping is relative to IDs as currently stored after previous applications.
	Returns the list that actually remapped tracks.
	'''
	ordered = sorted(decisions, key=lambda d: (d.remap_from_frame, d.event_id))
	applied = []
	for d in ordered:
		kind = None
		if event_kind_lookup and d.event_id in event_kind_lookup:
			kind = event_kind_lookup[d.event_id]
		if d.applies_remap:
			apply_decision_to_analyzer(analyzer, d, animal_kind=kind)
			applied.append(d)
	return applied


def _infer_kind(analyzer, ids: List[int]) -> Optional[str]:
	for kind in getattr(analyzer, 'animal_kinds', []) or []:
		centers = getattr(analyzer, 'animal_centers', {}).get(kind)
		if not centers:
			continue
		if all(i in centers for i in ids):
			return kind
	return None


def load_decisions(path: str) -> List[ReviewDecision]:
	'''Load decisions.jsonl (last decision per event_id wins).'''
	if not os.path.isfile(path):
		return []
	by_event: Dict[str, ReviewDecision] = {}
	with open(path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			d = ReviewDecision.from_dict(json.loads(line))
			by_event[d.event_id] = d
	return list(by_event.values())


def write_applied_corrections(directory: str, decisions: Sequence[ReviewDecision]) -> str:
	os.makedirs(directory, exist_ok=True)
	path = os.path.join(directory, 'applied_corrections.json')
	payload = {
		'count': len(decisions),
		'decisions': [d.to_dict() for d in decisions],
	}
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(payload, f, indent=2)
	return path


def apply_decisions_to_store(
	store,
	decisions: Sequence[ReviewDecision],
	animal_kind: Optional[str] = None,
) -> int:
	'''
	Apply remap decisions to a TrackletStore (in place).

	Returns number of decisions applied that remapped this store's kind.
	'''
	from .tracklets import apply_mapping_to_store

	ordered = sorted(decisions, key=lambda d: (d.remap_from_frame, d.event_id))
	n = 0
	for d in ordered:
		if not d.applies_remap:
			continue
		# Skip if decision is tagged for a different kind (when known)
		# ReviewDecision has no animal_kind field; filter via event_id only at caller.
		mapping = {int(k): int(v) for k, v in d.mapping.items()}
		if not mapping:
			continue
		# Only apply if involved ids exist in this store
		if not any(int(i) in store.ids for i in mapping.keys()):
			continue
		try:
			apply_mapping_to_store(store, mapping, int(d.remap_from_frame))
			n += 1
		except ValueError:
			# Non-permutation or missing ids — skip this decision
			continue
	return n


def write_tracklets_identity_status(
	directory: str,
	*,
	corrected: bool,
	n_decisions: int = 0,
	source: str = '',
) -> str:
	'''Mark whether on-disk tracklets already include ID remaps.'''
	os.makedirs(directory, exist_ok=True)
	path = os.path.join(directory, 'tracklets_identity_status.json')
	payload = {
		'corrected': bool(corrected),
		'n_decisions': int(n_decisions),
		'source': source,
	}
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(payload, f, indent=2)
	return path


def read_tracklets_identity_status(directory: str) -> Dict[str, Any]:
	path = os.path.join(directory, 'tracklets_identity_status.json')
	if not os.path.isfile(path):
		return {'corrected': False, 'n_decisions': 0, 'source': ''}
	try:
		with open(path, 'r', encoding='utf-8') as f:
			d = json.load(f)
		return {
			'corrected': bool(d.get('corrected', False)),
			'n_decisions': int(d.get('n_decisions') or 0),
			'source': str(d.get('source') or ''),
		}
	except Exception:
		return {'corrected': False, 'n_decisions': 0, 'source': ''}
