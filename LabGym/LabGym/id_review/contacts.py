'''Detect contact / separation events from tracklet centers.'''

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .types import ContactDetectorConfig, ContactEvent


def _animal_size(heights: np.ndarray, valid: np.ndarray, fallback: float = 20.0) -> float:
	vals = heights[valid & np.isfinite(heights)]
	if vals.size == 0:
		return float(fallback)
	return float(max(np.median(vals), 1.0))


def _pair_contact_mask(
	centers_a: np.ndarray,
	centers_b: np.ndarray,
	valid_a: np.ndarray,
	valid_b: np.ndarray,
	size: float,
	factor: float,
) -> np.ndarray:
	n = centers_a.shape[0]
	mask = np.zeros(n, dtype=bool)
	both = valid_a & valid_b
	if not np.any(both):
		return mask
	idx = np.where(both)[0]
	d = np.linalg.norm(centers_a[idx] - centers_b[idx], axis=1)
	mask[idx] = d < (factor * size)
	return mask


def _bridge_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
	if max_gap <= 0 or mask.size == 0:
		return mask.copy()
	out = mask.copy()
	n = len(out)
	i = 0
	while i < n:
		if out[i]:
			i += 1
			continue
		j = i
		while j < n and not out[j]:
			j += 1
		gap = j - i
		# only bridge interior gaps
		if i > 0 and j < n and out[i - 1] and (j < n and out[j]) and gap <= max_gap:
			out[i:j] = True
		i = j
	return out


def _runs_true(mask: np.ndarray) -> List[Tuple[int, int]]:
	'''Inclusive (start, end) index pairs where mask is True.'''
	runs: List[Tuple[int, int]] = []
	n = len(mask)
	i = 0
	while i < n:
		if not mask[i]:
			i += 1
			continue
		j = i
		while j + 1 < n and mask[j + 1]:
			j += 1
		runs.append((i, j))
		i = j + 1
	return runs


def _find_pre_frame(
	centers: np.ndarray,
	valid: np.ndarray,
	ids_idx: Sequence[int],
	start_frame: int,
	size: float,
	sep_factor: float,
) -> int:
	'''Last frame before contact where all involved are valid and pairwise separated.'''
	for f in range(start_frame - 1, -1, -1):
		if not all(valid[i, f] for i in ids_idx):
			continue
		ok = True
		for a in range(len(ids_idx)):
			for b in range(a + 1, len(ids_idx)):
				ia, ib = ids_idx[a], ids_idx[b]
				d = float(np.linalg.norm(centers[ia, f] - centers[ib, f]))
				if d < sep_factor * size:
					ok = False
					break
			if not ok:
				break
		if ok:
			return f
	# fallback: last frame with all valid before start
	for f in range(start_frame - 1, -1, -1):
		if all(valid[i, f] for i in ids_idx):
			return f
	return max(0, start_frame - 1)


def _find_post_frame(
	centers: np.ndarray,
	valid: np.ndarray,
	ids_idx: Sequence[int],
	end_frame: int,
	n_frames: int,
	size: float,
	sep_factor: float,
) -> int:
	'''First frame after contact where all involved are valid and pairwise separated.'''
	for f in range(end_frame + 1, n_frames):
		if not all(valid[i, f] for i in ids_idx):
			continue
		ok = True
		for a in range(len(ids_idx)):
			for b in range(a + 1, len(ids_idx)):
				ia, ib = ids_idx[a], ids_idx[b]
				d = float(np.linalg.norm(centers[ia, f] - centers[ib, f]))
				if d < sep_factor * size:
					ok = False
					break
			if not ok:
				break
		if ok:
			return f
	for f in range(end_frame + 1, n_frames):
		if all(valid[i, f] for i in ids_idx):
			return f
	return min(n_frames - 1, end_frame + 1) if end_frame + 1 < n_frames else end_frame


def _risk_score(
	centers: np.ndarray,
	valid: np.ndarray,
	ids_idx: Sequence[int],
	start: int,
	end: int,
	pre: int,
	post: int,
	size: float,
) -> Tuple[float, List[str]]:
	flags = ['close_contact']
	duration = end - start + 1
	# min distance during contact among pairs
	mins = []
	for a in range(len(ids_idx)):
		for b in range(a + 1, len(ids_idx)):
			ia, ib = ids_idx[a], ids_idx[b]
			for f in range(start, end + 1):
				if valid[ia, f] and valid[ib, f]:
					mins.append(float(np.linalg.norm(centers[ia, f] - centers[ib, f])))
	min_d = min(mins) if mins else size
	prox = float(np.clip(1.0 - (min_d / max(size, 1.0)), 0.0, 1.0))
	dur_term = float(np.clip(duration / 30.0, 0.0, 1.0))

	# trajectory discontinuity: compare pre->post displacement vs expected swap-free continuity
	disc = 0.0
	if all(valid[i, pre] and valid[i, post] for i in ids_idx) and len(ids_idx) == 2:
		ia, ib = ids_idx[0], ids_idx[1]
		# cost of keeping IDs vs swapping centers pre->post
		keep = (
			float(np.linalg.norm(centers[ia, post] - centers[ia, pre]))
			+ float(np.linalg.norm(centers[ib, post] - centers[ib, pre]))
		)
		swap = (
			float(np.linalg.norm(centers[ia, post] - centers[ib, pre]))
			+ float(np.linalg.norm(centers[ib, post] - centers[ia, pre]))
		)
		if swap + 1e-6 < keep:
			disc = float(np.clip((keep - swap) / max(size, 1.0), 0.0, 1.0))
			flags.append('possible_swap')

	score = 0.4 * prox + 0.3 * dur_term + 0.3 * disc
	return float(np.clip(score, 0.0, 1.0)), flags


def detect_contact_events_for_kind(
	ids: Sequence[int],
	centers: np.ndarray,
	valid: np.ndarray,
	heights: np.ndarray,
	animal_kind: str,
	config: Optional[ContactDetectorConfig] = None,
	event_id_start: int = 0,
) -> List[ContactEvent]:
	'''
	Detect pairwise contact bouts for one animal kind.

	centers: (n_ids, n_frames, 2)
	valid: (n_ids, n_frames)
	heights: (n_ids, n_frames)
	'''
	config = config or ContactDetectorConfig()
	ids = [int(i) for i in ids]
	n_ids, n_frames = valid.shape
	if n_ids < 2 or n_frames == 0:
		return []

	size = _animal_size(heights, valid)
	sep_factor = max(config.min_separation_gap, config.contact_distance_factor)

	# Collect raw pairwise runs, then merge overlapping runs that share animals
	# into multi-animal events via connected components per time-overlapping graph.
	pair_runs: List[Tuple[int, int, int, int]] = []  # i, j, start, end (indices into ids)

	for a in range(n_ids):
		for b in range(a + 1, n_ids):
			mask = _pair_contact_mask(
				centers[a], centers[b], valid[a], valid[b], size, config.contact_distance_factor
			)
			mask = _bridge_gaps(mask, config.gap_bridge_frames)
			for start, end in _runs_true(mask):
				if end - start + 1 >= config.min_contact_frames:
					pair_runs.append((a, b, start, end))

	if not pair_runs:
		return []

	# Greedy merge overlapping pair runs into events (connected by shared id + time overlap)
	used = [False] * len(pair_runs)
	merged: List[Tuple[List[int], int, int]] = []  # id indices, start, end

	for p in range(len(pair_runs)):
		if used[p]:
			continue
		a, b, s, e = pair_runs[p]
		comp_ids = {a, b}
		cs, ce = s, e
		used[p] = True
		changed = True
		while changed:
			changed = False
			for q in range(len(pair_runs)):
				if used[q]:
					continue
				qa, qb, qs, qe = pair_runs[q]
				# time overlap (with 1-frame tolerance) and shared animal
				if qe < cs - 1 or qs > ce + 1:
					continue
				if qa in comp_ids or qb in comp_ids:
					comp_ids.update((qa, qb))
					cs = min(cs, qs)
					ce = max(ce, qe)
					used[q] = True
					changed = True
		merged.append((sorted(comp_ids), cs, ce))

	events: List[ContactEvent] = []
	counter = event_id_start
	for id_indices, start, end in sorted(merged, key=lambda x: x[1]):
		pre = _find_pre_frame(centers, valid, id_indices, start, size, sep_factor)
		post = _find_post_frame(centers, valid, id_indices, end, n_frames, size, sep_factor)
		score, flags = _risk_score(centers, valid, id_indices, start, end, pre, post, size)
		involved = [ids[i] for i in id_indices]
		events.append(
			ContactEvent(
				event_id=f'e{counter:06d}',
				animal_kind=animal_kind,
				involved_ids=involved,
				start_frame=int(start),
				end_frame=int(end),
				pre_frame=int(pre),
				post_frame=int(post),
				risk_score=score,
				risk_flags=flags,
			)
		)
		counter += 1
	return events


def detect_contact_events(
	stores: Dict[str, 'TrackletStore'],  # noqa: F821
	config: Optional[ContactDetectorConfig] = None,
) -> List[ContactEvent]:
	'''Detect events across all animal kinds in a dict of TrackletStore.'''
	from .types import TrackletStore  # local import for typing only

	config = config or ContactDetectorConfig()
	all_events: List[ContactEvent] = []
	counter = 0
	for kind, store in stores.items():
		assert isinstance(store, TrackletStore)
		evs = detect_contact_events_for_kind(
			store.ids,
			store.centers,
			store.valid,
			store.heights,
			animal_kind=kind,
			config=config,
			event_id_start=counter,
		)
		all_events.extend(evs)
		counter += len(evs)
	all_events.sort(key=lambda e: (e.start_frame, e.event_id))
	return all_events
