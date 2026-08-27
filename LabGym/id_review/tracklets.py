'''Serialize tracklets and apply ID remapping to TrackletStore.'''

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .types import SCHEMA_VERSION, TrackletStore


def tracklets_from_analyzer(analyzer, animal_kind: str) -> TrackletStore:
	'''Build a TrackletStore from AnalyzeAnimalDetector in-memory state.'''
	ids = sorted(int(i) for i in analyzer.animal_centers[animal_kind].keys())
	if not ids:
		n_frames = len(getattr(analyzer, 'all_time', []) or [0])
		return TrackletStore(
			schema_version=SCHEMA_VERSION,
			animal_kind=animal_kind,
			ids=[],
			n_frames=n_frames,
			centers=np.zeros((0, n_frames, 2), dtype=np.float64),
			valid=np.zeros((0, n_frames), dtype=bool),
			heights=np.full((0, n_frames), np.nan, dtype=np.float64),
			contours=[],
			meta=_analyzer_meta(analyzer, animal_kind),
		)

	# Determine frame count from first ID series
	n_frames = len(analyzer.animal_centers[animal_kind][ids[0]])
	centers = np.zeros((len(ids), n_frames, 2), dtype=np.float64)
	valid = np.zeros((len(ids), n_frames), dtype=bool)
	heights = np.full((len(ids), n_frames), np.nan, dtype=np.float64)
	contours: List[List[Optional[np.ndarray]]] = []

	for row, tid in enumerate(ids):
		ctr_list = analyzer.animal_centers[animal_kind][tid]
		h_list = analyzer.animal_heights[animal_kind][tid]
		c_list = analyzer.animal_contours[animal_kind][tid]
		row_contours: List[Optional[np.ndarray]] = []
		for f in range(n_frames):
			c = ctr_list[f] if f < len(ctr_list) else None
			if c is not None:
				centers[row, f, 0] = float(c[0])
				centers[row, f, 1] = float(c[1])
				valid[row, f] = True
			h = h_list[f] if f < len(h_list) else None
			if h is not None:
				heights[row, f] = float(h)
			cnt = c_list[f] if f < len(c_list) else None
			if cnt is not None:
				row_contours.append(np.asarray(cnt, dtype=np.int32))
			else:
				row_contours.append(None)
		contours.append(row_contours)

	return TrackletStore(
		schema_version=SCHEMA_VERSION,
		animal_kind=animal_kind,
		ids=ids,
		n_frames=n_frames,
		centers=centers,
		valid=valid,
		heights=heights,
		contours=contours,
		meta=_analyzer_meta(analyzer, animal_kind),
	)


def tracklets_from_analyzer_all(analyzer) -> Dict[str, TrackletStore]:
	stores = {}
	for kind in analyzer.animal_kinds:
		if kind in analyzer.animal_centers:
			stores[kind] = tracklets_from_analyzer(analyzer, kind)
	return stores


def _analyzer_meta(analyzer, animal_kind: str) -> Dict[str, Any]:
	fps = getattr(analyzer, 'fps', None) or 0
	t = float(getattr(analyzer, 't', 0) or 0)
	length = int(getattr(analyzer, 'length', 0) or 0)
	# Matches acquire_information: start_t = max(0, t - length/fps)
	if fps:
		start_t = round(t - length / float(fps), 2)
		if start_t < 0:
			start_t = 0.0
	else:
		start_t = 0.0
	return {
		'video': getattr(analyzer, 'path_to_video', None),
		'fps': fps if fps else None,
		'framewidth': getattr(analyzer, 'framewidth', None),
		'frameheight': getattr(analyzer, 'frameheight', None),
		't': t,
		'length': length,
		'start_t': start_t,
		'duration': getattr(analyzer, 'duration', 0),
		'animal_kind': animal_kind,
		'animal_area': getattr(analyzer, 'animal_area', {}).get(animal_kind),
	}


def pack_contours(
	contours: Sequence[Sequence[Optional[np.ndarray]]],
) -> tuple:
	'''Flatten per-frame polygons into npz contour_lengths / contour_points.

	Args:
		contours: ``n_ids`` rows of ``n_frames`` polygons (or None).

	Returns:
		``(contour_lengths, contour_points)`` matching the on-disk npz layout.
	'''
	lengths: List[int] = []
	points_list: List[np.ndarray] = []
	for row in contours:
		for cnt in row:
			if cnt is None:
				lengths.append(0)
			else:
				arr = np.asarray(cnt).reshape(-1, 2)
				lengths.append(int(arr.shape[0]))
				points_list.append(arr.astype(np.int32, copy=False))
	if points_list:
		points = np.concatenate(points_list, axis=0)
	else:
		points = np.zeros((0, 2), dtype=np.int32)
	return np.asarray(lengths, dtype=np.int32), points


def _packed_contour_arrays(store: TrackletStore) -> tuple:
	'''Return (lengths, points) for npz write without expanding session-load stores.'''
	expected = len(store.ids) * int(store.n_frames)
	if store.contours is not None:
		return pack_contours(store.contours)
	lengths = store.contour_lengths
	points = store.contour_points
	if lengths is not None and int(lengths.size) == expected:
		if points is None:
			points = np.zeros((0, 2), dtype=np.int32)
		return np.asarray(lengths, dtype=np.int32), np.asarray(points, dtype=np.int32)
	return np.zeros((expected,), dtype=np.int32), np.zeros((0, 2), dtype=np.int32)


def save_tracklets(store: TrackletStore, directory: str, prefix: Optional[str] = None) -> str:
	'''
	Save one kind's tracklets under directory.

	Writes:
	  {prefix}tracklets_meta.json
	  {prefix}tracklets.npz  (centers, valid, heights, ids, contour payloads)
	'''
	os.makedirs(directory, exist_ok=True)
	prefix = prefix or f'{store.animal_kind}_'
	meta_path = os.path.join(directory, f'{prefix}tracklets_meta.json')
	npz_path = os.path.join(directory, f'{prefix}tracklets.npz')

	lengths, points = _packed_contour_arrays(store)

	np.savez_compressed(
		npz_path,
		centers=store.centers,
		valid=store.valid,
		heights=store.heights,
		ids=np.asarray(store.ids, dtype=np.int32),
		contour_lengths=np.asarray(lengths, dtype=np.int32),
		contour_points=points,
		n_frames=np.asarray([store.n_frames], dtype=np.int32),
	)

	meta = {
		'schema_version': SCHEMA_VERSION,
		'animal_kind': store.animal_kind,
		'ids': store.ids,
		'n_frames': store.n_frames,
		'meta': store.meta,
		'npz_file': os.path.basename(npz_path),
	}
	with open(meta_path, 'w', encoding='utf-8') as f:
		json.dump(meta, f, indent=2)

	return meta_path


def load_tracklets(directory: str, animal_kind: str, prefix: Optional[str] = None) -> TrackletStore:
	'''Load one animal kind's tracklets into packed RAM arrays.

	Does not expand a Python polygon for every analysis frame. Call
	``TrackletStore.contour_at`` to slice the outline for one frame.
	Other kinds in the same package are not opened.

	Args:
		directory: Folder containing ``{kind}_tracklets.npz`` and meta JSON.
		animal_kind: Detector class to load; other kinds are ignored.
		prefix: Filename prefix; default ``{animal_kind}_``.

	Returns:
		Store whose outlines live in packed arrays (``contours`` is None).
	'''
	prefix = prefix or f'{animal_kind}_'
	meta_path = os.path.join(directory, f'{prefix}tracklets_meta.json')
	with open(meta_path, 'r', encoding='utf-8') as f:
		meta = json.load(f)
	npz_path = os.path.join(directory, meta.get('npz_file', f'{prefix}tracklets.npz'))
	with np.load(npz_path, allow_pickle=False) as data:
		ids = [int(x) for x in data['ids'].tolist()]
		n_frames = int(data['n_frames'][0])
		centers = np.array(data['centers'], copy=True)
		valid = np.array(data['valid'], copy=True).astype(bool, copy=False)
		heights = np.array(data['heights'], copy=True)
		lengths = np.array(data['contour_lengths'], copy=True).astype(np.int32, copy=False)
		points = np.array(data['contour_points'], copy=True)
		if points.ndim != 2 or (points.size and points.shape[1] != 2):
			points = np.asarray(points).reshape(-1, 2).astype(np.int32, copy=False)
		else:
			points = points.astype(np.int32, copy=False)

	return TrackletStore(
		schema_version=int(meta.get('schema_version', SCHEMA_VERSION)),
		animal_kind=meta['animal_kind'],
		ids=ids,
		n_frames=n_frames,
		centers=centers,
		valid=valid,
		heights=heights,
		contours=None,
		contour_lengths=lengths,
		contour_points=points,
		meta=meta.get('meta') or {},
	)


def apply_mapping_to_store(
	store: TrackletStore,
	mapping: Dict[int, int],
	remap_from_frame: int,
) -> None:
	'''
	In-place remap for frames >= remap_from_frame.

	new[mapping[i]][f] = old[i][f] for each involved id i present in store.
	'''
	if not mapping:
		return
	involved = [int(i) for i in mapping.keys() if int(i) in store.ids]
	if not involved:
		return

	# Validate mapping is a permutation of involved
	targets = [int(mapping[i]) for i in involved]
	if sorted(targets) != sorted(involved):
		raise ValueError(f'mapping must be a permutation of involved ids: {mapping}')

	idx = {tid: store.id_index(tid) for tid in involved}
	f0 = max(0, int(remap_from_frame))
	n = store.n_frames
	if f0 >= n:
		return

	# Snapshot old rows for involved
	old_centers = {tid: store.centers[idx[tid], f0:].copy() for tid in involved}
	old_valid = {tid: store.valid[idx[tid], f0:].copy() for tid in involved}
	old_heights = {tid: store.heights[idx[tid], f0:].copy() for tid in involved}

	for src in involved:
		dst = int(mapping[src])
		di = idx[dst]
		store.centers[di, f0:] = old_centers[src]
		store.valid[di, f0:] = old_valid[src]
		store.heights[di, f0:] = old_heights[src]

	if store.contours is not None:
		old_contours = {tid: store.contours[idx[tid]][f0:] for tid in involved}
		for src in involved:
			dst = int(mapping[src])
			di = idx[dst]
			store.contours[di][f0:] = [
				(c.copy() if c is not None else None) for c in old_contours[src]
			]

	_remap_packed_contours(store, mapping, involved, idx, f0, n)


def _remap_packed_contours(
	store: TrackletStore,
	mapping: Dict[int, int],
	involved: Sequence[int],
	idx: Dict[int, int],
	f0: int,
	n_frames: int,
) -> None:
	'''Swap packed outline tails for frames >= f0. No-op when packed arrays are absent.'''
	lengths = store.contour_lengths
	points = store.contour_points
	n_ids = len(store.ids)
	expected = n_ids * n_frames
	if lengths is None or points is None or int(lengths.size) != expected:
		return
	offsets = store._packed_offsets()
	inv = {int(mapping[src]): int(src) for src in involved}
	new_lengths = np.array(lengths, copy=True)
	chunks: List[np.ndarray] = []
	for row, tid in enumerate(store.ids):
		head_i0 = row * n_frames
		head_i1 = row * n_frames + f0
		head = points[int(offsets[head_i0]) : int(offsets[head_i1])]
		if head.size:
			chunks.append(np.array(head, copy=True))
		src_tid = inv.get(int(tid))
		if src_tid is not None:
			si = idx[src_tid]
			tail_i0 = si * n_frames + f0
			tail_i1 = si * n_frames + n_frames
			new_lengths[row * n_frames + f0 : row * n_frames + n_frames] = lengths[tail_i0:tail_i1]
			tail = points[int(offsets[tail_i0]) : int(offsets[tail_i1])]
		else:
			tail_i0 = row * n_frames + f0
			tail_i1 = row * n_frames + n_frames
			tail = points[int(offsets[tail_i0]) : int(offsets[tail_i1])]
		if tail.size:
			chunks.append(np.array(tail, copy=True))
	if chunks:
		store.contour_points = np.concatenate(chunks, axis=0).astype(np.int32, copy=False)
	else:
		store.contour_points = np.zeros((0, 2), dtype=np.int32)
	store.contour_lengths = new_lengths.astype(np.int32, copy=False)
	store.invalidate_packed_offsets()
