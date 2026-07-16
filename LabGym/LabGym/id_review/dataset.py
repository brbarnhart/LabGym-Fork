'''Export review packs and decision / pair labels for training.'''

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .types import (
	SCHEMA_VERSION,
	ContactDetectorConfig,
	ContactEvent,
	ReviewDecision,
	SwitchMarker,
	TrackletStore,
	dumps_jsonl_line,
	identity_mapping,
	mapping_to_permutation,
	swap_mapping,
)
from .contacts import detect_contact_events
from .tracklets import (
	tracklets_from_analyzer_all,
	save_tracklets,
	load_tracklets,
	apply_mapping_to_store,
)
from .samples import export_event_samples, export_switch_samples
from .apply import (
	apply_decisions_to_analyzer,
	load_decisions,
	write_applied_corrections,
)


def review_dir(results_path: str) -> str:
	return os.path.join(results_path, 'id_review')


def export_review_pack(
	analyzer,
	config: Optional[ContactDetectorConfig] = None,
	extract_samples: bool = True,
) -> Tuple[str, List[ContactEvent]]:
	'''
	After craft_data: save tracklets, detect contacts, write events.jsonl + samples.

	Returns (id_review_directory, events).
	'''
	config = config or ContactDetectorConfig()
	out_dir = review_dir(analyzer.results_path)
	os.makedirs(out_dir, exist_ok=True)

	print('ID review: exporting tracklets...', flush=True)
	stores = tracklets_from_analyzer_all(analyzer)
	for kind, store in stores.items():
		save_tracklets(store, out_dir)
		print(
			f'ID review: saved tracklets for {kind} '
			f'({len(store.ids)} ids, {store.n_frames} frames)',
			flush=True,
		)

	# persist detector config
	with open(os.path.join(out_dir, 'contact_config.json'), 'w', encoding='utf-8') as f:
		json.dump({'schema_version': SCHEMA_VERSION, **config.to_dict()}, f, indent=2)

	print('ID review: detecting contact events...', flush=True)
	events = detect_contact_events(stores, config=config)
	print(f'ID review: found {len(events)} contact event(s)', flush=True)
	video = getattr(analyzer, 'path_to_video', None)
	fps = float(getattr(analyzer, 'fps', 0) or 0)
	framewidth = getattr(analyzer, 'framewidth', None)

	for i, ev in enumerate(events):
		ev.video = video
		ev.fps = fps
		# portable relative path in JSON
		ev.sample_dir = f'samples/{ev.event_id}'
		sample_abs = os.path.join(out_dir, 'samples', ev.event_id)
		if extract_samples and video and ev.animal_kind in stores:
			if (i + 1) == 1 or (i + 1) % 10 == 0 or (i + 1) == len(events):
				print(f'ID review: extracting samples {i + 1}/{len(events)}...', flush=True)
			try:
				export_event_samples(
					ev,
					stores[ev.animal_kind],
					video_path=video,
					sample_dir=sample_abs,
					fps=fps,
					framewidth=framewidth,
					analysis_meta=stores[ev.animal_kind].meta,
				)
			except Exception as exc:  # keep pack even if one sample fails
				os.makedirs(sample_abs, exist_ok=True)
				with open(os.path.join(sample_abs, 'meta.json'), 'w', encoding='utf-8') as f:
					json.dump({'event_id': ev.event_id, 'error': str(exc)}, f)

		# unlabeled pair stub for training pool
		write_pair_label(
			out_dir,
			ev,
			decision=None,
			status='unlabeled',
		)

	events_path = os.path.join(out_dir, 'events.jsonl')
	with open(events_path, 'w', encoding='utf-8') as f:
		for ev in events:
			f.write(dumps_jsonl_line(ev.to_dict()) + '\n')

	# ensure decisions file exists
	dec_path = os.path.join(out_dir, 'decisions.jsonl')
	if not os.path.isfile(dec_path):
		with open(dec_path, 'w', encoding='utf-8') as f:
			pass

	print(f'ID review: pack written to {out_dir}', flush=True)
	return out_dir, events


def load_events(path_or_dir: str) -> List[ContactEvent]:
	if os.path.isdir(path_or_dir):
		path = os.path.join(path_or_dir, 'events.jsonl')
	else:
		path = path_or_dir
	events = []
	if not os.path.isfile(path):
		return events
	with open(path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			events.append(ContactEvent.from_dict(json.loads(line)))
	return events


def append_decision(out_dir: str, decision: ReviewDecision) -> str:
	'''Append one decision line and update pairs/{event_id}.json.'''
	os.makedirs(out_dir, exist_ok=True)
	if not decision.timestamp_utc:
		decision.timestamp_utc = datetime.now(timezone.utc).isoformat()
	path = os.path.join(out_dir, 'decisions.jsonl')
	with open(path, 'a', encoding='utf-8') as f:
		f.write(dumps_jsonl_line(decision.to_dict()) + '\n')

	# update pair label if event known
	events = {e.event_id: e for e in load_events(out_dir)}
	ev = events.get(decision.event_id)
	if ev is not None:
		status = 'labeled' if decision.decision in ('keep', 'swap') else 'unlabeled'
		write_pair_label(out_dir, ev, decision=decision, status=status)
	return path


def write_pair_label(
	out_dir: str,
	event: ContactEvent,
	decision: Optional[ReviewDecision],
	status: str = 'unlabeled',
) -> str:
	pairs_dir = os.path.join(out_dir, 'pairs')
	os.makedirs(pairs_dir, exist_ok=True)
	sample_rel = event.sample_dir or os.path.join('samples', event.event_id)
	pre_crops = {str(i): os.path.join(sample_rel, f'pre_id_{i}.png') for i in event.involved_ids}
	post_crops = {str(i): os.path.join(sample_rel, f'post_id_{i}.png') for i in event.involved_ids}

	if decision is not None and status == 'labeled':
		perm = mapping_to_permutation(event.involved_ids, decision.mapping)
		label_source = decision.label_source
	else:
		perm = None
		label_source = None

	payload = {
		'schema_version': SCHEMA_VERSION,
		'event_id': event.event_id,
		'animal_kind': event.animal_kind,
		'involved_ids': event.involved_ids,
		'pre_crops': pre_crops,
		'post_crops': post_crops,
		'label_permutation': perm,
		'label_source': label_source,
		'status': status,
		'mapping_convention': (
			'label_permutation[j] = pre_identity continued by post crop of involved_ids[j]'
		),
	}
	path = os.path.join(pairs_dir, f'{event.event_id}.json')
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(payload, f, indent=2)
	return path


def default_remap_from_frame(event: ContactEvent) -> int:
	'''
	Default frame from which ID remaps apply (inclusive).

	Uses end_frame + 1 (first frame after the contact bout), not post_frame.
	post_frame is only for clean review crops and may be later than desired.
	'''
	return int(event.end_frame) + 1


def make_decision_for_event(
	event: ContactEvent,
	decision: str,
	annotator: str = 'local_user',
	notes: str = '',
	confidence: str = 'high',
	remap_from_frame: Optional[int] = None,
) -> ReviewDecision:
	'''
	Helper for GUI / tests: build ReviewDecision from keep|swap|uncertain|skip.

	remap_from_frame: if None, uses default_remap_from_frame(event) (= end_frame+1).
	'''
	decision = decision.lower().strip()
	ids = list(event.involved_ids)
	if decision == 'keep':
		mapping = identity_mapping(ids)
	elif decision == 'swap':
		if len(ids) != 2:
			raise ValueError('swap only supported for exactly 2 involved ids in MVP')
		mapping = swap_mapping(ids)
	elif decision in ('uncertain', 'skip'):
		mapping = identity_mapping(ids)
	else:
		raise ValueError(f'Unknown decision: {decision}')
	if remap_from_frame is None:
		remap_from_frame = default_remap_from_frame(event)
	return ReviewDecision(
		event_id=event.event_id,
		decision=decision,
		mapping=mapping,
		remap_from_frame=int(remap_from_frame),
		annotator=annotator,
		notes=notes,
		confidence=confidence,
		label_source='human',
	)


def save_switches(out_dir: str, markers: Sequence[SwitchMarker]) -> str:
	'''Overwrite switches.jsonl with the given markers (sorted by frame).'''
	os.makedirs(out_dir, exist_ok=True)
	path = os.path.join(out_dir, 'switches.jsonl')
	ordered = sorted(markers, key=lambda m: (m.frame, m.marker_id))
	with open(path, 'w', encoding='utf-8') as f:
		for m in ordered:
			if not m.timestamp_utc:
				m.timestamp_utc = datetime.now(timezone.utc).isoformat()
			f.write(dumps_jsonl_line(m.to_dict()) + '\n')
	return path


def load_switches(path_or_dir: str) -> List[SwitchMarker]:
	if os.path.isdir(path_or_dir):
		path = os.path.join(path_or_dir, 'switches.jsonl')
	else:
		path = path_or_dir
	if not os.path.isfile(path):
		return []
	markers = []
	with open(path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			markers.append(SwitchMarker.from_dict(json.loads(line)))
	markers.sort(key=lambda m: (m.frame, m.marker_id))
	return markers


def switches_to_decisions(markers: Sequence[SwitchMarker]) -> List[ReviewDecision]:
	return [m.to_review_decision() for m in markers if m.action == 'swap' and m.mapping]


def link_marker_to_event(marker: SwitchMarker, events: Sequence[ContactEvent]) -> Optional[str]:
	'''Nearest risk event containing the marker frame, else closest by center.'''
	best = None
	best_dist = None
	for ev in events:
		if ev.animal_kind != marker.animal_kind:
			continue
		if ev.start_frame <= marker.frame <= ev.end_frame + 5:
			return ev.event_id
		mid = 0.5 * (ev.start_frame + ev.end_frame)
		dist = abs(mid - marker.frame)
		if best_dist is None or dist < best_dist:
			best_dist = dist
			best = ev.event_id
	return best


def write_risk_labels(
	out_dir: str,
	events: Sequence[ContactEvent],
	markers: Sequence[SwitchMarker],
) -> str:
	'''
	Label each risk band: has_switch if any marker falls in [start, end+5].
	'''
	os.makedirs(out_dir, exist_ok=True)
	path = os.path.join(out_dir, 'risk_labels.jsonl')
	with open(path, 'w', encoding='utf-8') as f:
		for ev in events:
			hit = [
				m.marker_id
				for m in markers
				if m.animal_kind == ev.animal_kind
				and ev.start_frame <= m.frame <= ev.end_frame + 5
			]
			row = {
				'schema_version': SCHEMA_VERSION,
				'event_id': ev.event_id,
				'animal_kind': ev.animal_kind,
				'start_frame': ev.start_frame,
				'end_frame': ev.end_frame,
				'risk_score': ev.risk_score,
				'has_switch': bool(hit),
				'switch_marker_ids': hit,
				'label_source': 'human_timeline',
			}
			f.write(dumps_jsonl_line(row) + '\n')
	return path


def finalize_switch_annotations(
	out_dir: str,
	markers: Sequence[SwitchMarker],
	events: Optional[Sequence[ContactEvent]] = None,
	export_samples: bool = True,
) -> List[ReviewDecision]:
	'''
	Persist switches, sync decisions.jsonl for apply path, optional samples + risk labels.
	'''
	events = list(events) if events is not None else load_events(out_dir)
	# link events if missing
	final: List[SwitchMarker] = []
	for i, m in enumerate(sorted(markers, key=lambda x: (x.frame, x.marker_id))):
		if not m.marker_id:
			m.marker_id = f's{i:06d}'
		if not m.mapping and m.action == 'swap' and len(m.involved_ids) == 2:
			m.mapping = swap_mapping(m.involved_ids)
		if m.linked_event_id is None and events:
			m.linked_event_id = link_marker_to_event(m, events)
		final.append(m)

	save_switches(out_dir, final)

	# Rewrite decisions.jsonl from switches (authoritative for timeline UI)
	decisions = switches_to_decisions(final)
	dec_path = os.path.join(out_dir, 'decisions.jsonl')
	with open(dec_path, 'w', encoding='utf-8') as f:
		for d in decisions:
			if not d.timestamp_utc:
				d.timestamp_utc = datetime.now(timezone.utc).isoformat()
			f.write(dumps_jsonl_line(d.to_dict()) + '\n')

	write_risk_labels(out_dir, events, final)

	if export_samples:
		for m in final:
			try:
				export_switch_samples(out_dir, m)
			except Exception as exc:
				print(f'ID review: switch sample export failed for {m.marker_id}: {exc}', flush=True)

	return decisions


def make_swap_marker(
	frame: int,
	animal_kind: str,
	involved_ids: Sequence[int],
	fps: Optional[float] = None,
	marker_id: Optional[str] = None,
	linked_event_id: Optional[str] = None,
	notes: str = '',
) -> SwitchMarker:
	ids = [int(i) for i in involved_ids]
	if len(ids) != 2:
		raise ValueError('MVP swap markers require exactly 2 involved ids')
	mapping = swap_mapping(ids)
	time_sec = (float(frame) / float(fps)) if fps else None
	return SwitchMarker(
		marker_id=marker_id or f's{frame:06d}',
		frame=int(frame),
		animal_kind=animal_kind,
		involved_ids=ids,
		action='swap',
		mapping=mapping,
		time_sec=time_sec,
		linked_event_id=linked_event_id,
		notes=notes,
		timestamp_utc=datetime.now(timezone.utc).isoformat(),
	)


def run_id_review_pipeline(
	analyzer,
	config: Optional[ContactDetectorConfig] = None,
	review_callback=None,
	extract_samples: bool = True,
	auto_load_existing_decisions: bool = True,
) -> Tuple[str, List[ContactEvent], List[ReviewDecision]]:
	'''
	Full offline review hook for the analysis pipeline.

	review_callback(out_dir, events) -> list[ReviewDecision] | list[SwitchMarker] | None
	  GUI may return SwitchMarkers (preferred) or ReviewDecisions.
	  Timeline UI typically writes switches.jsonl itself and returns markers or [].

	Apply order: switches.jsonl if present and non-empty, else decisions.jsonl.
	'''
	# Per-event crop extraction is optional (timeline uses switch samples instead)
	out_dir, events = export_review_pack(
		analyzer,
		config=config,
		extract_samples=extract_samples,
	)

	if review_callback is not None:
		result = review_callback(out_dir, events) or []
		if result and isinstance(result[0], SwitchMarker):
			# GUI returned markers without writing; finalize now
			finalize_switch_annotations(
				out_dir,
				result,
				events=events,
				export_samples=True,
			)
		else:
			for d in result:
				if isinstance(d, ReviewDecision):
					append_decision(out_dir, d)

	# Prefer switches.jsonl for apply (timeline UI writes it on Done)
	markers = load_switches(out_dir)
	if markers:
		# Keep decisions.jsonl in sync for tooling that only reads decisions
		decisions = switches_to_decisions(markers)
		dec_path = os.path.join(out_dir, 'decisions.jsonl')
		with open(dec_path, 'w', encoding='utf-8') as f:
			for d in decisions:
				if not d.timestamp_utc:
					d.timestamp_utc = datetime.now(timezone.utc).isoformat()
				f.write(dumps_jsonl_line(d.to_dict()) + '\n')
		all_decisions = decisions
	elif auto_load_existing_decisions:
		all_decisions = load_decisions(os.path.join(out_dir, 'decisions.jsonl'))
	else:
		all_decisions = []

	kind_lookup = {e.event_id: e.animal_kind for e in events}
	for m in markers:
		kind_lookup[m.marker_id] = m.animal_kind

	applied = apply_decisions_to_analyzer(analyzer, all_decisions, event_kind_lookup=kind_lookup)
	if applied:
		write_applied_corrections(out_dir, applied)

	return out_dir, events, all_decisions
