'''Data structures for contact risk review and training export.'''

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

import numpy as np

SCHEMA_VERSION = 1


@dataclass
class ContactDetectorConfig:
	'''Tunable thresholds for contact / separation detection.'''

	contact_distance_factor: float = 1.5
	'''Centers closer than factor * animal_size are "in contact".'''

	min_contact_frames: int = 3
	'''Minimum consecutive contact frames to emit an event.'''

	gap_bridge_frames: int = 2
	'''Bridge brief non-contact gaps inside a longer bout.'''

	min_separation_gap: float = 1.0
	'''Centers farther than factor * size count as separated (for pre/post).'''

	window_before: int = 15
	'''Frames of trajectory context before contact (samples / features).'''

	window_after: int = 15
	'''Frames of trajectory context after contact.'''

	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)

	@classmethod
	def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'ContactDetectorConfig':
		if not d:
			return cls()
		known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
		return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ContactEvent:
	'''A close-proximity bout where identity switch is plausible.'''

	event_id: str
	animal_kind: str
	involved_ids: List[int]
	start_frame: int
	end_frame: int
	pre_frame: int
	post_frame: int
	risk_score: float = 0.0
	risk_flags: List[str] = field(default_factory=list)
	sample_dir: Optional[str] = None
	video: Optional[str] = None
	fps: Optional[float] = None

	def to_dict(self) -> Dict[str, Any]:
		d = asdict(self)
		d['schema_version'] = SCHEMA_VERSION
		return d

	@classmethod
	def from_dict(cls, d: Dict[str, Any]) -> 'ContactEvent':
		known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
		return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SwitchMarker:
	'''
	Human-confirmed identity switch at a specific analysis frame.

	From `frame` inclusive forward, `mapping` is applied (same convention as
	ReviewDecision: mapping[post_id] = pre_identity continued by that track).
	'''

	marker_id: str
	frame: int
	animal_kind: str
	involved_ids: List[int]
	action: str = 'swap'  # swap (MVP)
	mapping: Dict[int, int] = field(default_factory=dict)
	time_sec: Optional[float] = None
	linked_event_id: Optional[str] = None
	notes: str = ''
	annotator: str = 'local_user'
	timestamp_utc: str = ''
	label_source: str = 'human'

	def to_dict(self) -> Dict[str, Any]:
		return {
			'schema_version': SCHEMA_VERSION,
			'marker_id': self.marker_id,
			'frame': int(self.frame),
			'time_sec': self.time_sec,
			'animal_kind': self.animal_kind,
			'involved_ids': [int(i) for i in self.involved_ids],
			'action': self.action,
			'mapping': {str(k): int(v) for k, v in self.mapping.items()},
			'linked_event_id': self.linked_event_id,
			'notes': self.notes,
			'annotator': self.annotator,
			'timestamp_utc': self.timestamp_utc,
			'label_source': self.label_source,
		}

	@classmethod
	def from_dict(cls, d: Dict[str, Any]) -> 'SwitchMarker':
		raw_map = d.get('mapping') or {}
		mapping = {int(k): int(v) for k, v in raw_map.items()}
		return cls(
			marker_id=str(d.get('marker_id') or d.get('event_id') or 'm000000'),
			frame=int(d['frame'] if 'frame' in d else d.get('remap_from_frame', 0)),
			animal_kind=str(d.get('animal_kind') or 'animal'),
			involved_ids=[int(i) for i in (d.get('involved_ids') or list(mapping.keys()))],
			action=str(d.get('action') or d.get('decision') or 'swap'),
			mapping=mapping,
			time_sec=(float(d['time_sec']) if d.get('time_sec') is not None else None),
			linked_event_id=d.get('linked_event_id'),
			notes=d.get('notes', ''),
			annotator=d.get('annotator', 'local_user'),
			timestamp_utc=d.get('timestamp_utc', ''),
			label_source=d.get('label_source', 'human'),
		)

	def to_review_decision(self) -> 'ReviewDecision':
		'''Compat: convert to ReviewDecision for apply_decisions_to_analyzer.'''
		return ReviewDecision(
			event_id=self.marker_id,
			decision='swap' if self.action == 'swap' else self.action,
			mapping=dict(self.mapping),
			remap_from_frame=int(self.frame),
			timestamp_utc=self.timestamp_utc,
			annotator=self.annotator,
			notes=self.notes,
			confidence='high',
			label_source=self.label_source,
		)


@dataclass
class ReviewDecision:
	'''
	Human (or future auto-confirmed) decision for one contact event.

	mapping: post_id -> pre_identity (see package docstring).
	Keys may be int or str in JSON; always normalized to int on load.
	'''

	event_id: str
	decision: str  # keep | swap | uncertain | skip
	mapping: Dict[int, int]
	remap_from_frame: int
	timestamp_utc: str = ''
	annotator: str = 'local_user'
	notes: str = ''
	confidence: str = 'high'
	label_source: str = 'human'

	def to_dict(self) -> Dict[str, Any]:
		return {
			'schema_version': SCHEMA_VERSION,
			'event_id': self.event_id,
			'timestamp_utc': self.timestamp_utc,
			'decision': self.decision,
			'mapping': {str(k): int(v) for k, v in self.mapping.items()},
			'remap_from_frame': self.remap_from_frame,
			'annotator': self.annotator,
			'notes': self.notes,
			'confidence': self.confidence,
			'label_source': self.label_source,
		}

	@classmethod
	def from_dict(cls, d: Dict[str, Any]) -> 'ReviewDecision':
		raw_map = d.get('mapping') or {}
		mapping = {int(k): int(v) for k, v in raw_map.items()}
		return cls(
			event_id=d['event_id'],
			decision=d['decision'],
			mapping=mapping,
			remap_from_frame=int(d['remap_from_frame']),
			timestamp_utc=d.get('timestamp_utc', ''),
			annotator=d.get('annotator', 'local_user'),
			notes=d.get('notes', ''),
			confidence=d.get('confidence', 'high'),
			label_source=d.get('label_source', 'human'),
		)

	@property
	def applies_remap(self) -> bool:
		if self.decision in ('uncertain', 'skip'):
			return False
		return any(int(k) != int(v) for k, v in self.mapping.items())


@dataclass
class TrackletStore:
	'''
	Per-kind multi-ID trajectories for offline review.

	centers: shape (n_ids, n_frames, 2), invalid where valid==False
	heights: shape (n_ids, n_frames), NaN if missing
	contours: list length n_ids, each a list length n_frames of ndarray|None
	'''

	schema_version: int
	animal_kind: str
	ids: List[int]
	n_frames: int
	centers: np.ndarray
	valid: np.ndarray
	heights: np.ndarray
	contours: List[List[Optional[np.ndarray]]]
	meta: Dict[str, Any] = field(default_factory=dict)

	def id_index(self, track_id: int) -> int:
		return self.ids.index(track_id)


def identity_mapping(ids: Sequence[int]) -> Dict[int, int]:
	return {int(i): int(i) for i in ids}


def swap_mapping(ids: Sequence[int]) -> Dict[int, int]:
	'''Pairwise swap for exactly two IDs; raises if not length 2.'''
	ids = [int(i) for i in ids]
	if len(ids) != 2:
		raise ValueError('swap_mapping requires exactly two involved ids')
	a, b = ids
	return {a: b, b: a}


def mapping_to_permutation(ids: Sequence[int], mapping: Dict[int, int]) -> List[int]:
	'''
	label_permutation[j] = pre_identity that post crop of ids[j] continues.

	ids order is the involved_ids order used in the event.
	'''
	return [int(mapping[int(i)]) for i in ids]


def dumps_jsonl_line(obj: Dict[str, Any]) -> str:
	return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
