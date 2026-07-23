'''
Contact-based identity review for LabGym detector tracks.

Mapping convention (schema_version 1)
-------------------------------------
Crops and tracklet fields are labeled by the tracker's IDs at a given frame.

A mapping maps post_id -> pre_identity continued by that track::

    mapping[post_id] = pre_identity

Example (2-animal swap)::

    mapping = {0: 1, 1: 0}

Applying reassigns per-frame data for frames >= remap_from_frame so that::

    new[mapping[i]][f] = old[i][f]

Timeline UI records SwitchMarkers in switches.jsonl; each converts to a
ReviewDecision for apply. Risk contact events remain navigation hints.

Do not change this convention without bumping schema_version.
'''

from .types import (
	SCHEMA_VERSION,
	ContactEvent,
	ContactDetectorConfig,
	ReviewDecision,
	SwitchMarker,
	TrackletStore,
)
from .contacts import detect_contact_events
from .tracklets import (
	tracklets_from_analyzer,
	save_tracklets,
	load_tracklets,
	apply_mapping_to_store,
)
from .apply import (
	apply_decision_to_analyzer,
	apply_decisions_to_analyzer,
	apply_decisions_to_store,
	load_decisions,
	write_tracklets_identity_status,
	read_tracklets_identity_status,
)
from .dataset import (
	export_review_pack,
	append_decision,
	write_pair_label,
	load_events,
	run_id_review_pipeline,
	make_decision_for_event,
	default_remap_from_frame,
	save_switches,
	load_switches,
	switches_to_decisions,
	finalize_switch_annotations,
	make_swap_marker,
)

__all__ = [
	'SCHEMA_VERSION',
	'ContactEvent',
	'ContactDetectorConfig',
	'ReviewDecision',
	'SwitchMarker',
	'TrackletStore',
	'detect_contact_events',
	'tracklets_from_analyzer',
	'save_tracklets',
	'load_tracklets',
	'apply_mapping_to_store',
	'apply_decision_to_analyzer',
	'apply_decisions_to_analyzer',
	'apply_decisions_to_store',
	'load_decisions',
	'write_tracklets_identity_status',
	'read_tracklets_identity_status',
	'export_review_pack',
	'append_decision',
	'write_pair_label',
	'load_events',
	'run_id_review_pipeline',
	'make_decision_for_event',
	'default_remap_from_frame',
	'save_switches',
	'load_switches',
	'switches_to_decisions',
	'finalize_switch_annotations',
	'make_swap_marker',
]
