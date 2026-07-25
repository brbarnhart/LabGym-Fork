"""Corrected tracklets must be re-saved after ID review remaps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from LabGym.id_review.apply import (
	apply_decisions_to_store,
	read_tracklets_identity_status,
	write_tracklets_identity_status,
)
from LabGym.id_review.tracklets import (
	TrackletStore,
	apply_mapping_to_store,
	save_tracklets,
	load_tracklets,
)
from LabGym.id_review.types import SCHEMA_VERSION, ReviewDecision


def _make_store(n_frames: int = 20) -> "TrackletStore":
	# ID 0 stays left, ID 1 stays right — after swap from frame 10, positions flip
	centers = np.zeros((2, n_frames, 2), dtype=np.float64)
	valid = np.ones((2, n_frames), dtype=bool)
	heights = np.full((2, n_frames), 10.0)
	contours = [[None] * n_frames, [None] * n_frames]
	for f in range(n_frames):
		centers[0, f] = (10.0, 10.0)
		centers[1, f] = (90.0, 10.0)
		contours[0][f] = np.array([[[8, 8]], [[12, 8]], [[12, 12]], [[8, 12]]], dtype=np.int32)
		contours[1][f] = np.array([[[88, 8]], [[92, 8]], [[92, 12]], [[88, 12]]], dtype=np.int32)
	return TrackletStore(
		schema_version=SCHEMA_VERSION,
		animal_kind="mouse",
		ids=[0, 1],
		n_frames=n_frames,
		centers=centers,
		valid=valid,
		heights=heights,
		contours=contours,
		meta={"fps": 10},
	)


def test_apply_mapping_swaps_from_frame():
	store = _make_store()
	apply_mapping_to_store(store, {0: 1, 1: 0}, remap_from_frame=10)
	# Before remap: id0 left
	assert store.centers[0, 5, 0] == 10.0
	assert store.centers[1, 5, 0] == 90.0
	# After remap: id0 has what was id1's trajectory from f10
	assert store.centers[0, 10, 0] == 90.0
	assert store.centers[1, 10, 0] == 10.0


def test_apply_decisions_to_store_and_roundtrip(tmp_path: Path):
	store = _make_store()
	save_tracklets(store, str(tmp_path))
	write_tracklets_identity_status(str(tmp_path), corrected=False)

	d = ReviewDecision(
		event_id="m1",
		decision="swap",
		mapping={0: 1, 1: 0},
		remap_from_frame=10,
	)
	loaded = load_tracklets(str(tmp_path), "mouse")
	n = apply_decisions_to_store(loaded, [d])
	assert n == 1
	assert loaded.centers[0, 15, 0] == 90.0
	save_tracklets(loaded, str(tmp_path))
	write_tracklets_identity_status(
		str(tmp_path), corrected=True, n_decisions=1, source="test"
	)

	again = load_tracklets(str(tmp_path), "mouse")
	assert again.centers[0, 15, 0] == 90.0
	assert again.centers[1, 15, 0] == 10.0
	st = read_tracklets_identity_status(str(tmp_path))
	assert st["corrected"] is True


def test_annotator_lazy_apply(tmp_path: Path):
	"""Simulates old pack: uncorrected npz + switches.jsonl."""
	from LabGym.annotator.core.tracklets_bridge import load_tracklets_for_annotator
	from LabGym.id_review.types import SwitchMarker
	from LabGym.id_review.dataset import save_switches

	store = _make_store()
	save_tracklets(store, str(tmp_path))
	# No corrected flag
	marker = SwitchMarker(
		marker_id="s000001_mouse",
		frame=10,
		animal_kind="mouse",
		involved_ids=[0, 1],
		action="swap",
		mapping={0: 1, 1: 0},
	)
	save_switches(str(tmp_path), [marker])

	loaded = load_tracklets_for_annotator(str(tmp_path), video_total_frames=20)
	# Overlays / centers for subject 0 at analysis frame 15 should be remapped (right)
	st = loaded.stores["mouse"]
	assert st.centers[0, 15, 0] == 90.0
	assert read_tracklets_identity_status(str(tmp_path))["corrected"] is True
