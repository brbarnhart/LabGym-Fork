'''Tests for Analyze Behaviors parameter presets.'''

import json
import os
import tempfile

from LabGym.analysis_presets import (
	SCHEMA_VERSION,
	apply_preset_to_panel,
	load_preset,
	panel_to_preset,
	save_preset,
)


class _FakePanel:
	def __init__(self):
		self.behavior_mode = 0
		self.use_detector = True
		self.detector_path = None
		self.path_to_detector = r'C:\detectors\mouse'
		self.detector_batch = 2
		self.detection_threshold = 0.0
		self.animal_kinds = ['mouse']
		self.background_path = None
		self.model_path = None
		self.path_to_categorizer = None
		self.path_to_videos = [r'C:\videos\a.avi', r'C:\videos\b.avi']
		self.result_path = r'C:\results'
		self.framewidth = 480
		self.delta = 10000
		self.decode_animalnumber = False
		self.animal_number = {'mouse': 2}
		self.autofind_t = False
		self.decode_t = False
		self.t = 5.0
		self.duration = 60
		self.decode_extraction = False
		self.ex_start = 0
		self.ex_end = None
		self.behaviornames_and_colors = {}
		self.dim_tconv = 8
		self.dim_conv = 8
		self.channel = 1
		self.length = 15
		self.animal_vs_bg = 0
		self.stable_illumination = True
		self.animation_analyzer = False
		self.animal_to_include = ['mouse']
		self.ID_colors = [(255, 0, 0), (0, 255, 0)]
		self.behavior_to_include = ['all']
		self.parameter_to_analyze = ['4 locomotion parameters']
		self.include_bodyparts = False
		self.std = 0
		self.uncertain = 0.1
		self.min_length = None
		self.show_legend = True
		self.background_free = True
		self.black_background = True
		self.normalize_distance = True
		self.social_distance = float('inf')
		self.color_costar = False
		self.specific_behaviors = {}
		self.correct_ID = False
		self.id_review_enabled = True
		self.id_review_contact_distance_factor = 1.2
		self.id_review_min_contact_frames = 4
		self.id_review_gap_bridge_frames = 1


def test_panel_to_preset_roundtrip_json():
	panel = _FakePanel()
	preset = panel_to_preset(panel)
	assert preset['schema_version'] == SCHEMA_VERSION
	assert preset['kind'] == 'labgym_analyze_behaviors'
	assert preset['parameters']['social_distance'] == 'inf'
	assert preset['parameters']['animal_number'] == {'mouse': 2}
	assert preset['parameters']['id_review_enabled'] is True

	with tempfile.TemporaryDirectory() as td:
		path = os.path.join(td, 'preset.json')
		save_preset(path, preset)
		loaded = load_preset(path)
		assert loaded['parameters']['framewidth'] == 480

	other = _FakePanel()
	other.t = 0
	other.animal_number = 1
	other.id_review_enabled = False
	other.social_distance = 0
	other.path_to_videos = None
	other.result_path = None
	other.path_to_detector = None
	warnings = apply_preset_to_panel(other, loaded)
	# missing paths may warn
	assert other.t == 5.0
	assert other.animal_number == {'mouse': 2}
	assert other.social_distance == float('inf')
	assert other.id_review_contact_distance_factor == 1.2
	assert other.ID_colors == [(255, 0, 0), (0, 255, 0)]
	assert other.duration == 60


def test_animal_number_int_roundtrip():
	panel = _FakePanel()
	panel.animal_number = 3
	panel.use_detector = False
	preset = panel_to_preset(panel)
	assert preset['parameters']['animal_number'] == 3
	other = _FakePanel()
	apply_preset_to_panel(other, preset)
	assert other.animal_number == 3


def test_json_serializable():
	panel = _FakePanel()
	preset = panel_to_preset(panel)
	# must not raise
	s = json.dumps(preset)
	assert 'labgym_analyze_behaviors' in s
