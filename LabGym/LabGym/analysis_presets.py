'''
Save / load analysis-parameter presets for the Analyze Behaviors panel.

Format: JSON with schema_version. Paths are stored as absolute strings when set.
float('inf') for social_distance is stored as the string "inf".
'''

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1
PRESET_KIND = 'labgym_analyze_behaviors'

# Attributes on PanelLv2_AnalyzeBehaviors that define an analysis pipeline
PARAM_KEYS = [
	'behavior_mode',
	'use_detector',
	'detector_path',
	'path_to_detector',
	'detector_batch',
	'detection_threshold',
	'animal_kinds',
	'background_path',
	'model_path',
	'path_to_categorizer',
	'path_to_videos',
	'result_path',
	'framewidth',
	'delta',
	'decode_animalnumber',
	'animal_number',
	'autofind_t',
	'decode_t',
	't',
	'duration',
	'decode_extraction',
	'ex_start',
	'ex_end',
	'behaviornames_and_colors',
	'dim_tconv',
	'dim_conv',
	'channel',
	'length',
	'animal_vs_bg',
	'stable_illumination',
	'animation_analyzer',
	'animal_to_include',
	'ID_colors',
	'behavior_to_include',
	'parameter_to_analyze',
	'include_bodyparts',
	'std',
	'uncertain',
	'min_length',
	'show_legend',
	'background_free',
	'black_background',
	'normalize_distance',
	'social_distance',
	'color_costar',
	'specific_behaviors',
	'correct_ID',
	'id_review_enabled',
	'id_review_contact_distance_factor',
	'id_review_min_contact_frames',
	'id_review_gap_bridge_frames',
]


def _jsonable(value: Any) -> Any:
	if value is None:
		return None
	if isinstance(value, float) and value == float('inf'):
		return 'inf'
	if isinstance(value, (str, int, float, bool)):
		return value
	if isinstance(value, (list, tuple)):
		return [_jsonable(v) for v in value]
	if isinstance(value, dict):
		return {str(k): _jsonable(v) for k, v in value.items()}
	# pathlib etc.
	return str(value)


def _from_jsonable(value: Any, key: Optional[str] = None) -> Any:
	if value is None:
		return None
	if value == 'inf' and key == 'social_distance':
		return float('inf')
	if isinstance(value, list):
		# ID_colors: list of [b,g,r] -> list of tuples
		if key == 'ID_colors':
			out = []
			for item in value:
				if isinstance(item, (list, tuple)) and len(item) >= 3:
					out.append((int(item[0]), int(item[1]), int(item[2])))
				else:
					out.append(item)
			return out
		return [_from_jsonable(v) for v in value]
	if isinstance(value, dict):
		# animal_number may be dict of str->int
		if key == 'animal_number':
			# could also be a plain int stored without going through dict
			return {k: int(v) if not isinstance(v, dict) else v for k, v in value.items()}
		if key == 'specific_behaviors':
			# {animal: {behavior: null}}
			return {
				ak: {bk: None for bk in (av or {})}
				for ak, av in value.items()
			}
		if key == 'behaviornames_and_colors':
			return {k: list(v) if isinstance(v, list) else v for k, v in value.items()}
		return {k: _from_jsonable(v) for k, v in value.items()}
	return value


def panel_to_preset(panel) -> Dict[str, Any]:
	'''Serialize Analyze Behaviors panel state to a preset dict.'''
	params = {}
	for key in PARAM_KEYS:
		if hasattr(panel, key):
			params[key] = _jsonable(getattr(panel, key))
	# Sync id-review spins if present
	if hasattr(panel, 'spin_contact_dist'):
		params['id_review_contact_distance_factor'] = float(panel.spin_contact_dist.GetValue())
		params['id_review_min_contact_frames'] = int(panel.spin_min_contact.GetValue())
		params['id_review_gap_bridge_frames'] = int(panel.spin_gap_bridge.GetValue())
		params['id_review_enabled'] = bool(panel.checkbox_id_review.GetValue())

	return {
		'schema_version': SCHEMA_VERSION,
		'kind': PRESET_KIND,
		'saved_at_utc': datetime.now(timezone.utc).isoformat(),
		'description': '',
		'parameters': params,
	}


def apply_preset_to_panel(panel, preset: Dict[str, Any]) -> List[str]:
	'''
	Apply preset parameters onto the panel object (does not refresh labels).

	Returns a list of warning strings (missing paths, version notes, etc.).
	'''
	warnings: List[str] = []
	if not isinstance(preset, dict):
		raise ValueError('Preset must be a JSON object')
	kind = preset.get('kind')
	if kind is not None and kind != PRESET_KIND:
		warnings.append(f'Unexpected preset kind {kind!r}; expected {PRESET_KIND!r}.')
	ver = int(preset.get('schema_version', 1))
	if ver > SCHEMA_VERSION:
		warnings.append(
			f'Preset schema_version {ver} is newer than this LabGym ({SCHEMA_VERSION}); '
			'some fields may be ignored.'
		)
	params = preset.get('parameters')
	if not isinstance(params, dict):
		raise ValueError('Preset missing "parameters" object')

	for key in PARAM_KEYS:
		if key not in params:
			continue
		val = _from_jsonable(params[key], key=key)
		# animal_number stored as int in JSON stays int
		if key == 'animal_number' and isinstance(params[key], (int, float)) and not isinstance(params[key], bool):
			val = int(params[key])
		setattr(panel, key, val)

	# Path existence checks
	for path_key, label in (
		('path_to_categorizer', 'Categorizer'),
		('path_to_detector', 'Detector'),
		('result_path', 'Results folder'),
		('background_path', 'Background folder'),
	):
		p = getattr(panel, path_key, None)
		if p and not os.path.exists(p):
			warnings.append(f'{label} path not found: {p}')

	videos = getattr(panel, 'path_to_videos', None)
	if videos:
		missing = [v for v in videos if not os.path.isfile(v)]
		if missing:
			warnings.append(
				f'{len(missing)} of {len(videos)} input video/image path(s) not found '
				f'(first missing: {missing[0]}).'
			)

	return warnings


def save_preset(path: str, preset: Dict[str, Any]) -> None:
	parent = os.path.dirname(os.path.abspath(path))
	if parent:
		os.makedirs(parent, exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		json.dump(preset, f, indent=2, ensure_ascii=False)


def load_preset(path: str) -> Dict[str, Any]:
	with open(path, 'r', encoding='utf-8') as f:
		return json.load(f)


def refresh_analyze_behaviors_labels(panel) -> None:
	'''Update Analyze Behaviors StaticText labels and ID-review widgets from panel state.'''

	# Categorizer
	if panel.path_to_categorizer is None:
		panel.text_selectcategorizer.SetLabel(
			'No behavior classification; the time window to measure kinematics of tracked animals is: '
			+ str(panel.length)
			+ ' frames.'
		)
	else:
		uncertain_pct = int(round(float(panel.uncertain) * 100))
		base = f'The path to the Categorizer is: {panel.path_to_categorizer} with uncertainty of {uncertain_pct}%'
		if panel.min_length is not None:
			base += f'; minimun length of {panel.min_length}'
		panel.text_selectcategorizer.SetLabel(base + '.')

	# Videos
	if panel.path_to_videos:
		path = os.path.dirname(panel.path_to_videos[0])
		n = len(panel.path_to_videos)
		if panel.framewidth is not None:
			panel.text_inputvideos.SetLabel(
				f'Selected {n} file(s) in: {path} (proportionally resize frame / image width to {panel.framewidth}).'
			)
		else:
			panel.text_inputvideos.SetLabel(
				f'Selected {n} file(s) in: {path} (original frame / image size).'
			)
	else:
		panel.text_inputvideos.SetLabel('None.')

	# Output
	if panel.result_path:
		panel.text_outputfolder.SetLabel(f'Results will be in: {panel.result_path}.')
	else:
		panel.text_outputfolder.SetLabel('None.')

	# Detection method
	if panel.use_detector:
		det_name = (
			os.path.basename(panel.path_to_detector)
			if panel.path_to_detector
			else 'Detector'
		)
		if panel.behavior_mode >= 3:
			thr = int(round(float(panel.detection_threshold) * 100))
			panel.text_detection.SetLabel(
				f'Detector: {det_name} (detection threshold: {thr}%); '
				f'The animals/objects: {panel.animal_kinds}.'
			)
		else:
			panel.text_detection.SetLabel(
				f'Detector: {det_name}; The animals/objects: {panel.animal_kinds}.'
			)
	else:
		contrast = {
			0: 'animal brighter',
			1: 'animal darker',
			2: 'animal partially brighter/darker',
		}.get(panel.animal_vs_bg, 'animal')
		if panel.background_path:
			panel.text_detection.SetLabel(
				f'Background subtraction: {contrast}, loaded background from {panel.background_path}.'
			)
		elif panel.decode_extraction:
			panel.text_detection.SetLabel(
				f'Background subtraction: {contrast}, using time window decoded from filenames "_xst_" and "_xet_".'
			)
		elif panel.ex_end is None and panel.ex_start == 0:
			panel.text_detection.SetLabel(
				f'Background subtraction: {contrast}, using the entire duration.'
			)
		elif panel.ex_end is None:
			panel.text_detection.SetLabel(
				f'Background subtraction: {contrast}, using time window (in seconds) from {panel.ex_start} to the end.'
			)
		else:
			panel.text_detection.SetLabel(
				f'Background subtraction: {contrast}, using time window (in seconds) from {panel.ex_start} to {panel.ex_end}.'
			)

	# Timing
	if panel.behavior_mode >= 3:
		panel.text_startanalyze.SetLabel(
			'No need to specify this since the selected behavior mode is "Static images".'
		)
		panel.text_duration.SetLabel(
			'No need to specify this since the selected behavior mode is "Static images".'
		)
		panel.text_animalnumber.SetLabel(
			'No need to specify this since the selected behavior mode is "Static images".'
		)
		panel.text_selectparameters.SetLabel(
			'No need to specify this since the selected behavior mode is "Static images".'
		)
	else:
		if panel.autofind_t:
			panel.text_startanalyze.SetLabel(
				'Automatically find the onset of the 1st time when light on / off as the beginning time.'
			)
		elif panel.decode_t:
			panel.text_startanalyze.SetLabel(
				'Decode the beginning time from the filenames: the "t" immediately after the letter "b"" in "_bt_".'
			)
		else:
			panel.text_startanalyze.SetLabel(
				f'Analysis will begin at the: {panel.t} second.'
			)

		if panel.duration != 0:
			panel.text_duration.SetLabel(
				f'The analysis duration is {panel.duration} seconds.'
			)
		else:
			panel.text_duration.SetLabel(
				'The analysis duration is from the specified beginning time to the end of a video.'
			)

		if panel.decode_animalnumber:
			panel.text_animalnumber.SetLabel(
				'Decode from the filenames: the "n" immediately after the letter "n" in _"nn"_.'
			)
		elif panel.use_detector and isinstance(panel.animal_number, dict):
			panel.text_animalnumber.SetLabel(
				f'The number of {panel.animal_kinds} is: {list(panel.animal_number.values())}.'
			)
		elif panel.animal_number is not None:
			panel.text_animalnumber.SetLabel(
				f'The total number of animals in a video is {panel.animal_number}.'
			)
		else:
			panel.text_animalnumber.SetLabel('Default: 1.')

		# Parameters
		if panel.parameter_to_analyze:
			norm = (
				' with normalization of distance'
				if panel.normalize_distance
				else ' NO normalization of distance'
			)
			# only mention norm if locomotion selected
			if '4 locomotion parameters' in panel.parameter_to_analyze:
				panel.text_selectparameters.SetLabel(
					'Selected: ' + str(panel.parameter_to_analyze) + ';' + norm + '.'
				)
			else:
				panel.text_selectparameters.SetLabel(
					'Selected: ' + str(panel.parameter_to_analyze) + '.'
				)
		else:
			panel.text_selectparameters.SetLabel('Default: none.')

	# Behaviors
	if panel.path_to_categorizer is None:
		panel.text_selectbehaviors.SetLabel(
			'No behavior classification. Just track animals and quantify motion kinematics.'
		)
	elif panel.behavior_to_include:
		if panel.correct_ID:
			panel.text_selectbehaviors.SetLabel(
				'Selected: '
				+ str(panel.behavior_to_include)
				+ '. Specific behaviors: '
				+ str(panel.specific_behaviors)
				+ '.'
			)
		else:
			panel.text_selectbehaviors.SetLabel(
				'Selected: ' + str(panel.behavior_to_include) + '.'
			)
	else:
		panel.text_selectbehaviors.SetLabel(
			'All the behaviors in the selected Categorizer with default colors.'
		)

	# ID review widgets
	if hasattr(panel, 'checkbox_id_review'):
		panel.checkbox_id_review.SetValue(bool(panel.id_review_enabled))
		panel.spin_contact_dist.SetValue(float(panel.id_review_contact_distance_factor))
		panel.spin_min_contact.SetValue(int(panel.id_review_min_contact_frames))
		panel.spin_gap_bridge.SetValue(int(panel.id_review_gap_bridge_frames))
		if hasattr(panel, '_set_id_review_params_enabled'):
			panel._set_id_review_params_enabled(bool(panel.id_review_enabled))
