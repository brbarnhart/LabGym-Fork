'''Extract pre/post review images and simple marker features from video.'''

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .types import ContactEvent, SwitchMarker, TrackletStore
from .types import SCHEMA_VERSION


def analysis_frame_to_video_frame(analyzer_meta: Dict[str, Any], analysis_frame: int, fps: float) -> int:
	'''
	Map analysis frame index to absolute video frame index.

	LabGym reads frames with time=(frame_count+1)/fps and begins analysis when
	time >= start_t where start_t = max(0, t - length/fps). The first analysis
	frame therefore corresponds to the first video frame_count meeting that.
	'''
	if not fps:
		return max(0, int(analysis_frame))
	start_t = analyzer_meta.get('start_t')
	if start_t is None:
		t = float(analyzer_meta.get('t') or 0)
		length = float(analyzer_meta.get('length') or 0)
		start_t = max(0.0, round(t - length / float(fps), 2))
	# Smallest frame_count with (frame_count+1)/fps >= start_t
	# => frame_count+1 >= start_t*fps => frame_count >= start_t*fps - 1
	first = max(0, int(round(float(start_t) * float(fps) - 1)))
	return max(0, first + int(analysis_frame))


def draw_tracklet_overlay(
	frame: np.ndarray,
	store: TrackletStore,
	frame_idx: int,
	highlight_ids: Optional[List[int]] = None,
	remap_frame: Optional[int] = None,
) -> np.ndarray:
	out = frame.copy()
	highlight_ids = highlight_ids or store.ids
	colors = {}
	for i, tid in enumerate(store.ids):
		# distinct BGR colors
		colors[tid] = (
			int(37 * (i + 1) % 255),
			int(91 * (i + 3) % 255),
			int(173 * (i + 5) % 255),
		)
	for tid in store.ids:
		row = store.id_index(tid)
		if frame_idx >= store.n_frames or not store.valid[row, frame_idx]:
			continue
		cnt = store.contours[row][frame_idx]
		cx, cy = store.centers[row, frame_idx]
		color = colors[tid]
		thick = 3 if tid in highlight_ids else 1
		if cnt is not None:
			cv2.drawContours(out, [cnt], 0, color, thick)
		cv2.circle(out, (int(cx), int(cy)), 4, color, -1)
		cv2.putText(
			out,
			str(tid),
			(int(cx) + 6, int(cy) - 6),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			color,
			2,
			cv2.LINE_AA,
		)
	# HUD: analysis frame + markers
	label = f'analysis f={frame_idx}'
	if remap_frame is not None:
		label += f'  remap_from={remap_frame}'
		if frame_idx >= remap_frame:
			label += '  [remap active region]'
	cv2.putText(out, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
	return out


# Back-compat aliases used internally
_analysis_frame_to_video_frame = analysis_frame_to_video_frame
_draw_overlay = draw_tracklet_overlay


def _crop_rgb(frame: np.ndarray, contour: Optional[np.ndarray], center: Tuple[float, float], pad: int = 8) -> np.ndarray:
	h, w = frame.shape[:2]
	if contour is not None and len(contour) > 0:
		x, y, bw, bh = cv2.boundingRect(contour)
		x0 = max(0, x - pad)
		y0 = max(0, y - pad)
		x1 = min(w, x + bw + pad)
		y1 = min(h, y + bh + pad)
		crop = frame[y0:y1, x0:x1].copy()
		# mask to contour in crop coords
		mask = np.zeros(crop.shape[:2], dtype=np.uint8)
		shifted = contour.copy()
		shifted[:, 0, 0] -= x0
		shifted[:, 0, 1] -= y0
		cv2.drawContours(mask, [shifted], 0, 255, -1)
		masked = crop.copy()
		masked[mask == 0] = 0
		return masked
	# fallback square around center
	cx, cy = int(center[0]), int(center[1])
	s = 40
	x0, y0 = max(0, cx - s), max(0, cy - s)
	x1, y1 = min(w, cx + s), min(h, cy + s)
	return frame[y0:y1, x0:x1].copy()


def compute_marker_features(crop_bgr: np.ndarray) -> Dict[str, float]:
	'''Deterministic marker-oriented features for future automation.'''
	if crop_bgr is None or crop_bgr.size == 0:
		return {
			'mean_b': 0.0, 'mean_g': 0.0, 'mean_r': 0.0,
			'red_frac': 0.0, 'bright_frac': 0.0, 'max_bright': 0.0,
		}
	pix = crop_bgr.reshape(-1, 3)
	# ignore pure black background from mask
	mask = np.any(pix > 5, axis=1)
	if not np.any(mask):
		pix_m = pix
	else:
		pix_m = pix[mask]
	mean = pix_m.mean(axis=0)
	b, g, r = float(mean[0]), float(mean[1]), float(mean[2])
	# red-ish: R high relative to G and B
	red_frac = float(np.mean((pix_m[:, 2] > 80) & (pix_m[:, 2] > pix_m[:, 1] * 1.2) & (pix_m[:, 2] > pix_m[:, 0] * 1.2)))
	gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).reshape(-1)
	if np.any(mask):
		gray_m = gray[mask]
	else:
		gray_m = gray
	bright_frac = float(np.mean(gray_m > 200))
	max_bright = float(np.max(gray_m)) if gray_m.size else 0.0
	return {
		'mean_b': b, 'mean_g': g, 'mean_r': r,
		'red_frac': red_frac, 'bright_frac': bright_frac, 'max_bright': max_bright,
	}


def _read_video_frame(capture: cv2.VideoCapture, frame_index: int, framewidth: Optional[int]) -> Optional[np.ndarray]:
	capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
	ok, frame = capture.read()
	if not ok or frame is None:
		return None
	if framewidth is not None:
		h, w = frame.shape[:2]
		fh = int(h * framewidth / w)
		frame = cv2.resize(frame, (framewidth, fh), interpolation=cv2.INTER_AREA)
	return frame


def export_event_samples(
	event: ContactEvent,
	store: TrackletStore,
	video_path: str,
	sample_dir: str,
	fps: float,
	framewidth: Optional[int] = None,
	analysis_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
	'''
	Write pre/post full frames and per-ID RGB crops for one event.

	Returns meta dict (also written to sample_dir/meta.json).
	'''
	os.makedirs(sample_dir, exist_ok=True)
	analysis_meta = analysis_meta or store.meta or {}
	fps = float(fps or analysis_meta.get('fps') or 30)
	fw = framewidth if framewidth is not None else analysis_meta.get('framewidth')

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise IOError(f'Cannot open video: {video_path}')

	try:
		pre_v = _analysis_frame_to_video_frame(analysis_meta, event.pre_frame, fps)
		post_v = _analysis_frame_to_video_frame(analysis_meta, event.post_frame, fps)
		pre_frame = _read_video_frame(cap, pre_v, fw)
		post_frame = _read_video_frame(cap, post_v, fw)
	finally:
		cap.release()

	if pre_frame is None or post_frame is None:
		raise IOError(f'Failed to read pre/post frames for {event.event_id}')

	pre_full = draw_tracklet_overlay(pre_frame, store, event.pre_frame, event.involved_ids)
	post_full = draw_tracklet_overlay(post_frame, store, event.post_frame, event.involved_ids)
	cv2.imwrite(os.path.join(sample_dir, 'pre_full.jpg'), pre_full)
	cv2.imwrite(os.path.join(sample_dir, 'post_full.jpg'), post_full)

	pre_crops = {}
	post_crops = {}
	features = {'pre': {}, 'post': {}}
	for tid in event.involved_ids:
		row = store.id_index(tid)
		# pre
		cnt = store.contours[row][event.pre_frame] if store.valid[row, event.pre_frame] else None
		ctr = tuple(store.centers[row, event.pre_frame]) if store.valid[row, event.pre_frame] else (0, 0)
		crop = _crop_rgb(pre_frame, cnt, ctr)
		pre_name = f'pre_id_{tid}.png'
		cv2.imwrite(os.path.join(sample_dir, pre_name), crop)
		pre_crops[str(tid)] = pre_name
		features['pre'][str(tid)] = compute_marker_features(crop)
		# post
		cnt = store.contours[row][event.post_frame] if store.valid[row, event.post_frame] else None
		ctr = tuple(store.centers[row, event.post_frame]) if store.valid[row, event.post_frame] else (0, 0)
		crop = _crop_rgb(post_frame, cnt, ctr)
		post_name = f'post_id_{tid}.png'
		cv2.imwrite(os.path.join(sample_dir, post_name), crop)
		post_crops[str(tid)] = post_name
		features['post'][str(tid)] = compute_marker_features(crop)

	# short center snippets
	def _snippet(f0: int, span: int = 15) -> Dict[str, List[Optional[List[float]]]]:
		out: Dict[str, List[Optional[List[float]]]] = {}
		for tid in event.involved_ids:
			row = store.id_index(tid)
			pts = []
			for f in range(max(0, f0 - span), min(store.n_frames, f0 + span + 1)):
				if store.valid[row, f]:
					pts.append([float(store.centers[row, f, 0]), float(store.centers[row, f, 1])])
				else:
					pts.append(None)
			out[str(tid)] = pts
		return out

	pre_centers = _snippet(event.pre_frame)
	post_centers = _snippet(event.post_frame)
	with open(os.path.join(sample_dir, 'pre_centers.json'), 'w', encoding='utf-8') as f:
		json.dump(pre_centers, f)
	with open(os.path.join(sample_dir, 'post_centers.json'), 'w', encoding='utf-8') as f:
		json.dump(post_centers, f)

	meta = {
		'schema_version': SCHEMA_VERSION,
		'event_id': event.event_id,
		'animal_kind': event.animal_kind,
		'involved_ids': event.involved_ids,
		'pre_frame': event.pre_frame,
		'post_frame': event.post_frame,
		'pre_video_frame': pre_v,
		'post_video_frame': post_v,
		'pre_crops': pre_crops,
		'post_crops': post_crops,
		'marker_features': features,
		'mapping_convention': (
			'Crops labeled by tracker IDs at pre/post. '
			'decision.mapping[post_id] = pre_identity continued by that post crop.'
		),
	}
	with open(os.path.join(sample_dir, 'meta.json'), 'w', encoding='utf-8') as f:
		json.dump(meta, f, indent=2)
	return meta


def detections_at_frame_after_markers(
	store: TrackletStore,
	frame_idx: int,
	markers: Sequence['SwitchMarker'],
) -> Dict[int, Tuple[bool, Tuple[float, float], Optional[np.ndarray]]]:
	'''
	Single-frame detections under each track ID after applying swap markers
	with marker.frame <= frame_idx (for live preview of corrected labels).

	Returns tid -> (valid, center, contour).
	'''
	# type: ignore[name-defined]
	out: Dict[int, Tuple[bool, Tuple[float, float], Optional[np.ndarray]]] = {}
	for tid in store.ids:
		row = store.id_index(tid)
		if frame_idx < store.n_frames and store.valid[row, frame_idx]:
			c = store.centers[row, frame_idx]
			out[tid] = (True, (float(c[0]), float(c[1])), store.contours[row][frame_idx])
		else:
			out[tid] = (False, (0.0, 0.0), None)

	for m in sorted(markers, key=lambda x: x.frame):
		if m.frame > frame_idx:
			break
		if m.action != 'swap' or not m.mapping:
			continue
		# new[mapping[src]] = old[src]
		old = {tid: out[tid] for tid in m.mapping.keys() if tid in out}
		for src, dst in m.mapping.items():
			if src in old:
				out[int(dst)] = old[int(src)]
	return out


def draw_detections_overlay(
	frame: np.ndarray,
	detections: Dict[int, Tuple[bool, Tuple[float, float], Optional[np.ndarray]]],
	highlight_ids: Optional[List[int]] = None,
	frame_idx: int = 0,
	n_markers_applied: int = 0,
) -> np.ndarray:
	'''Draw ID labels/contours from a detections dict (post-marker preview).'''
	out = frame.copy()
	highlight_ids = highlight_ids or list(detections.keys())
	for i, tid in enumerate(sorted(detections.keys())):
		valid, center, cnt = detections[tid]
		if not valid:
			continue
		color = (
			int(37 * (i + 1) % 255),
			int(91 * (i + 3) % 255),
			int(173 * (i + 5) % 255),
		)
		thick = 3 if tid in highlight_ids else 1
		if cnt is not None:
			cv2.drawContours(out, [cnt], 0, color, thick)
		cx, cy = int(center[0]), int(center[1])
		cv2.circle(out, (cx, cy), 4, color, -1)
		cv2.putText(
			out,
			str(tid),
			(cx + 6, cy - 6),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			color,
			2,
			cv2.LINE_AA,
		)
	cv2.putText(
		out,
		f'analysis f={frame_idx}  markers<={n_markers_applied} applied (preview)',
		(10, 28),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.65,
		(0, 255, 255),
		2,
		cv2.LINE_AA,
	)
	return out


def export_switch_samples(out_dir: str, marker: SwitchMarker, window: int = 5) -> Dict[str, Any]:
	'''
	Export pre/post crops around a switch marker for training.

	pre = frame-1 (or frame-window), post = frame (switch applies from frame).
	'''
	from .tracklets import load_tracklets

	kind = marker.animal_kind
	store = load_tracklets(out_dir, kind)
	meta = store.meta or {}
	video = meta.get('video')
	fps = float(meta.get('fps') or 30)
	fw = meta.get('framewidth')
	if not video or not os.path.isfile(str(video)):
		raise IOError(f'Video not found for switch samples: {video}')

	sample_rel = os.path.join('samples', f'switch_{marker.marker_id}')
	sample_dir = os.path.join(out_dir, sample_rel)
	os.makedirs(sample_dir, exist_ok=True)

	pre_f = max(0, int(marker.frame) - 1)
	post_f = min(store.n_frames - 1, int(marker.frame))
	# also grab slightly earlier "clean" pre if possible
	pre_f = max(0, int(marker.frame) - max(1, window))

	cap = cv2.VideoCapture(str(video))
	if not cap.isOpened():
		raise IOError(f'Cannot open video: {video}')
	try:
		pre_v = analysis_frame_to_video_frame(meta, pre_f, fps)
		post_v = analysis_frame_to_video_frame(meta, post_f, fps)
		pre_img = _read_video_frame(cap, pre_v, fw)
		post_img = _read_video_frame(cap, post_v, fw)
	finally:
		cap.release()
	if pre_img is None or post_img is None:
		raise IOError(f'Failed to read frames for switch {marker.marker_id}')

	pre_full = draw_tracklet_overlay(pre_img, store, pre_f, marker.involved_ids)
	post_full = draw_tracklet_overlay(post_img, store, post_f, marker.involved_ids)
	cv2.imwrite(os.path.join(sample_dir, 'pre_full.jpg'), pre_full)
	cv2.imwrite(os.path.join(sample_dir, 'post_full.jpg'), post_full)

	pre_crops = {}
	post_crops = {}
	features = {'pre': {}, 'post': {}}
	for tid in marker.involved_ids:
		row = store.id_index(tid)
		for tag, f_idx, img, crops_dict, feat_key in (
			('pre', pre_f, pre_img, pre_crops, 'pre'),
			('post', post_f, post_img, post_crops, 'post'),
		):
			cnt = store.contours[row][f_idx] if store.valid[row, f_idx] else None
			ctr = tuple(store.centers[row, f_idx]) if store.valid[row, f_idx] else (0, 0)
			crop = _crop_rgb(img, cnt, ctr)
			name = f'{tag}_id_{tid}.png'
			cv2.imwrite(os.path.join(sample_dir, name), crop)
			crops_dict[str(tid)] = name
			features[feat_key][str(tid)] = compute_marker_features(crop)

	meta_out = {
		'schema_version': SCHEMA_VERSION,
		'marker_id': marker.marker_id,
		'frame': marker.frame,
		'pre_frame': pre_f,
		'post_frame': post_f,
		'animal_kind': marker.animal_kind,
		'involved_ids': marker.involved_ids,
		'mapping': {str(k): int(v) for k, v in marker.mapping.items()},
		'pre_crops': pre_crops,
		'post_crops': post_crops,
		'marker_features': features,
		'sample_dir': sample_rel.replace('\\', '/'),
	}
	with open(os.path.join(sample_dir, 'meta.json'), 'w', encoding='utf-8') as f:
		json.dump(meta_out, f, indent=2)

	# pairs label for matching
	pairs_dir = os.path.join(out_dir, 'pairs')
	os.makedirs(pairs_dir, exist_ok=True)
	perm = [int(marker.mapping[int(i)]) for i in marker.involved_ids]
	pair = {
		'schema_version': SCHEMA_VERSION,
		'event_id': marker.marker_id,
		'animal_kind': marker.animal_kind,
		'involved_ids': marker.involved_ids,
		'pre_crops': {str(i): os.path.join(sample_rel.replace('\\', '/'), f'pre_id_{i}.png') for i in marker.involved_ids},
		'post_crops': {str(i): os.path.join(sample_rel.replace('\\', '/'), f'post_id_{i}.png') for i in marker.involved_ids},
		'label_permutation': perm,
		'label_source': 'human',
		'status': 'labeled',
		'mapping_convention': (
			'label_permutation[j] = pre_identity continued by post crop of involved_ids[j]'
		),
	}
	with open(os.path.join(pairs_dir, f'{marker.marker_id}.json'), 'w', encoding='utf-8') as f:
		json.dump(pair, f, indent=2)
	return meta_out
