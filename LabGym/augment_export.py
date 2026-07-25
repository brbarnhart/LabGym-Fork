"""Lightweight training-example augmentation (no TensorFlow).

Used by :mod:`LabGym.categorizer` and by process-pool workers so Windows
``spawn`` does not re-import Keras/TF in every child.
"""

from __future__ import annotations

from collections import deque
import itertools
import os
import random

import cv2
import numpy as np
from scipy import ndimage
from skimage import transform
from skimage.transform import AffineTransform


def img_to_array(img):
	"""Minimal Keras-compatible float32 HWC array (avoids importing keras)."""
	x = np.asarray(img, dtype="float32")
	if x.ndim == 2:
		x = np.expand_dims(x, axis=-1)
	return x


def resolve_aug_methods(aug_methods):
	"""Expand user-facing aug method names into internal method code strings."""
	if not aug_methods:
		return ["orig"]

	remove = []
	all_methods = [
		"orig",
		"rot1",
		"rot2",
		"rot3",
		"rot4",
		"rot5",
		"rot6",
		"shrp",
		"shrn",
		"sclh",
		"sclw",
		"del1",
		"del2",
	]
	options = ["rot7", "flph", "flpv", "brih", "bril", "shrr", "sclr", "delr"]
	for r in range(1, len(options) + 1):
		all_methods.extend(["".join(c) for c in itertools.combinations(options, r)])

	for i in all_methods:
		if "random rotation" not in aug_methods:
			if "rot" in i:
				remove.append(i)
		if "horizontal flipping" not in aug_methods:
			if "flph" in i:
				remove.append(i)
		if "vertical flipping" not in aug_methods:
			if "flpv" in i:
				remove.append(i)
		if "random brightening" not in aug_methods:
			if "brih" in i:
				remove.append(i)
		if "random dimming" not in aug_methods:
			if "bril" in i:
				remove.append(i)
		if "random shearing" not in aug_methods:
			if "shr" in i:
				remove.append(i)
		if "random rescaling" not in aug_methods:
			if "scl" in i:
				remove.append(i)
		if "random deletion" not in aug_methods:
			if "del" in i:
				remove.append(i)

	return list(set(all_methods) - set(remove))


def default_aug_workers(export: bool = True) -> int:
	"""Recommended worker count for export augmentation."""
	cpu = os.cpu_count() or 1
	if export:
		return max(1, min(8, cpu - 1 if cpu > 1 else 1))
	return 1


def init_augment_worker() -> None:
	"""Process-pool initializer: avoid OpenCV oversubscription."""
	try:
		cv2.setNumThreads(1)
	except Exception:
		pass


def augment_export_task(payload: dict):
	"""Picklable process-pool entry point (export path only).

	``payload`` keys: animation_path, methods, dim_tconv, dim_conv, channel,
	time_step, background_free, black_background, behavior_mode, out_path, seed.
	"""
	return augment_one_example(
		payload["animation_path"],
		payload["methods"],
		dim_tconv=payload.get("dim_tconv", 0),
		dim_conv=payload.get("dim_conv", 64),
		channel=payload.get("channel", 1),
		time_step=payload.get("time_step", 15),
		background_free=payload.get("background_free", True),
		black_background=payload.get("black_background", True),
		behavior_mode=payload.get("behavior_mode", 0),
		out_path=payload.get("out_path"),
		seed=payload.get("seed"),
	)


def augment_one_example(
	animation_path,
	methods,
	dim_tconv=0,
	dim_conv=64,
	channel=1,
	time_step=15,
	background_free=True,
	black_background=True,
	behavior_mode=0,
	out_path=None,
	seed=None,
):
	"""
	Augment a single source example with all method codes in ``methods``.

	Picklable top-level function for process-pool use.

	Returns
	-------
	tuple
		``(animations, pattern_images, labels, amount, warnings)`` where the
		first three are lists (or None when exporting) and ``warnings`` is a
		list of log strings.
	"""
	if seed is not None:
		random.seed(seed)
		np.random.seed(int(seed) % (2**32 - 1))

	methods = list(methods)
	random.shuffle(methods)

	name = os.path.splitext(os.path.basename(animation_path))[0].split("_")[0]
	label = os.path.splitext(animation_path)[0].split("_")[-1]
	path_to_pattern_image = os.path.splitext(animation_path)[0] + ".jpg"

	animations_out = []
	patterns_out = []
	labels_out = []
	warnings = []
	amount = 0

	for m in methods:

		if "rot1" in m:
			angle = np.random.uniform(5, 45)
		elif "rot2" in m:
			angle = np.random.uniform(45, 85)
		elif "rot3" in m:
			angle = 90.0
		elif "rot4" in m:
			angle = np.random.uniform(95, 135)
		elif "rot5" in m:
			angle = np.random.uniform(135, 175)
		elif "rot6" in m:
			angle = 180.0
		elif "rot7" in m:
			angle = np.random.uniform(5, 175)
		else:
			angle = None

		if "flphflpv" in m:
			code = -1
		elif "flph" in m:
			code = 1
		elif "flpv" in m:
			code = 0
		else:
			code = None

		if "brihbril" in m:
			beta = np.random.uniform(-50, 50)
		elif "brih" in m:
			beta = np.random.uniform(10, 50)
		elif "bril" in m:
			beta = np.random.uniform(-50, -10)
		else:
			beta = None

		if "shrp" in m:
			shear = np.random.uniform(0.15, 0.21)
		elif "shrn" in m:
			shear = np.random.uniform(-0.21, -0.15)
		elif "shrr" in m:
			shear = np.random.uniform(-0.21, 0.21)
		else:
			shear = None

		if "sclh" in m:
			width = 0
			scale = np.random.uniform(0.6, 0.9)
		elif "sclw" in m:
			width = 1
			scale = np.random.uniform(0.6, 0.9)
		elif "sclr" in m:
			width = random.randint(0, 1)
			scale = np.random.uniform(0.6, 0.9)
		else:
			scale = None

		if "del1" in m:
			if time_step >= 30:
				idx1 = random.randint(0, round(time_step / 3))
				idx2 = random.randint(round(time_step / 3) + 1, round(time_step * 2 / 3))
				to_delete = [idx1, idx2]
			else:
				to_delete = [random.randint(0, round(time_step / 3))]
		elif "del2" in m:
			to_delete = [random.randint(0, round(time_step / 2) + 1)]
		elif "delr" in m:
			to_delete = [random.randint(0, time_step - 1)]
		else:
			to_delete = None

		if dim_tconv != 0:

			capture = cv2.VideoCapture(animation_path)
			if out_path is not None:
				fps = round(capture.get(cv2.CAP_PROP_FPS))
				w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
				h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
				writer = cv2.VideoWriter(
					os.path.join(out_path, name + "_" + m + "_" + label + ".avi"),
					cv2.VideoWriter_fourcc(*"MJPG"),
					int(fps),
					(w, h),
					True,
				)
			animation = deque()
			frames = deque(maxlen=time_step)
			original_frame = None
			n = 0

			while True:
				retval, frame = capture.read()
				if original_frame is None:
					original_frame = frame
				if frame is None:
					break
				frames.append(frame)

			capture.release()

			frames_length = len(frames)
			if frames_length < time_step:
				for _diff in range(time_step - frames_length):
					frames.append(np.zeros_like(original_frame))
				warnings.append(
					"Inconsistent duration of animation detected at: "
					+ str(animation_path)
					+ "."
				)
				warnings.append(
					"Zero padding has been used, which may decrease the training accuracy."
				)

			for frame in frames:

				if to_delete is not None and n in to_delete:

					if black_background is False:
						frame = np.uint8(np.zeros_like(original_frame) + 255)
					else:
						frame = np.zeros_like(original_frame)

				else:

					if code is not None:
						frame = cv2.flip(frame, code)

					if beta is not None:
						frame = frame.astype("float")
						if background_free:
							if black_background:
								frame[frame > 30] += beta
							else:
								frame[frame < 225] += beta
						else:
							frame += beta
						frame = np.uint8(np.clip(frame, 0, 255))

					if angle is not None:
						frame = ndimage.rotate(frame, angle, reshape=False, prefilter=False)

					if shear is not None:
						tf = AffineTransform(shear=shear)
						frame = transform.warp(
							frame, tf, order=1, preserve_range=True, mode="constant"
						)

					if scale is not None:
						frame_black = np.zeros_like(frame)
						if black_background is False:
							frame_black = np.uint8(frame_black + 255)
						if width == 0:
							frame_scl = cv2.resize(
								frame,
								(frame.shape[1], int(frame.shape[0] * scale)),
								interpolation=cv2.INTER_AREA,
							)
						else:
							frame_scl = cv2.resize(
								frame,
								(int(frame.shape[1] * scale), frame.shape[0]),
								interpolation=cv2.INTER_AREA,
							)
						frame_scl = img_to_array(frame_scl)
						x = (frame_black.shape[1] - frame_scl.shape[1]) // 2
						y = (frame_black.shape[0] - frame_scl.shape[0]) // 2
						frame_black[
							y : y + frame_scl.shape[0], x : x + frame_scl.shape[1]
						] = frame_scl
						frame = frame_black

				if out_path is None:
					if channel == 1:
						frame = cv2.cvtColor(np.uint8(frame), cv2.COLOR_BGR2GRAY)
					frame = cv2.resize(
						frame, (dim_tconv, dim_tconv), interpolation=cv2.INTER_AREA
					)
					frame = img_to_array(frame)
					animation.append(frame)
				else:
					writer.write(np.uint8(frame))

				n += 1

			if out_path is None:
				animations_out.append(np.array(animation))
			else:
				writer.release()

		pattern_image = cv2.imread(path_to_pattern_image)

		if code is not None:
			pattern_image = cv2.flip(pattern_image, code)

		if behavior_mode == 3:
			if beta is not None:
				pattern_image = pattern_image.astype("float")
				if background_free:
					if black_background:
						pattern_image[pattern_image > 30] += beta
					else:
						pattern_image[pattern_image < 225] += beta
				else:
					pattern_image += beta
				pattern_image = np.uint8(np.clip(pattern_image, 0, 255))

		if angle is not None:
			pattern_image = ndimage.rotate(
				pattern_image, angle, reshape=False, prefilter=False
			)

		if shear is not None:
			tf = AffineTransform(shear=shear)
			pattern_image = transform.warp(
				pattern_image, tf, order=1, preserve_range=True, mode="constant"
			)

		if scale is not None:
			pattern_image_black = np.zeros_like(pattern_image)
			if width == 0:
				pattern_image_scl = cv2.resize(
					pattern_image,
					(pattern_image.shape[1], int(pattern_image.shape[0] * scale)),
					interpolation=cv2.INTER_AREA,
				)
			else:
				pattern_image_scl = cv2.resize(
					pattern_image,
					(int(pattern_image.shape[1] * scale), pattern_image.shape[0]),
					interpolation=cv2.INTER_AREA,
				)
			x = (pattern_image_black.shape[1] - pattern_image_scl.shape[1]) // 2
			y = (pattern_image_black.shape[0] - pattern_image_scl.shape[0]) // 2
			pattern_image_black[
				y : y + pattern_image_scl.shape[0],
				x : x + pattern_image_scl.shape[1],
				:,
			] = pattern_image_scl
			pattern_image = pattern_image_black

		if out_path is None:

			if behavior_mode == 3:
				if channel == 1:
					pattern_image = cv2.cvtColor(
						np.uint8(pattern_image), cv2.COLOR_BGR2GRAY
					)

			pattern_image = cv2.resize(
				pattern_image, (dim_conv, dim_conv), interpolation=cv2.INTER_AREA
			)
			patterns_out.append(img_to_array(pattern_image))
			labels_out.append(label)
			amount += 1

		else:

			cv2.imwrite(
				os.path.join(out_path, name + "_" + m + "_" + label + ".jpg"),
				np.uint8(pattern_image),
			)
			amount += 1

	if out_path is not None:
		return None, None, None, amount, warnings
	return animations_out, patterns_out, labels_out, amount, warnings
