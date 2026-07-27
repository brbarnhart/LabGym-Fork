"""Backend helpers for classic dense generate + sort (no Qt)."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence


ProgressCb = Callable[[str], None]


@dataclass
class DenseGenerateConfig:
    path_to_videos: List[str]
    result_path: str
    behavior_mode: int = 0  # 0 non-int, 1 basic, 2 advanced, 3 static images
    use_detector: bool = True
    path_to_detector: Optional[str] = None
    animal_kinds: List[str] = field(default_factory=list)
    animal_number: Optional[int | Dict[str, int]] = None
    framewidth: Optional[int] = None
    t: float = 0.0
    duration: float = 0.0
    length: int = 15
    skip_redundant: int = 1
    social_distance: float = 0.0
    include_bodyparts: bool = False
    std: int = 0
    background_free: bool = True
    black_background: bool = True
    color_costar: bool = False
    detection_threshold: float = 0.0
    # background subtraction path
    background_path: Optional[str] = None
    stable_illumination: bool = True
    animal_vs_bg: int = 0
    delta: int = 10000
    autofind_t: bool = False
    decode_t: bool = False
    decode_animalnumber: bool = False
    decode_extraction: bool = False
    ex_start: int = 0
    ex_end: Optional[int] = None


def run_dense_generate(
    cfg: DenseGenerateConfig,
    progress: Optional[ProgressCb] = None,
) -> None:
    """Run classic LabGym unsorted example generation (dense Tools path)."""
    log = progress or (lambda _m: None)

    if cfg.behavior_mode >= 3:
        if not cfg.path_to_detector:
            raise RuntimeError("Static images mode requires a Detector.")
        from LabGym.analyzebehavior_dt import AnalyzeAnimalDetector

        log("Generating static-image examples…")
        aad = AnalyzeAnimalDetector()
        aad.analyze_images_individuals(
            cfg.path_to_detector,
            cfg.path_to_videos,
            cfg.result_path,
            cfg.animal_kinds,
            generate=True,
            imagewidth=cfg.framewidth,
            detection_threshold=cfg.detection_threshold,
            background_free=cfg.background_free,
            black_background=cfg.black_background,
        )
        log("Static-image generation completed.")
        return

    from LabGym.analyzebehavior import AnalyzeAnimal
    from LabGym.analyzebehavior_dt import AnalyzeAnimalDetector

    os.makedirs(cfg.result_path, exist_ok=True)
    n = len(cfg.path_to_videos)
    for vi, video in enumerate(cfg.path_to_videos, start=1):
        log(f"[{vi}/{n}] {os.path.basename(video)}")
        filename = os.path.splitext(os.path.basename(video))[0].split("_")
        animal_number = cfg.animal_number
        t = cfg.t
        ex_start = cfg.ex_start
        ex_end = cfg.ex_end

        if cfg.decode_animalnumber:
            if cfg.use_detector:
                animal_number = {}
                number = [x[1:] for x in filename if len(x) > 1 and x[0] == "n"]
                for a, animal_name in enumerate(cfg.animal_kinds):
                    animal_number[animal_name] = int(number[a])
            else:
                for x in filename:
                    if len(x) > 1 and x[0] == "n":
                        animal_number = int(x[1:])
        if cfg.decode_t:
            for x in filename:
                if len(x) > 1 and x[0] == "b":
                    t = float(x[1:])
        if cfg.decode_extraction:
            for x in filename:
                if len(x) > 2:
                    if x[:2] == "xs":
                        ex_start = int(x[2:])
                    if x[:2] == "xe":
                        ex_end = int(x[2:])

        if animal_number is None:
            if cfg.use_detector:
                animal_number = {name: 1 for name in cfg.animal_kinds}
            else:
                animal_number = 1

        if not cfg.use_detector:
            aa = AnalyzeAnimal()
            aa.prepare_analysis(
                video,
                cfg.result_path,
                animal_number,
                delta=cfg.delta,
                framewidth=cfg.framewidth,
                stable_illumination=cfg.stable_illumination,
                channel=3,
                include_bodyparts=cfg.include_bodyparts,
                std=cfg.std,
                categorize_behavior=False,
                animation_analyzer=False,
                path_background=cfg.background_path,
                autofind_t=cfg.autofind_t,
                t=t,
                duration=cfg.duration,
                ex_start=ex_start,
                ex_end=ex_end,
                length=cfg.length,
                animal_vs_bg=cfg.animal_vs_bg,
            )
            if cfg.behavior_mode == 0:
                aa.generate_data(
                    background_free=cfg.background_free,
                    black_background=cfg.black_background,
                    skip_redundant=cfg.skip_redundant,
                )
            else:
                aa.generate_data_interact_basic(
                    background_free=cfg.background_free,
                    black_background=cfg.black_background,
                    skip_redundant=cfg.skip_redundant,
                )
        else:
            if not cfg.path_to_detector:
                raise RuntimeError("Detector path required when use_detector is True.")
            aad = AnalyzeAnimalDetector()
            aad.prepare_analysis(
                cfg.path_to_detector,
                video,
                cfg.result_path,
                animal_number,
                cfg.animal_kinds,
                cfg.behavior_mode,
                framewidth=cfg.framewidth,
                channel=3,
                include_bodyparts=cfg.include_bodyparts,
                std=cfg.std,
                categorize_behavior=False,
                animation_analyzer=False,
                t=t,
                duration=cfg.duration,
                length=cfg.length,
                social_distance=cfg.social_distance,
            )
            if cfg.behavior_mode == 0:
                aad.generate_data(
                    background_free=cfg.background_free,
                    black_background=cfg.black_background,
                    skip_redundant=cfg.skip_redundant,
                )
            elif cfg.behavior_mode == 1:
                aad.generate_data_interact_basic(
                    background_free=cfg.background_free,
                    black_background=cfg.black_background,
                    skip_redundant=cfg.skip_redundant,
                )
            else:
                aad.generate_data_interact_advance(
                    background_free=cfg.background_free,
                    black_background=cfg.black_background,
                    skip_redundant=cfg.skip_redundant,
                    color_costar=cfg.color_costar,
                )
    log("Dense example generation completed.")


def run_manual_sort(
    input_path: str,
    result_path: str,
    keys_behaviors: Dict[str, str],
    progress: Optional[ProgressCb] = None,
) -> None:
    """Interactive OpenCV sort UI (same key bindings as classic LabGym)."""
    import cv2
    import numpy as np

    log = progress or (lambda _m: None)
    reserved = {"o", "p", "q", "u"}
    for key in keys_behaviors:
        if len(key) != 1:
            raise ValueError(f"Key must be a single character: {key!r}")
        if key.lower() in reserved:
            raise ValueError(f"Key {key!r} is reserved (o/p/q/u).")

    keys_behaviorpaths = {}
    for key, name in keys_behaviors.items():
        path = os.path.join(result_path, name)
        os.makedirs(path, exist_ok=True)
        keys_behaviorpaths[key] = path

    check_animations = [i for i in os.listdir(input_path) if i.endswith(".avi")]
    only_image = False
    if not check_animations:
        check_images = [i for i in os.listdir(input_path) if i.endswith(".jpg")]
        if not check_images:
            raise RuntimeError("No examples found in input folder.")
        only_image = True

    log("Opening OpenCV sorting window (o prev, p next, q quit, u undo)…")
    cv2.namedWindow("Sorting Behavior Examples", cv2.WINDOW_NORMAL)
    actions: List[List[str]] = []
    index = 0
    stop = False
    moved = False
    example_name = ""
    shortcutkey = ""

    while not stop:
        if moved:
            moved = False
            if not only_image:
                shutil.move(
                    os.path.join(input_path, example_name + ".avi"),
                    os.path.join(keys_behaviorpaths[shortcutkey], example_name + ".avi"),
                )
            shutil.move(
                os.path.join(input_path, example_name + ".jpg"),
                os.path.join(keys_behaviorpaths[shortcutkey], example_name + ".jpg"),
            )

        pattern_images = [i for i in os.listdir(input_path) if i.endswith(".jpg")]
        pattern_images = sorted(
            pattern_images,
            key=lambda name: int(name.split("_len")[0].split("_")[-1]),
        )

        if pattern_images and index < len(pattern_images):
            example_name = pattern_images[index].split(".jpg")[0]
            pattern_image = cv2.resize(
                cv2.imread(os.path.join(input_path, example_name + ".jpg")),
                (600, 600),
                interpolation=cv2.INTER_AREA,
            )
            animation = None
            fps = 30.0
            if not only_image:
                frame_count = example_name.split("_len")[0].split("_")[-1]
                animation = cv2.VideoCapture(
                    os.path.join(input_path, example_name + ".avi")
                )
                fps = animation.get(cv2.CAP_PROP_FPS) or 30.0
            else:
                frame_count = ""

            while True:
                if not only_image and animation is not None:
                    ret, frame = animation.read()
                    if not ret:
                        animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    frame = cv2.resize(frame, (600, 600), interpolation=cv2.INTER_AREA)
                    combined = np.hstack((frame, pattern_image))
                    cv2.putText(
                        combined,
                        "frame count: " + str(frame_count),
                        (10, 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        1,
                    )
                    x_begin = 550
                else:
                    combined = pattern_image
                    x_begin = 5

                n = 1
                for i in ["o: Prev", "p: Next", "q: Quit", "u: Undo"]:
                    cv2.putText(
                        combined,
                        i,
                        (x_begin, 15 * n),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        1,
                    )
                    n += 1
                n += 1
                for i in keys_behaviors:
                    cv2.putText(
                        combined,
                        i + ": " + keys_behaviors[i],
                        (x_begin, 15 * n),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 255),
                        1,
                    )
                    n += 1

                cv2.imshow("Sorting Behavior Examples", combined)
                cv2.moveWindow("Sorting Behavior Examples", 50, 0)

                if not only_image:
                    key = cv2.waitKey(int(1000 / max(fps, 1))) & 0xFF
                else:
                    key = cv2.waitKey(1) & 0xFF

                for sk in keys_behaviorpaths:
                    if key == ord(sk):
                        example_name = pattern_images[index].split(".")[0]
                        shortcutkey = sk
                        actions.append([sk, example_name])
                        moved = True
                        break
                if moved:
                    break

                if key == ord("u"):
                    if actions:
                        last = actions.pop()
                        shortcutkey = last[0]
                        example_name = last[1]
                        if not only_image:
                            shutil.move(
                                os.path.join(
                                    keys_behaviorpaths[shortcutkey], example_name + ".avi"
                                ),
                                os.path.join(input_path, example_name + ".avi"),
                            )
                        shutil.move(
                            os.path.join(
                                keys_behaviorpaths[shortcutkey], example_name + ".jpg"
                            ),
                            os.path.join(input_path, example_name + ".jpg"),
                        )
                        break
                    continue

                if key == ord("p"):
                    index += 1
                    break
                if key == ord("o"):
                    if index >= 1:
                        index -= 1
                    break
                if key == ord("q"):
                    stop = True
                    break

            if animation is not None:
                animation.release()
        else:
            if not pattern_images:
                log("Behavior example sorting completed.")
                stop = True
            else:
                index = 0

    cv2.destroyAllWindows()


def load_detector_animal_kinds(path_to_detector: str) -> List[str]:
    """Read animal/object category names from a trained detector folder."""
    import json

    path = os.path.join(path_to_detector, "model_parameters.txt")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        data = json.loads(f.read())
    names = data.get("animal_names") or data.get("animal_kinds") or []
    return list(names)
