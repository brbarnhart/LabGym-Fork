"""Headless: detect/track + categorize + export LabGym analysis products.

Uses AnalyzeAnimalDetector (detector path). Optionally applies ID remaps from
an existing id_review package after craft_data so Review IDs corrections stick.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

ProgressCb = Optional[Callable[[str], None]]

# Default BGR-ish ID palette (LabGym-style)
_DEFAULT_ID_COLORS = [
    (0, 255, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 255, 0),
    (255, 0, 0),
]


@dataclass
class ProcessVideoConfig:
    video_path: str
    detector_path: str
    categorizer_path: str
    results_root: str
    animal_kinds: List[str] = field(default_factory=list)
    animal_number: Dict[str, int] = field(default_factory=dict)
    # If set, apply remaps from this id_review folder after tracking
    id_review_dir: str = ""
    framewidth: Optional[int] = None
    t: float = 0.0
    duration: float = 0.0  # 0 = full video
    detector_batch: int = 1
    uncertain: float = 0.0
    min_length: Optional[int] = None
    background_free: bool = True
    black_background: bool = True
    color_costar: bool = False
    social_distance: float = 0.0
    show_legend: bool = True
    normalize_distance: bool = True
    parameter_to_analyze: Optional[List[str]] = None
    # Override categorizer meta if needed (usually auto from model_parameters.txt)
    behavior_mode: Optional[int] = None
    length: Optional[int] = None


@dataclass
class ProcessVideoResult:
    video_path: str
    results_path: str
    ok: bool = True
    error: str = ""
    log: List[str] = field(default_factory=list)
    behaviors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_categorizer_metadata(categorizer_path: str | Path) -> Dict[str, Any]:
    """Parse LabGym categorizer model_parameters.txt into a plain dict."""
    path = Path(categorizer_path) / "model_parameters.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Categorizer parameters not found: {path}")
    parameters = pd.read_csv(path)
    out: Dict[str, Any] = {}
    # classnames may be a single cell with list repr or multiple columns
    if "classnames" in parameters.columns:
        names = list(parameters["classnames"].dropna())
        # sometimes stored as one stringified list
        if len(names) == 1 and isinstance(names[0], str) and names[0].startswith("["):
            try:
                import ast

                names = list(ast.literal_eval(names[0]))
            except Exception:
                pass
        out["classnames"] = [str(n) for n in names if str(n) not in ("nan", "None")]
    for key in (
        "dim_conv",
        "dim_tconv",
        "channel",
        "time_step",
        "network",
        "inner_code",
        "std",
        "background_free",
        "black_background",
        "behavior_kind",
        "social_distance",
        "color_code",
        "level_tconv",
        "level_conv",
    ):
        if key in parameters.columns:
            try:
                out[key] = parameters[key].iloc[0]
            except Exception:
                pass
    return out


def _behavior_colors(classnames: Sequence[str]) -> Dict[str, List[str]]:
    """Behavior display colors: {name: ['#ffffff', '#hex']} (LabGym annotate legend)."""
    palette = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
        "#f032e6",
        "#bcf60c",
        "#fabebe",
        "#008080",
        "#e6beff",
        "#9a6324",
        "#fffac8",
        "#800000",
        "#aaffc3",
    ]
    out: Dict[str, List[str]] = {}
    for i, name in enumerate(classnames):
        out[str(name)] = ["#ffffff", palette[i % len(palette)]]
    return out


def _apply_existing_id_review(analyzer, id_review_dir: str, prog: Callable[[str], None]) -> int:
    """Apply switches/decisions from a prior Review IDs package to the analyzer."""
    from LabGym.id_review.apply import apply_decisions_to_analyzer, load_decisions
    from LabGym.id_review.dataset import load_switches, switches_to_decisions

    review = Path(id_review_dir)
    if not review.is_dir():
        return 0
    markers = load_switches(str(review))
    if markers:
        decisions = switches_to_decisions(markers)
    else:
        decisions = load_decisions(str(review / "decisions.jsonl"))
    if not decisions:
        prog("No ID remaps found in id_review package.")
        return 0
    applied = apply_decisions_to_analyzer(analyzer, decisions)
    prog(f"Applied {len(applied)} ID remap decision(s) from {review}")
    return len(applied)


def process_video(
    config: ProcessVideoConfig,
    *,
    progress: ProgressCb = None,
) -> ProcessVideoResult:
    """Run detector tracking + categorizer + export for one video."""
    log: List[str] = []

    def _prog(msg: str) -> None:
        log.append(msg)
        if progress:
            progress(msg)

    video = Path(config.video_path)
    if not video.is_file():
        return ProcessVideoResult(
            video_path=str(video), results_path="", ok=False, error=f"Video not found: {video}", log=log
        )
    if not Path(config.detector_path).is_dir():
        return ProcessVideoResult(
            video_path=str(video),
            results_path="",
            ok=False,
            error=f"Detector not found: {config.detector_path}",
            log=log,
        )
    if not Path(config.categorizer_path).is_dir():
        return ProcessVideoResult(
            video_path=str(video),
            results_path="",
            ok=False,
            error=f"Categorizer not found: {config.categorizer_path}",
            log=log,
        )

    try:
        from LabGym.analyzebehavior_dt import AnalyzeAnimalDetector
        from LabGym.detection.batch_detect import load_detector_animal_kinds
    except Exception as exc:
        return ProcessVideoResult(
            video_path=str(video),
            results_path="",
            ok=False,
            error=f"Import failed: {exc}",
            log=log,
        )

    try:
        meta = load_categorizer_metadata(config.categorizer_path)
        classnames = meta.get("classnames") or []
        if not classnames:
            return ProcessVideoResult(
                video_path=str(video),
                results_path="",
                ok=False,
                error="Categorizer has no classnames in model_parameters.txt",
                log=log,
            )
        names_and_colors = _behavior_colors(classnames)

        kinds = list(config.animal_kinds) or load_detector_animal_kinds(config.detector_path)
        numbers = dict(config.animal_number)
        if not numbers:
            numbers = {k: 1 for k in kinds}
        elif len(numbers) == 1 and not all(k in numbers for k in kinds):
            n = max(1, int(next(iter(numbers.values()))))
            numbers = {k: n for k in kinds}

        dim_conv = int(meta.get("dim_conv", 32) or 32)
        dim_tconv = int(meta.get("dim_tconv", 32) or 32)
        channel = int(meta.get("channel", 1) or 1)
        length = int(config.length if config.length is not None else meta.get("time_step", 15) or 15)
        if length < 3:
            length = 3
        network = int(meta.get("network", 2) or 2)
        animation_analyzer = network == 2
        include_bodyparts = int(meta.get("inner_code", 1) or 1) == 0
        std = int(meta.get("std", 0) or 0)
        background_free = int(meta.get("background_free", 0) or 0) == 0
        if config.background_free is False:
            background_free = False
        black_background = int(meta.get("black_background", 0) or 0) != 1
        if not config.black_background:
            black_background = False
        behavior_mode = int(
            config.behavior_mode
            if config.behavior_mode is not None
            else meta.get("behavior_kind", 0) or 0
        )
        social_distance = float(
            config.social_distance
            if config.social_distance
            else meta.get("social_distance", 0) or 0
        )
        color_costar = int(meta.get("color_code", 1) or 1) == 0 or bool(config.color_costar)

        results_root = Path(config.results_root)
        results_root.mkdir(parents=True, exist_ok=True)

        _prog(f"Preparing analysis for {video.name}…")
        aad = None
        try:
            aad = AnalyzeAnimalDetector()
            aad.prepare_analysis(
                str(Path(config.detector_path).resolve()),
                str(video.resolve()),
                str(results_root.resolve()),
                numbers,
                kinds,
                behavior_mode,
                names_and_colors=names_and_colors,
                framewidth=config.framewidth,
                dim_tconv=dim_tconv,
                dim_conv=dim_conv,
                channel=channel,
                include_bodyparts=include_bodyparts,
                std=std,
                categorize_behavior=True,
                animation_analyzer=animation_analyzer,
                t=float(config.t),
                duration=float(config.duration),
                length=length,
                social_distance=social_distance,
            )
            results_path = aad.results_path
            # JobProgress (and similar) expose .frame(current, total); plain callables do not.
            _frame = getattr(progress, "frame", None) if progress is not None else None
            frame_progress = _frame if callable(_frame) else None

            _prog("Detect + track…")
            if behavior_mode == 1:
                aad.acquire_information_interact_basic(
                    batch_size=int(config.detector_batch),
                    background_free=background_free,
                    black_background=black_background,
                    frame_progress=frame_progress,
                    status_progress=_prog,
                )
            else:
                aad.acquire_information(
                    batch_size=int(config.detector_batch),
                    background_free=background_free,
                    black_background=black_background,
                    color_costar=color_costar,
                    frame_progress=frame_progress,
                    status_progress=_prog,
                )
            if behavior_mode != 1:
                _prog("Crafting track data…")
                aad.craft_data()
                if config.id_review_dir:
                    _apply_existing_id_review(aad, config.id_review_dir, _prog)

            _prog("Categorizing behaviors…")
            aad.categorize_behaviors(
                str(Path(config.categorizer_path).resolve()),
                uncertain=float(config.uncertain),
                min_length=config.min_length,
            )

            animal_to_include = list(kinds)
            id_colors = list(_DEFAULT_ID_COLORS)
            while len(id_colors) < max(1, len(animal_to_include)):
                id_colors.extend(_DEFAULT_ID_COLORS)
            behavior_to_include = list(classnames)

            _prog("Annotating video…")
            aad.annotate_video(
                animal_to_include,
                id_colors,
                behavior_to_include,
                show_legend=bool(config.show_legend),
            )

            params = config.parameter_to_analyze
            if params is None:
                params = [
                    "count",
                    "duration",
                    "latency",
                    "distance",
                    "speed",
                    "intensity_area",
                ]
            _prog("Exporting results…")
            aad.export_results(
                normalize_distance=bool(config.normalize_distance),
                parameter_to_analyze=params,
            )

            manifest = {
                "video_path": str(video.resolve()),
                "detector_path": str(Path(config.detector_path).resolve()),
                "categorizer_path": str(Path(config.categorizer_path).resolve()),
                "results_path": results_path,
                "id_review_dir": config.id_review_dir or "",
                "behaviors": list(classnames),
                "animal_kinds": kinds,
                "behavior_mode": behavior_mode,
            }
            Path(results_path).joinpath("process_video_job.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            _prog(f"Done: {results_path}")
            return ProcessVideoResult(
                video_path=str(video.resolve()),
                results_path=str(results_path),
                ok=True,
                log=log,
                behaviors=list(classnames),
            )
        finally:
            try:
                from LabGym.gpu_utils import release_analyzer_gpu

                release_analyzer_gpu(aad)
                _prog("Released detector GPU memory.")
            except Exception as exc:
                try:
                    _prog(f"Warning: GPU release incomplete: {exc}")
                except Exception:
                    pass
    except Exception as exc:
        _prog(f"ERROR: {exc}")
        return ProcessVideoResult(
            video_path=str(video),
            results_path="",
            ok=False,
            error=str(exc),
            log=log,
        )
