"""Headless detect + track → identity package (id_review tracklets + events).

Does not open the wx ID review UI. Users review IDs later in the PySide
Detector → Review IDs tab.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

ProgressCb = Optional[Callable[[str], None]]


@dataclass
class DetectTrackConfig:
    """Parameters for one detect+track run (batch or single video)."""

    video_path: str
    detector_path: str
    results_root: str  # parent folder; LabGym creates <stem>/ under this
    animal_kinds: List[str] = field(default_factory=list)
    # counts per kind; if empty, 1 of each kind
    animal_number: Dict[str, int] = field(default_factory=dict)
    behavior_mode: int = 0  # 0 non-interactive tracking; 2 interactive advanced tracking
    framewidth: Optional[int] = None
    t: float = 0.0  # start time (s)
    duration: float = 0.0  # 0 = entire video
    length: int = 15  # history length used by analyzer craft
    detector_batch: int = 1
    background_free: bool = True
    black_background: bool = True
    color_costar: bool = False
    social_distance: float = 0.0
    # Contact risk pack for later ID review
    export_id_review: bool = True
    extract_contact_samples: bool = False
    contact_distance_factor: float = 1.0
    min_contact_frames: int = 3
    gap_bridge_frames: int = 2
    write_default_subjects: bool = True

    def resolved_animal_kinds(self) -> List[str]:
        if self.animal_kinds:
            return list(self.animal_kinds)
        return load_detector_animal_kinds(self.detector_path)

    def resolved_animal_number(self) -> Dict[str, int]:
        kinds = self.resolved_animal_kinds()
        out: Dict[str, int] = {}
        for k in kinds:
            if self.animal_number and k in self.animal_number:
                out[k] = max(1, int(self.animal_number[k]))
            elif self.animal_number and len(self.animal_number) == 1:
                # single value applied to all kinds
                out[k] = max(1, int(next(iter(self.animal_number.values()))))
            else:
                out[k] = 1
        return out


@dataclass
class DetectTrackResult:
    video_path: str
    results_path: str  # <results_root>/<stem>
    id_review_dir: str  # <results_path>/id_review
    n_events: int = 0
    animal_kinds: List[str] = field(default_factory=list)
    log: List[str] = field(default_factory=list)
    ok: bool = True
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_detector_animal_kinds(detector_path: str | Path) -> List[str]:
    """Read animal category names from a LabGym detector folder."""
    path = Path(detector_path) / "model_parameters.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Detector parameters not found: {path}")
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    names = data.get("animal_names") or []
    if not names:
        mapping = data.get("animal_mapping") or {}
        names = list(mapping.values()) if isinstance(mapping, dict) else []
    if not names:
        raise ValueError(f"No animal_names in {path}")
    return [str(n) for n in names]


def list_detectors(models_root: str | Path) -> List[Path]:
    """Find detector folders (contain model_parameters.txt) under a root."""
    root = Path(models_root)
    if not root.is_dir():
        return []
    found: List[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if "model_parameters.txt" in filenames:
            found.append(Path(dirpath))
    # also one level of LabGym bundled detectors if models_root is empty
    return sorted(found)


def detect_and_track_video(
    config: DetectTrackConfig,
    *,
    progress: ProgressCb = None,
) -> DetectTrackResult:
    """Run detector tracking on one video and write an id_review identity pack.

    Heavy imports are deferred so unit tests can mock this function without
    loading Detectron2.
    """
    log: List[str] = []

    def _prog(msg: str) -> None:
        log.append(msg)
        if progress:
            progress(msg)

    video = Path(config.video_path)
    if not video.is_file():
        return DetectTrackResult(
            video_path=str(video),
            results_path="",
            id_review_dir="",
            ok=False,
            error=f"Video not found: {video}",
            log=log,
        )
    if not Path(config.detector_path).is_dir():
        return DetectTrackResult(
            video_path=str(video),
            results_path="",
            id_review_dir="",
            ok=False,
            error=f"Detector folder not found: {config.detector_path}",
            log=log,
        )

    try:
        kinds = config.resolved_animal_kinds()
        numbers = config.resolved_animal_number()
    except Exception as exc:
        return DetectTrackResult(
            video_path=str(video),
            results_path="",
            id_review_dir="",
            ok=False,
            error=str(exc),
            log=log,
        )

    results_root = Path(config.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    try:
        from LabGym.analyzebehavior_dt import AnalyzeAnimalDetector
        from LabGym.id_review.dataset import export_review_pack, review_dir
        from LabGym.id_review.types import ContactDetectorConfig
        from LabGym.identity.package import (
            save_subjects,
            subjects_from_track_ids,
        )
        from LabGym.id_review.tracklets import load_tracklets
        from LabGym.annotator.core.tracklets_bridge import discover_tracklet_kinds
        from LabGym.id_review.apply import write_tracklets_identity_status
    except Exception as exc:
        return DetectTrackResult(
            video_path=str(video),
            results_path="",
            id_review_dir="",
            ok=False,
            error=f"Failed to import analysis stack: {exc}",
            log=log,
        )

    _prog(f"Preparing analysis for {video.name}…")
    try:
        aad = AnalyzeAnimalDetector()
        aad.prepare_analysis(
            str(Path(config.detector_path).resolve()),
            str(video.resolve()),
            str(results_root.resolve()),
            numbers,
            kinds,
            int(config.behavior_mode),
            names_and_colors=None,
            framewidth=config.framewidth,
            categorize_behavior=False,
            animation_analyzer=False,
            t=float(config.t),
            duration=float(config.duration),
            length=int(config.length),
            social_distance=float(config.social_distance),
        )
        results_path = aad.results_path
        _prog("Running detector tracking (acquire_information)…")
        if int(config.behavior_mode) == 1:
            aad.acquire_information_interact_basic(
                batch_size=int(config.detector_batch),
                background_free=bool(config.background_free),
                black_background=bool(config.black_background),
            )
        else:
            aad.acquire_information(
                batch_size=int(config.detector_batch),
                background_free=bool(config.background_free),
                black_background=bool(config.black_background),
                color_costar=bool(config.color_costar),
            )
        if int(config.behavior_mode) != 1:
            _prog("Crafting track data…")
            aad.craft_data()

        id_review_path = ""
        n_events = 0
        if config.export_id_review and int(config.behavior_mode) != 1:
            _prog("Exporting id_review identity package…")
            cfg = ContactDetectorConfig(
                contact_distance_factor=float(config.contact_distance_factor),
                min_contact_frames=int(config.min_contact_frames),
                gap_bridge_frames=int(config.gap_bridge_frames),
            )
            out_dir, events = export_review_pack(
                aad,
                config=cfg,
                extract_samples=bool(config.extract_contact_samples),
            )
            id_review_path = out_dir
            n_events = len(events)
            write_tracklets_identity_status(
                out_dir,
                corrected=False,
                n_decisions=0,
                source="batch_detect",
            )
            if config.write_default_subjects:
                try:
                    kind_ids: Dict[str, List[int]] = {}
                    for kind in discover_tracklet_kinds(out_dir):
                        store = load_tracklets(out_dir, kind)
                        kind_ids[kind] = list(store.ids)
                    if kind_ids:
                        save_subjects(out_dir, subjects_from_track_ids(kind_ids))
                except Exception as exc:
                    _prog(f"Warning: could not write default subjects.json: {exc}")
            # Job manifest
            manifest = {
                "video_path": str(video.resolve()),
                "detector_path": str(Path(config.detector_path).resolve()),
                "results_path": results_path,
                "id_review_dir": id_review_path,
                "animal_kinds": kinds,
                "animal_number": numbers,
                "behavior_mode": int(config.behavior_mode),
                "n_contact_events": n_events,
            }
            Path(out_dir).joinpath("detect_track_job.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        else:
            id_review_path = review_dir(results_path)
            os.makedirs(id_review_path, exist_ok=True)

        _prog(f"Done: {id_review_path or results_path}")
        return DetectTrackResult(
            video_path=str(video.resolve()),
            results_path=str(results_path),
            id_review_dir=str(id_review_path),
            n_events=n_events,
            animal_kinds=kinds,
            log=log,
            ok=True,
        )
    except Exception as exc:
        _prog(f"ERROR: {exc}")
        return DetectTrackResult(
            video_path=str(video),
            results_path="",
            id_review_dir="",
            ok=False,
            error=str(exc),
            log=log,
            animal_kinds=kinds if "kinds" in dir() else [],
        )
