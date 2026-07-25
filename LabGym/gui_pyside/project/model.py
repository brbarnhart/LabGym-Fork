"""On-disk project format (*.labproj.json) for the workbench shell."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_SCHEMA_VERSION = 1
PROJECT_FILE_SUFFIX = ".labproj.json"


@dataclass
class ProjectVideo:
    """One video entry in the project manifest."""

    path: str
    enabled: bool = True
    # Optional per-video overrides (filled by later phases)
    detection_dir: str = ""
    annotations_path: str = ""
    notes: str = ""

    def resolved_path(self, root_dir: str = "") -> Path:
        p = Path(self.path)
        if p.is_absolute() or not root_dir:
            return p
        return Path(root_dir) / p

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"path": self.path, "enabled": bool(self.enabled)}
        if self.detection_dir:
            d["detection_dir"] = self.detection_dir
        if self.annotations_path:
            d["annotations_path"] = self.annotations_path
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectVideo":
        return cls(
            path=str(d.get("path") or ""),
            enabled=bool(d.get("enabled", True)),
            detection_dir=str(d.get("detection_dir") or ""),
            annotations_path=str(d.get("annotations_path") or ""),
            notes=str(d.get("notes") or ""),
        )


@dataclass
class ProjectPaths:
    detection_output_root: str = "detection"
    annotations_root: str = ""
    examples_root: str = "examples"
    models_root: str = "models"
    processed_root: str = "processed"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ProjectPaths":
        d = d or {}
        return cls(
            detection_output_root=str(d.get("detection_output_root") or "detection"),
            annotations_root=str(d.get("annotations_root") or ""),
            examples_root=str(d.get("examples_root") or "examples"),
            models_root=str(d.get("models_root") or "models"),
            processed_root=str(d.get("processed_root") or "processed"),
        )


@dataclass
class ProjectDefaults:
    behavior_mode: int = 0
    exclusive_mode: bool = True
    window_length: int = 15
    sampling: str = "dense_in_bout"
    stride: int = 0
    min_bout_frames: int = 1
    social_distance: float = 0.0
    write_soft_labels: bool = True
    background_free: bool = True
    detector_name: str = ""
    categorizer_name: str = ""
    # Currently selected video path (absolute or project-relative)
    current_video: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "ProjectDefaults":
        d = d or {}
        return cls(
            behavior_mode=int(d.get("behavior_mode") or 0),
            exclusive_mode=bool(d.get("exclusive_mode", True)),
            window_length=int(d.get("window_length") or 15),
            sampling=str(d.get("sampling") or "dense_in_bout"),
            stride=int(d.get("stride") or 0),
            min_bout_frames=int(d.get("min_bout_frames") or 1),
            social_distance=float(d.get("social_distance") or 0.0),
            write_soft_labels=bool(d.get("write_soft_labels", True)),
            background_free=bool(d.get("background_free", True)),
            detector_name=str(d.get("detector_name") or ""),
            categorizer_name=str(d.get("categorizer_name") or ""),
            current_video=str(d.get("current_video") or ""),
        )


@dataclass
class Project:
    """Experiment-level project shared across workbenches."""

    schema_version: int = PROJECT_SCHEMA_VERSION
    name: str = "Untitled"
    root_dir: str = ""
    videos: List[ProjectVideo] = field(default_factory=list)
    paths: ProjectPaths = field(default_factory=ProjectPaths)
    defaults: ProjectDefaults = field(default_factory=ProjectDefaults)
    notes: str = ""
    # Runtime only (not always persisted as absolute)
    file_path: str = ""

    # --- helpers ---

    def display_name(self) -> str:
        if self.file_path:
            return Path(self.file_path).name
        return self.name or "Untitled"

    def status_summary(self) -> str:
        n = len(self.videos)
        n_on = sum(1 for v in self.videos if v.enabled)
        root = Path(self.root_dir).name if self.root_dir else "—"
        cur = self.defaults.current_video
        cur_name = Path(cur).name if cur else "—"
        dirty = ""  # filled by controller in UI
        return (
            f"{self.display_name()}  ·  root: {root}  ·  "
            f"videos: {n_on}/{n}  ·  current: {cur_name}{dirty}"
        )

    def enabled_videos(self) -> List[ProjectVideo]:
        return [v for v in self.videos if v.enabled]

    def resolve_path(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute() or not self.root_dir:
            return p
        return Path(self.root_dir) / p

    def absolute_video_paths(self, *, enabled_only: bool = True) -> List[Path]:
        out: List[Path] = []
        for v in self.videos:
            if enabled_only and not v.enabled:
                continue
            if not v.path:
                continue
            out.append(v.resolved_path(self.root_dir))
        return out

    def add_video(self, path: str, *, enabled: bool = True) -> ProjectVideo:
        path = str(path).strip()
        if not path:
            raise ValueError("Empty video path")
        # Prefer path relative to root when possible
        rel = path
        if self.root_dir:
            try:
                rel = str(Path(path).resolve().relative_to(Path(self.root_dir).resolve()))
            except (ValueError, OSError):
                rel = path
        # Dedupe by resolved absolute path
        abs_new = str(Path(path).resolve()) if Path(path).is_absolute() else str(
            self.resolve_path(rel).resolve()
        )
        for existing in self.videos:
            try:
                if str(existing.resolved_path(self.root_dir).resolve()) == abs_new:
                    existing.enabled = enabled
                    return existing
            except OSError:
                if existing.path == rel or existing.path == path:
                    existing.enabled = enabled
                    return existing
        entry = ProjectVideo(path=rel, enabled=enabled)
        self.videos.append(entry)
        return entry

    def remove_video_at(self, index: int) -> None:
        if 0 <= index < len(self.videos):
            del self.videos[index]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "name": self.name,
            "root_dir": self.root_dir,
            "videos": [v.to_dict() for v in self.videos],
            "paths": self.paths.to_dict(),
            "defaults": self.defaults.to_dict(),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, file_path: str = "") -> "Project":
        version = int(d.get("schema_version") or PROJECT_SCHEMA_VERSION)
        if version > PROJECT_SCHEMA_VERSION:
            raise ValueError(
                f"Project schema version {version} is newer than supported "
                f"({PROJECT_SCHEMA_VERSION}). Update LabGym."
            )
        videos = [ProjectVideo.from_dict(x) for x in (d.get("videos") or [])]
        proj = cls(
            schema_version=PROJECT_SCHEMA_VERSION,
            name=str(d.get("name") or "Untitled"),
            root_dir=str(d.get("root_dir") or ""),
            videos=videos,
            paths=ProjectPaths.from_dict(d.get("paths")),
            defaults=ProjectDefaults.from_dict(d.get("defaults")),
            notes=str(d.get("notes") or ""),
            file_path=file_path,
        )
        return proj

    def save(self, path: Optional[str] = None) -> Path:
        target = path or self.file_path
        if not target:
            raise ValueError("No path for project save")
        out = Path(target)
        # Prefer *.labproj.json when no extension given
        if out.suffix == "":
            out = out.with_name(out.name + PROJECT_FILE_SUFFIX)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        self.file_path = str(out.resolve())
        return out

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Project file must be a JSON object")
        return cls.from_dict(raw, file_path=str(path.resolve()))

    @classmethod
    def new(cls, name: str = "Untitled", root_dir: str = "") -> "Project":
        return cls(name=name, root_dir=root_dir)
