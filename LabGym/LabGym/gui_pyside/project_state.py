"""Shared project settings for the ethogram-first PySide workflow shell."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, QSettings, Signal


@dataclass
class ProjectState:
    """Paths and defaults for one ethogram-first training project."""

    video_path: str = ""
    tracklets_dir: str = ""
    annotations_path: str = ""
    examples_out_dir: str = ""
    # Annotator / ethogram
    behavior_mode: int = 0  # 0 non-int, 1 interactive basic, 2 interactive advanced
    exclusive_mode: bool = True
    # Example generation defaults (Stage C)
    window_length: int = 15
    sampling: str = "dense_in_bout"
    stride: int = 0
    min_bout_frames: int = 1
    social_distance: float = 0.0
    write_soft_labels: bool = True
    background_free: bool = True
    notes: str = ""

    def inferred_annotations_path(self) -> str:
        if self.annotations_path.strip():
            return self.annotations_path.strip()
        if self.video_path.strip():
            return str(Path(self.video_path).with_suffix(".annotations.json"))
        return ""

    def inferred_examples_dir(self) -> str:
        if self.examples_out_dir.strip():
            return self.examples_out_dir.strip()
        if self.video_path.strip():
            stem = Path(self.video_path).stem
            return str(Path(self.video_path).parent / f"{stem}_examples_from_ethogram")
        return ""

    def status_summary(self) -> str:
        parts = []
        parts.append(
            f"video: {Path(self.video_path).name}" if self.video_path else "video: —"
        )
        parts.append(
            f"tracklets: {Path(self.tracklets_dir).name}"
            if self.tracklets_dir
            else "tracklets: —"
        )
        ann = self.inferred_annotations_path()
        parts.append(f"ann: {Path(ann).name}" if ann else "ann: —")
        parts.append(f"mode: {self.behavior_mode}")
        return "  ·  ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectState":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class ProjectController(QObject):
    """Mutable project settings with change notifications + QSettings persistence."""

    changed = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.state = ProjectState()
        self._settings = QSettings("LabGym", "workflow")

    def update(self, **kwargs: Any) -> None:
        dirty = False
        for key, value in kwargs.items():
            if not hasattr(self.state, key):
                continue
            if getattr(self.state, key) != value:
                setattr(self.state, key, value)
                dirty = True
        if dirty:
            self.changed.emit()

    def replace(self, state: ProjectState) -> None:
        self.state = state
        self.changed.emit()

    def save_settings(self) -> None:
        d = self.state.to_dict()
        self._settings.beginGroup("project")
        for k, v in d.items():
            self._settings.setValue(k, v)
        self._settings.endGroup()

    def load_settings(self) -> None:
        self._settings.beginGroup("project")
        data: Dict[str, Any] = {}
        for key in ProjectState.__dataclass_fields__:  # type: ignore[attr-defined]
            if self._settings.contains(key):
                val = self._settings.value(key)
                # QSettings may return strings for bools/ints
                field_type = ProjectState.__dataclass_fields__[key].type  # type: ignore[attr-defined]
                data[key] = _coerce(val, field_type)
        self._settings.endGroup()
        if data:
            self.state = ProjectState.from_dict({**self.state.to_dict(), **data})
            self.changed.emit()


def _coerce(val: Any, type_hint: Any) -> Any:
    if val is None:
        return val
    hint = str(type_hint)
    if "bool" in hint:
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("1", "true", "yes")
    if "int" in hint:
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0
    if "float" in hint:
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0
    return str(val) if val is not None else ""
