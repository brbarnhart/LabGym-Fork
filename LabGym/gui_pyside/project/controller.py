"""ProjectController: mutable project + dirty flag + recent files."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QSettings, Signal

from .model import Project
from .paths import (
    ResolvedVideoContext,
    current_video_path,
    resolve_video_context,
    set_current_video,
)

_MAX_RECENT = 12


class ProjectController(QObject):
    """Owns the open Project and notifies the shell when it changes."""

    changed = Signal()  # any field / dirty / path change
    project_replaced = Signal()  # new/open replaced whole project

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.project = Project.new()
        self._dirty = False
        self._settings = QSettings("LabGym", "workbench")

    # --- video session helpers (Phase 2+) ---

    def current_video_path(self) -> str:
        return current_video_path(self.project)

    def set_current_video(self, path: str, *, dirty: bool = True) -> None:
        set_current_video(self.project, path)
        if dirty:
            self.mark_dirty()
        else:
            self.changed.emit()

    def resolve_context(self, video_path: Optional[str] = None) -> ResolvedVideoContext:
        return resolve_video_context(self.project, video_path=video_path)

    # --- dirty ---

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            self.changed.emit()
        else:
            self.changed.emit()

    def mark_clean(self) -> None:
        self._dirty = False
        self.changed.emit()

    def status_summary(self) -> str:
        base = self.project.status_summary()
        if self._dirty:
            return base + "  ·  unsaved"
        return base

    # --- lifecycle ---

    def new_project(self, name: str = "Untitled", root_dir: str = "") -> None:
        self.project = Project.new(name=name, root_dir=root_dir)
        self._dirty = True
        self.project_replaced.emit()
        self.changed.emit()

    def load_from_path(self, path: str | Path) -> None:
        self.project = Project.load(path)
        self._dirty = False
        self.add_recent(str(self.project.file_path))
        self.project_replaced.emit()
        self.changed.emit()

    def save(self, path: Optional[str] = None) -> Path:
        out = self.project.save(path)
        self._dirty = False
        self.add_recent(str(out))
        self.changed.emit()
        return out

    def replace(self, project: Project, *, dirty: bool = False) -> None:
        self.project = project
        self._dirty = dirty
        self.project_replaced.emit()
        self.changed.emit()

    # --- recent ---

    def recent_paths(self) -> List[str]:
        raw = self._settings.value("recent_projects", [])
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw] if raw else []
        return [str(x) for x in raw if x]

    def add_recent(self, path: str) -> None:
        path = str(Path(path).resolve())
        items = [p for p in self.recent_paths() if p != path]
        items.insert(0, path)
        items = items[:_MAX_RECENT]
        self._settings.setValue("recent_projects", items)

    def clear_recent(self) -> None:
        self._settings.setValue("recent_projects", [])

    def last_workbench_id(self) -> str:
        return str(self._settings.value("last_workbench", "categorizer") or "categorizer")

    def set_last_workbench_id(self, workbench_id: str) -> None:
        self._settings.setValue("last_workbench", workbench_id)

    def last_project_path(self) -> str:
        return str(self._settings.value("last_project", "") or "")

    def set_last_project_path(self, path: str) -> None:
        self._settings.setValue("last_project", path)
