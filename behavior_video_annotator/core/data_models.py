"""Core dataclasses for the annotation session.

Follows the exact structure specified in project_plan.md.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Behavior:
    """A user-defined behavior to annotate."""
    name: str
    color: str = "#FF5555"      # hex color for timeline + video overlay
    hotkey: Optional[str] = None  # e.g. "1", "g", "r"  (single char preferred)


@dataclass
class Bout:
    """A contiguous annotated interval for a behavior."""
    start_frame: int
    end_frame: int

    def duration_frames(self) -> int:
        """Return number of frames in the bout (inclusive)."""
        return max(0, self.end_frame - self.start_frame + 1)

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame


@dataclass
class AnnotationSession:
    """Complete state for one video annotation session."""
    video_path: str
    fps: float
    total_frames: int
    behaviors: List[Behavior] = field(default_factory=list)
    bouts: Dict[str, List[Bout]] = field(default_factory=dict)  # behavior_name -> list[Bout]
    exclusive_mode: bool = False  # if True, only one behavior active at a time

    def get_behavior(self, name: str) -> Optional[Behavior]:
        for b in self.behaviors:
            if b.name == name:
                return b
        return None

    def add_behavior(self, behavior: Behavior) -> None:
        if any(b.name == behavior.name for b in self.behaviors):
            raise ValueError(f"Behavior '{behavior.name}' already exists")
        self.behaviors.append(behavior)
        if behavior.name not in self.bouts:
            self.bouts[behavior.name] = []

    def remove_behavior(self, name: str) -> None:
        self.behaviors = [b for b in self.behaviors if b.name != name]
        self.bouts.pop(name, None)

    def rename_behavior(self, old_name: str, new_name: str) -> None:
        if old_name == new_name:
            return
        if any(b.name == new_name for b in self.behaviors):
            raise ValueError(f"Behavior '{new_name}' already exists")
        for b in self.behaviors:
            if b.name == old_name:
                b.name = new_name
                break
        if old_name in self.bouts:
            self.bouts[new_name] = self.bouts.pop(old_name)

    def set_behavior_color(self, name: str, color: str) -> None:
        for b in self.behaviors:
            if b.name == name:
                b.color = color
                return

    def set_behavior_hotkey(self, name: str, hotkey: Optional[str]) -> None:
        for b in self.behaviors:
            if b.name == name:
                b.hotkey = hotkey
                return

    # Bout operations (delegated to manager in practice, here for convenience)
    def add_bout(self, behavior_name: str, bout: Bout) -> None:
        if behavior_name not in self.bouts:
            self.bouts[behavior_name] = []
        # Caller (manager) is responsible for validation
        self.bouts[behavior_name].append(bout)
        # Keep bouts sorted by start
        self.bouts[behavior_name].sort(key=lambda b: b.start_frame)

    def remove_bout(self, behavior_name: str, index: int) -> None:
        if behavior_name in self.bouts and 0 <= index < len(self.bouts[behavior_name]):
            del self.bouts[behavior_name][index]

    def clear_bouts(self, behavior_name: Optional[str] = None) -> None:
        if behavior_name is None:
            for k in list(self.bouts.keys()):
                self.bouts[k] = []
        else:
            self.bouts[behavior_name] = []
