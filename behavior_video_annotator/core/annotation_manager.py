"""AnnotationManager: owns the AnnotationSession and provides safe mutation + I/O.

Responsibilities:
- Load / save to the exact JSON schema in project_plan.md
- Bout creation with validation (no overlaps within a behavior)
- Toggle / start / end semantics
- Query active behaviors at a frame
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

from .data_models import AnnotationSession, Behavior, Bout


class AnnotationManager:
    def __init__(self, session: Optional[AnnotationSession] = None):
        self.session: AnnotationSession = session or AnnotationSession(
            video_path="", fps=30.0, total_frames=0
        )
        self._active_starts: dict[str, Optional[int]] = {}  # behavior_name -> start_frame while "on"
        # Sync exclusive mode from session
        self.exclusive_mode = self.session.exclusive_mode
        # On init/load, no bouts are open (they are all recorded as closed intervals)

        # Undo support
        self._undo_stack: List[Tuple[dict, dict, str]] = []  # (bouts_snapshot, active_snapshot, description)
        self._max_undo = 50

    # --- Session / I/O ---

    @classmethod
    def load_from_json(cls, path: str | Path) -> "AnnotationManager":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        behaviors = [Behavior(**b) for b in data.get("behaviors", [])]
        bouts: dict[str, list[Bout]] = {}
        for name, blist in data.get("bouts", {}).items():
            bouts[name] = [Bout(**b) for b in blist]

        sess = AnnotationSession(
            video_path=data["video_path"],
            fps=float(data["fps"]),
            total_frames=int(data["total_frames"]),
            behaviors=behaviors,
            bouts=bouts,
            exclusive_mode=data.get("exclusive_mode", False),
        )
        return cls(sess)

    def save_to_json(self, path: str | Path) -> None:
        data = {
            "video_path": self.session.video_path,
            "fps": self.session.fps,
            "total_frames": self.session.total_frames,
            "behaviors": [asdict(b) for b in self.session.behaviors],
            "bouts": {
                name: [asdict(b) for b in blist]
                for name, blist in self.session.bouts.items()
            },
            "exclusive_mode": self.session.exclusive_mode,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    # --- Behavior management (thin wrappers) ---

    def add_behavior(self, name: str, color: str = "#FF5555", hotkey: Optional[str] = None) -> Behavior:
        beh = Behavior(name=name, color=color, hotkey=hotkey)
        self.session.add_behavior(beh)
        return beh

    def remove_behavior(self, name: str) -> None:
        self._snapshot(f"Remove behavior {name}")
        self.session.remove_behavior(name)

    def rename_behavior(self, old: str, new: str) -> None:
        self.session.rename_behavior(old, new)

    def set_color(self, name: str, color: str) -> None:
        self.session.set_behavior_color(name, color)

    def set_hotkey(self, name: str, hotkey: Optional[str]) -> None:
        self.session.set_behavior_hotkey(name, hotkey)

    # --- Bout logic (the heart of annotation) ---

    def _get_bouts(self, name: str) -> List[Bout]:
        return self.session.bouts.setdefault(name, [])

    def _validate_no_overlap(self, name: str, start: int, end: int, *, exclude_index: Optional[int] = None) -> None:
        """Raise if the proposed [start, end] overlaps any existing bout for this behavior."""
        for i, b in enumerate(self._get_bouts(name)):
            if exclude_index is not None and i == exclude_index:
                continue
            if not (end < b.start_frame or start > b.end_frame):
                raise ValueError(f"Overlap with existing bout [{b.start_frame}-{b.end_frame}] for '{name}'")

    def add_bout(self, name: str, start_frame: int, end_frame: int) -> Bout:
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame
        self._snapshot(f"Add bout for {name}")
        self._validate_no_overlap(name, start_frame, end_frame)
        bout = Bout(start_frame=start_frame, end_frame=end_frame)
        self.session.add_bout(name, bout)
        return bout

    def delete_bout(self, name: str, index: int) -> None:
        self._snapshot(f"Delete bout for {name}")
        self.session.remove_bout(name, index)

    def change_bout_type(self, from_name: str, index: int, to_name: str) -> Bout:
        """Reassign a bout from one behavior type to another.

        Validates that the interval does not overlap any existing bout of the
        destination behavior. Supports undo via the standard snapshot stack.
        """
        if from_name == to_name:
            raise ValueError("New behavior type is the same as the current type")
        if self.session.get_behavior(to_name) is None:
            raise ValueError(f"Unknown behavior '{to_name}'")
        blist = self._get_bouts(from_name)
        if not (0 <= index < len(blist)):
            raise IndexError(f"Bout index {index} out of range for '{from_name}'")

        bout = blist[index]
        # Validate before snapshot so a failed change does not pollute undo history
        self._validate_no_overlap(to_name, bout.start_frame, bout.end_frame)

        self._snapshot(f"Change bout type {from_name} → {to_name}")
        # Remove from source, add to destination
        del self._get_bouts(from_name)[index]
        moved = Bout(bout.start_frame, bout.end_frame)
        self.session.add_bout(to_name, moved)
        return moved

    def update_bout_frames(
        self,
        name: str,
        index: int,
        start_frame: int,
        end_frame: int,
    ) -> Bout:
        """Change the start/end of an existing bout (with overlap validation + undo)."""
        blist = self._get_bouts(name)
        if not (0 <= index < len(blist)):
            raise IndexError(f"Bout index {index} out of range for '{name}'")

        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame

        total = self.session.total_frames
        if total > 0:
            start_frame = max(0, min(int(start_frame), total - 1))
            end_frame = max(0, min(int(end_frame), total - 1))
        else:
            start_frame = max(0, int(start_frame))
            end_frame = max(0, int(end_frame))
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame

        old = blist[index]
        if old.start_frame == start_frame and old.end_frame == end_frame:
            return old

        # Validate before snapshot so a failed edit does not pollute undo history
        self._validate_no_overlap(name, start_frame, end_frame, exclude_index=index)

        self._snapshot(
            f"Edit bout {name} [{old.start_frame}-{old.end_frame}] → [{start_frame}-{end_frame}]"
        )
        blist[index] = Bout(start_frame, end_frame)
        blist.sort(key=lambda b: b.start_frame)
        # Return the updated bout (index may have changed after sort)
        for b in blist:
            if b.start_frame == start_frame and b.end_frame == end_frame:
                return b
        return Bout(start_frame, end_frame)

    def set_exclusive_mode(self, exclusive: bool) -> None:
        """Switch between multi-behavior (overlapping OK) and exclusive (one at a time) mode."""
        if self.exclusive_mode != exclusive:
            # If turning on exclusive, close any currently active except possibly one?
            # For simplicity, close all opens when changing mode (user can re-toggle)
            self.close_all_open_bouts()
            self.exclusive_mode = exclusive
            self.session.exclusive_mode = exclusive

    def _close_behavior(self, name: str, end_frame: int) -> Optional[Bout]:
        """Close an open bout for this behavior at end_frame (if one is open).

        If end_frame < start, the open bout is simply cancelled (no bout is recorded).
        This is used in exclusive mode to ensure the transition frame has only the new behavior.
        """
        start = self._active_starts.get(name)
        if start is None:
            return None
        start = max(0, start)
        if end_frame < start:
            # Closing before the bout even began (e.g. auto-close at frame-1 when started at frame)
            # Just discard the open start without creating a (invalid) bout.
            self._active_starts[name] = None
            return None
        end = end_frame
        bouts = self._get_bouts(name)
        bout = Bout(start, end)
        self._validate_no_overlap(name, start, end)
        bouts.append(bout)
        bouts.sort(key=lambda b: b.start_frame)
        self._active_starts[name] = None
        return bout

    def close_all_open_bouts(self, end_frame: Optional[int] = None) -> None:
        """Force close any currently open bouts. Useful before export or mode switch."""
        if end_frame is None:
            end_frame = self.session.total_frames - 1 if self.session.total_frames > 0 else 0
        for name in list(self._active_starts.keys()):
            if self._active_starts.get(name) is not None:
                self._close_behavior(name, end_frame)

    def toggle_bout(self, name: str, frame: int) -> Tuple[str, Optional[Bout]]:
        """
        Toggle annotation for a behavior at the current frame.

        Two modes (controlled by self.exclusive_mode / session):

        - Multi (exclusive_mode=False): Each behavior is independent.
          Pressing its hotkey starts a bout if none open for it, or closes the open one.

        - Exclusive (exclusive_mode=True): Only one behavior active at a time.
          - Pressing a different behavior's hotkey: auto-closes any other active behavior(s), then starts this one.
          - Pressing the hotkey of the currently active behavior: closes it (manual "off" / no behavior).
          - This allows marking periods of "nothing interesting".

        Always respects no-overlap within the same behavior.
        Returns (action, bout or None). Actions: "started", "closed", etc.
        """
        frame = max(0, frame)

        current_start = self._active_starts.get(name)

        # Record for undo before any change
        action_desc = f"Toggle {name} at frame {frame}"
        self._snapshot(action_desc)

        if self.exclusive_mode and current_start is None:
            # Auto-close any other active behaviors at the *previous* frame so that
            # the transition frame has only the new behavior (no 1-frame overlap).
            for other in list(self._active_starts.keys()):
                if other != name and self._active_starts.get(other) is not None:
                    self._close_behavior(other, frame - 1)

        if current_start is not None:
            # This behavior is currently on → close it
            bout = self._close_behavior(name, frame)
            return ("closed", bout)
        else:
            # We are starting a new behavior at this frame.
            if self.exclusive_mode:
                # Ensure this frame is exclusively claimed by the new behavior:
                # Trim any other behavior's most recent bout if it ends exactly at this frame
                # (covers both auto-close cases and manual off + immediate switch at same frame).
                for other_name in list(self.session.bouts.keys()):
                    if other_name == name:
                        continue
                    blist = self.session.bouts.get(other_name, [])
                    if not blist:
                        continue
                    last = blist[-1]
                    if last.end_frame == frame and last.start_frame <= frame:
                        if last.start_frame <= frame - 1:
                            blist[-1] = Bout(last.start_frame, frame - 1)
                        else:
                            blist.pop()  # remove the pointless single-frame or invalid bout

            # Start a new open bout for this behavior
            self._active_starts[name] = frame
            return ("started", None)

    def get_open_behaviors_at_frame(self, frame: int) -> List[str]:
        """Behaviors currently toggled on (open / recording) that cover this frame.

        These are live annotations not yet closed into a saved bout.
        """
        open_names: List[str] = []
        for name, start in self._active_starts.items():
            if start is not None and start <= frame:
                open_names.append(name)
        return open_names

    def get_annotated_behaviors_at_frame(self, frame: int) -> List[str]:
        """Behaviors with a completed (saved) bout covering this frame.

        Does not include currently open/toggled-on starts — only recorded bouts.
        """
        annotated: List[str] = []
        for name, blist in self.session.bouts.items():
            for b in blist:
                if b.contains(frame):
                    annotated.append(name)
                    break
        return annotated

    def get_active_behaviors(self, frame: int) -> List[str]:
        """Return all behavior names present at this frame (open and/or saved).

        Always returns both currently toggled-on behaviors and completed bout
        annotations so re-annotation can still see prior labels. Exclusive mode
        does not hide existing annotations from this query (use the open vs
        annotated helpers when you need them separated).
        """
        seen: set[str] = set()
        active: List[str] = []
        for name in self.get_annotated_behaviors_at_frame(frame):
            if name not in seen:
                active.append(name)
                seen.add(name)
        for name in self.get_open_behaviors_at_frame(frame):
            if name not in seen:
                active.append(name)
                seen.add(name)
        return active

    def get_bouts_for_behavior(self, name: str) -> List[Bout]:
        return list(self._get_bouts(name))

    def is_behavior_active(self, name: str) -> bool:
        """Whether this behavior currently has an open (unclosed) bout."""
        return self._active_starts.get(name) is not None

    def get_active_start_frame(self, name: str) -> Optional[int]:
        """If the behavior is currently active, return the frame it was started at."""
        return self._active_starts.get(name)

    def get_open_starts(self) -> dict[str, int]:
        """Return {behavior_name: start_frame} for all currently open bouts."""
        return {k: v for k, v in self._active_starts.items() if v is not None}

    # --- Undo support ---

    def _snapshot(self, description: str = "Change"):
        """Save current state for undo."""
        # Deep copy bouts
        bouts_copy: dict[str, list[Bout]] = {}
        for k, blist in self.session.bouts.items():
            bouts_copy[k] = [Bout(b.start_frame, b.end_frame) for b in blist]
        active_copy = dict(self._active_starts)
        self._undo_stack.append((bouts_copy, active_copy, description))
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def undo(self) -> str:
        """Undo the last annotation change. Returns a description of what was undone."""
        if not self._undo_stack:
            return "Nothing to undo"
        bouts_snap, active_snap, desc = self._undo_stack.pop()
        self.session.bouts = bouts_snap
        self._active_starts = active_snap
        # Ensure lists are sorted
        for blist in self.session.bouts.values():
            blist.sort(key=lambda b: b.start_frame)
        return f"Undid: {desc}"

    # --- Convenience ---

    def clear_all(self) -> None:
        for name in list(self.session.bouts.keys()):
            self.session.bouts[name] = []

    def is_empty(self) -> bool:
        return all(len(bl) == 0 for bl in self.session.bouts.values())

    # --- Behavior Templates (reusable sets of behaviors + hotkeys + colors) ---

    def save_behavior_template(self, path: str | Path) -> None:
        """Save only the behavior definitions (name, color, hotkey) as a reusable template."""
        data = {
            "behaviors": [asdict(b) for b in self.session.behaviors]
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_behavior_template(self, path: str | Path) -> None:
        """Load a behavior template and apply it to the current session.

        - Replaces the current behavior list.
        - Preserves bouts for any behaviors whose names still exist.
        - Drops bouts for behaviors no longer in the template.
        - Keeps currently active/open bouts only for behaviors still present.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        new_behaviors = [Behavior(**b) for b in data.get("behaviors", [])]
        if not new_behaviors:
            raise ValueError("Template contains no behaviors")

        old_bouts = dict(self.session.bouts)  # copy
        old_active = dict(self._active_starts)

        self.session.behaviors = new_behaviors
        self.session.bouts = {}

        behavior_names = {b.name for b in new_behaviors}
        for b in new_behaviors:
            if b.name in old_bouts:
                self.session.bouts[b.name] = old_bouts[b.name]
            else:
                self.session.bouts[b.name] = []

        # Keep active starts only for behaviors that still exist
        self._active_starts = {k: v for k, v in old_active.items() if k in behavior_names}
