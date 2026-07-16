"""AnnotationManager: owns the AnnotationSession and provides safe mutation + I/O.

Responsibilities:
- Load / save schema v2 (with v1 migration)
- Per-subject bout creation with validation (no overlaps within a behavior)
- Toggle / start / end semantics on the active subject
- Query active behaviors at a frame
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .data_models import (
    BEHAVIOR_MODE_INTERACTIVE_BASIC,
    SCHEMA_VERSION,
    AnnotationSession,
    Behavior,
    Bout,
    Subject,
)

GROUP_KEY = "group"


class AnnotationManager:
    def __init__(self, session: Optional[AnnotationSession] = None):
        self.session: AnnotationSession = session or AnnotationSession(
            video_path="", fps=30.0, total_frames=0
        )
        # Open starts scoped by subject: subject_key -> {behavior_name: start_frame}
        self._active_starts: Dict[str, Dict[str, Optional[int]]] = {}
        self.exclusive_mode = self.session.exclusive_mode
        self._ensure_active_maps()

        # Undo: (bouts_snapshot, interaction_snapshot, active_snapshot, description)
        self._undo_stack: List[Tuple[dict, dict, dict, str]] = []
        self._max_undo = 50

    def _ensure_active_maps(self) -> None:
        for subj in self.session.subjects:
            key = str(subj.subject_id)
            self._active_starts.setdefault(key, {})

    def _sk(self, subject_id: Optional[int] = None) -> str:
        return self.session.subject_key(subject_id)

    def _open_map(self, subject_id: Optional[int] = None) -> Dict[str, Optional[int]]:
        key = self._sk(subject_id)
        return self._active_starts.setdefault(key, {})

    # --- Session / I/O ---

    @classmethod
    def load_from_json(cls, path: str | Path) -> "AnnotationManager":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        sess = AnnotationSession.from_dict(data)
        return cls(sess)

    def save_to_json(self, path: str | Path) -> None:
        data = self.session.to_dict()
        data["schema_version"] = SCHEMA_VERSION
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def set_active_subject(self, subject_id: int) -> None:
        self.session.set_active_subject(subject_id)
        self._ensure_active_maps()

    def add_subject(
        self,
        subject_id: int,
        animal_kind: str = "animal",
        display_name: str = "",
        color: str = "#4FC3F7",
    ) -> Subject:
        subj = Subject(
            subject_id=subject_id,
            animal_kind=animal_kind,
            display_name=display_name,
            color=color,
        )
        self.session.ensure_subject(subj)
        self._ensure_active_maps()
        return subj

    # --- Behavior management ---

    def add_behavior(
        self, name: str, color: str = "#FF5555", hotkey: Optional[str] = None
    ) -> Behavior:
        beh = Behavior(name=name, color=color, hotkey=hotkey)
        self.session.add_behavior(beh)
        return beh

    def remove_behavior(self, name: str) -> None:
        self._snapshot(f"Remove behavior {name}")
        self.session.remove_behavior(name)
        for omap in self._active_starts.values():
            omap.pop(name, None)

    def rename_behavior(self, old: str, new: str) -> None:
        self.session.rename_behavior(old, new)
        for omap in self._active_starts.values():
            if old in omap:
                omap[new] = omap.pop(old)

    def set_color(self, name: str, color: str) -> None:
        self.session.set_behavior_color(name, color)

    def set_hotkey(self, name: str, hotkey: Optional[str]) -> None:
        self.session.set_behavior_hotkey(name, hotkey)

    # --- Bout logic (active subject / group for interactive basic) ---

    def uses_group_ethogram(self) -> bool:
        """Interactive basic stores group-level labels, not per-subject."""
        return int(self.session.behavior_mode) == BEHAVIOR_MODE_INTERACTIVE_BASIC

    def _group_bouts_map(self) -> Dict[str, List[Bout]]:
        g = self.session.interaction_bouts.setdefault(GROUP_KEY, {})
        for beh in self.session.behaviors:
            g.setdefault(beh.name, [])
        return g

    def _get_bouts(
        self, name: str, subject_id: Optional[int] = None
    ) -> List[Bout]:
        # Interactive basic always uses the group ethogram unless a specific
        # subject_id is requested for multi-subject timeline inspection.
        if self.uses_group_ethogram() and subject_id is None:
            return self._group_bouts_map().setdefault(name, [])
        if self.uses_group_ethogram() and subject_id is not None:
            # Explicit subject lookups still return empty for group mode
            # unless we later add per-subject interaction roles here.
            return self.session.bouts_for_subject(subject_id).setdefault(name, [])
        bmap = self.session.bouts_for_subject(subject_id)
        return bmap.setdefault(name, [])

    def _validate_no_overlap(
        self,
        name: str,
        start: int,
        end: int,
        *,
        exclude_index: Optional[int] = None,
        subject_id: Optional[int] = None,
    ) -> None:
        for i, b in enumerate(self._get_bouts(name, subject_id)):
            if exclude_index is not None and i == exclude_index:
                continue
            if not (end < b.start_frame or start > b.end_frame):
                raise ValueError(
                    f"Overlap with existing bout [{b.start_frame}-{b.end_frame}] for '{name}'"
                )

    def add_bout(
        self,
        name: str,
        start_frame: int,
        end_frame: int,
        subject_id: Optional[int] = None,
        partner_ids: Optional[List[int]] = None,
    ) -> Bout:
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame
        self._snapshot(f"Add bout for {name}")
        bout_subject = None if self.uses_group_ethogram() else subject_id
        self._validate_no_overlap(
            name, start_frame, end_frame, subject_id=bout_subject
        )
        bout = Bout(
            start_frame=start_frame,
            end_frame=end_frame,
            partner_ids=list(partner_ids or []),
        )
        if self.uses_group_ethogram():
            blist = self._group_bouts_map().setdefault(name, [])
            blist.append(bout)
            blist.sort(key=lambda b: b.start_frame)
        else:
            self.session.add_bout(name, bout, subject_id=subject_id)
        return bout

    def delete_bout(
        self, name: str, index: int, subject_id: Optional[int] = None
    ) -> None:
        self._snapshot(f"Delete bout for {name}")
        self.session.remove_bout(name, index, subject_id=subject_id)

    def change_bout_type(
        self,
        from_name: str,
        index: int,
        to_name: str,
        subject_id: Optional[int] = None,
    ) -> Bout:
        if from_name == to_name:
            raise ValueError("New behavior type is the same as the current type")
        if self.session.get_behavior(to_name) is None:
            raise ValueError(f"Unknown behavior '{to_name}'")
        blist = self._get_bouts(from_name, subject_id)
        if not (0 <= index < len(blist)):
            raise IndexError(f"Bout index {index} out of range for '{from_name}'")

        bout = blist[index]
        self._validate_no_overlap(
            to_name, bout.start_frame, bout.end_frame, subject_id=subject_id
        )

        self._snapshot(f"Change bout type {from_name} → {to_name}")
        del self._get_bouts(from_name, subject_id)[index]
        moved = Bout(
            bout.start_frame,
            bout.end_frame,
            partner_ids=list(bout.partner_ids),
            subjects=list(bout.subjects),
        )
        self.session.add_bout(to_name, moved, subject_id=subject_id)
        return moved

    def update_bout_frames(
        self,
        name: str,
        index: int,
        start_frame: int,
        end_frame: int,
        subject_id: Optional[int] = None,
    ) -> Bout:
        blist = self._get_bouts(name, subject_id)
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

        self._validate_no_overlap(
            name, start_frame, end_frame, exclude_index=index, subject_id=subject_id
        )

        self._snapshot(
            f"Edit bout {name} [{old.start_frame}-{old.end_frame}] → [{start_frame}-{end_frame}]"
        )
        blist[index] = Bout(
            start_frame,
            end_frame,
            partner_ids=list(old.partner_ids),
            subjects=list(old.subjects),
        )
        blist.sort(key=lambda b: b.start_frame)
        for b in blist:
            if b.start_frame == start_frame and b.end_frame == end_frame:
                return b
        return Bout(start_frame, end_frame)

    def set_exclusive_mode(self, exclusive: bool) -> None:
        if self.exclusive_mode != exclusive:
            self.close_all_open_bouts()
            self.exclusive_mode = exclusive
            self.session.exclusive_mode = exclusive

    def _close_behavior(
        self,
        name: str,
        end_frame: int,
        subject_id: Optional[int] = None,
    ) -> Optional[Bout]:
        # Group ethogram open-starts still live under the active subject key
        omap = self._open_map(subject_id)
        start = omap.get(name)
        if start is None:
            return None
        start = max(0, start)
        if end_frame < start:
            omap[name] = None
            return None
        end = end_frame
        # For group mode, write into interaction_bouts regardless of subject_id
        bout_subject = None if self.uses_group_ethogram() else subject_id
        bouts = self._get_bouts(name, bout_subject)
        partners: List[int] = []
        if hasattr(self, "_pending_partners") and subject_id is not None:
            partners = list(
                self._pending_partners.get(str(subject_id), {}).pop(name, [])
            )
        bout = Bout(start, end, partner_ids=partners)
        self._validate_no_overlap(
            name, start, end, subject_id=bout_subject
        )
        bouts.append(bout)
        bouts.sort(key=lambda b: b.start_frame)
        omap[name] = None
        return bout

    def close_all_open_bouts(self, end_frame: Optional[int] = None) -> None:
        if end_frame is None:
            end_frame = (
                self.session.total_frames - 1 if self.session.total_frames > 0 else 0
            )
        for sk, omap in list(self._active_starts.items()):
            sid = int(sk)
            for name in list(omap.keys()):
                if omap.get(name) is not None:
                    self._close_behavior(name, end_frame, subject_id=sid)

    def toggle_bout(
        self,
        name: str,
        frame: int,
        subject_id: Optional[int] = None,
        partner_ids: Optional[List[int]] = None,
    ) -> Tuple[str, Optional[Bout]]:
        """Toggle annotation for a behavior at the current frame (active subject)."""
        frame = max(0, frame)
        sid = (
            self.session.active_subject_id
            if subject_id is None
            else int(subject_id)
        )
        omap = self._open_map(sid)
        current_start = omap.get(name)
        # stash partners for the open bout (applied on close)
        if not hasattr(self, "_pending_partners"):
            self._pending_partners: Dict[str, Dict[str, List[int]]] = {}
        sk = str(sid)
        self._pending_partners.setdefault(sk, {})

        self._snapshot(f"Toggle {name} at frame {frame} (subject {sid})")

        if self.exclusive_mode and current_start is None:
            for other in list(omap.keys()):
                if other != name and omap.get(other) is not None:
                    self._close_behavior(other, frame - 1, subject_id=sid)

        if current_start is not None:
            bout = self._close_behavior(name, frame, subject_id=sid)
            return ("closed", bout)

        if self.exclusive_mode:
            if self.uses_group_ethogram():
                bmap = self._group_bouts_map()
            else:
                bmap = self.session.bouts_for_subject(sid)
            for other_name in list(bmap.keys()):
                if other_name == name:
                    continue
                blist = bmap.get(other_name, [])
                if not blist:
                    continue
                last = blist[-1]
                if last.end_frame == frame and last.start_frame <= frame:
                    if last.start_frame <= frame - 1:
                        blist[-1] = Bout(
                            last.start_frame,
                            frame - 1,
                            partner_ids=list(last.partner_ids),
                            subjects=list(last.subjects),
                        )
                    else:
                        blist.pop()

        omap[name] = frame
        self._pending_partners.setdefault(sk, {})[name] = list(partner_ids or [])
        return ("started", None)

    def get_open_behaviors_at_frame(
        self, frame: int, subject_id: Optional[int] = None
    ) -> List[str]:
        omap = self._open_map(subject_id)
        open_names: List[str] = []
        for name, start in omap.items():
            if start is not None and start <= frame:
                open_names.append(name)
        return open_names

    def get_annotated_behaviors_at_frame(
        self, frame: int, subject_id: Optional[int] = None
    ) -> List[str]:
        annotated: List[str] = []
        if self.uses_group_ethogram() and subject_id is None:
            bmap = self._group_bouts_map()
        else:
            bmap = self.session.bouts_for_subject(subject_id)
        for name, blist in bmap.items():
            for b in blist:
                if b.contains(frame):
                    annotated.append(name)
                    break
        return annotated

    def get_active_behaviors(
        self, frame: int, subject_id: Optional[int] = None
    ) -> List[str]:
        seen: set = set()
        active: List[str] = []
        for name in self.get_annotated_behaviors_at_frame(frame, subject_id):
            if name not in seen:
                active.append(name)
                seen.add(name)
        for name in self.get_open_behaviors_at_frame(frame, subject_id):
            if name not in seen:
                active.append(name)
                seen.add(name)
        return active

    def get_bouts_for_behavior(
        self, name: str, subject_id: Optional[int] = None
    ) -> List[Bout]:
        return list(self._get_bouts(name, subject_id))

    def is_behavior_active(
        self, name: str, subject_id: Optional[int] = None
    ) -> bool:
        return self._open_map(subject_id).get(name) is not None

    def get_active_start_frame(
        self, name: str, subject_id: Optional[int] = None
    ) -> Optional[int]:
        return self._open_map(subject_id).get(name)

    def get_open_starts(self, subject_id: Optional[int] = None) -> dict:
        omap = self._open_map(subject_id)
        return {k: v for k, v in omap.items() if v is not None}

    # --- Undo ---

    def _snapshot(self, description: str = "Change") -> None:
        bouts_copy: Dict[str, Dict[str, List[Bout]]] = {}
        for sk, bmap in self.session.bouts.items():
            bouts_copy[sk] = {
                name: [
                    Bout(
                        b.start_frame,
                        b.end_frame,
                        partner_ids=list(b.partner_ids),
                        subjects=list(b.subjects),
                    )
                    for b in blist
                ]
                for name, blist in bmap.items()
            }
        inter_copy: Dict[str, Dict[str, List[Bout]]] = {}
        for gk, bmap in self.session.interaction_bouts.items():
            inter_copy[gk] = {
                name: [
                    Bout(
                        b.start_frame,
                        b.end_frame,
                        partner_ids=list(b.partner_ids),
                        subjects=list(b.subjects),
                    )
                    for b in blist
                ]
                for name, blist in bmap.items()
            }
        active_copy = {
            sk: dict(omap) for sk, omap in self._active_starts.items()
        }
        self._undo_stack.append((bouts_copy, inter_copy, active_copy, description))
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def undo(self) -> str:
        if not self._undo_stack:
            return "Nothing to undo"
        bouts_snap, inter_snap, active_snap, desc = self._undo_stack.pop()
        self.session.bouts = bouts_snap
        self.session.interaction_bouts = inter_snap
        self._active_starts = active_snap
        for bmap in self.session.bouts.values():
            for blist in bmap.values():
                blist.sort(key=lambda b: b.start_frame)
        return f"Undid: {desc}"

    # --- Convenience ---

    def clear_all(self) -> None:
        self.session.clear_bouts()

    def is_empty(self) -> bool:
        for bmap in self.session.bouts.values():
            for bl in bmap.values():
                if bl:
                    return False
        for bmap in self.session.interaction_bouts.values():
            for bl in bmap.values():
                if bl:
                    return False
        return True

    def get_bouts_for_behavior_all_subjects(
        self, name: str
    ) -> Dict[int, List[Bout]]:
        """Return {subject_id: bouts} for multi-subject timeline views."""
        out: Dict[int, List[Bout]] = {}
        for subj in self.session.subjects:
            out[subj.subject_id] = list(
                self._get_bouts(name, subject_id=subj.subject_id)
            )
        return out

    # --- Behavior Templates ---

    def save_behavior_template(self, path: str | Path) -> None:
        data = {
            "behaviors": [
                {"name": b.name, "color": b.color, "hotkey": b.hotkey}
                for b in self.session.behaviors
            ]
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_behavior_template(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        new_behaviors = [
            Behavior(
                name=b["name"],
                color=b.get("color", "#FF5555"),
                hotkey=b.get("hotkey"),
            )
            for b in data.get("behaviors", [])
        ]
        if not new_behaviors:
            raise ValueError("Template contains no behaviors")

        old_bouts = {
            sk: {n: list(bl) for n, bl in bmap.items()}
            for sk, bmap in self.session.bouts.items()
        }
        old_active = {sk: dict(omap) for sk, omap in self._active_starts.items()}

        self.session.behaviors = new_behaviors
        behavior_names = {b.name for b in new_behaviors}
        new_nested: Dict[str, Dict[str, List[Bout]]] = {}
        for sk in old_bouts:
            new_nested[sk] = {}
            for b in new_behaviors:
                new_nested[sk][b.name] = old_bouts[sk].get(b.name, [])
        self.session.bouts = new_nested

        self._active_starts = {
            sk: {k: v for k, v in omap.items() if k in behavior_names}
            for sk, omap in old_active.items()
        }
        self.session._ensure_bout_maps()
