"""Core dataclasses for multi-subject annotation sessions (schema v2).

Schema v1 (behavior_video_annotator) is supported via load migration:
  bouts: {behavior: [Bout, ...]}  →  single subject "0"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = 2

# Behavior mode codes (aligned with LabGym categorizer/analysis)
BEHAVIOR_MODE_NON_INTERACTIVE = 0
BEHAVIOR_MODE_INTERACTIVE_BASIC = 1
BEHAVIOR_MODE_INTERACTIVE_ADVANCED = 2
BEHAVIOR_MODE_STATIC_IMAGE = 3


@dataclass
class Behavior:
    """A user-defined behavior to annotate."""

    name: str
    color: str = "#FF5555"
    hotkey: Optional[str] = None


@dataclass
class Bout:
    """A contiguous annotated interval for a behavior."""

    start_frame: int
    end_frame: int
    # Optional interaction partners (mode 2 / advanced)
    partner_ids: List[int] = field(default_factory=list)
    # Optional subject membership for group/interaction bouts
    subjects: List[int] = field(default_factory=list)

    def duration_frames(self) -> int:
        """Return number of frames in the bout (inclusive)."""
        return max(0, self.end_frame - self.start_frame + 1)

    def contains(self, frame: int) -> bool:
        return self.start_frame <= frame <= self.end_frame

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "start_frame": int(self.start_frame),
            "end_frame": int(self.end_frame),
        }
        if self.partner_ids:
            d["partner_ids"] = [int(x) for x in self.partner_ids]
        if self.subjects:
            d["subjects"] = [int(x) for x in self.subjects]
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Bout":
        return cls(
            start_frame=int(d["start_frame"]),
            end_frame=int(d["end_frame"]),
            partner_ids=[int(x) for x in d.get("partner_ids") or []],
            subjects=[int(x) for x in d.get("subjects") or []],
        )


@dataclass
class Subject:
    """An individual (or logical) annotation target in a video."""

    subject_id: int
    animal_kind: str = "animal"
    display_name: str = ""
    color: str = "#4FC3F7"

    def __post_init__(self) -> None:
        if not self.display_name:
            self.display_name = f"{self.animal_kind}_{self.subject_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": int(self.subject_id),
            "animal_kind": self.animal_kind,
            "display_name": self.display_name,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Subject":
        return cls(
            subject_id=int(d["subject_id"]),
            animal_kind=str(d.get("animal_kind") or "animal"),
            display_name=str(d.get("display_name") or ""),
            color=str(d.get("color") or "#4FC3F7"),
        )


@dataclass
class TracksRef:
    """Reference to LabGym tracklets produced by detection + ID review."""

    path: Optional[str] = None
    meta_path: Optional[str] = None
    analysis_start_frame: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "meta_path": self.meta_path,
            "analysis_start_frame": int(self.analysis_start_frame),
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> Optional["TracksRef"]:
        if not d:
            return None
        return cls(
            path=d.get("path"),
            meta_path=d.get("meta_path"),
            analysis_start_frame=int(d.get("analysis_start_frame") or 0),
        )


def _empty_behavior_map(behavior_names: List[str]) -> Dict[str, List[Bout]]:
    return {name: [] for name in behavior_names}


@dataclass
class AnnotationSession:
    """Complete state for one video annotation session (schema v2).

    ``bouts`` maps subject_id (as string) → behavior_name → list of Bout.
    ``interaction_bouts`` maps group key (e.g. ``\"group\"``) → behavior → bouts
    for interactive-basic (mode 1) group-level labels.
    """

    video_path: str
    fps: float
    total_frames: int
    behaviors: List[Behavior] = field(default_factory=list)
    subjects: List[Subject] = field(default_factory=list)
    bouts: Dict[str, Dict[str, List[Bout]]] = field(default_factory=dict)
    interaction_bouts: Dict[str, Dict[str, List[Bout]]] = field(default_factory=dict)
    exclusive_mode: bool = False
    behavior_mode: int = BEHAVIOR_MODE_NON_INTERACTIVE
    tracks_ref: Optional[TracksRef] = None
    schema_version: int = SCHEMA_VERSION
    active_subject_id: int = 0

    def __post_init__(self) -> None:
        if not self.subjects:
            self.subjects = [
                Subject(subject_id=0, animal_kind="animal", display_name="subject_0")
            ]
            self.active_subject_id = 0
        self._ensure_bout_maps()

    # --- helpers ---

    def subject_key(self, subject_id: Optional[int] = None) -> str:
        sid = self.active_subject_id if subject_id is None else int(subject_id)
        return str(sid)

    def get_subject(self, subject_id: int) -> Optional[Subject]:
        for s in self.subjects:
            if s.subject_id == subject_id:
                return s
        return None

    def ensure_subject(self, subject: Subject) -> None:
        if self.get_subject(subject.subject_id) is None:
            self.subjects.append(subject)
        self._ensure_bout_maps()

    def set_active_subject(self, subject_id: int) -> None:
        if self.get_subject(subject_id) is None:
            raise ValueError(f"Unknown subject_id {subject_id}")
        self.active_subject_id = int(subject_id)

    def _ensure_bout_maps(self) -> None:
        names = [b.name for b in self.behaviors]
        for subj in self.subjects:
            key = str(subj.subject_id)
            if key not in self.bouts:
                self.bouts[key] = _empty_behavior_map(names)
            else:
                for name in names:
                    self.bouts[key].setdefault(name, [])
        # Drop orphaned behavior keys carefully is not done here (manager handles renames)

    def bouts_for_subject(self, subject_id: Optional[int] = None) -> Dict[str, List[Bout]]:
        key = self.subject_key(subject_id)
        if key not in self.bouts:
            self.bouts[key] = _empty_behavior_map([b.name for b in self.behaviors])
        return self.bouts[key]

    def get_behavior(self, name: str) -> Optional[Behavior]:
        for b in self.behaviors:
            if b.name == name:
                return b
        return None

    def add_behavior(self, behavior: Behavior) -> None:
        if any(b.name == behavior.name for b in self.behaviors):
            raise ValueError(f"Behavior '{behavior.name}' already exists")
        self.behaviors.append(behavior)
        for key in self.bouts:
            self.bouts[key].setdefault(behavior.name, [])
        for key in self.interaction_bouts:
            self.interaction_bouts[key].setdefault(behavior.name, [])

    def remove_behavior(self, name: str) -> None:
        self.behaviors = [b for b in self.behaviors if b.name != name]
        for key in list(self.bouts.keys()):
            self.bouts[key].pop(name, None)
        for key in list(self.interaction_bouts.keys()):
            self.interaction_bouts[key].pop(name, None)

    def rename_behavior(self, old_name: str, new_name: str) -> None:
        if old_name == new_name:
            return
        if any(b.name == new_name for b in self.behaviors):
            raise ValueError(f"Behavior '{new_name}' already exists")
        for b in self.behaviors:
            if b.name == old_name:
                b.name = new_name
                break
        for key, bmap in self.bouts.items():
            if old_name in bmap:
                bmap[new_name] = bmap.pop(old_name)
        for key, bmap in self.interaction_bouts.items():
            if old_name in bmap:
                bmap[new_name] = bmap.pop(old_name)

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

    def add_bout(
        self,
        behavior_name: str,
        bout: Bout,
        subject_id: Optional[int] = None,
    ) -> None:
        bmap = self.bouts_for_subject(subject_id)
        if behavior_name not in bmap:
            bmap[behavior_name] = []
        bmap[behavior_name].append(bout)
        bmap[behavior_name].sort(key=lambda b: b.start_frame)

    def remove_bout(
        self,
        behavior_name: str,
        index: int,
        subject_id: Optional[int] = None,
    ) -> None:
        bmap = self.bouts_for_subject(subject_id)
        if behavior_name in bmap and 0 <= index < len(bmap[behavior_name]):
            del bmap[behavior_name][index]

    def clear_bouts(
        self,
        behavior_name: Optional[str] = None,
        subject_id: Optional[int] = None,
    ) -> None:
        if subject_id is None and behavior_name is None:
            for key in list(self.bouts.keys()):
                for k in list(self.bouts[key].keys()):
                    self.bouts[key][k] = []
            return
        bmap = self.bouts_for_subject(subject_id)
        if behavior_name is None:
            for k in list(bmap.keys()):
                bmap[k] = []
        else:
            bmap[behavior_name] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "video_path": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "behavior_mode": int(self.behavior_mode),
            "behaviors": [
                {"name": b.name, "color": b.color, "hotkey": b.hotkey}
                for b in self.behaviors
            ],
            "subjects": [s.to_dict() for s in self.subjects],
            "active_subject_id": int(self.active_subject_id),
            "tracks_ref": self.tracks_ref.to_dict() if self.tracks_ref else None,
            "bouts": {
                sk: {
                    bname: [bout.to_dict() for bout in blist]
                    for bname, blist in bmap.items()
                }
                for sk, bmap in self.bouts.items()
            },
            "interaction_bouts": {
                gk: {
                    bname: [bout.to_dict() for bout in blist]
                    for bname, blist in bmap.items()
                }
                for gk, bmap in self.interaction_bouts.items()
            },
            "exclusive_mode": bool(self.exclusive_mode),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationSession":
        """Load schema v2 or migrate schema v1 (flat bouts)."""
        version = int(data.get("schema_version") or 1)
        behaviors = [
            Behavior(
                name=b["name"],
                color=b.get("color", "#FF5555"),
                hotkey=b.get("hotkey"),
            )
            for b in data.get("behaviors", [])
        ]

        if version >= 2 and "subjects" in data:
            subjects = [Subject.from_dict(s) for s in data.get("subjects", [])]
            if not subjects:
                subjects = [Subject(subject_id=0)]
            bouts: Dict[str, Dict[str, List[Bout]]] = {}
            for sk, bmap in (data.get("bouts") or {}).items():
                bouts[str(sk)] = {
                    bname: [Bout.from_dict(b) for b in (blist or [])]
                    for bname, blist in bmap.items()
                }
            interaction_bouts: Dict[str, Dict[str, List[Bout]]] = {}
            for gk, bmap in (data.get("interaction_bouts") or {}).items():
                interaction_bouts[str(gk)] = {
                    bname: [Bout.from_dict(b) for b in (blist or [])]
                    for bname, blist in bmap.items()
                }
            tracks_ref = TracksRef.from_dict(data.get("tracks_ref"))
            active = int(data.get("active_subject_id", subjects[0].subject_id))
            return cls(
                video_path=data.get("video_path", ""),
                fps=float(data.get("fps", 30.0)),
                total_frames=int(data.get("total_frames", 0)),
                behaviors=behaviors,
                subjects=subjects,
                bouts=bouts,
                interaction_bouts=interaction_bouts,
                exclusive_mode=bool(data.get("exclusive_mode", False)),
                behavior_mode=int(data.get("behavior_mode", 0)),
                tracks_ref=tracks_ref,
                schema_version=SCHEMA_VERSION,
                active_subject_id=active,
            )

        # --- v1 migration: flat behavior → bouts map ---
        subjects = [Subject(subject_id=0, animal_kind="animal", display_name="subject_0")]
        flat = data.get("bouts") or {}
        nested: Dict[str, Dict[str, List[Bout]]] = {"0": {}}
        for bname, blist in flat.items():
            nested["0"][bname] = [Bout.from_dict(b) for b in (blist or [])]
        # Ensure all defined behaviors exist under subject 0
        for beh in behaviors:
            nested["0"].setdefault(beh.name, [])

        return cls(
            video_path=data.get("video_path", ""),
            fps=float(data.get("fps", 30.0)),
            total_frames=int(data.get("total_frames", 0)),
            behaviors=behaviors,
            subjects=subjects,
            bouts=nested,
            interaction_bouts={},
            exclusive_mode=bool(data.get("exclusive_mode", False)),
            behavior_mode=int(data.get("behavior_mode", 0)),
            tracks_ref=None,
            schema_version=SCHEMA_VERSION,
            active_subject_id=0,
        )
