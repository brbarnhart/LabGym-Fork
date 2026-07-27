"""Resolve video / annotations / tracklets / examples paths for a Project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .model import Project, ProjectVideo

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}

# Cache for discover_tracklets_dir (cleared on project replace/load).
# Key: (root_dir, detection_root, video_path, entry.detection_dir)
_tracklets_cache: Dict[Tuple[str, str, str, str], str] = {}


def clear_tracklets_discovery_cache() -> None:
    """Drop cached tracklet/id_review discovery results."""
    _tracklets_cache.clear()


@dataclass
class ResolvedVideoContext:
    """Paths for the project's current (or selected) video session."""

    video_path: str
    video_entry: Optional[ProjectVideo]
    annotations_path: str
    annotations_exists: bool
    tracklets_dir: str
    tracklets_exists: bool
    examples_out_dir: str
    behavior_mode: int
    exclusive_mode: bool
    window_length: int
    sampling: str
    stride: int
    min_bout_frames: int
    social_distance: float
    write_soft_labels: bool
    background_free: bool

    def summary_lines(self) -> List[str]:
        tr = "yes" if self.tracklets_exists else "missing"
        ann = "yes" if self.annotations_exists else "missing"
        return [
            f"Video: {self.video_path or '—'}",
            f"Annotations ({ann}): {self.annotations_path or '—'}",
            f"Tracklets ({tr}): {self.tracklets_dir or '—'}",
            f"Examples out: {self.examples_out_dir or '—'}",
            f"Mode: {self.behavior_mode}  ·  exclusive: {self.exclusive_mode}  ·  "
            f"length: {self.window_length}  ·  sampling: {self.sampling}",
        ]


def find_video_entry(project: Project, video_path: str) -> Optional[ProjectVideo]:
    if not video_path:
        return None
    try:
        target = str(Path(video_path).resolve())
    except OSError:
        target = str(Path(video_path))
    for v in project.videos:
        try:
            if str(v.resolved_path(project.root_dir).resolve()) == target:
                return v
        except OSError:
            if v.path == video_path:
                return v
    # Match by basename as weak fallback
    name = Path(video_path).name
    for v in project.videos:
        if Path(v.path).name == name:
            return v
    return None


def current_video_path(project: Project) -> str:
    """Absolute (or best-effort) path of the current video."""
    cur = (project.defaults.current_video or "").strip()
    if cur:
        p = project.resolve_path(cur)
        return str(p)
    # Fall back to first enabled video
    enabled = project.enabled_videos()
    if enabled:
        return str(enabled[0].resolved_path(project.root_dir))
    return ""


def set_current_video(project: Project, path: str) -> None:
    """Store current video as root-relative when possible."""
    path = str(path).strip()
    if not path:
        project.defaults.current_video = ""
        return
    rel = path
    if project.root_dir:
        try:
            rel = str(Path(path).resolve().relative_to(Path(project.root_dir).resolve()))
        except (ValueError, OSError):
            rel = path
    project.defaults.current_video = rel


def annotations_path_for(project: Project, video_path: str) -> str:
    entry = find_video_entry(project, video_path)
    if entry and entry.annotations_path.strip():
        return str(project.resolve_path(entry.annotations_path))
    if not video_path:
        return ""
    # Sidecar next to video
    return str(Path(video_path).with_suffix(".annotations.json"))


def _dir_has_tracklet_markers(directory: Path) -> bool:
    """True if *directory* itself looks like a tracklets / id_review folder.

    Prefer cheap exact-name checks before globs (network FS friendly).
    """
    if not directory.is_dir():
        return False
    # Cheap exact files first
    for name in ("meta.json", "events.json", "switches.json", "subjects.json"):
        if (directory / name).is_file():
            return True
    # Short-circuiting globs (do not materialize full lists)
    for pat in ("*_tracklets.npz", "*_tracklets_meta.json", "*tracklets*.npz"):
        if next(directory.glob(pat), None) is not None:
            return True
    return False


def _looks_like_tracklets_dir(directory: Path) -> bool:
    if _dir_has_tracklet_markers(directory):
        return True
    nested = directory / "id_review"
    if nested.is_dir() and _dir_has_tracklet_markers(nested):
        return True
    return False


def discover_tracklets_dir(project: Project, video_path: str) -> str:
    """Best-effort tracklets / id_review folder for a video.

    Results are cached until :func:`clear_tracklets_discovery_cache` (called when
    the project is replaced/loaded). Prefer an explicit ``detection_dir`` on the
    video entry when set — that path is not glob-searched.
    """
    entry = find_video_entry(project, video_path)
    entry_det = (entry.detection_dir if entry else "") or ""

    # Explicit entry: resolve once, no scan, still cache for consistency
    cache_key = (
        project.root_dir or "",
        (project.paths.detection_output_root or "").strip(),
        video_path or "",
        entry_det.strip(),
    )
    if cache_key in _tracklets_cache:
        return _tracklets_cache[cache_key]

    result = ""
    if entry and entry_det.strip():
        d = project.resolve_path(entry_det)
        if d.is_dir():
            nested = d / "id_review"
            result = str(nested) if nested.is_dir() else str(d)
            _tracklets_cache[cache_key] = result
            return result

    if not video_path:
        _tracklets_cache[cache_key] = ""
        return ""

    vp = Path(video_path)
    stem = vp.stem
    candidates: List[Path] = []

    # Project detection root / stem (cheap is_dir checks only)
    det_root = project.paths.detection_output_root.strip()
    if det_root:
        base = project.resolve_path(det_root)
        candidates.extend(
            [
                base / stem / "id_review",
                base / stem,
                base / f"{stem}_processed" / "id_review",
                base / "id_review",
            ]
        )

    # Sibling conventions (same as annotator try_autoload)
    candidates.extend(
        [
            vp.parent / "id_review",
            vp.with_suffix("") / "id_review",
            vp.parent / stem / "id_review",
            vp.parent / f"{stem}_processed" / "id_review",
            vp.parent / f"{stem}_processed",
        ]
    )

    for c in candidates:
        if _looks_like_tracklets_dir(c):
            nested = c / "id_review"
            if nested.is_dir() and _dir_has_tracklet_markers(nested):
                result = str(nested)
            else:
                result = str(c)
            break
        # Prefer nested id_review under an existing stem folder without full markers
        if c.is_dir() and (c / "id_review").is_dir():
            result = str(c / "id_review")
            break

    _tracklets_cache[cache_key] = result
    return result


def examples_out_dir_for(project: Project, video_path: str) -> str:
    root_ex = project.paths.examples_root.strip() or "examples"
    base = project.resolve_path(root_ex)
    if video_path:
        stem = Path(video_path).stem
        return str(base / f"{stem}_examples_from_ethogram")
    return str(base)


def resolve_video_context(
    project: Project, *, video_path: Optional[str] = None
) -> ResolvedVideoContext:
    """Build the full path context for annotate / generate tabs."""
    vp = (video_path or current_video_path(project) or "").strip()
    if vp and not Path(vp).is_absolute() and project.root_dir:
        vp = str(project.resolve_path(vp))

    entry = find_video_entry(project, vp) if vp else None
    ann = annotations_path_for(project, vp) if vp else ""
    tracks = discover_tracklets_dir(project, vp) if vp else ""
    examples = examples_out_dir_for(project, vp)
    d = project.defaults
    return ResolvedVideoContext(
        video_path=vp,
        video_entry=entry,
        annotations_path=ann,
        annotations_exists=bool(ann) and Path(ann).is_file(),
        tracklets_dir=tracks,
        tracklets_exists=bool(tracks) and Path(tracks).is_dir(),
        examples_out_dir=examples,
        behavior_mode=int(d.behavior_mode),
        exclusive_mode=bool(d.exclusive_mode),
        window_length=int(d.window_length),
        sampling=str(d.sampling),
        stride=int(d.stride),
        min_bout_frames=int(d.min_bout_frames),
        social_distance=float(d.social_distance),
        write_soft_labels=bool(d.write_soft_labels),
        background_free=bool(d.background_free),
    )


def list_project_video_choices(project: Project) -> List[Tuple[str, str]]:
    """Return (display_label, absolute_or_resolved_path) for enabled videos."""
    out: List[Tuple[str, str]] = []
    for v in project.enabled_videos():
        resolved = str(v.resolved_path(project.root_dir))
        label = v.path
        out.append((label, resolved))
    return out
