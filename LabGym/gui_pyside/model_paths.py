"""Shared discovery of detector / categorizer folders for workbench tabs."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

from LabGym.detection.batch_detect import list_detectors
from LabGym.mypkg_resources import resource_filename

PathLike = Union[str, Path]


def bundled_detectors_root() -> Optional[Path]:
    try:
        p = Path(resource_filename("LabGym", "detectors"))
        return p if p.is_dir() else None
    except Exception:
        return None


def bundled_models_root() -> Optional[Path]:
    try:
        p = Path(resource_filename("LabGym", "models"))
        return p if p.is_dir() else None
    except Exception:
        return None


def project_models_root(project) -> Optional[Path]:
    """Return the project models folder if configured and resolvable."""
    if not getattr(project, "root_dir", None):
        return None
    rel = (getattr(project.paths, "models_root", None) or "models").strip() or "models"
    try:
        return Path(project.resolve_path(rel))
    except Exception:
        return None


def model_search_roots(
    project=None,
    *,
    include_bundled_detectors: bool = True,
    include_bundled_models: bool = True,
) -> List[Path]:
    """Ordered roots used when scanning for detectors / categorizers."""
    roots: List[Path] = []
    seen: set[str] = set()

    def _add(p: Optional[Path]) -> None:
        if p is None:
            return
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            return
        seen.add(key)
        roots.append(p)

    if project is not None:
        _add(project_models_root(project))
    if include_bundled_detectors:
        _add(bundled_detectors_root())
    if include_bundled_models:
        _add(bundled_models_root())
    return roots


def list_categorizer_folders(models_root: PathLike) -> List[Path]:
    """Find categorizer folders (CSV model_parameters.txt with classnames/network)."""
    root = Path(models_root)
    if not root.is_dir():
        return []
    found: List[Path] = []
    for p in root.rglob("model_parameters.txt"):
        try:
            # categorizer uses CSV; detector uses JSON
            text = p.read_text(encoding="utf-8", errors="ignore")[:80]
            if "classnames" in text or "network" in text or "time_step" in text:
                found.append(p.parent)
        except Exception:
            continue
    return sorted(set(found))


def scan_detector_paths(project=None, *, roots: Optional[Sequence[PathLike]] = None) -> List[str]:
    """Unique detector folder paths (string), sorted by discovery order."""
    search = [Path(r) for r in roots] if roots is not None else model_search_roots(project)
    out: List[str] = []
    seen: set[str] = set()
    for root in search:
        for d in list_detectors(root):
            s = str(d)
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def scan_categorizer_paths(
    project=None, *, roots: Optional[Sequence[PathLike]] = None
) -> List[str]:
    """Unique categorizer folder paths (string)."""
    search = [Path(r) for r in roots] if roots is not None else model_search_roots(project)
    out: List[str] = []
    seen: set[str] = set()
    for root in search:
        for c in list_categorizer_folders(root):
            s = str(c)
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out
