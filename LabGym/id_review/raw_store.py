"""Raw tracklet snapshot and accepted-identities status for an identity package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Union

from LabGym.annotator.core.tracklets_bridge import discover_tracklet_kinds

from .apply import read_tracklets_identity_status, write_tracklets_identity_status
from .tracklets import load_tracklets, save_tracklets

RAW_SUBDIR = "raw"
_TRACKLET_SUFFIXES = ("_tracklets.npz", "_tracklets_meta.json")

PathLike = Union[str, Path]


def _tracklet_kind_files(
    directory: PathLike, kinds: Iterable[str]
) -> Iterator[Path]:
    """Yield public tracklet npz/meta paths for each kind under *directory*."""
    root = Path(directory)
    for kind in kinds:
        for suffix in _TRACKLET_SUFFIXES:
            yield root / f"{kind}{suffix}"


def raw_dir(directory: PathLike) -> Path:
    """Directory that holds immutable detector tracklets."""
    return Path(directory) / RAW_SUBDIR


def save_raw_tracklets(directory: PathLike, stores: Dict[str, Any]) -> Path:
    """Write raw tracklets under ``raw/``; does not publish remapped files."""
    dest = raw_dir(directory)
    dest.mkdir(parents=True, exist_ok=True)
    for store in stores.values():
        save_tracklets(store, str(dest))
    return dest


def load_kind_stores(directory: PathLike) -> Dict[str, Any]:
    """Load every discovered ``*_tracklets.npz`` kind from *directory*.

    Args:
        directory: Folder that may contain remapped or raw tracklet files.

    Returns:
        Mapping of animal kind to ``TrackletStore``. Empty if none found.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    kinds = discover_tracklet_kinds(directory)
    return {kind: load_tracklets(str(directory), kind) for kind in kinds}


def load_raw_tracklets(directory: PathLike) -> Dict[str, Any]:
    """Load raw tracklets from ``raw/``. Empty dict if none."""
    return load_kind_stores(raw_dir(directory))


def has_raw_snapshot(directory: PathLike) -> bool:
    """True when immutable detector tracklets exist under ``raw/``."""
    return bool(discover_tracklet_kinds(raw_dir(directory)))


def has_accepted_identities(directory: PathLike) -> bool:
    """True when Review IDs has published remapped tracklets at the package root.

    Legacy packs with ``corrected: true`` and root tracklets count as accepted.
    An unsaved detect pack (``corrected: false`` or missing status) does not.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return False
    if not discover_tracklet_kinds(directory):
        return False
    status = read_tracklets_identity_status(str(directory))
    return bool(status.get("accepted"))


def snapshot_uncorrected_root_to_raw(directory: PathLike) -> bool:
    """Move unpublished root tracklets into ``raw/``.

    Used for legacy detect-only packs whose canonical npz is still raw.
    Returns True if any files were moved. Accepted / legacy corrected
    packs are refused so remapped geometry is never copied in as raw.
    """
    directory = Path(directory)
    if has_accepted_identities(directory):
        return False
    kinds = discover_tracklet_kinds(directory)
    if not kinds:
        return False
    dest = raw_dir(directory)
    dest.mkdir(parents=True, exist_ok=True)
    moved = False
    for src in _tracklet_kind_files(directory, kinds):
        if not src.is_file():
            continue
        target = dest / src.name
        if target.exists():
            target.unlink()
        src.replace(target)
        moved = True
    if moved:
        prev = read_tracklets_identity_status(str(directory))
        write_tracklets_identity_status(
            str(directory),
            corrected=False,
            accepted=False,
            has_raw=True,
            n_decisions=int(prev.get("n_decisions") or 0),
            source=str(prev.get("source") or "snapshot_uncorrected_root"),
        )
    return moved


def unpublish_remapped_tracklets(directory: PathLike) -> None:
    """Remove published remapped tracklets; leave raw snapshot in place."""
    directory = Path(directory)
    for path in _tracklet_kind_files(directory, discover_tracklet_kinds(directory)):
        if path.is_file():
            path.unlink()
    prev = read_tracklets_identity_status(str(directory))
    write_tracklets_identity_status(
        str(directory),
        corrected=False,
        accepted=False,
        has_raw=has_raw_snapshot(directory),
        n_decisions=0,
        source=str(prev.get("source") or "unpublish"),
    )


def clear_switch_records(directory: PathLike) -> None:
    """Remove switch/decision files so a new detect is a new identity world."""
    directory = Path(directory)
    for name in ("switches.jsonl", "decisions.jsonl", "applied_corrections.json"):
        path = directory / name
        if path.is_file():
            path.unlink()


def clear_raw_snapshot(directory: PathLike) -> None:
    """Remove raw tracklet files so a new detect can replace the snapshot."""
    dest = raw_dir(directory)
    if not dest.is_dir():
        return
    for path in _tracklet_kind_files(dest, discover_tracklet_kinds(dest)):
        if path.is_file():
            path.unlink()


def reset_identity_package_for_new_detect(directory: PathLike) -> None:
    """Replace raw, unpublish remapped, and clear switches for a new detect."""
    unpublish_remapped_tracklets(directory)
    clear_switch_records(directory)
    clear_raw_snapshot(directory)
