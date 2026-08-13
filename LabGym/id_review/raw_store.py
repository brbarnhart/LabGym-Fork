"""Raw tracklet snapshot and accepted-identities status for an identity package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

from LabGym.annotator.core.tracklets_bridge import discover_tracklet_kinds

from .apply import read_tracklets_identity_status, write_tracklets_identity_status
from .tracklets import load_tracklets, save_tracklets

RAW_SUBDIR = "raw"

PathLike = Union[str, Path]


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


def load_raw_tracklets(directory: PathLike) -> Dict[str, Any]:
    """Load raw tracklets from ``raw/``. Empty dict if none."""
    dest = raw_dir(directory)
    if not dest.is_dir():
        return {}
    kinds = discover_tracklet_kinds(dest)
    return {kind: load_tracklets(str(dest), kind) for kind in kinds}


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
    if "accepted" in status and status["accepted"] is not None:
        return bool(status["accepted"])
    return bool(status.get("corrected"))


def snapshot_uncorrected_root_to_raw(directory: PathLike) -> bool:
    """Move unpublished root tracklets into ``raw/``.

    Used for legacy detect-only packs whose canonical npz is still raw.
    Returns True if any files were moved.
    """
    directory = Path(directory)
    kinds = discover_tracklet_kinds(directory)
    if not kinds:
        return False
    dest = raw_dir(directory)
    dest.mkdir(parents=True, exist_ok=True)
    moved = False
    for kind in kinds:
        for suffix in ("_tracklets.npz", "_tracklets_meta.json"):
            src = directory / f"{kind}{suffix}"
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
    for kind in discover_tracklet_kinds(directory):
        for suffix in ("_tracklets.npz", "_tracklets_meta.json"):
            path = directory / f"{kind}{suffix}"
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


def reset_identity_package_for_new_detect(directory: PathLike) -> None:
    """Unpublish remapped tracklets and clear switches for a new tracking run."""
    unpublish_remapped_tracklets(directory)
    clear_switch_records(directory)
