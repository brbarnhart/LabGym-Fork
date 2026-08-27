"""Per-video on-disk identity-package probes for list tabs.

Project swap must not call this module. Tabs request status on first show or
explicit refresh. A probe never fabricates pending or done; until it finishes
the status is unknown.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union

from PySide6.QtCore import QThread, Signal

from . import paths as paths_mod
from .model import Project

STATUS_UNKNOWN = "unknown"
STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_ERROR = "error"

STATUS_UNKNOWN_LABEL = "…"

# Hung NAS ``is_dir`` / glob: fail this video and continue with the rest.
DEFAULT_PROBE_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one video disk-status probe."""

    video_path: str
    status: str
    tracklets_dir: str = ""
    detection_dir: str = ""
    error: str = ""
    cancelled: bool = False
    timed_out: bool = False


def status_label(status: str) -> str:
    """Short table/combo label for a disk-status value."""
    if status == STATUS_UNKNOWN:
        return STATUS_UNKNOWN_LABEL
    return status


def _relative_detection_dir(project: Project, tracklets_dir: str) -> str:
    if not tracklets_dir:
        return ""
    root = (project.root_dir or "").strip()
    try:
        if root:
            return str(
                Path(tracklets_dir).resolve().relative_to(Path(root).resolve())
            )
    except (ValueError, OSError):
        pass
    return tracklets_dir


def _probe_impl(project: Project, video_path: str) -> ProbeResult:
    tracks = paths_mod.discover_tracklets_dir(project, video_path)
    if tracks:
        det = _relative_detection_dir(project, tracks)
        return ProbeResult(
            video_path=video_path,
            status=STATUS_DONE,
            tracklets_dir=tracks,
            detection_dir=det,
        )
    return ProbeResult(video_path=video_path, status=STATUS_PENDING)


def probe_video(
    project: Project,
    video_path: str,
    *,
    timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
    cancel_event: Optional[threading.Event] = None,
) -> ProbeResult:
    """Probe one video for an identity package / tracklets folder.

    Uses a stored ``detection_dir`` as a hint (via discovery). Does not mark
    the project dirty. Until this returns, callers must treat the row as
    **status unknown**.
    """
    if cancel_event is not None and cancel_event.is_set():
        return ProbeResult(
            video_path=video_path,
            status=STATUS_UNKNOWN,
            cancelled=True,
        )

    box: List[Union[ProbeResult, Exception]] = []
    done = threading.Event()

    def work() -> None:
        try:
            box.append(_probe_impl(project, video_path))
        except Exception as exc:
            box.append(exc)
        finally:
            done.set()

    # Daemon so a hung NAS call cannot block process exit or the next probe.
    threading.Thread(target=work, name="labgym-video-probe", daemon=True).start()
    if not done.wait(timeout_s):
        return ProbeResult(
            video_path=video_path,
            status=STATUS_ERROR,
            error=f"Timed out after {timeout_s:.0f}s probing {video_path}",
            timed_out=True,
        )

    if cancel_event is not None and cancel_event.is_set():
        return ProbeResult(
            video_path=video_path,
            status=STATUS_UNKNOWN,
            cancelled=True,
        )
    if not box:
        return ProbeResult(
            video_path=video_path,
            status=STATUS_ERROR,
            error=f"Probe returned no result for {video_path}",
        )
    outcome = box[0]
    if isinstance(outcome, Exception):
        return ProbeResult(
            video_path=video_path,
            status=STATUS_ERROR,
            error=f"{outcome} ({video_path})",
        )
    return outcome


def apply_probe_to_project(project: Project, result: ProbeResult) -> None:
    """Remember a found package in memory. Does not dirty or save the project."""
    if result.cancelled or result.status not in (STATUS_DONE, STATUS_PENDING):
        return
    if result.status == STATUS_DONE and result.tracklets_dir:
        entry = paths_mod.find_video_entry(project, result.video_path)
        if entry is not None and result.detection_dir:
            entry.detection_dir = result.detection_dir
        paths_mod.cache_tracklets_dir(
            project, result.video_path, result.tracklets_dir
        )
    else:
        paths_mod.cache_tracklets_dir(project, result.video_path, "")


def probe_videos(
    project: Project,
    video_paths: Iterable[str],
    *,
    timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
    cancel_event: Optional[threading.Event] = None,
    on_result: Optional[Callable[[ProbeResult], None]] = None,
) -> List[ProbeResult]:
    """Probe videos sequentially. Cancel skips remaining items; one failure continues."""
    out: List[ProbeResult] = []
    for path in video_paths:
        if cancel_event is not None and cancel_event.is_set():
            break
        result = probe_video(
            project, path, timeout_s=timeout_s, cancel_event=cancel_event
        )
        out.append(result)
        if on_result is not None:
            on_result(result)
    return out


class VideoDiskStatusProbeThread(QThread):
    """Background sequential probes. Caller applies results on the UI thread."""

    probe_finished = Signal(object)
    batch_finished = Signal()

    def __init__(
        self,
        project: Project,
        video_paths: Iterable[str],
        *,
        timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
        cancel_event: Optional[threading.Event] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._project = project
        self._paths = list(video_paths)
        self._timeout_s = timeout_s
        self._cancel = cancel_event if cancel_event is not None else threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        probe_videos(
            self._project,
            self._paths,
            timeout_s=self._timeout_s,
            cancel_event=self._cancel,
            on_result=self.probe_finished.emit,
        )
        self.batch_finished.emit()
