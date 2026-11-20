"""Per-video on-disk identity-package probes for list tabs.

Project swap must not call this module. Tabs request status on first show or
explicit refresh. A probe never fabricates pending or done; until it finishes
the status is unknown.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import QTableWidgetItem

from . import paths as paths_mod
from .model import Project

STATUS_UNKNOWN = "unknown"
STATUS_PENDING = "pending"
STATUS_DONE = "done"
STATUS_ERROR = "error"

STATUS_UNKNOWN_LABEL = "…"
STATUS_DONE_MARK = "✓"

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


def combo_item_label(name: str, status: str) -> str:
    """Video combo text: unknown mark until probed; never a fake done check.

    Args:
        name: Display name from the project (file name or relative path).
        status: Disk-status value (``unknown`` / ``pending`` / ``done`` / ``error``).

    Returns:
        ``name`` plus ``…`` while unknown, ``✓`` only after a successful probe,
        ``error`` after a failed probe, or the bare name for pending (no package).
    """
    if status == STATUS_UNKNOWN:
        return f"{name}  {STATUS_UNKNOWN_LABEL}"
    if status == STATUS_DONE:
        return f"{name}  {STATUS_DONE_MARK}"
    if status == STATUS_ERROR:
        return f"{name}  error"
    return name


def combo_item_tooltip(status: str, note: str = "") -> str:
    """Hover text for a Review IDs combo row.

    Args:
        status: Disk-status value.
        note: Probe error text (includes the failing path).

    Returns:
        *note* when *status* is error, otherwise empty (no fake pending/done).
    """
    if status == STATUS_ERROR:
        return (note or "").strip()
    return ""


def run_blocked_unknown_paths(
    checked_paths: Iterable[str],
    status_by_path: Mapping[str, Sequence[str]],
) -> List[str]:
    """Return checked videos still **status unknown**.

    Run on Detect + track or Process videos must refuse (or stay disabled)
    while this list is non-empty. Missing map entries are unknown.

    Args:
        checked_paths: Selected video paths.
        status_by_path: ``path -> (status, note, ...)`` from in-session probes.

    Returns:
        Paths among *checked_paths* that have not been probed yet.
    """
    blocked: List[str] = []
    for path in checked_paths:
        entry = status_by_path.get(path)
        status = entry[0] if entry else STATUS_UNKNOWN
        if status == STATUS_UNKNOWN:
            blocked.append(path)
    return blocked


def relative_detection_dir(project: Project, tracklets_dir: str) -> str:
    """Project-root-relative detection path for an in-memory ``detection_dir``.

    Args:
        project: Open project document.
        tracklets_dir: Absolute (or already relative) identity-package folder.

    Returns:
        Path relative to ``project.root_dir`` when possible, else *tracklets_dir*.
    """
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


def _relative_detection_dir(project: Project, tracklets_dir: str) -> str:
    """Backward-compatible alias for :func:`relative_detection_dir`."""
    return relative_detection_dir(project, tracklets_dir)


def _probe_impl(project: Project, video_path: str) -> ProbeResult:
    """Discover the identity package for one video (may hit disk / NAS).

    Args:
        project: Open project document (``detection_dir`` is a hint).
        video_path: Video to probe.

    Returns:
        ``STATUS_DONE`` with package paths, or ``STATUS_PENDING`` when none.
    """
    tracks = paths_mod.discover_tracklets_dir(project, video_path)
    if tracks:
        det = relative_detection_dir(project, tracks)
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
    **status unknown**. A cache hit returns immediately without a timeout
    thread. ``timeout_s <= 0`` also runs in-process (no hang watchdog).
    """
    if cancel_event is not None and cancel_event.is_set():
        return ProbeResult(
            video_path=video_path,
            status=STATUS_UNKNOWN,
            cancelled=True,
        )

    cached = paths_mod.cached_tracklets_dir(project, video_path)
    if cached is not None:
        if cached:
            det = relative_detection_dir(project, cached)
            return ProbeResult(
                video_path=video_path,
                status=STATUS_DONE,
                tracklets_dir=cached,
                detection_dir=det,
            )
        return ProbeResult(video_path=video_path, status=STATUS_PENDING)

    if timeout_s <= 0:
        try:
            return _probe_impl(project, video_path)
        except Exception as exc:
            return ProbeResult(
                video_path=video_path,
                status=STATUS_ERROR,
                error=f"{exc} ({video_path})",
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
        parent: Optional[QObject] = None,
    ) -> None:
        """Create a sequential probe worker.

        Args:
            project: Open project document.
            video_paths: Videos still **status unknown**.
            timeout_s: Per-video hang timeout (NAS). Cached hits skip the wrapper.
            cancel_event: Shared cancel; a new event is created when omitted.
            parent: Qt owner (usually the list tab).
        """
        super().__init__(parent)
        self._project = project
        self._paths = list(video_paths)
        self._timeout_s = timeout_s
        self._cancel = cancel_event if cancel_event is not None else threading.Event()

    def cancel(self) -> None:
        """Abort remaining probes; the current video's I/O may still finish."""
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


class VideoDiskStatusMixin:
    """Probe-thread lifecycle and **status unknown** bookkeeping for list tabs.

    Tabs call :meth:`_init_video_disk_status` from ``__init__`` and implement
    :meth:`_unknown_video_paths`. Table tabs should use
    :class:`VideoTableDiskStatusMixin` instead.
    """

    def _init_video_disk_status(self) -> None:
        """Allocate probe-thread fields (status map, generation, counters)."""
        self._status_by_path: Dict[str, Tuple[str, str]] = {}
        self._probe_thread: Optional[VideoDiskStatusProbeThread] = None
        self._probe_cancel: Optional[threading.Event] = None
        self._probe_gen = 0
        self._probe_total = 0
        self._probe_done = 0

    def _status_for_video(self, path: str) -> Tuple[str, str]:
        """Sticky in-session status, else **status unknown** until a probe finishes."""
        if path in self._status_by_path:
            return self._status_by_path[path]
        return STATUS_UNKNOWN, ""

    def _set_probe_counts(self, n_videos: int, n_unknown: int) -> None:
        """Set ``N / M checked`` from a freshly painted list."""
        self._probe_total = n_videos
        self._probe_done = n_videos - n_unknown
        self._update_probe_progress()

    def _update_probe_progress(self) -> None:
        """Paint catch-up text on ``lbl_probe`` when that label exists."""
        label = getattr(self, "lbl_probe", None)
        if label is None:
            return
        total = self._probe_total
        if total <= 0:
            label.setText("")
            return
        label.setText(f"{self._probe_done} / {total} checked")

    def _unknown_video_paths(self) -> List[str]:
        """Return video paths still **status unknown**. Tabs must override."""
        raise NotImplementedError

    def _probe_result_note(self, result: ProbeResult) -> str:
        """Per-row note stored with the probe status (default: error text)."""
        return result.error

    def _apply_probe_result(self, result: ProbeResult) -> None:
        """Record one probe on the status map and optional combo labels."""
        self._status_by_path[result.video_path] = (
            result.status,
            self._probe_result_note(result),
        )
        apply_combo = getattr(self, "_apply_combo_status", None)
        if apply_combo is not None:
            apply_combo(result.video_path, result.status)

    def _cancel_status_probes(self) -> None:
        """Bump generation and cancel the in-flight probe thread."""
        self._probe_gen += 1
        if self._probe_cancel is not None:
            self._probe_cancel.set()
        thread = self._probe_thread
        if thread is not None:
            thread.cancel()
        self._probe_thread = None
        self._probe_cancel = None

    def _maybe_start_status_probes(self) -> None:
        """Start probes when the tab is visible and no batch is running."""
        if getattr(self, "_batch_active", False) or not self.isVisible():
            return
        self._start_status_probes()

    def _start_status_probes(self) -> None:
        """Probe unknown videos sequentially off the UI thread."""
        if getattr(self, "_batch_active", False):
            return
        paths = self._unknown_video_paths()
        self._update_probe_progress()
        if not paths:
            return
        self._cancel_status_probes()
        gen = self._probe_gen
        cancel = threading.Event()
        self._probe_cancel = cancel
        thread = VideoDiskStatusProbeThread(
            self.project.project,
            paths,
            cancel_event=cancel,
            parent=self,
        )
        thread.probe_finished.connect(
            lambda result, g=gen: self._on_probe_result(result, g)
        )
        thread.batch_finished.connect(
            lambda g=gen: self._on_probe_batch_finished(g)
        )
        self._probe_thread = thread
        thread.start()

    def _on_probe_result(self, result: object, gen: int) -> None:
        """Apply one probe on the UI thread if this generation is still current."""
        if gen != self._probe_gen:
            return
        if not isinstance(result, ProbeResult) or result.cancelled:
            return
        apply_probe_to_project(self.project.project, result)
        self._apply_probe_result(result)
        self._probe_done += 1
        self._update_probe_progress()
        self._update_run_enabled()

    def _on_probe_batch_finished(self, gen: int) -> None:
        """Clear the thread handle after the last probe of this generation."""
        if gen != self._probe_gen:
            return
        self._probe_thread = None
        self._update_probe_progress()
        self._update_run_enabled()

    def _update_run_enabled(self) -> None:
        """Disable Run while any checked row is still **status unknown**."""
        btn = getattr(self, "btn_run", None)
        if btn is None:
            return
        queue = getattr(self, "queue", None)
        if (queue is not None and queue.is_running) or getattr(
            self, "_batch_active", False
        ):
            btn.setEnabled(False)
            return
        selected = getattr(self, "_selected_videos", None)
        if selected is None:
            return
        blocked = run_blocked_unknown_paths(selected(), self._status_by_path)
        btn.setEnabled(not blocked)


class VideoTableDiskStatusMixin(VideoDiskStatusMixin):
    """Video-table list: checked rows, sticky status cells, Run disable."""

    def _unknown_video_paths(self) -> List[str]:
        """Return table video paths still **status unknown**."""
        out: List[str] = []
        table = self.table
        for row in range(table.rowCount()):
            item0 = table.item(row, 0)
            if not item0:
                continue
            path = str(item0.data(Qt.ItemDataRole.UserRole) or "")
            if not path:
                continue
            status, _note = self._status_for_video(path)
            if status == STATUS_UNKNOWN:
                out.append(path)
        return out

    def _row_for_path(self, path: str) -> Optional[int]:
        """Return the table row whose UserRole is *path*, or None."""
        table = self.table
        for row in range(table.rowCount()):
            item0 = table.item(row, 0)
            if item0 and str(item0.data(Qt.ItemDataRole.UserRole) or "") == path:
                return row
        return None

    def _selected_videos(self) -> List[str]:
        """Checked video paths in table column 0."""
        out: List[str] = []
        table = self.table
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                path = item.data(Qt.ItemDataRole.UserRole)
                if path:
                    out.append(str(path))
        return out

    def _set_status(self, row: int, path: str, status: str, note: str = "") -> None:
        """Write sticky status onto the map and status/note cells."""
        self._status_by_path[path] = (status, note)
        table = self.table
        label = status_label(status)
        if table.item(row, 2):
            table.item(row, 2).setText(label)
        else:
            table.setItem(row, 2, QTableWidgetItem(label))
        if table.item(row, 3):
            table.item(row, 3).setText(note)
        else:
            table.setItem(row, 3, QTableWidgetItem(note))
        if note and status == "error":
            item2 = table.item(row, 2)
            item3 = table.item(row, 3)
            if item2:
                item2.setToolTip(note)
            if item3:
                item3.setToolTip(note)

    def _apply_probe_result(self, result: ProbeResult) -> None:
        """Update the matching table row, or the map if the row is gone."""
        note = self._probe_result_note(result)
        row = self._row_for_path(result.video_path)
        if row is not None:
            self._set_status(row, result.video_path, result.status, note)
        else:
            self._status_by_path[result.video_path] = (result.status, note)

    def _probe_result_note(self, result: ProbeResult) -> str:
        """Detect + track default: package path or probe error."""
        return result.tracklets_dir or result.error

    def _manual_refresh_videos(self) -> None:
        """Clear discovery cache and re-probe the table."""
        self._cancel_status_probes()
        paths_mod.clear_tracklets_discovery_cache()
        self._status_by_path.clear()
        self.refresh_videos()
        self._start_status_probes()

    def refresh_videos(self) -> None:
        """Rebuild the checked-video table from the project document.

        Mid-batch ``project.changed`` must not wipe live status; skip rebuilds
        while a batch is active. Until probes finish, status is unknown.
        """
        if getattr(self, "_batch_active", False):
            return

        prev_checked: Dict[str, bool] = {}
        table = self.table
        for row in range(table.rowCount()):
            item0 = table.item(row, 0)
            if not item0:
                continue
            path = str(item0.data(Qt.ItemDataRole.UserRole) or "")
            if path:
                prev_checked[path] = item0.checkState() == Qt.CheckState.Checked

        choices = paths_mod.list_project_video_choices(self.project.project)
        table.setRowCount(0)
        table.setRowCount(len(choices))
        unknown = 0
        for row, (label, resolved) in enumerate(choices):
            path = str(resolved)
            chk = QTableWidgetItem()
            chk.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            if path in prev_checked:
                chk.setCheckState(
                    Qt.CheckState.Checked
                    if prev_checked[path]
                    else Qt.CheckState.Unchecked
                )
            else:
                chk.setCheckState(Qt.CheckState.Checked)
            chk.setData(Qt.ItemDataRole.UserRole, path)
            table.setItem(row, 0, chk)
            table.setItem(row, 1, QTableWidgetItem(label))
            status, note = self._status_for_video(path)
            if status == STATUS_UNKNOWN:
                unknown += 1
            st_item = QTableWidgetItem(status_label(status))
            if status == STATUS_UNKNOWN:
                st_item.setToolTip("Status unknown — not yet checked on disk.")
            table.setItem(row, 2, st_item)
            table.setItem(row, 3, QTableWidgetItem(note))
        self._set_probe_counts(len(choices), unknown)
        self._update_run_enabled()

