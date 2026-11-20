"""Review IDs session-load UI (hunt, QThread wiring, swap protocol).

Extracted from the tab so Review IDs stays under the ~1k LOC guideline.
Worker I/O lives in ``review_ids_session_load``; this module owns cancel/replace
and the **project swap** session protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from LabGym.gui_pyside.project.paths import (
    list_project_video_choices,
    paths_refer_to_same_video,
)
from LabGym.gui_pyside.project.video_disk_status import (
    STATUS_UNKNOWN,
    ProbeResult,
    apply_probe_to_project,
    combo_item_label,
    combo_item_tooltip,
)
from LabGym.gui_pyside.session_swap import switch_markers_signature
from LabGym.gui_pyside.widgets.path_browse import browse_existing_directory
from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
    ReviewPackageMetadata,
    clone_markers,
)
from LabGym.gui_pyside.workbenches.detector.review_ids_session_load import (
    ReviewSessionLoadThread,
)
from LabGym.identity.package import (
    clone_store,
    migrate_uncorrected_public_to_raw,
    needs_uncorrected_raw_migrate,
    switch_edits_allowed,
)


class ReviewIdsSessionMixin:
    """Session load, video-combo hunt, and swap-protocol methods for Review IDs."""

    def refresh_video_list(self) -> None:
        """Paint video names from the project document.

        Rows stay **status unknown** until a probe finishes. Does not hunt
        disk or start a **session load**. Preserves the selected video across
        rebuilds.
        """
        current = self.combo_video.currentData()
        self.combo_video.blockSignals(True)
        self.combo_video.clear()
        self.combo_video.addItem("(open package folder…)", "")
        self._video_names.clear()
        unknown = 0
        choices = list_project_video_choices(self.project.project)
        for label, resolved in choices:
            path = str(resolved)
            self._video_names[path] = label
            status, note = self._status_for_video(path)
            if status == STATUS_UNKNOWN:
                unknown += 1
            self.combo_video.addItem(combo_item_label(label, status), path)
            tip = combo_item_tooltip(status, note)
            if tip:
                self.combo_video.setItemData(
                    self.combo_video.count() - 1,
                    tip,
                    Qt.ItemDataRole.ToolTipRole,
                )
        if current:
            for i in range(self.combo_video.count()):
                if self.combo_video.itemData(i) == current:
                    self.combo_video.setCurrentIndex(i)
                    break
        self.combo_video.blockSignals(False)
        self._set_probe_counts(len(choices), unknown)
        self._sync_combo_error_tooltip()

    def _on_video_combo(self, _i: int) -> None:
        self._sync_combo_error_tooltip()
        if self._session_thread is None:
            return
        path = self.combo_video.currentData()
        if path:
            self._load_selected_video_package()
        else:
            self._cancel_session_load()

    def _on_project_replaced(self) -> None:
        """Drop the previous session; do not **session load** the new current video."""
        self._cancel_status_probes()
        self._status_by_path.clear()
        self.drop_session()
        self.refresh_video_list()
        self._maybe_start_status_probes()

    def _unknown_video_paths(self) -> List[str]:
        out: List[str] = []
        for i in range(1, self.combo_video.count()):
            path = str(self.combo_video.itemData(i) or "")
            if not path:
                continue
            status, _note = self._status_for_video(path)
            if status == STATUS_UNKNOWN:
                out.append(path)
        return out

    def _apply_combo_status(self, path: str, status: str) -> None:
        """Update combo text and error tooltip (failing path) for *path*."""
        name = self._video_names.get(path, Path(path).name)
        _status, note = self._status_for_video(path)
        tip = combo_item_tooltip(status, note)
        for i in range(1, self.combo_video.count()):
            if str(self.combo_video.itemData(i) or "") == path:
                self.combo_video.setItemText(i, combo_item_label(name, status))
                self.combo_video.setItemData(i, tip, Qt.ItemDataRole.ToolTipRole)
                self._sync_combo_error_tooltip()
                return

    def _sync_combo_error_tooltip(self) -> None:
        """Show the failing path on the closed combo when the current row errored."""
        path = str(self.combo_video.currentData() or "")
        if not path:
            self.combo_video.setToolTip("")
            return
        status, note = self._status_for_video(path)
        self.combo_video.setToolTip(combo_item_tooltip(status, note))

    def _browse_package(self) -> None:
        start = self.review_dir or self.project.project.root_dir or ""
        d = browse_existing_directory(
            self, start, "Select id_review / tracklets folder"
        )
        if d:
            self._package_video_path = ""
            self.load_package(d)

    def _load_selected_video_package(self) -> None:
        """Hunt the identity package off the UI thread, then session-load it.

        A successful hunt remembers ``detection_dir`` in memory (same as a
        probe) and does not mark the **project** dirty.
        """
        path = self.combo_video.currentData()
        if not path:
            self._browse_package()
            return
        self._cancel_session_load()
        gen = self._session_gen
        video_path = str(path)
        thread = ReviewSessionLoadThread(
            "",
            hunt_video=video_path,
            hunt_project=self.project.project,
            parent=self,
        )
        thread.hunt_ready.connect(
            lambda result, g=gen, vp=video_path: self._on_session_hunt_ready(
                result, g, vp
            )
        )
        thread.hunt_failed.connect(
            lambda msg, g=gen, vp=video_path: self._on_session_hunt_failed(
                msg, g, vp
            )
        )
        thread.finished.connect(lambda g=gen: self._on_session_thread_finished(g))
        self._session_thread = thread
        self.btn_cancel_load.setEnabled(True)
        self.status.setText("Finding identity package…")
        thread.start()

    def _on_session_hunt_ready(
        self, result: object, gen: int, video_path: str
    ) -> None:
        if gen != self._session_gen:
            return
        if not isinstance(result, ProbeResult) or not result.tracklets_dir:
            self._on_session_hunt_failed("", gen, video_path)
            return
        apply_probe_to_project(self.project.project, result)
        cur = self.project.current_video_path()
        if not paths_refer_to_same_video(cur, video_path):
            self.project.set_current_video(video_path, dirty=True)
        self._package_video_path = video_path
        self.load_package(result.tracklets_dir)

    def _on_session_hunt_failed(self, message: str, gen: int, video_path: str) -> None:
        if gen != self._session_gen:
            return
        self._session_thread = None
        self.btn_cancel_load.setEnabled(False)
        detail = (message or "").strip()
        body = (
            "No tracklets / id_review folder found for this video.\n"
            "Run Detect + track first, or Open package folder…\n\n"
            f"Video:\n{video_path}"
        )
        if detail:
            body = f"{body}\n\n{detail}"
        QMessageBox.warning(self, "Review IDs", body)

    def load_package(self, review_dir: str) -> bool:
        """Start a cancellable **session load** for an identity package.

        Offers migrate-to-raw when an uncorrected public pack has no raw
        snapshot. Declining leaves files unchanged and keeps switch edits
        locked. Package JSON unlocks the tab; pixels and outlines arrive
        independently off the UI thread. Not an analysis job-queue item.

        Args:
            review_dir: Path to the ``id_review`` package directory.

        Returns:
            True if the load was started; False if it failed before starting.
        """
        if needs_uncorrected_raw_migrate(review_dir):
            reply = QMessageBox.question(
                self,
                "Move tracklets into raw?",
                "This package has public tracklets but no raw snapshot. "
                "That can mean the data is in a bad state.\n\n"
                "Move the public tracklets into raw so switch edits can be "
                "rebuilt later? Declining leaves the files unchanged and "
                "keeps switch edits locked.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                migrate_uncorrected_public_to_raw(review_dir)
        if not Path(review_dir).is_dir():
            QMessageBox.warning(
                self, "Review IDs", f"Not a folder:\n{review_dir}"
            )
            self.setEnabled_controls(False)
            return False
        self._start_session_load(review_dir)
        return True

    def _start_session_load(
        self,
        review_dir: str,
        *,
        visible_kind: Optional[str] = None,
        skip_metadata: bool = False,
        skip_video: bool = False,
    ) -> None:
        """Replace any in-flight session load with a new one."""
        kind_only = bool(skip_metadata)
        self._cancel_session_load()
        if not kind_only:
            self._reset_session_viewer()
        gen = self._session_gen
        thread = ReviewSessionLoadThread(
            review_dir,
            visible_kind=visible_kind,
            video_hint=self._package_video_path or None,
            skip_metadata=skip_metadata,
            skip_video=skip_video,
            metadata=self._session_meta if kind_only else None,
            parent=self,
        )
        thread.metadata_ready.connect(
            lambda meta, g=gen: self._on_session_metadata(meta, g)
        )
        thread.kind_ready.connect(
            lambda kind, store, g=gen: self._on_session_kind(kind, store, g)
        )
        thread.frame_ready.connect(
            lambda frame, g=gen: self._on_session_frame(frame, g)
        )
        thread.kind_error.connect(
            lambda kind, msg, g=gen: self._on_session_kind_error(kind, msg, g)
        )
        thread.video_error.connect(
            lambda msg, g=gen: self._on_session_video_error(msg, g)
        )
        thread.load_error.connect(
            lambda msg, g=gen: self._on_session_load_error(msg, g)
        )
        thread.finished.connect(
            lambda g=gen: self._on_session_thread_finished(g)
        )
        self._session_thread = thread
        self.btn_cancel_load.setEnabled(True)
        if kind_only:
            self.status.setText(
                f"Loading outlines for {visible_kind or 'kind'}…"
            )
        else:
            self.lbl_pkg.setText(f"Loading: {review_dir}")
            self.status.setText("Loading package JSON…")
        thread.start()

    def _cancel_session_load(self) -> None:
        """Abort the in-flight session load without dropping loaded metadata."""
        self._session_gen += 1
        thread = self._session_thread
        if thread is not None:
            thread.cancel()
            thread.finished.connect(thread.deleteLater)
        self._session_thread = None
        self.btn_cancel_load.setEnabled(False)

    def _reset_session_viewer(self) -> None:
        """Clear capture and tracklets so a new package can replace this one."""
        self._stop_play()
        self._release_cap()
        self.review_dir = ""
        self.events = []
        self.markers = []
        self._saved_markers = []
        self._undo_stack.clear()
        self._stores.clear()
        self._baseline_stores.clear()
        self._stores_cloned = False
        self._already_corrected = False
        self._has_raw = False
        self._training_ranges = []
        self._range_start = None
        self._dirty = False
        self.frame = 0
        self.n_frames = 1
        self._pending_bgr = None
        self._session_canvas_error = ""
        self._session_store_meta = {}
        self._session_video_path = ""
        self._session_meta = None
        self.video_label.clear()
        self.status.setText("")
        self.markers_panel.set_markers([])
        self.subjects_table.set_subjects([])
        self.kind_combo.blockSignals(True)
        self.kind_combo.clear()
        self.kind_combo.blockSignals(False)
        self.setEnabled_controls(False)

    def _on_session_metadata(self, meta: object, gen: int) -> None:
        if gen != self._session_gen:
            return
        if not isinstance(meta, ReviewPackageMetadata):
            return
        self._apply_session_metadata(meta, gen)

    def _apply_session_metadata(self, meta: ReviewPackageMetadata, gen: int) -> None:
        """Unlock timeline / subjects / n_frames from package JSON."""
        if gen != self._session_gen:
            return
        self._session_meta = meta
        self.review_dir = meta.review_dir
        self.events = meta.events
        self.markers = meta.markers
        self._undo_stack.clear()
        self._already_corrected = meta.already_corrected
        self._has_raw = bool(meta.has_raw)
        self.animal_kind = meta.animal_kind
        self.n_frames = meta.n_frames
        info = meta.kind_info.get(self.animal_kind)
        self.involved_ids = list(info.ids) if info is not None else []
        self.fps = meta.fps
        self.frame = 0
        self._dirty = False
        self._saved_markers = clone_markers(self.markers)
        self._training_ranges = []
        self._range_start = None
        self._session_store_meta = dict(info.store_meta) if info is not None else {}
        self._session_video_path = str(meta.video_path or "")
        self._refresh_range_list()
        self._prefills_hard_case_out()

        self.kind_combo.blockSignals(True)
        self.kind_combo.clear()
        self.kind_combo.addItems(list(meta.kinds))
        self.kind_combo.setCurrentText(self.animal_kind)
        self.kind_combo.blockSignals(False)
        self._refresh_id_combos()

        self.subjects_table.set_subjects(meta.subjects)
        self.slider.setMaximum(max(0, self.n_frames - 1))
        self.setEnabled_controls(True)
        self._set_switch_edits_enabled(switch_edits_allowed(self.review_dir))
        self._set_package_label(accepted=meta.accepted, has_raw=self._has_raw)
        self._refresh_marker_list()
        self._update_undo_button()
        self._refresh_timeline()
        self.status.setText(
            f"Frame 0/{max(0, self.n_frames - 1)}  |  loading video and outlines…"
        )

    def _on_session_kind(self, kind: str, store: object, gen: int) -> None:
        if gen != self._session_gen:
            return
        self._stores[kind] = store
        self._baseline_stores[kind] = store
        self._stores_cloned = False
        if kind == self.animal_kind:
            self.involved_ids = list(getattr(store, "ids", []) or [])
            n_frames = int(getattr(store, "n_frames", 0) or 0)
            if n_frames:
                self.n_frames = max(1, n_frames)
            meta = dict(getattr(store, "meta", {}) or {})
            if meta:
                self._session_store_meta = meta
                self.fps = float(meta.get("fps") or self.fps)
            self._refresh_id_combos()
        self._session_canvas_error = ""
        self._render()
        self._refresh_timeline()

    def _on_session_frame(self, frame: object, gen: int) -> None:
        if gen != self._session_gen:
            return
        self._pending_bgr = frame
        if self.frame == 0:
            self._render()

    def _on_session_kind_error(self, kind: str, message: str, gen: int) -> None:
        if gen != self._session_gen:
            return
        self._session_canvas_error = (
            f"Could not load outlines for {kind}: {message}"
        )
        self._render()

    def _on_session_video_error(self, message: str, gen: int) -> None:
        if gen != self._session_gen:
            return
        self._session_canvas_error = message
        if self._pending_bgr is None:
            self._render()

    def _on_session_load_error(self, message: str, gen: int) -> None:
        if gen != self._session_gen:
            return
        QMessageBox.warning(self, "Review IDs", message)
        self.setEnabled_controls(False)
        self.lbl_pkg.setText("No package loaded.")
        self.status.setText(message)

    def _on_session_thread_finished(self, gen: int) -> None:
        if gen != self._session_gen:
            return
        self._session_thread = None
        self.btn_cancel_load.setEnabled(False)
        if self.review_dir:
            self._render()
        self.session_load_finished.emit(bool(self.review_dir))

    def _ensure_mutable_stores(self) -> None:
        """Clone packed stores on the first **switch marker** edit, not at load."""
        if self._stores_cloned:
            return
        for kind, store in list(self._baseline_stores.items()):
            self._stores[kind] = clone_store(store)
        self._stores_cloned = True

    def _on_kind(self, kind: str) -> None:
        if not kind:
            return
        self.animal_kind = kind
        info = None
        if self._session_meta is not None:
            info = self._session_meta.kind_info.get(kind)
        store = self._stores.get(kind)
        if store:
            self.n_frames = store.n_frames
            self.involved_ids = list(store.ids)
            self.fps = float(store.meta.get("fps") or self.fps)
            self._session_store_meta = dict(store.meta or {})
            self._refresh_id_combos()
            self._seek(min(self.frame, self.n_frames - 1))
            return
        if info is not None:
            self.n_frames = max(1, info.n_frames)
            self.involved_ids = list(info.ids)
            self.fps = float(info.fps or self.fps)
            self._session_store_meta = dict(info.store_meta)
            self._refresh_id_combos()
            self._refresh_timeline()
        if not self.review_dir:
            return
        self._start_session_load(
            self.review_dir,
            visible_kind=kind,
            skip_metadata=True,
            skip_video=True,
        )

    def has_unsaved_switch_markers(self) -> bool:
        """True when **switch markers** differ from the last load or save.

        Names/roles and proposed switch markers do not count.
        """
        return switch_markers_signature(self.markers) != switch_markers_signature(
            self._saved_markers
        )

    def save_switch_markers(self) -> bool:
        """Persist the current switch-marker list via Review IDs save."""
        return self.save_package(notify=False)

    def discard_unsaved_switch_markers(self) -> None:
        """Revert in-memory switch markers to the last load/save; do not write."""
        self.markers = clone_markers(self._saved_markers)
        self._undo_stack.clear()
        if self.review_dir:
            self._refresh_marker_list()
            self._refresh_timeline()
            self._render()
            self._update_undo_button()

    def drop_session(self) -> None:
        """Release capture and tracklet arrays; clear the viewer.

        Does not **session load** a video and does not mark the project dirty.
        """
        self._cancel_session_load()
        self._package_video_path = ""
        self._reset_session_viewer()
        self.lbl_pkg.setText("No package loaded.")
