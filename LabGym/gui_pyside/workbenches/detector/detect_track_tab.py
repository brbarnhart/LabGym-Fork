"""Detector → Detect + track subjects (batch, headless)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
    QHeaderView,
)

from LabGym.detection.batch_detect import (
    DetectTrackConfig,
    DetectTrackResult,
    detect_and_track_video,
    load_detector_animal_kinds,
)
from LabGym.gui_pyside.jobs.sequential_queue import (
    JobItem,
    JobProgress,
    SequentialJobQueue,
    summarize_job_statuses,
)
from LabGym.gui_pyside.model_paths import scan_detector_paths
from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.project.paths import list_project_video_choices
from LabGym.gui_pyside.widgets.path_browse import browse_existing_directory
from LabGym.gui_pyside.workbenches.detector.detect_track_progress import (
    DetectTrackProgressDialog,
)


class DetectTrackTab(QWidget):
    """Batch detect+track project videos into identity packages."""

    request_edit_project = Signal()
    request_review_ids = Signal()
    batch_finished = Signal()

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self.queue = SequentialJobQueue(self)
        self.queue.job_started.connect(self._on_job_started)
        self.queue.job_progress.connect(self._on_progress)
        self.queue.job_frame_progress.connect(self._on_frame_progress)
        self.queue.job_finished.connect(self._on_job_done)
        self.queue.job_failed.connect(self._on_job_fail)
        self.queue.queue_finished.connect(self._on_queue_done)
        # job_id is the video path; maps to table row for the current batch
        self._job_rows: Dict[str, int] = {}
        # path -> (status, note) so post-batch project.changed refreshes keep results
        self._status_by_path: Dict[str, Tuple[str, str]] = {}
        self._batch_active = False
        self._progress_dlg: Optional[DetectTrackProgressDialog] = None

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Run the LabGym detector to <b>detect and track</b> animals on project "
            "videos. Writes identity packages (<code>id_review/</code> with tracklets "
            "+ contact-risk events) for later Review IDs. One video at a time."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(intro)

        # Detector
        det_box = QGroupBox("Detector")
        det_form = QFormLayout(det_box)
        self.ed_detector = QComboBox()
        self.ed_detector.setEditable(True)
        self.ed_detector.setMinimumWidth(280)
        self.ed_detector.setToolTip(
            "Folder of a trained LabGym detector (contains model_final.pth, "
            "config.yaml, model_parameters.txt). Animal category names are read "
            "from model_parameters.txt."
        )
        btn_browse = QPushButton("Browse…")
        btn_browse.setToolTip("Choose a detector folder on disk.")
        btn_browse.clicked.connect(self._browse_detector)
        btn_scan = QPushButton("Scan models folder")
        btn_scan.setToolTip(
            "Search the project models folder and LabGym’s bundled detectors "
            "for folders that contain model_parameters.txt."
        )
        btn_scan.clicked.connect(self._scan_detectors)
        row = QHBoxLayout()
        row.addWidget(self.ed_detector, 1)
        row.addWidget(btn_browse)
        row.addWidget(btn_scan)
        det_form.addRow(
            self._lab(
                "Detector folder:",
                "Path to a trained LabGym Mask R-CNN detector used to find and "
                "track animals in each video.",
            ),
            row,
        )
        self.lbl_kinds = QLabel("—")
        self.lbl_kinds.setToolTip(
            "Animal/object category names defined inside the selected detector."
        )
        det_form.addRow(
            self._lab(
                "Animal kinds:",
                "Categories the detector was trained to find (e.g. mouse). "
                "Filled automatically when a valid detector is selected.",
            ),
            self.lbl_kinds,
        )
        self.ed_detector.currentTextChanged.connect(self._on_detector_changed)
        layout.addWidget(det_box)

        # Params
        p_box = QGroupBox("Tracking parameters")
        p_box.setToolTip(
            "Controls how far/how long tracking runs and how identity packages "
            "are exported for later ID review."
        )
        p_form = QFormLayout(p_box)
        self.spin_animals = QSpinBox()
        self.spin_animals.setRange(1, 50)
        self.spin_animals.setValue(2)
        tip_animals = (
            "How many individuals of each animal kind are expected in the video "
            "(same count applied to every kind). Used to allocate track IDs "
            "(e.g. 2 mice → IDs 0 and 1). Set this to the typical number of "
            "animals you want tracked."
        )
        self.spin_animals.setToolTip(tip_animals)
        p_form.addRow(self._lab("Animals per kind:", tip_animals), self.spin_animals)

        self.combo_mode = QComboBox()
        self.combo_mode.addItem("0 — Non-interactive", 0)
        self.combo_mode.addItem("2 — Interactive advanced (tracking)", 2)
        tip_mode = (
            "LabGym behavior/tracking geometry:\n"
            "• 0 Non-interactive — track each animal alone (standard multi-animal "
            "tracking; recommended for most ID-review work).\n"
            "• 2 Interactive advanced — tracking suited to social/interaction "
            "setups (main animal + nearby others / costars).\n"
            "Mode 1 (interactive basic) is not used here for batch package export."
        )
        self.combo_mode.setToolTip(tip_mode)
        p_form.addRow(self._lab("Behavior mode:", tip_mode), self.combo_mode)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 32)
        self.spin_batch.setValue(1)
        tip_batch = (
            "How many frames the detector processes together (GPU batch size). "
            "Higher values can be faster on a strong GPU but use more VRAM. "
            "Use 1 if you run out of memory or see CUDA OOM errors."
        )
        self.spin_batch.setToolTip(tip_batch)
        p_form.addRow(self._lab("Detector batch size:", tip_batch), self.spin_batch)

        self.spin_length = QSpinBox()
        self.spin_length.setRange(1, 200)
        self.spin_length.setValue(15)
        tip_length = (
            "History length (frames) — LabGym’s internal “length” / time window "
            "used while building track data (pattern/animation history buffers).\n\n"
            "• Not the length of the whole video.\n"
            "• Think of it as “how many recent frames the tracker keeps in memory "
            "for each animal when crafting tracks.”\n"
            "• Default 15 matches common LabGym settings. Larger can help "
            "smooth short gaps but uses more memory and is slower.\n"
            "• When you later generate training examples from ethograms, that "
            "window length is chosen separately in Generate examples."
        )
        self.spin_length.setToolTip(tip_length)
        p_form.addRow(
            self._lab("History length (frames):", tip_length), self.spin_length
        )

        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.0, 1e7)
        self.spin_duration.setValue(0.0)
        self.spin_duration.setSpecialValueText("full video")
        tip_duration = (
            "How many seconds of video to analyze after the start time. "
            "0 (or “full video”) processes from the start time through the end "
            "of the file. Use a shorter duration for quick tests."
        )
        self.spin_duration.setToolTip(tip_duration)
        p_form.addRow(
            self._lab("Duration (seconds):", tip_duration), self.spin_duration
        )

        self.spin_t = QDoubleSpinBox()
        self.spin_t.setRange(0.0, 1e7)
        self.spin_t.setValue(0.0)
        tip_t = (
            "Start time in seconds from the beginning of the video. "
            "0 starts at the first frame. Use this to skip an intro or to "
            "analyze only a later segment (together with Duration)."
        )
        self.spin_t.setToolTip(tip_t)
        p_form.addRow(self._lab("Start time (s):", tip_t), self.spin_t)

        self.spin_fw = QSpinBox()
        self.spin_fw.setRange(0, 4000)
        self.spin_fw.setValue(0)
        self.spin_fw.setSpecialValueText("original")
        tip_fw = (
            "Frame width (resize) — if set (e.g. 480), every frame is scaled so "
            "its width is this many pixels, keeping aspect ratio (height is "
            "scaled to match).\n\n"
            "• “original” / 0 = do not resize; use native resolution.\n"
            "• Smaller widths make detection much faster and use less memory, "
            "which LabGym strongly recommends for large videos.\n"
            "• Too small can hurt detection of tiny animals; too large is slow.\n"
            "• 480 is a common LabGym default when resizing."
        )
        self.spin_fw.setToolTip(tip_fw)
        p_form.addRow(self._lab("Frame width (resize):", tip_fw), self.spin_fw)

        self.chk_export = QCheckBox("Export id_review package (tracklets + contact risk)")
        self.chk_export.setChecked(True)
        tip_export = (
            "When checked (recommended), writes an identity package under "
            "detection/<video_stem>/id_review/ with tracklets, contact-risk "
            "events, and default subjects.json for Review IDs and annotation. "
            "Uncheck only if you only want intermediate analysis folders."
        )
        self.chk_export.setToolTip(tip_export)
        p_form.addRow(self.chk_export)

        self.spin_contact = QDoubleSpinBox()
        self.spin_contact.setRange(0.1, 20.0)
        self.spin_contact.setValue(1.0)
        tip_contact = (
            "Contact-risk distance threshold as a multiple of typical animal "
            "size. When two animals come within about (factor × body size), a "
            "risk band is marked on the Review IDs timeline for possible ID "
            "swaps.\n\n"
            "• Higher → fewer, only closer contacts flagged.\n"
            "• Lower → more risk bands (more candidate swap regions).\n"
            "Default 1.0 is a reasonable starting point."
        )
        self.spin_contact.setToolTip(tip_contact)
        p_form.addRow(
            self._lab("Contact distance × size:", tip_contact), self.spin_contact
        )

        layout.addWidget(p_box)

        # Video table
        v_box = QGroupBox("Videos to process")
        v_l = QVBoxLayout(v_box)
        row2 = QHBoxLayout()
        btn_refresh = QPushButton("Refresh from project")
        btn_refresh.clicked.connect(self._manual_refresh_videos)
        btn_edit = QPushButton("Edit project…")
        btn_edit.clicked.connect(self.request_edit_project.emit)
        btn_all = QPushButton("Select all")
        btn_all.clicked.connect(lambda: self._set_all_checked(True))
        btn_none = QPushButton("Select none")
        btn_none.clicked.connect(lambda: self._set_all_checked(False))
        row2.addWidget(btn_refresh)
        row2.addWidget(btn_edit)
        row2.addWidget(btn_all)
        row2.addWidget(btn_none)
        row2.addStretch(1)
        v_l.addLayout(row2)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["", "Video", "Status", "id_review / note"])
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setColumnWidth(0, 36)
        v_l.addWidget(self.table)
        layout.addWidget(v_box, 1)

        # Run
        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Run detect + track on selected")
        self.btn_run.clicked.connect(self._start_batch)
        self.btn_cancel = QPushButton("Cancel queue")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.queue.cancel)
        self.btn_review = QPushButton("Go to Review IDs…")
        self.btn_review.clicked.connect(self.request_review_ids.emit)
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_cancel)
        run_row.addWidget(self.btn_review)
        run_row.addStretch(1)
        layout.addLayout(run_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(120)
        layout.addWidget(self.log)

        # Full rebuild / defaults only on project replace/open/edit — not every dirty/save.
        self.project.project_replaced.connect(self._on_project_replaced)
        self._on_project_replaced()

    @staticmethod
    def _lab(text: str, tip: str) -> QLabel:
        lab = QLabel(text)
        lab.setToolTip(tip)
        return lab

    def _on_project_replaced(self) -> None:
        self._init_detector_defaults()
        self._apply_project_defaults()
        self.refresh_videos()

    def _apply_project_defaults(self) -> None:
        """Sync Edit Project defaults into tracking params."""
        d = self.project.project.defaults
        mode = int(d.behavior_mode)
        # This tab only offers 0 and 2; map interactive-basic (1) → 0.
        if mode == 1:
            mode = 0
        idx = self.combo_mode.findData(mode)
        if idx >= 0:
            self.combo_mode.setCurrentIndex(idx)
        self.spin_length.setValue(max(1, int(d.window_length or 15)))

    def _init_detector_defaults(self) -> None:
        p = self.project.project
        cur = (p.defaults.detector_name or self.ed_detector.currentText() or "").strip()
        self.ed_detector.blockSignals(True)
        self.ed_detector.clear()
        for s in scan_detector_paths(p):
            self.ed_detector.addItem(s)
        if cur:
            self.ed_detector.setEditText(cur)
        elif self.ed_detector.count():
            self.ed_detector.setCurrentIndex(0)
        self.ed_detector.blockSignals(False)
        self._on_detector_changed(self.ed_detector.currentText())

    def _scan_detectors(self) -> None:
        cur = self.ed_detector.currentText()
        self.ed_detector.clear()
        for s in scan_detector_paths(self.project.project):
            self.ed_detector.addItem(s)
        if cur:
            self.ed_detector.setEditText(cur)
        self._on_detector_changed(self.ed_detector.currentText())

    def _browse_detector(self) -> None:
        start = self.ed_detector.currentText() or self.project.project.root_dir or ""
        d = browse_existing_directory(self, start, "Select detector folder")
        if d:
            self.ed_detector.setEditText(d)
            self._on_detector_changed(d)

    def _on_detector_changed(self, path: str) -> None:
        path = (path or "").strip()
        if path and Path(path).is_dir():
            try:
                kinds = load_detector_animal_kinds(path)
                self.lbl_kinds.setText(", ".join(kinds))
            except Exception as exc:
                self.lbl_kinds.setText(f"(error: {exc})")
        else:
            self.lbl_kinds.setText("—")

    def _manual_refresh_videos(self) -> None:
        from LabGym.gui_pyside.project.paths import clear_tracklets_discovery_cache

        clear_tracklets_discovery_cache()
        self.refresh_videos()

    def refresh_videos(self) -> None:
        # Mid-batch project.changed (mark_dirty after each finished video) must not
        # rebuild the table — that was wiping live status. Keep sticky statuses for
        # post-batch rebuilds via _status_by_path.
        if self._batch_active:
            return

        prev_checked: Dict[str, bool] = {}
        for r in range(self.table.rowCount()):
            item0 = self.table.item(r, 0)
            if not item0:
                continue
            path = str(item0.data(Qt.ItemDataRole.UserRole) or "")
            if path:
                prev_checked[path] = item0.checkState() == Qt.CheckState.Checked

        choices = list_project_video_choices(self.project.project)
        self.table.setRowCount(0)
        self.table.setRowCount(len(choices))
        for r, (label, resolved) in enumerate(choices):
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
            self.table.setItem(r, 0, chk)
            self.table.setItem(r, 1, QTableWidgetItem(label))
            status, note = self._status_by_path.get(path, ("pending", ""))
            self.table.setItem(r, 2, QTableWidgetItem(status))
            self.table.setItem(r, 3, QTableWidgetItem(note))

    def _set_status(self, row: int, path: str, status: str, note: str = "") -> None:
        self._status_by_path[path] = (status, note)
        if self.table.item(row, 2):
            self.table.item(row, 2).setText(status)
        else:
            self.table.setItem(row, 2, QTableWidgetItem(status))
        if self.table.item(row, 3):
            self.table.item(row, 3).setText(note)
        else:
            self.table.setItem(row, 3, QTableWidgetItem(note))

    def _set_all_checked(self, on: bool) -> None:
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item:
                item.setCheckState(state)

    def _selected_videos(self) -> List[str]:
        out = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item and item.checkState() == Qt.CheckState.Checked:
                path = item.data(Qt.ItemDataRole.UserRole)
                if path:
                    out.append(str(path))
        return out

    def _results_root(self) -> Path:
        p = self.project.project
        rel = p.paths.detection_output_root.strip() or "detection"
        if p.root_dir:
            return p.resolve_path(rel)
        # fall back beside first video
        vids = self._selected_videos()
        if vids:
            return Path(vids[0]).parent / rel
        return Path(rel)

    def _start_batch(self) -> None:
        if self.queue.is_running:
            QMessageBox.information(self, "Busy", "A batch is already running.")
            return
        detector = self.ed_detector.currentText().strip()
        if not detector or not Path(detector).is_dir():
            QMessageBox.warning(self, "Detect + track", "Select a valid detector folder.")
            return
        videos = self._selected_videos()
        if not videos:
            QMessageBox.warning(
                self, "Detect + track", "Select at least one project video."
            )
            return

        # Block table rebuilds for the whole batch (including mark_dirty below).
        self._batch_active = True

        # Persist detector choice on project
        self.project.project.defaults.detector_name = detector
        self.project.mark_dirty()

        results_root = self._results_root()
        results_root.mkdir(parents=True, exist_ok=True)

        items: List[JobItem] = []
        self._job_rows.clear()
        for r in range(self.table.rowCount()):
            item0 = self.table.item(r, 0)
            if not item0 or item0.checkState() != Qt.CheckState.Checked:
                continue
            path = str(item0.data(Qt.ItemDataRole.UserRole))
            # Stable id = path (not row index).
            label = Path(path).name
            self._job_rows[path] = r
            self._set_status(r, path, "queued", "")
            items.append(JobItem(job_id=path, label=label, payload=path))

        mode = int(self.combo_mode.currentData())
        n_per = int(self.spin_animals.value())
        fw = int(self.spin_fw.value()) or None

        def runner(job: JobItem, prog: JobProgress) -> DetectTrackResult:
            cfg = DetectTrackConfig(
                video_path=str(job.payload),
                detector_path=detector,
                results_root=str(results_root),
                animal_number={"_all": n_per},  # resolved per kind in batch_detect
                behavior_mode=mode,
                framewidth=fw,
                t=float(self.spin_t.value()),
                duration=float(self.spin_duration.value()),
                length=int(self.spin_length.value()),
                detector_batch=int(self.spin_batch.value()),
                export_id_review=self.chk_export.isChecked(),
                contact_distance_factor=float(self.spin_contact.value()),
            )
            # Expand animal_number to each kind
            try:
                kinds = load_detector_animal_kinds(detector)
                cfg.animal_kinds = kinds
                cfg.animal_number = {k: n_per for k in kinds}
            except Exception:
                pass
            return detect_and_track_video(cfg, progress=prog)

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.log.append(f"Starting batch: {len(items)} video(s) → {results_root}")
        dlg = self._ensure_progress_dialog()
        dlg.begin_batch(len(items))
        self.queue.start(items, runner)

    def _ensure_progress_dialog(self) -> DetectTrackProgressDialog:
        if self._progress_dlg is None:
            self._progress_dlg = DetectTrackProgressDialog(self)
            self._progress_dlg.cancel_requested.connect(self.queue.cancel)
        return self._progress_dlg

    def _job_index(self, job_id: str) -> int:
        for i, it in enumerate(self.queue.items):
            if it.job_id == job_id:
                return i
        return 0

    def _on_job_started(self, job_id: str) -> None:
        row = self._job_rows.get(job_id)
        label = Path(job_id).name
        if row is not None:
            self._set_status(row, job_id, "running", "Starting…")
        dlg = self._progress_dlg
        if dlg is not None:
            dlg.set_current_video(label, self._job_index(job_id))

    def _on_progress(self, job_id: str, msg: str) -> None:
        row = self._job_rows.get(job_id)
        if row is not None:
            self._set_status(row, job_id, "running", msg[:200])
        if self._progress_dlg is not None:
            self._progress_dlg.set_status_message(msg[:300])
        self.log.append(f"[{Path(job_id).name}] {msg}")

    def _on_frame_progress(self, job_id: str, current: int, total: int) -> None:
        row = self._job_rows.get(job_id)
        note = f"frames {current}/{total}" if total else f"frames {current}"
        if row is not None:
            self._set_status(row, job_id, "running", note)
        if self._progress_dlg is not None:
            self._progress_dlg.set_frame_progress(current, total)
            self._progress_dlg.set_status_message(note)

    def _on_job_done(self, job_id: str, result: object) -> None:
        # Queue only emits finished for non-soft-failures (ok is not False).
        row = self._job_rows.get(job_id)
        if self._progress_dlg is not None:
            self._progress_dlg.mark_file_finished()

        if not isinstance(result, DetectTrackResult):
            if row is not None:
                self._set_status(row, job_id, "done", "finished")
            return
        path = job_id or result.video_path
        note = result.id_review_dir or result.results_path or "finished"
        if row is not None:
            self._set_status(row, path, "done", note)
        if result.ok:
            self._register_detection_dir(result)
        self.log.append(
            f"[{Path(job_id).name}] OK → {result.id_review_dir} "
            f"({result.n_events} risk events)"
        )

    def _on_job_fail(self, job_id: str, error: str) -> None:
        row = self._job_rows.get(job_id)
        if self._progress_dlg is not None:
            self._progress_dlg.mark_file_finished()
        if row is not None:
            self._set_status(row, job_id, "error", error[:200])
        self.log.append(f"[{Path(job_id).name}] FAIL: {error}")

    def _register_detection_dir(self, result: DetectTrackResult) -> None:
        """Store detection_dir on the matching project video entry."""
        if not result.ok or not result.id_review_dir:
            return
        from LabGym.gui_pyside.project.paths import find_video_entry

        entry = find_video_entry(self.project.project, result.video_path)
        if entry is None:
            return
        root = self.project.project.root_dir
        det = result.id_review_dir
        try:
            if root:
                det = str(
                    Path(result.id_review_dir).resolve().relative_to(
                        Path(root).resolve()
                    )
                )
        except (ValueError, OSError):
            det = result.id_review_dir
        if entry.detection_dir != det:
            entry.detection_dir = det
            self.project.mark_dirty()

    def _on_queue_done(self) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._batch_active = False
        self.log.append("Batch finished.")
        self.batch_finished.emit()
        if self._progress_dlg is not None:
            self._progress_dlg.finish_batch()

        for it in self.queue.items:
            if it.status == "cancelled":
                row = self._job_rows.get(it.job_id)
                if row is not None:
                    self._set_status(row, it.job_id, "cancelled", "")

        # Sync table with any project video-list changes deferred during the batch.
        self.refresh_videos()
        n_ok, n_err, _n_cancel = summarize_job_statuses(self.queue.items)

        QMessageBox.information(
            self,
            "Detect + track",
            f"Batch finished.\n\nSucceeded: {n_ok}\nFailed: {n_err}\n\n"
            "Open Detector → Review IDs to fix swaps and assign names.",
        )
