"""Detector → Detect + track subjects (batch, headless)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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
    list_detectors,
    load_detector_animal_kinds,
)
from LabGym.gui_pyside.jobs.sequential_queue import JobItem, SequentialJobQueue
from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.project.paths import list_project_video_choices
from LabGym.mypkg_resources import resource_filename


class DetectTrackTab(QWidget):
    """Batch detect+track project videos into identity packages."""

    request_edit_project = Signal()
    request_review_ids = Signal()
    batch_finished = Signal()

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self.queue = SequentialJobQueue(self)
        self.queue.job_progress.connect(self._on_progress)
        self.queue.job_finished.connect(self._on_job_done)
        self.queue.job_failed.connect(self._on_job_fail)
        self.queue.queue_finished.connect(self._on_queue_done)
        self._job_rows: Dict[str, int] = {}

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
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_detector)
        btn_scan = QPushButton("Scan models folder")
        btn_scan.clicked.connect(self._scan_detectors)
        row = QHBoxLayout()
        row.addWidget(self.ed_detector, 1)
        row.addWidget(btn_browse)
        row.addWidget(btn_scan)
        det_form.addRow("Detector folder:", row)
        self.lbl_kinds = QLabel("—")
        det_form.addRow("Animal kinds:", self.lbl_kinds)
        self.ed_detector.currentTextChanged.connect(self._on_detector_changed)
        layout.addWidget(det_box)

        # Params
        p_box = QGroupBox("Tracking parameters")
        p_form = QFormLayout(p_box)
        self.spin_animals = QSpinBox()
        self.spin_animals.setRange(1, 50)
        self.spin_animals.setValue(2)
        self.spin_animals.setToolTip(
            "Number of individuals of each animal kind (same count for every kind)"
        )
        p_form.addRow("Animals per kind:", self.spin_animals)

        self.combo_mode = QComboBox()
        self.combo_mode.addItem("0 — Non-interactive", 0)
        self.combo_mode.addItem("2 — Interactive advanced (tracking)", 2)
        p_form.addRow("Behavior mode:", self.combo_mode)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 32)
        self.spin_batch.setValue(1)
        p_form.addRow("Detector batch size:", self.spin_batch)

        self.spin_length = QSpinBox()
        self.spin_length.setRange(1, 200)
        self.spin_length.setValue(15)
        p_form.addRow("History length (frames):", self.spin_length)

        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.0, 1e7)
        self.spin_duration.setValue(0.0)
        self.spin_duration.setSpecialValueText("full video")
        self.spin_duration.setToolTip("0 = analyze entire video")
        p_form.addRow("Duration (seconds):", self.spin_duration)

        self.spin_t = QDoubleSpinBox()
        self.spin_t.setRange(0.0, 1e7)
        self.spin_t.setValue(0.0)
        p_form.addRow("Start time (s):", self.spin_t)

        self.spin_fw = QSpinBox()
        self.spin_fw.setRange(0, 4000)
        self.spin_fw.setValue(0)
        self.spin_fw.setSpecialValueText("original")
        p_form.addRow("Frame width (resize):", self.spin_fw)

        self.chk_export = QCheckBox("Export id_review package (tracklets + contact risk)")
        self.chk_export.setChecked(True)
        p_form.addRow(self.chk_export)

        self.spin_contact = QDoubleSpinBox()
        self.spin_contact.setRange(0.1, 20.0)
        self.spin_contact.setValue(1.0)
        p_form.addRow("Contact distance × size:", self.spin_contact)

        layout.addWidget(p_box)

        # Video table
        v_box = QGroupBox("Videos to process")
        v_l = QVBoxLayout(v_box)
        row2 = QHBoxLayout()
        btn_refresh = QPushButton("Refresh from project")
        btn_refresh.clicked.connect(self.refresh_videos)
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

        self.project.changed.connect(self.refresh_videos)
        self.project.project_replaced.connect(self.refresh_videos)
        self._init_detector_defaults()
        self.refresh_videos()

    def _init_detector_defaults(self) -> None:
        # Prefer project models root, then bundled detectors
        roots = []
        p = self.project.project
        if p.root_dir:
            roots.append(p.resolve_path(p.paths.models_root))
        if p.defaults.detector_name:
            self.ed_detector.setEditText(p.defaults.detector_name)
        try:
            bundled = Path(resource_filename("LabGym", "detectors"))
            if bundled.is_dir():
                roots.append(bundled)
        except Exception:
            pass
        for root in roots:
            for d in list_detectors(root):
                self.ed_detector.addItem(str(d))
        if self.ed_detector.count() and not self.ed_detector.currentText():
            self.ed_detector.setCurrentIndex(0)
        self._on_detector_changed(self.ed_detector.currentText())

    def _scan_detectors(self) -> None:
        p = self.project.project
        roots = []
        if p.root_dir:
            roots.append(p.resolve_path(p.paths.models_root))
        try:
            bundled = Path(resource_filename("LabGym", "detectors"))
            roots.append(bundled)
        except Exception:
            pass
        cur = self.ed_detector.currentText()
        self.ed_detector.clear()
        seen = set()
        for root in roots:
            for d in list_detectors(root):
                s = str(d)
                if s not in seen:
                    seen.add(s)
                    self.ed_detector.addItem(s)
        if cur:
            self.ed_detector.setEditText(cur)
        self._on_detector_changed(self.ed_detector.currentText())

    def _browse_detector(self) -> None:
        start = self.ed_detector.currentText() or self.project.project.root_dir or ""
        d = QFileDialog.getExistingDirectory(self, "Select detector folder", start)
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

    def refresh_videos(self) -> None:
        choices = list_project_video_choices(self.project.project)
        self.table.setRowCount(0)
        self.table.setRowCount(len(choices))
        for r, (label, resolved) in enumerate(choices):
            chk = QTableWidgetItem()
            chk.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            chk.setCheckState(Qt.CheckState.Checked)
            chk.setData(Qt.ItemDataRole.UserRole, resolved)
            self.table.setItem(r, 0, chk)
            self.table.setItem(r, 1, QTableWidgetItem(label))
            self.table.setItem(r, 2, QTableWidgetItem("pending"))
            self.table.setItem(r, 3, QTableWidgetItem(""))

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
            jid = f"v{r}"
            self._job_rows[jid] = r
            self.table.item(r, 2).setText("queued")
            self.table.item(r, 3).setText("")
            items.append(JobItem(job_id=jid, label=Path(path).name, payload=path))

        mode = int(self.combo_mode.currentData())
        n_per = int(self.spin_animals.value())
        fw = int(self.spin_fw.value()) or None

        def runner(job: JobItem, prog) -> DetectTrackResult:
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
        self.queue.start(items, runner)

    def _on_progress(self, job_id: str, msg: str) -> None:
        row = self._job_rows.get(job_id)
        if row is not None and self.table.item(row, 2):
            self.table.item(row, 2).setText("running")
            self.table.item(row, 3).setText(msg[:200])
        self.log.append(f"[{job_id}] {msg}")

    def _on_job_done(self, job_id: str, result: object) -> None:
        row = self._job_rows.get(job_id)
        if not isinstance(result, DetectTrackResult):
            return
        if row is not None:
            if result.ok:
                self.table.item(row, 2).setText("done")
                self.table.item(row, 3).setText(result.id_review_dir or result.results_path)
                self._register_detection_dir(result)
            else:
                self.table.item(row, 2).setText("error")
                self.table.item(row, 3).setText(result.error[:200])
        self.log.append(
            f"[{job_id}] "
            + (
                f"OK → {result.id_review_dir} ({result.n_events} risk events)"
                if result.ok
                else f"FAIL: {result.error}"
            )
        )

    def _on_job_fail(self, job_id: str, error: str) -> None:
        row = self._job_rows.get(job_id)
        if row is not None:
            self.table.item(row, 2).setText("error")
            self.table.item(row, 3).setText(error[:200])
        self.log.append(f"[{job_id}] FAIL: {error}")

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
        self.log.append("Batch finished.")
        self.batch_finished.emit()
        n_ok = sum(1 for it in self.queue.items if it.status == "done")
        n_err = sum(1 for it in self.queue.items if it.status == "error")
        QMessageBox.information(
            self,
            "Detect + track",
            f"Batch finished.\n\nSucceeded: {n_ok}\nFailed: {n_err}\n\n"
            "Open Detector → Review IDs to fix swaps and assign names.",
        )
