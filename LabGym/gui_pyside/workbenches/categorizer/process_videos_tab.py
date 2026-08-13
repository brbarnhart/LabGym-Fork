"""Categorizer → Process videos (categorize accepted identities)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from LabGym.analysis.process_videos import (
    ProcessVideoConfig,
    ProcessVideoResult,
    load_categorizer_metadata,
    process_video,
)
from LabGym.gui_pyside.jobs.sequential_queue import (
    JobItem,
    JobProgress,
    SequentialJobQueue,
    summarize_job_statuses,
)
from LabGym.gui_pyside.model_paths import scan_categorizer_paths
from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.project.paths import (
    discover_tracklets_dir,
    list_project_video_choices,
)
from LabGym.gui_pyside.widgets.path_browse import (
    path_edit_row,
    set_line_edit_directory,
)


class ProcessVideosTab(QWidget):
    """Batch-categorize project videos from accepted remapped tracklets."""

    request_edit_project = Signal()
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
        # job_id is the video path
        self._job_rows: Dict[str, int] = {}
        # path -> (status, note) survives post-batch project.changed rebuilds
        self._status_by_path: Dict[str, Tuple[str, str]] = {}
        self._batch_active = False

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Process project videos that already have <b>accepted identities</b> "
            "(Detector → Review IDs → Save). Rebuilds categorizer inputs from "
            "those remapped outlines — it does <b>not</b> run the detector. "
            "One video at a time."
        ))

        # Models
        mbox = QGroupBox("Models")
        mform = QFormLayout(mbox)
        self.ed_categorizer = QLineEdit()
        self.ed_categorizer.setToolTip(
            "Trained LabGym categorizer folder (Keras model + model_parameters.txt "
            "with classnames, time_step, network type, etc.). Behavior names and "
            "input sizes are loaded automatically from that file."
        )
        b_c = QPushButton("Browse…")
        b_c.clicked.connect(
            lambda: set_line_edit_directory(
                self, self.ed_categorizer, caption="Select categorizer folder"
            )
        )
        mform.addRow(
            self._lab(
                "Categorizer:",
                "Model that assigns behavior labels over time for each tracked animal.",
            ),
            path_edit_row(self.ed_categorizer, b_c),
        )
        self.lbl_behaviors = QLabel("—")
        self.lbl_behaviors.setToolTip(
            "Behavior class names stored in the categorizer’s model_parameters.txt."
        )
        mform.addRow(
            self._lab("Behaviors:", "Classes this categorizer can predict."),
            self.lbl_behaviors,
        )
        self.ed_categorizer.textChanged.connect(self._on_categorizer_changed)

        btn_scan = QPushButton("Scan project / bundled models")
        btn_scan.setToolTip(
            "Search the project models folder and LabGym’s bundled detectors/models "
            "directories for available detectors and categorizers."
        )
        btn_scan.clicked.connect(self._scan_models)
        mform.addRow(btn_scan)
        layout.addWidget(mbox)

        # Params
        pbox = QGroupBox("Analysis parameters")
        pbox.setToolTip(
            "How long to analyze, at what resolution, and whether to reuse ID fixes "
            "from Review IDs before labeling behaviors."
        )
        pform = QFormLayout(pbox)
        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.0, 1e7)
        self.spin_duration.setValue(0.0)
        self.spin_duration.setSpecialValueText("full video")
        tip_dur = (
            "Seconds of video to analyze after the start time. 0 / “full video” = "
            "from start time to end of file."
        )
        self.spin_duration.setToolTip(tip_dur)
        pform.addRow(self._lab("Duration (s):", tip_dur), self.spin_duration)

        self.spin_t = QDoubleSpinBox()
        self.spin_t.setRange(0.0, 1e7)
        tip_t = "Start time in seconds from the beginning of the video (0 = first frame)."
        self.spin_t.setToolTip(tip_t)
        pform.addRow(self._lab("Start time (s):", tip_t), self.spin_t)

        self.spin_fw = QSpinBox()
        self.spin_fw.setRange(0, 4000)
        self.spin_fw.setSpecialValueText("original")
        tip_fw = (
            "Resize each frame to this width in pixels (height scales to keep aspect "
            "ratio). “original”/0 = no resize. Smaller (e.g. 480) is much faster; "
            "must match what you used for training/detection when possible."
        )
        self.spin_fw.setToolTip(tip_fw)
        pform.addRow(self._lab("Frame width:", tip_fw), self.spin_fw)

        self.spin_uncertain = QDoubleSpinBox()
        self.spin_uncertain.setRange(0.0, 1.0)
        self.spin_uncertain.setSingleStep(0.05)
        self.spin_uncertain.setValue(0.0)
        tip_unc = (
            "Minimum gap between the top and second-best behavior probabilities "
            "required to accept a label. 0 = always take the top class. Higher "
            "values produce more “NA” (uncertain) labels when the model is unsure."
        )
        self.spin_uncertain.setToolTip(tip_unc)
        pform.addRow(
            self._lab("Uncertainty threshold:", tip_unc), self.spin_uncertain
        )

        self.chk_legend = QCheckBox("Show legend on annotated video")
        self.chk_legend.setChecked(True)
        self.chk_legend.setToolTip(
            "Draw a behavior-name color legend on the exported annotated video."
        )
        pform.addRow(self.chk_legend)

        self.ed_out = QLineEdit()
        self.ed_out.setToolTip(
            "Parent folder for analysis outputs. LabGym creates a subfolder per "
            "video stem (annotated video, spreadsheets, etc.)."
        )
        b_o = QPushButton("Browse…")
        b_o.clicked.connect(
            lambda: set_line_edit_directory(
                self, self.ed_out, caption="Select results root folder"
            )
        )
        pform.addRow(
            self._lab(
                "Results root:",
                "Where per-video analysis folders are written (default: project "
                "analysis/).",
            ),
            path_edit_row(self.ed_out, b_o),
        )
        layout.addWidget(pbox)

        # Videos
        vbox = QGroupBox("Videos")
        vl = QVBoxLayout(vbox)
        row = QHBoxLayout()
        btn_ref = QPushButton("Refresh from project")
        btn_ref.clicked.connect(self._manual_refresh_videos)
        btn_edit = QPushButton("Edit project…")
        btn_edit.clicked.connect(self.request_edit_project.emit)
        btn_all = QPushButton("Select all")
        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none = QPushButton("Select none")
        btn_none.clicked.connect(lambda: self._set_all(False))
        for b in (btn_ref, btn_edit, btn_all, btn_none):
            row.addWidget(b)
        row.addStretch(1)
        vl.addLayout(row)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["", "Video", "Status", "Accepted identities"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setColumnWidth(0, 36)
        vl.addWidget(self.table)
        layout.addWidget(vbox, 1)

        run_row = QHBoxLayout()
        self.btn_run = QPushButton("Process selected videos")
        self.btn_run.clicked.connect(self._start)
        self.btn_cancel = QPushButton("Cancel queue")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.queue.cancel)
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_cancel)
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
        self._init_defaults()
        self.refresh_videos()

    def _init_defaults(self) -> None:
        p = self.project.project
        if p.defaults.categorizer_name:
            self.ed_categorizer.setText(p.defaults.categorizer_name)
        if p.root_dir:
            # analysis results: under root/analysis or detection sibling
            self.ed_out.setText(str(p.resolve_path("analysis")))
        self._on_categorizer_changed(self.ed_categorizer.text())

    def _scan_models(self) -> None:
        p = self.project.project
        cats = scan_categorizer_paths(p)
        if cats and not self.ed_categorizer.text().strip():
            self.ed_categorizer.setText(cats[0])
        self.log.append(f"Scan found {len(cats)} categorizer(s).")
        if cats:
            self.log.append("Categorizers: " + "; ".join(cats[:5]))

    def _on_categorizer_changed(self, path: str) -> None:
        path = (path or "").strip()
        if path and Path(path).is_dir():
            try:
                meta = load_categorizer_metadata(path)
                names = meta.get("classnames") or []
                self.lbl_behaviors.setText(", ".join(names) if names else "(no classnames)")
            except Exception as exc:
                self.lbl_behaviors.setText(f"(error: {exc})")
        else:
            self.lbl_behaviors.setText("—")

    def _manual_refresh_videos(self) -> None:
        from LabGym.gui_pyside.project.paths import clear_tracklets_discovery_cache

        clear_tracklets_discovery_cache()
        self.refresh_videos()

    def refresh_videos(self) -> None:
        # Skip rebuild mid-batch so mark_dirty cannot wipe live status.
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
            if path in self._status_by_path:
                status, note = self._status_by_path[path]
            else:
                tracks = discover_tracklets_dir(self.project.project, path)
                from LabGym.id_review.raw_store import has_accepted_identities

                if tracks and has_accepted_identities(tracks):
                    status, note = ("pending", tracks)
                elif tracks:
                    status, note = (
                        "blocked",
                        "needs Review IDs save",
                    )
                else:
                    status, note = (
                        "blocked",
                        "(no identity package — Detect + track, then Review IDs)",
                    )
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

    def _set_all(self, on: bool) -> None:
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it:
                it.setCheckState(state)

    def _selected_videos(self) -> List[str]:
        out = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it and it.checkState() == Qt.CheckState.Checked:
                out.append(str(it.data(Qt.ItemDataRole.UserRole)))
        return out

    def _start(self) -> None:
        if self.queue.is_running:
            QMessageBox.information(self, "Busy", "A batch is already running.")
            return
        categorizer = self.ed_categorizer.text().strip()
        out = self.ed_out.text().strip()
        videos = self._selected_videos()
        if not categorizer or not Path(categorizer).is_dir():
            QMessageBox.warning(self, "Process", "Select a valid categorizer folder.")
            return
        if not out:
            QMessageBox.warning(self, "Process", "Choose a results root folder.")
            return
        if not videos:
            QMessageBox.warning(self, "Process", "Select at least one video.")
            return

        from LabGym.id_review.raw_store import has_accepted_identities

        project = self.project.project
        missing = []
        for path in videos:
            pkg = discover_tracklets_dir(project, path) or ""
            if not pkg or not has_accepted_identities(pkg):
                missing.append(Path(path).name)
        if missing:
            QMessageBox.warning(
                self,
                "Review IDs required",
                "These videos do not have accepted identities. "
                "Open Detector → Review IDs, save, then try again:\n\n"
                + "\n".join(f"  • {n}" for n in missing[:12]),
            )
            return

        # Block table rebuilds for the whole batch (including mark_dirty below).
        self._batch_active = True
        self.project.project.defaults.categorizer_name = categorizer
        self.project.mark_dirty()
        Path(out).mkdir(parents=True, exist_ok=True)

        fw = int(self.spin_fw.value()) or None

        items: List[JobItem] = []
        self._job_rows.clear()
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if not it or it.checkState() != Qt.CheckState.Checked:
                continue
            path = str(it.data(Qt.ItemDataRole.UserRole))
            label = Path(path).name
            self._job_rows[path] = r
            self._set_status(r, path, "queued", "")
            items.append(JobItem(job_id=path, label=label, payload=path))

        def runner(job: JobItem, prog: JobProgress) -> ProcessVideoResult:
            video = str(job.payload)
            id_dir = discover_tracklets_dir(project, video) or ""
            cfg = ProcessVideoConfig(
                video_path=video,
                categorizer_path=categorizer,
                results_root=out,
                id_review_dir=id_dir,
                framewidth=fw,
                t=float(self.spin_t.value()),
                duration=float(self.spin_duration.value()),
                uncertain=float(self.spin_uncertain.value()),
                show_legend=self.chk_legend.isChecked(),
            )
            return process_video(cfg, progress=prog)

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.log.append(f"Starting process batch: {len(items)} video(s) → {out}")
        self.queue.start(items, runner)

    def _on_job_started(self, job_id: str) -> None:
        row = self._job_rows.get(job_id)
        if row is not None:
            self._set_status(row, job_id, "running", "Starting…")

    def _on_progress(self, job_id: str, msg: str) -> None:
        row = self._job_rows.get(job_id)
        if row is not None:
            self._set_status(row, job_id, "running", msg[:200])
        self.log.append(f"[{Path(job_id).name}] {msg}")

    def _on_frame_progress(self, job_id: str, current: int, total: int) -> None:
        row = self._job_rows.get(job_id)
        note = f"frames {current}/{total}" if total else f"frames {current}"
        if row is not None:
            self._set_status(row, job_id, "running", note)

    def _on_job_done(self, job_id: str, result: object) -> None:
        row = self._job_rows.get(job_id)
        if not isinstance(result, ProcessVideoResult):
            if row is not None:
                self._set_status(row, job_id, "done", "finished")
            return
        if row is not None:
            note = result.results_path or "finished"
            self._set_status(row, job_id, "done", note)
        self.log.append(f"[{Path(job_id).name}] OK → {result.results_path}")

    def _on_job_fail(self, job_id: str, error: str) -> None:
        row = self._job_rows.get(job_id)
        if row is not None:
            self._set_status(row, job_id, "error", error[:200])
        self.log.append(f"[{Path(job_id).name}] FAIL: {error}")

    def _on_queue_done(self) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self._batch_active = False
        self.log.append("Batch finished.")
        self.batch_finished.emit()
        for it in self.queue.items:
            if it.status == "cancelled":
                row = self._job_rows.get(it.job_id)
                if row is not None:
                    self._set_status(row, it.job_id, "cancelled", "")
        self.refresh_videos()
        n_ok, n_err, _n_cancel = summarize_job_statuses(self.queue.items)
        QMessageBox.information(
            self,
            "Process videos",
            f"Batch finished.\n\nSucceeded: {n_ok}\nFailed: {n_err}",
        )
