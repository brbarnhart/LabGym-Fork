"""Categorizer → Process videos (categorize accepted identities)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

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
    resolve_video_context,
)
from LabGym.gui_pyside.project.video_disk_status import (
    STATUS_DONE,
    STATUS_UNKNOWN,
    ProbeResult,
    VideoTableDiskStatusMixin,
    run_blocked_unknown_paths,
)
from LabGym.gui_pyside.widgets.path_browse import (
    path_edit_row,
    set_line_edit_directory,
)
from LabGym.identity.downstream import may_use_downstream
from LabGym.identity.package import has_identity_package


class ProcessVideosTab(VideoTableDiskStatusMixin, QWidget):
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
        self._batch_active = False
        self._init_video_disk_status()
        self._models_need_scan = True
        self._behaviors_need_refresh = True

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
            "Search the project models folder and LabGym’s bundled models "
            "directory for available categorizers."
        )
        btn_scan.clicked.connect(self._scan_models)
        mform.addRow(btn_scan)
        layout.addWidget(mbox)

        # Params
        pbox = QGroupBox("Analysis parameters")
        pbox.setToolTip(
            "How long to analyze and at what resolution. Behaviors are labeled "
            "from accepted remapped identities — this tab does not run the detector."
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
        self.lbl_probe = QLabel("")
        self.lbl_probe.setToolTip(
            "Per-video disk check after Open. Rows stay status unknown (… ) "
            "until each probe finishes."
        )
        row.addWidget(self.lbl_probe)
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
        self.table.itemChanged.connect(self._on_table_item_changed)
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
        self._update_run_enabled()

    @staticmethod
    def _lab(text: str, tip: str) -> QLabel:
        lab = QLabel(text)
        lab.setToolTip(tip)
        return lab

    def showEvent(self, event) -> None:  # noqa: N802 — Qt
        super().showEvent(event)
        self._maybe_refresh_behaviors()
        self._maybe_scan_models()
        self._maybe_start_status_probes()

    def hideEvent(self, event) -> None:  # noqa: N802 — Qt
        self._cancel_status_probes()
        super().hideEvent(event)

    def _on_project_replaced(self) -> None:
        self._cancel_status_probes()
        self._status_by_path.clear()
        self._models_need_scan = True
        self._behaviors_need_refresh = True
        self._init_defaults()
        self.refresh_videos()
        self._maybe_start_status_probes()

    def _init_defaults(self) -> None:
        p = self.project.project
        self.ed_categorizer.blockSignals(True)
        if p.defaults.categorizer_name:
            self.ed_categorizer.setText(p.defaults.categorizer_name)
        self.ed_categorizer.blockSignals(False)
        if p.root_dir:
            # analysis results: under root/analysis or detection sibling
            self.ed_out.setText(str(p.resolve_path("analysis")))
        # Do not read categorizer files during project swap.
        self.lbl_behaviors.setText("—")

    def _maybe_scan_models(self) -> None:
        """Walk categorizer folders when this tab is shown, not during project swap."""
        if not self._models_need_scan or not self.isVisible():
            return
        self._models_need_scan = False
        self._scan_models()

    def _maybe_refresh_behaviors(self) -> None:
        if not self._behaviors_need_refresh or not self.isVisible():
            return
        self._behaviors_need_refresh = False
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

    def _probe_result_note(self, result: ProbeResult) -> str:
        """Accepted-identities note for Process videos, not a package path."""
        if result.status == STATUS_DONE and result.tracklets_dir:
            from LabGym.id_review.raw_store import has_accepted_identities

            return (
                "yes"
                if has_accepted_identities(result.tracklets_dir)
                else "no — save Review IDs"
            )
        if result.status == STATUS_UNKNOWN:
            return ""
        return result.error

    def _on_table_item_changed(self, _item) -> None:
        self._update_run_enabled()

    def _set_all(self, on: bool) -> None:
        state = Qt.CheckState.Checked if on else Qt.CheckState.Unchecked
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it:
                it.setCheckState(state)

    def _start(self) -> None:
        if self.queue.is_running:
            QMessageBox.information(self, "Busy", "A batch is already running.")
            return
        categorizer = self.ed_categorizer.text().strip()
        out = self.ed_out.text().strip()
        videos = self._selected_videos()
        if not videos:
            QMessageBox.warning(self, "Process", "Select at least one video.")
            return
        unknown = run_blocked_unknown_paths(videos, self._status_by_path)
        if unknown:
            QMessageBox.warning(
                self,
                "Process",
                "Status is still unknown for checked video(s). "
                "Wait until disk checks finish, or uncheck those rows:\n\n"
                + "\n".join(f"  • {Path(p).name}" for p in unknown[:12]),
            )
            return
        if not categorizer or not Path(categorizer).is_dir():
            QMessageBox.warning(self, "Process", "Select a valid categorizer folder.")
            return
        if not out:
            QMessageBox.warning(self, "Process", "Choose a results root folder.")
            return

        project = self.project.project
        missing = []
        for path in videos:
            ctx = resolve_video_context(project, video_path=path)
            pkg = bool(ctx.tracklets_dir) and has_identity_package(ctx.tracklets_dir)
            if not may_use_downstream(
                int(ctx.behavior_mode),
                bool(ctx.accepted_identities),
                identity_package=pkg,
            ):
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
            ctx = resolve_video_context(project, video_path=video)
            cfg = ProcessVideoConfig(
                video_path=video,
                categorizer_path=categorizer,
                results_root=out,
                id_review_dir=ctx.tracklets_dir or "",
                framewidth=fw,
                t=float(self.spin_t.value()),
                duration=float(self.spin_duration.value()),
                uncertain=float(self.spin_uncertain.value()),
                show_legend=self.chk_legend.isChecked(),
                behavior_mode=int(ctx.behavior_mode),
            )
            return process_video(cfg, progress=prog)

        self._update_run_enabled()
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
        self._maybe_start_status_probes()
        n_ok, n_err, _n_cancel = summarize_job_statuses(self.queue.items)
        QMessageBox.information(
            self,
            "Process videos",
            f"Batch finished.\n\nSucceeded: {n_ok}\nFailed: {n_err}",
        )
