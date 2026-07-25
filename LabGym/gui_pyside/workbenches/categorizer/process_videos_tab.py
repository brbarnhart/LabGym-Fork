"""Categorizer → Process videos (batch analysis with detector + categorizer)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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
from LabGym.detection.batch_detect import (
    list_detectors,
    load_detector_animal_kinds,
)
from LabGym.gui_pyside.jobs.sequential_queue import JobItem, SequentialJobQueue
from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.project.paths import (
    discover_tracklets_dir,
    list_project_video_choices,
)
from LabGym.mypkg_resources import resource_filename


def _list_categorizers(models_root: str | Path) -> List[Path]:
    root = Path(models_root)
    if not root.is_dir():
        return []
    found: List[Path] = []
    for p in root.rglob("model_parameters.txt"):
        # skip detector folders that use json not csv - try read
        try:
            # categorizer uses CSV; detector uses JSON
            text = p.read_text(encoding="utf-8", errors="ignore")[:80]
            if "classnames" in text or "network" in text or "time_step" in text:
                found.append(p.parent)
        except Exception:
            continue
    return sorted(set(found))


class ProcessVideosTab(QWidget):
    """Batch-run detector + categorizer; optional apply Review IDs remaps."""

    request_edit_project = Signal()
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
        layout.addWidget(QLabel(
            "Process project videos: <b>detect + track</b>, optionally apply "
            "ID remaps from Review IDs packages, <b>categorize</b>, annotate, "
            "and export LabGym analysis outputs. One video at a time."
        ))

        # Models
        mbox = QGroupBox("Models")
        mform = QFormLayout(mbox)
        self.ed_detector = QLineEdit()
        self.ed_detector.setToolTip(
            "Trained LabGym detector folder (model_final.pth + model_parameters.txt). "
            "Used to find and track animals before categorization."
        )
        b_d = QPushButton("Browse…")
        b_d.clicked.connect(lambda: self._browse_dir(self.ed_detector))
        mform.addRow(
            self._lab(
                "Detector:",
                "Path to the detector used for locating animals. Same type as "
                "Detect + track.",
            ),
            self._row(self.ed_detector, b_d),
        )
        self.lbl_kinds = QLabel("—")
        self.lbl_kinds.setToolTip("Categories the detector was trained to detect.")
        mform.addRow(
            self._lab("Animal kinds:", "Read from the detector’s model_parameters.txt."),
            self.lbl_kinds,
        )
        self.ed_detector.textChanged.connect(self._on_detector_changed)

        self.ed_categorizer = QLineEdit()
        self.ed_categorizer.setToolTip(
            "Trained LabGym categorizer folder (Keras model + model_parameters.txt "
            "with classnames, time_step, network type, etc.). Behavior names and "
            "input sizes are loaded automatically from that file."
        )
        b_c = QPushButton("Browse…")
        b_c.clicked.connect(lambda: self._browse_dir(self.ed_categorizer))
        mform.addRow(
            self._lab(
                "Categorizer:",
                "Model that assigns behavior labels over time for each tracked animal.",
            ),
            self._row(self.ed_categorizer, b_c),
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
        self.spin_animals = QSpinBox()
        self.spin_animals.setRange(1, 50)
        self.spin_animals.setValue(2)
        tip_animals = (
            "Expected number of individuals of each animal kind (same count for "
            "every kind). Allocates track slots for the detector (e.g. 2 → IDs 0,1)."
        )
        self.spin_animals.setToolTip(tip_animals)
        pform.addRow(self._lab("Animals per kind:", tip_animals), self.spin_animals)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 32)
        self.spin_batch.setValue(1)
        tip_batch = (
            "GPU batch size for detector inference. Higher can be faster; lower "
            "if you hit out-of-memory errors."
        )
        self.spin_batch.setToolTip(tip_batch)
        pform.addRow(self._lab("Detector batch size:", tip_batch), self.spin_batch)

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

        self.chk_apply_id = QCheckBox(
            "Apply ID remaps from Review IDs package when available"
        )
        self.chk_apply_id.setChecked(True)
        self.chk_apply_id.setToolTip(
            "If an id_review package (from Detect + track / Review IDs) is found "
            "for the video, apply saved identity-swap corrections after tracking "
            "and before categorizing. Recommended when you already fixed ID swaps."
        )
        pform.addRow(self.chk_apply_id)

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
        b_o.clicked.connect(lambda: self._browse_dir(self.ed_out))
        pform.addRow(
            self._lab(
                "Results root:",
                "Where per-video analysis folders are written (default: project "
                "analysis/).",
            ),
            self._row(self.ed_out, b_o),
        )
        layout.addWidget(pbox)

        # Videos
        vbox = QGroupBox("Videos")
        vl = QVBoxLayout(vbox)
        row = QHBoxLayout()
        btn_ref = QPushButton("Refresh from project")
        btn_ref.clicked.connect(self.refresh_videos)
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
            ["", "Video", "Status", "Results / id_review"]
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

        self.project.changed.connect(self.refresh_videos)
        self.project.project_replaced.connect(self.refresh_videos)
        self._init_defaults()
        self.refresh_videos()

    @staticmethod
    def _lab(text: str, tip: str) -> QLabel:
        lab = QLabel(text)
        lab.setToolTip(tip)
        return lab

    @staticmethod
    def _row(edit, btn):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        h.addWidget(btn)
        return w

    def _browse_dir(self, edit: QLineEdit) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select folder", edit.text())
        if d:
            edit.setText(d)

    def _init_defaults(self) -> None:
        p = self.project.project
        if p.defaults.detector_name:
            self.ed_detector.setText(p.defaults.detector_name)
        if p.defaults.categorizer_name:
            self.ed_categorizer.setText(p.defaults.categorizer_name)
        if p.root_dir:
            # analysis results: under root/analysis or detection sibling
            self.ed_out.setText(str(p.resolve_path("analysis")))
        self._on_detector_changed(self.ed_detector.text())
        self._on_categorizer_changed(self.ed_categorizer.text())

    def _scan_models(self) -> None:
        roots: List[Path] = []
        p = self.project.project
        if p.root_dir:
            roots.append(p.resolve_path(p.paths.models_root or "models"))
        try:
            roots.append(Path(resource_filename("LabGym", "detectors")))
            roots.append(Path(resource_filename("LabGym", "models")))
        except Exception:
            pass
        dets: List[str] = []
        cats: List[str] = []
        for root in roots:
            for d in list_detectors(root):
                dets.append(str(d))
            for c in _list_categorizers(root):
                cats.append(str(c))
        if dets and not self.ed_detector.text().strip():
            self.ed_detector.setText(dets[0])
        if cats and not self.ed_categorizer.text().strip():
            self.ed_categorizer.setText(cats[0])
        self.log.append(
            f"Scan found {len(dets)} detector(s), {len(cats)} categorizer(s)."
        )
        if dets:
            self.log.append("Detectors: " + "; ".join(dets[:5]))
        if cats:
            self.log.append("Categorizers: " + "; ".join(cats[:5]))

    def _on_detector_changed(self, path: str) -> None:
        path = (path or "").strip()
        if path and Path(path).is_dir():
            try:
                self.lbl_kinds.setText(", ".join(load_detector_animal_kinds(path)))
            except Exception as exc:
                self.lbl_kinds.setText(f"(error: {exc})")
        else:
            self.lbl_kinds.setText("—")

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

    def refresh_videos(self) -> None:
        choices = list_project_video_choices(self.project.project)
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
            tracks = discover_tracklets_dir(self.project.project, resolved)
            self.table.setItem(r, 2, QTableWidgetItem("pending"))
            self.table.setItem(
                r, 3, QTableWidgetItem(tracks or "(no id_review yet — will re-detect)")
            )

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
        detector = self.ed_detector.text().strip()
        categorizer = self.ed_categorizer.text().strip()
        out = self.ed_out.text().strip()
        videos = self._selected_videos()
        if not detector or not Path(detector).is_dir():
            QMessageBox.warning(self, "Process", "Select a valid detector folder.")
            return
        if not categorizer or not Path(categorizer).is_dir():
            QMessageBox.warning(self, "Process", "Select a valid categorizer folder.")
            return
        if not out:
            QMessageBox.warning(self, "Process", "Choose a results root folder.")
            return
        if not videos:
            QMessageBox.warning(self, "Process", "Select at least one video.")
            return

        self.project.project.defaults.detector_name = detector
        self.project.project.defaults.categorizer_name = categorizer
        self.project.mark_dirty()
        Path(out).mkdir(parents=True, exist_ok=True)

        try:
            kinds = load_detector_animal_kinds(detector)
        except Exception as exc:
            QMessageBox.warning(self, "Process", f"Detector metadata: {exc}")
            return
        n_per = int(self.spin_animals.value())
        numbers = {k: n_per for k in kinds}
        fw = int(self.spin_fw.value()) or None

        items: List[JobItem] = []
        self._job_rows.clear()
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if not it or it.checkState() != Qt.CheckState.Checked:
                continue
            path = str(it.data(Qt.ItemDataRole.UserRole))
            jid = f"p{r}"
            self._job_rows[jid] = r
            self.table.item(r, 2).setText("queued")
            items.append(JobItem(job_id=jid, label=Path(path).name, payload=path))

        apply_id = self.chk_apply_id.isChecked()
        project = self.project.project

        def runner(job: JobItem, prog) -> ProcessVideoResult:
            video = str(job.payload)
            id_dir = ""
            if apply_id:
                id_dir = discover_tracklets_dir(project, video) or ""
            cfg = ProcessVideoConfig(
                video_path=video,
                detector_path=detector,
                categorizer_path=categorizer,
                results_root=out,
                animal_kinds=kinds,
                animal_number=numbers,
                id_review_dir=id_dir,
                framewidth=fw,
                t=float(self.spin_t.value()),
                duration=float(self.spin_duration.value()),
                detector_batch=int(self.spin_batch.value()),
                uncertain=float(self.spin_uncertain.value()),
                show_legend=self.chk_legend.isChecked(),
            )
            return process_video(cfg, progress=prog)

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.log.append(f"Starting process batch: {len(items)} video(s) → {out}")
        self.queue.start(items, runner)

    def _on_progress(self, job_id: str, msg: str) -> None:
        row = self._job_rows.get(job_id)
        if row is not None:
            self.table.item(row, 2).setText("running")
            self.table.item(row, 3).setText(msg[:200])
        self.log.append(f"[{job_id}] {msg}")

    def _on_job_done(self, job_id: str, result: object) -> None:
        row = self._job_rows.get(job_id)
        if not isinstance(result, ProcessVideoResult):
            return
        if row is not None:
            if result.ok:
                self.table.item(row, 2).setText("done")
                self.table.item(row, 3).setText(result.results_path)
            else:
                self.table.item(row, 2).setText("error")
                self.table.item(row, 3).setText(result.error[:200])
        self.log.append(
            f"[{job_id}] "
            + (f"OK → {result.results_path}" if result.ok else f"FAIL: {result.error}")
        )

    def _on_job_fail(self, job_id: str, error: str) -> None:
        row = self._job_rows.get(job_id)
        if row is not None:
            self.table.item(row, 2).setText("error")
            self.table.item(row, 3).setText(error[:200])
        self.log.append(f"[{job_id}] FAIL: {error}")

    def _on_queue_done(self) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.log.append("Batch finished.")
        self.batch_finished.emit()
        n_ok = sum(1 for it in self.queue.items if it.status == "done")
        n_err = sum(1 for it in self.queue.items if it.status == "error")
        QMessageBox.information(
            self,
            "Process videos",
            f"Batch finished.\n\nSucceeded: {n_ok}\nFailed: {n_err}",
        )
