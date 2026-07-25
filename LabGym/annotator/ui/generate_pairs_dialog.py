"""Dialog: generate LabGym training pairs from the open ethogram + tracklets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from LabGym.annotator.core.annotation_manager import AnnotationManager
from LabGym.annotator.core.tracklets_bridge import LoadedTracklets
from LabGym.training.ethogram_examples import GenerationConfig, generate_examples_from_ethogram


class _GenWorker(QObject):
    finished = Signal(dict)
    progress = Signal(int, int, str)
    error = Signal(str)

    def __init__(self, config: GenerationConfig, session, loaded: Optional[LoadedTracklets]):
        super().__init__()
        self.config = config
        self.session = session
        self.loaded = loaded

    def run(self):
        try:
            def _cb(done, tot, msg):
                self.progress.emit(done, tot, msg)

            result = generate_examples_from_ethogram(
                self.config,
                session=self.session,
                loaded_tracklets=self.loaded,
                progress=_cb,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class GeneratePairsDialog(QDialog):
    """Generate sorted LabGym .avi/.jpg pairs from ethogram (Stage C)."""

    def __init__(
        self,
        manager: AnnotationManager,
        video_path: str,
        loaded_tracklets: Optional[LoadedTracklets] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Generate LabGym training pairs from ethogram")
        self.resize(560, 420)
        self.manager = manager
        self.video_path = video_path
        self.loaded = loaded_tracklets
        self._thread: Optional[QThread] = None
        self._worker: Optional[_GenWorker] = None

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Creates already-sorted LabGym animation + pattern pairs from "
                "the saved ethogram and fixed tracklets. Re-run anytime with a "
                "new length without re-annotating."
            )
        )
        form = QFormLayout()

        self.ed_out = QLineEdit()
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._browse_out)
        row = QHBoxLayout()
        row.addWidget(self.ed_out, 1)
        row.addWidget(btn_out)
        form.addRow("Output folder:", row)

        self.ed_tracklets = QLineEdit()
        if loaded_tracklets:
            self.ed_tracklets.setText(loaded_tracklets.directory)
        elif manager.session.tracks_ref and manager.session.tracks_ref.path:
            self.ed_tracklets.setText(
                str(Path(manager.session.tracks_ref.path).parent)
            )
        btn_tr = QPushButton("Browse…")
        btn_tr.clicked.connect(self._browse_tracklets)
        row2 = QHBoxLayout()
        row2.addWidget(self.ed_tracklets, 1)
        row2.addWidget(btn_tr)
        form.addRow("Tracklets folder:", row2)

        self.spin_length = QSpinBox()
        self.spin_length.setRange(1, 500)
        self.spin_length.setValue(15)
        form.addRow("Window length (frames):", self.spin_length)

        self.combo_sampling = QComboBox()
        for s in ("dense_in_bout", "bout_end", "bout_center", "coverage"):
            self.combo_sampling.addItem(s, s)
        form.addRow("Sampling:", self.combo_sampling)

        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(0, 500)
        self.spin_stride.setValue(0)
        self.spin_stride.setSpecialValueText("auto (length/3)")
        form.addRow("Stride (dense):", self.spin_stride)

        self.spin_min_bout = QSpinBox()
        self.spin_min_bout.setRange(1, 500)
        self.spin_min_bout.setValue(1)
        form.addRow("Min bout frames:", self.spin_min_bout)

        self.spin_social = QDoubleSpinBox()
        self.spin_social.setRange(0.0, 100.0)
        self.spin_social.setValue(0.0)
        self.spin_social.setToolTip("Mode 2 only; 0 = include all other IDs as costars")
        form.addRow("Social distance (mode 2):", self.spin_social)

        self.chk_soft = QCheckBox("Write soft_labels.csv")
        self.chk_soft.setChecked(True)
        form.addRow(self.chk_soft)

        self.chk_bg_free = QCheckBox("Background-free blobs")
        self.chk_bg_free.setChecked(True)
        form.addRow(self.chk_bg_free)

        layout.addLayout(form)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.lbl_status = QLabel("")
        layout.addWidget(self.progress)
        layout.addWidget(self.lbl_status)

        btns = QHBoxLayout()
        self.btn_run = QPushButton("Generate")
        self.btn_run.clicked.connect(self._run)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_close)
        layout.addLayout(btns)

        # Default output next to video
        if video_path:
            stem = Path(video_path).stem
            self.ed_out.setText(
                str(Path(video_path).parent / f"{stem}_examples_from_ethogram")
            )

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.ed_out.text())
        if d:
            self.ed_out.setText(d)

    def _browse_tracklets(self):
        d = QFileDialog.getExistingDirectory(
            self, "Tracklets / id_review folder", self.ed_tracklets.text()
        )
        if d:
            self.ed_tracklets.setText(d)

    def _run(self):
        out = self.ed_out.text().strip()
        tracks = self.ed_tracklets.text().strip()
        if not out:
            QMessageBox.warning(self, "Generate", "Choose an output folder.")
            return
        if not tracks or not Path(tracks).is_dir():
            QMessageBox.warning(
                self,
                "Generate",
                "Tracklets folder is required (post–ID-review id_review directory).",
            )
            return
        if not self.video_path:
            QMessageBox.warning(self, "Generate", "No video loaded.")
            return

        # Ensure open bouts closed for complete ethogram
        # (caller should have closed; still ok)

        ann_path = str(
            Path(self.video_path).with_suffix(".annotations.json")
        )
        # Prefer in-memory session; write temp if needed for config record
        cfg = GenerationConfig(
            video_path=self.video_path,
            annotations_path=ann_path,
            tracklets_dir=tracks,
            output_dir=out,
            length=int(self.spin_length.value()),
            behavior_mode=int(self.manager.session.behavior_mode),
            sampling=str(self.combo_sampling.currentData()),
            stride=int(self.spin_stride.value()),
            min_bout_frames=int(self.spin_min_bout.value()),
            social_distance=float(self.spin_social.value()),
            write_soft_labels=self.chk_soft.isChecked(),
            background_free=self.chk_bg_free.isChecked(),
            analysis_start_frame=(
                self.loaded.analysis_start_frame if self.loaded else None
            ),
        )

        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.lbl_status.setText("Starting…")

        self._thread = QThread(self)
        self._worker = _GenWorker(cfg, self.manager.session, self.loaded)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.start()

    def _on_progress(self, done: int, tot: int, msg: str):
        if tot > 0:
            self.progress.setValue(int(100 * done / tot))
        self.lbl_status.setText(msg)

    def _on_finished(self, result: dict):
        self.btn_run.setEnabled(True)
        self.progress.setValue(100)
        self.lbl_status.setText(
            f"Wrote {result.get('written', 0)} pairs "
            f"(skipped {result.get('skipped', 0)})"
        )
        QMessageBox.information(
            self,
            "Generation complete",
            f"Output: {result.get('output_dir')}\n"
            f"Written: {result.get('written')}\n"
            f"Skipped: {result.get('skipped')}\n"
            f"Counts: {result.get('counts')}\n"
            f"Config: {result.get('generation_config')}\n"
            f"Soft labels: {result.get('soft_labels')}",
        )
        self.accept()

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Error")
        QMessageBox.critical(self, "Generation failed", msg)
