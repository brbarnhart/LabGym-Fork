"""Categorizer → Generate training data → Generate examples (ethogram-first)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from LabGym.annotator.core.annotation_manager import AnnotationManager
from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.project.paths import list_project_video_choices
from LabGym.identity.downstream import may_use_downstream
from LabGym.training.ethogram_examples import GenerationConfig, generate_examples_from_ethogram


class _Worker(QObject):
    finished = Signal(dict)
    progress = Signal(int, int, str)
    error = Signal(str)

    def __init__(self, config: GenerationConfig):
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            def _cb(done, tot, msg):
                self.progress.emit(done, tot, msg)

            result = generate_examples_from_ethogram(self.config, progress=_cb)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class GenerateExamplesTab(QWidget):
    """Generate sorted LabGym pairs from ethogram + fixed tracklets."""

    request_edit_project = Signal()
    request_annotate = Signal()

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None
        self._block = False

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Create <b>already-sorted</b> LabGym animation + pattern pairs from the "
            "ethogram JSON and fixed tracklets. Change window length later without "
            "re-annotating. Dense generate-then-sort is not offered."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # Video picker
        top = QHBoxLayout()
        top.addWidget(QLabel("Video:"))
        self.combo_video = QComboBox()
        self.combo_video.currentIndexChanged.connect(self._on_video_combo)
        top.addWidget(self.combo_video, 1)
        btn_edit = QPushButton("Edit project…")
        btn_edit.clicked.connect(self.request_edit_project.emit)
        top.addWidget(btn_edit)
        btn_ann = QPushButton("Go to Annotate ethogram")
        btn_ann.clicked.connect(self.request_annotate.emit)
        top.addWidget(btn_ann)
        layout.addLayout(top)

        self.lbl_summary = QLabel()
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.lbl_summary.setStyleSheet(
            "QLabel { background: #2a2a2a; color: #eee; padding: 8px; border-radius: 4px; }"
        )
        layout.addWidget(self.lbl_summary)

        # Generation params (synced from / to project defaults)
        params = QGroupBox("Generation parameters (saved with project)")
        params.setToolTip(
            "How short video clips (examples) are cut from annotated bouts for "
            "categorizer training. Stored on the project so Annotate/Generate stay in sync."
        )
        form = QFormLayout(params)

        self.spin_length = QSpinBox()
        self.spin_length.setRange(1, 500)
        self.spin_length.setValue(15)
        tip_len = (
            "Window length (frames) — duration of each training clip (LabGym "
            "time_step). Example: 15 at 30 fps ≈ 0.5 s of video + matching pattern "
            "image. Must match the length you will train the categorizer with. "
            "You can re-run generation with a new length without re-annotating."
        )
        self.spin_length.setToolTip(tip_len)
        form.addRow(self._lab("Window length (frames):", tip_len), self.spin_length)

        self.combo_sampling = QComboBox()
        for s in ("dense_in_bout", "bout_end", "bout_center", "coverage"):
            self.combo_sampling.addItem(s, s)
        tip_samp = (
            "Where windows are placed inside each bout:\n"
            "• dense_in_bout — many overlapping windows along the bout (most examples).\n"
            "• bout_end — one window ending at the bout’s last frame.\n"
            "• bout_center — one window centered on the bout.\n"
            "• coverage — spaced windows that cover the bout."
        )
        self.combo_sampling.setToolTip(tip_samp)
        form.addRow(self._lab("Sampling:", tip_samp), self.combo_sampling)

        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(0, 500)
        self.spin_stride.setSpecialValueText("auto (length/3)")
        tip_stride = (
            "Step between successive windows when sampling is dense. "
            "0 = auto (about length/3). Smaller stride → more examples and more overlap."
        )
        self.spin_stride.setToolTip(tip_stride)
        form.addRow(self._lab("Stride (dense):", tip_stride), self.spin_stride)

        self.spin_min_bout = QSpinBox()
        self.spin_min_bout.setRange(1, 500)
        tip_min = (
            "Skip annotated bouts shorter than this many frames. Filters out "
            "tiny accidental toggles that are too short for a full training window."
        )
        self.spin_min_bout.setToolTip(tip_min)
        form.addRow(self._lab("Min bout frames:", tip_min), self.spin_min_bout)

        self.spin_social = QDoubleSpinBox()
        self.spin_social.setRange(0.0, 100.0)
        tip_soc = (
            "Interactive advanced (mode 2) only: how near another animal must be "
            "(in folds of body size) to be included as a costar in the crop. "
            "0 = include all other IDs (no distance filter)."
        )
        self.spin_social.setToolTip(tip_soc)
        form.addRow(self._lab("Social distance (mode 2):", tip_soc), self.spin_social)

        self.chk_soft = QCheckBox("Write soft_labels.csv")
        self.chk_soft.setChecked(True)
        self.chk_soft.setToolTip(
            "Also write soft_labels.csv next to examples for hard_soft_aux / "
            "soft_primary categorizer training (probability targets from overlapping labels)."
        )
        form.addRow(self.chk_soft)

        self.chk_bg = QCheckBox("Background-free blobs")
        self.chk_bg.setChecked(True)
        self.chk_bg.setToolTip(
            "If checked, animal blobs are extracted without the video background "
            "(LabGym background-free style). Matches typical LabGym training examples."
        )
        form.addRow(self.chk_bg)

        layout.addWidget(params)

        run_box = QGroupBox("Run")
        run_l = QVBoxLayout(run_box)
        self.btn_run = QPushButton("Generate training pairs")
        self.btn_run.clicked.connect(self._run)
        run_l.addWidget(self.btn_run)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        run_l.addWidget(self.progress)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        run_l.addWidget(self.log)
        layout.addWidget(run_box)
        layout.addStretch(1)

        for w in (
            self.spin_length,
            self.spin_stride,
            self.spin_min_bout,
            self.spin_social,
        ):
            w.valueChanged.connect(self._commit_params_to_project)
        self.combo_sampling.currentIndexChanged.connect(self._commit_params_to_project)
        self.chk_soft.toggled.connect(self._commit_params_to_project)
        self.chk_bg.toggled.connect(self._commit_params_to_project)

        # Full video-list rebuild only on replace; dirty/param commits stay light.
        self.project.project_replaced.connect(self._on_project_replaced)
        self.project.changed.connect(self._on_project_light)
        self._on_project_replaced()

    @staticmethod
    def _lab(text: str, tip: str) -> QLabel:
        lab = QLabel(text)
        lab.setToolTip(tip)
        return lab

    def _on_project_replaced(self) -> None:
        self._load_params_from_project()
        self._refresh_video_list()
        self._refresh_summary()

    def _on_project_light(self) -> None:
        """Dirty flag / param writes — do not rebuild video combo."""
        if self._block:
            return
        self._refresh_summary()

    def _load_params_from_project(self) -> None:
        self._block = True
        d = self.project.project.defaults
        self.spin_length.setValue(int(d.window_length))
        idx = self.combo_sampling.findData(d.sampling)
        self.combo_sampling.setCurrentIndex(max(0, idx))
        self.spin_stride.setValue(int(d.stride))
        self.spin_min_bout.setValue(int(d.min_bout_frames))
        self.spin_social.setValue(float(d.social_distance))
        self.chk_soft.setChecked(bool(d.write_soft_labels))
        self.chk_bg.setChecked(bool(d.background_free))
        self._block = False

    def _commit_params_to_project(self) -> None:
        if self._block:
            return
        d = self.project.project.defaults
        d.window_length = int(self.spin_length.value())
        d.sampling = str(self.combo_sampling.currentData())
        d.stride = int(self.spin_stride.value())
        d.min_bout_frames = int(self.spin_min_bout.value())
        d.social_distance = float(self.spin_social.value())
        d.write_soft_labels = self.chk_soft.isChecked()
        d.background_free = self.chk_bg.isChecked()
        self.project.mark_dirty()

    def _refresh_video_list(self) -> None:
        self._block = True
        self.combo_video.clear()
        choices = list_project_video_choices(self.project.project)
        cur = self.project.current_video_path()
        select = 0
        for i, (label, resolved) in enumerate(choices):
            self.combo_video.addItem(label, resolved)
            try:
                if cur and Path(resolved).resolve() == Path(cur).resolve():
                    select = i
            except OSError:
                pass
        if choices:
            self.combo_video.setCurrentIndex(select)
            self.btn_run.setEnabled(True)
        else:
            self.btn_run.setEnabled(False)
        self._block = False

    def _on_video_combo(self, _i: int) -> None:
        if self._block:
            return
        path = self.combo_video.currentData()
        if path:
            self.project.set_current_video(str(path), dirty=True)
        self._refresh_summary()

    def _selected_video(self) -> str:
        data = self.combo_video.currentData()
        return str(data) if data else self.project.current_video_path()

    def _refresh_summary(self) -> None:
        ctx = self.project.resolve_context(self._selected_video() or None)
        html = "<br>".join(
            f"<b>{line.split(':', 1)[0]}:</b>{line.split(':', 1)[1]}"
            if ":" in line
            else line
            for line in ctx.summary_lines()
        )
        warnings = []
        if not ctx.video_path:
            warnings.append("Add videos via Edit project…")
        elif not Path(ctx.video_path).is_file():
            warnings.append("Video file missing on disk")
        if ctx.video_path and not ctx.annotations_exists:
            warnings.append("Annotations missing — annotate and save first")
        if ctx.video_path and not may_use_downstream(
            int(ctx.behavior_mode), bool(ctx.accepted_identities)
        ):
            warnings.append(
                "Accepted identities missing — save Review IDs before generating"
            )
        if warnings:
            html += "<br><br><span style='color:#f6a'>" + " · ".join(warnings) + "</span>"
        self.lbl_summary.setText(html)

    def _run(self) -> None:
        self._commit_params_to_project()
        video = self._selected_video()
        ctx = self.project.resolve_context(video or None)

        if not ctx.video_path or not Path(ctx.video_path).is_file():
            QMessageBox.warning(
                self, "Generate", "Select a valid project video first."
            )
            return
        if not ctx.annotations_exists:
            QMessageBox.warning(
                self,
                "Generate",
                "Annotations JSON not found.\n"
                "Use Annotate ethogram, save (Ctrl+S), then try again.\n\n"
                f"Expected:\n{ctx.annotations_path}",
            )
            self.request_annotate.emit()
            return
        behavior_mode = int(ctx.behavior_mode)
        try:
            sess = AnnotationManager.load_from_json(ctx.annotations_path).session
            behavior_mode = int(sess.behavior_mode)
        except (OSError, ValueError, KeyError, TypeError):
            pass

        if not may_use_downstream(behavior_mode, bool(ctx.accepted_identities)):
            QMessageBox.warning(
                self,
                "Generate",
                "Accepted identities are required.\n"
                "Open Detector → Review IDs, review or accept the detector "
                "IDs, and save.\n\n"
                f"Package:\n{ctx.tracklets_dir or '(none found)'}",
            )
            return

        cfg = GenerationConfig(
            video_path=ctx.video_path,
            annotations_path=ctx.annotations_path,
            tracklets_dir=ctx.tracklets_dir,
            output_dir=ctx.examples_out_dir,
            length=int(self.spin_length.value()),
            behavior_mode=behavior_mode,
            sampling=str(self.combo_sampling.currentData()),
            stride=int(self.spin_stride.value()),
            min_bout_frames=int(self.spin_min_bout.value()),
            social_distance=float(self.spin_social.value()),
            write_soft_labels=self.chk_soft.isChecked(),
            background_free=self.chk_bg.isChecked(),
        )

        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.log.append(f"Starting… → {cfg.output_dir}")

        self._thread = QThread(self)
        self._worker = _Worker(cfg)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._thread.start()

    def _cleanup_thread(self) -> None:
        self._worker = None
        self._thread = None

    def _on_progress(self, done: int, tot: int, msg: str) -> None:
        if tot > 0:
            self.progress.setValue(int(100 * done / tot))
        self.log.append(msg)

    def _on_finished(self, result: dict) -> None:
        self.btn_run.setEnabled(True)
        self.progress.setValue(100)
        written = result.get("written", 0)
        skipped = result.get("skipped", 0)
        out = result.get("output_dir") or ""
        self.log.append(f"Done. Wrote {written} pairs (skipped {skipped}).\n→ {out}")
        QMessageBox.information(
            self,
            "Generate complete",
            f"Wrote {written} pairs (skipped {skipped}).\n\n{out}",
        )

    def _on_error(self, msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.log.append(f"ERROR: {msg}")
        QMessageBox.critical(self, "Generate failed", msg)
