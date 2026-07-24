"""Generate LabGym training pairs from ethogram (Stage C) using project settings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from LabGym.annotator.core.annotation_manager import AnnotationManager
from LabGym.gui_pyside.project_state import ProjectController
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

            result = generate_examples_from_ethogram(
                self.config,
                progress=_cb,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class GenerateTab(QWidget):
    """Run ethogram → sorted LabGym pairs using Project settings."""

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Generate <b>already-sorted</b> LabGym animation + pattern pairs from "
            "the ethogram JSON and fixed tracklets. Change window length later "
            "without re-annotating."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        summary_box = QGroupBox("Will use")
        summary_l = QVBoxLayout(summary_box)
        self.lbl_summary = QLabel()
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        summary_l.addWidget(self.lbl_summary)
        layout.addWidget(summary_box)

        form_box = QGroupBox("Run")
        form = QFormLayout(form_box)
        self.btn_refresh = QPushButton("Refresh summary from Project")
        self.btn_refresh.clicked.connect(self._refresh)
        form.addRow(self.btn_refresh)

        self.btn_run = QPushButton("Generate training pairs")
        self.btn_run.clicked.connect(self._run)
        form.addRow(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        form.addRow("Progress:", self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        form.addRow("Log:", self.log)

        layout.addWidget(form_box)
        layout.addStretch(1)

        self.project.changed.connect(self._refresh)
        self._refresh()

    def _refresh(self) -> None:
        s = self.project.state
        ann = s.inferred_annotations_path()
        out = s.inferred_examples_dir()
        lines = [
            f"<b>Video:</b> {s.video_path or '—'}",
            f"<b>Annotations:</b> {ann or '—'}",
            f"<b>Tracklets:</b> {s.tracklets_dir or '—'}",
            f"<b>Output:</b> {out or '—'}",
            f"<b>Length:</b> {s.window_length}  ·  <b>Sampling:</b> {s.sampling}  ·  "
            f"<b>Mode:</b> {s.behavior_mode}",
            f"<b>Soft labels:</b> {s.write_soft_labels}  ·  "
            f"<b>Background-free:</b> {s.background_free}",
        ]
        self.lbl_summary.setText("<br>".join(lines))

    def _run(self) -> None:
        s = self.project.state
        video = s.video_path.strip()
        ann = s.inferred_annotations_path()
        tracks = s.tracklets_dir.strip()
        out = s.inferred_examples_dir()

        if not video or not Path(video).is_file():
            QMessageBox.warning(self, "Generate", "Set a valid video on the Project tab.")
            return
        if not ann or not Path(ann).is_file():
            QMessageBox.warning(
                self,
                "Generate",
                "Annotations JSON not found.\n"
                "Annotate and save (e.g. video.annotations.json) first.\n\n"
                f"Expected: {ann}",
            )
            return
        if not tracks or not Path(tracks).is_dir():
            QMessageBox.warning(
                self,
                "Generate",
                "Set the post–ID-review tracklets folder on the Project tab.",
            )
            return
        if not out:
            QMessageBox.warning(self, "Generate", "Set an examples output folder.")
            return

        # Prefer mode from annotations file if present
        behavior_mode = int(s.behavior_mode)
        try:
            sess = AnnotationManager.load_from_json(ann).session
            behavior_mode = int(sess.behavior_mode)
        except Exception:
            pass

        cfg = GenerationConfig(
            video_path=video,
            annotations_path=ann,
            tracklets_dir=tracks,
            output_dir=out,
            length=int(s.window_length),
            behavior_mode=behavior_mode,
            sampling=str(s.sampling),
            stride=int(s.stride),
            min_bout_frames=int(s.min_bout_frames),
            social_distance=float(s.social_distance),
            write_soft_labels=bool(s.write_soft_labels),
            background_free=bool(s.background_free),
        )

        self.btn_run.setEnabled(False)
        self.progress.setValue(0)
        self.log.append("Starting generation…")

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
        out = result.get("output_dir") or self.project.state.inferred_examples_dir()
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
