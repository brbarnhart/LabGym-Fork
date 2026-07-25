"""Detector → Generate training data → Generate images (frame extraction)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.project.controller import ProjectController

_VIDEO_FILTER = (
    "Video files (*.avi *.mpg *.mpeg *.mp4 *.mkv *.m4v *.mov *.wmv);;All files (*.*)"
)


class _ExtractWorker(QObject):
    finished = Signal(int)  # number of videos processed
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        videos: List[str],
        out_path: str,
        framewidth: Optional[int],
        start_t: float,
        duration: float,
        skip_redundant: int,
    ):
        super().__init__()
        self.videos = videos
        self.out_path = out_path
        self.framewidth = framewidth
        self.start_t = start_t
        self.duration = duration
        self.skip_redundant = skip_redundant

    def run(self) -> None:
        try:
            from LabGym.tools import extract_frames

            Path(self.out_path).mkdir(parents=True, exist_ok=True)
            n = len(self.videos)
            for i, path in enumerate(self.videos, start=1):
                name = Path(path).name
                self.progress.emit(f"[{i}/{n}] Extracting frames from {name}…")
                extract_frames(
                    path,
                    self.out_path,
                    framewidth=self.framewidth,
                    start_t=self.start_t,
                    duration=self.duration,
                    skip_redundant=self.skip_redundant,
                )
            self.finished.emit(n)
        except Exception as exc:
            self.error.emit(str(exc))


class GenerateImagesTab(QWidget):
    """Extract still frames from videos for detector training (classic LabGym)."""

    request_annotate = Signal()
    request_train = Signal()

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._videos: List[str] = []
        self._thread: Optional[QThread] = None

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Extract still frames from videos to use as detector training images. "
                "After extraction, annotate them externally (see the <b>Annotate images</b> "
                "subtab), then use <b>Train detector</b> with the image folder and COCO JSON."
            )
        )

        form = QFormLayout()

        self.ed_videos = QLineEdit()
        self.ed_videos.setReadOnly(True)
        self.ed_videos.setPlaceholderText("No videos selected")
        self.ed_videos.setToolTip(
            "One or more videos. Common formats (mp4, mov, avi, m4v, mkv, mpg, mpeg) "
            "are supported."
        )
        row_v = QHBoxLayout()
        row_v.addWidget(self.ed_videos, 1)
        b_vid = QPushButton("Browse…")
        b_vid.clicked.connect(self._browse_videos)
        b_proj = QPushButton("Use project videos")
        b_proj.setToolTip("Load the video list from the open project.")
        b_proj.clicked.connect(self._use_project_videos)
        row_v.addWidget(b_vid)
        row_v.addWidget(b_proj)
        form.addRow("Videos:", row_v)

        self.chk_resize = QCheckBox("Proportionally resize frame width")
        self.chk_resize.setToolTip(
            "Optional. Reducing frame size speeds up later detector training. "
            "Leave unchecked to keep original resolution."
        )
        self.spin_width = QSpinBox()
        self.spin_width.setRange(10, 10000)
        self.spin_width.setValue(480)
        self.spin_width.setEnabled(False)
        self.spin_width.setToolTip("Target frame width in pixels (height scales with aspect ratio).")
        self.chk_resize.toggled.connect(self.spin_width.setEnabled)
        row_r = QHBoxLayout()
        row_r.addWidget(self.chk_resize)
        row_r.addWidget(QLabel("Width (px):"))
        row_r.addWidget(self.spin_width)
        row_r.addStretch(1)
        form.addRow("Resize:", row_r)

        self.ed_out = QLineEdit()
        self.ed_out.setToolTip("Folder where extracted JPG frames will be written.")
        b_out = QPushButton("Browse…")
        b_out.clicked.connect(self._browse_out)
        form.addRow("Output folder:", self._row(self.ed_out, b_out))

        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0.0, 1e9)
        self.spin_start.setDecimals(2)
        self.spin_start.setValue(0.0)
        self.spin_start.setSuffix(" s")
        self.spin_start.setToolTip("Beginning time (seconds) for extraction on every selected video.")
        form.addRow("Start time:", self.spin_start)

        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0.0, 1e9)
        self.spin_duration.setDecimals(2)
        self.spin_duration.setValue(0.0)
        self.spin_duration.setSuffix(" s")
        self.spin_duration.setToolTip(
            "How long to extract after start time. 0 = from start time to end of each video."
        )
        form.addRow("Duration (0 = to end):", self.spin_duration)

        self.spin_skip = QSpinBox()
        self.spin_skip.setRange(1, 1_000_000)
        self.spin_skip.setValue(1000)
        self.spin_skip.setToolTip(
            "Write one image every N frames. Larger intervals make training images more diverse "
            "and keep the set smaller. Classic LabGym default is 1000."
        )
        form.addRow("Skip interval (frames):", self.spin_skip)

        layout.addLayout(form)

        row_btn = QHBoxLayout()
        self.btn_run = QPushButton("Start generating images")
        self.btn_run.setToolTip("Extract frames from all selected videos into the output folder.")
        self.btn_run.clicked.connect(self._run)
        row_btn.addWidget(self.btn_run)
        row_btn.addStretch(1)
        layout.addLayout(row_btn)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        layout.addWidget(self.log)
        layout.addStretch(1)

        self.project.changed.connect(self._maybe_prefills)
        self._maybe_prefills()

    def _row(self, line: QLineEdit, button: QPushButton) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(line, 1)
        h.addWidget(button)
        return w

    def _maybe_prefills(self) -> None:
        p = self.project.project
        if not self.ed_out.text().strip() and p.root_dir:
            default = Path(p.root_dir) / "detector_training_images"
            self.ed_out.setText(str(default))

    def _browse_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select video(s)", "", _VIDEO_FILTER)
        if paths:
            self._set_videos(paths)

    def _use_project_videos(self) -> None:
        root = self.project.project.root_dir or ""
        paths = [
            str(v.resolved_path(root))
            for v in self.project.project.enabled_videos()
            if v.path
        ]
        if not paths:
            QMessageBox.information(
                self,
                "No project videos",
                "The open project has no enabled videos. Add them via Project → Edit "
                "project, or browse for files.",
            )
            return
        self._set_videos(paths)

    def _set_videos(self, paths: List[str]) -> None:
        self._videos = list(paths)
        n = len(self._videos)
        parent = str(Path(self._videos[0]).parent) if self._videos else ""
        self.ed_videos.setText(f"{n} video(s)" + (f" in {parent}" if parent else ""))

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output folder for images")
        if d:
            self.ed_out.setText(d)

    def _run(self) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "Busy", "Extraction is already running.")
            return
        if not self._videos:
            QMessageBox.warning(self, "Missing input", "Select one or more videos first.")
            return
        out = self.ed_out.text().strip()
        if not out:
            QMessageBox.warning(self, "Missing output", "Select an output folder for the images.")
            return

        reply = QMessageBox.question(
            self,
            "Start generating images?",
            f"Extract frames from {len(self._videos)} video(s) into:\n{out}",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        framewidth = self.spin_width.value() if self.chk_resize.isChecked() else None
        worker = _ExtractWorker(
            videos=list(self._videos),
            out_path=out,
            framewidth=framewidth,
            start_t=float(self.spin_start.value()),
            duration=float(self.spin_duration.value()),
            skip_redundant=int(self.spin_skip.value()),
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self._on_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._clear_thread)

        self._thread = thread
        self.btn_run.setEnabled(False)
        self.log.append("Starting image extraction…")
        thread.start()

    def _clear_thread(self) -> None:
        self._thread = None
        self.btn_run.setEnabled(True)

    def _on_progress(self, msg: str) -> None:
        self.log.append(msg)

    def _on_finished(self, n: int) -> None:
        out = self.ed_out.text().strip()
        self.log.append(f"Done. Processed {n} video(s). Images in: {out}")
        QMessageBox.information(
            self,
            "Image generation completed",
            "Frames were written to the output folder.\n\n"
            "Next: annotate them (Annotate images subtab / EZannot), then open "
            "Train detector with the image folder and COCO JSON.",
        )
        self.request_annotate.emit()

    def _on_error(self, err: str) -> None:
        self.log.append(f"Error: {err}")
        QMessageBox.critical(self, "Extraction failed", err)
