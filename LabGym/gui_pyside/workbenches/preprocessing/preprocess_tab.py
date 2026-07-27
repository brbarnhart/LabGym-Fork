"""Preprocessing → Preprocess videos (PySide + tools.preprocess_video)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.widgets.path_browse import (
    browse_existing_directory,
    path_edit_row,
    set_line_edit_directory,
)
from LabGym.gui_pyside.project.paths import list_project_video_choices
from LabGym.tools import preprocess_video


class _PreprocessWorker(QObject):
    finished = Signal(int, int)  # ok, fail
    # path, short status for table, log line
    progress = Signal(str, str, str)

    def __init__(self, videos: List[str], out_dir: str, kwargs: dict):
        super().__init__()
        self.videos = videos
        self.out_dir = out_dir
        self.kwargs = kwargs

    def run(self) -> None:
        ok = fail = 0
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        for path in self.videos:
            name = Path(path).name
            self.progress.emit(path, "running", f"Processing {name}…")
            try:
                preprocess_video(path, self.out_dir, **self.kwargs)
                ok += 1
                self.progress.emit(path, "done", f"OK {name}")
            except Exception as exc:
                fail += 1
                self.progress.emit(path, "error", f"ERROR {name}: {exc}")
        self.finished.emit(ok, fail)


class PreprocessTab(QWidget):
    request_edit_project = Signal()

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None
        self._worker: Optional[_PreprocessWorker] = None
        self._batch_active = False
        self._status_by_path: Dict[str, str] = {}
        self._row_by_path: Dict[str, int] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Trim, crop, resize, enhance brightness/contrast, and/or reduce FPS. "
            "Uses LabGym <code>preprocess_video</code>."
        ))

        out_row = QHBoxLayout()
        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("Output folder for *_processed.avi")
        self.ed_out.setToolTip(
            "Folder where preprocessed videos are written as "
            "<name>_processed.avi (and with trim windows in the filename if trimming)."
        )
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._browse_out)
        lab_out = QLabel("Output:")
        lab_out.setToolTip(self.ed_out.toolTip())
        out_row.addWidget(lab_out)
        out_row.addWidget(self.ed_out, 1)
        out_row.addWidget(btn_out)
        layout.addLayout(out_row)

        opts = QGroupBox("Options")
        form = QFormLayout(opts)
        self.spin_width = QSpinBox()
        self.spin_width.setRange(0, 4000)
        self.spin_width.setSpecialValueText("original")
        self.spin_width.setValue(0)
        tip_w = (
            "Resize output frames so width is this many pixels (height scales to "
            "keep aspect ratio). “original”/0 = keep native size. Smaller widths "
            "speed later detection; 480 is a common LabGym choice."
        )
        self.spin_width.setToolTip(tip_w)
        form.addRow(self._lab("Frame width (resize):", tip_w), self.spin_width)

        self.chk_trim = QCheckBox("Trim to time windows")
        self.chk_trim.setToolTip(
            "If checked, keep only the listed time segments and concatenate them "
            "into one shorter output video."
        )
        form.addRow(self.chk_trim)
        self.ed_windows = QLineEdit("0-10")
        self.ed_windows.setToolTip(
            "Time windows in seconds: start-end,start-end,… "
            "Example: 0-30,120-180 keeps the first 30 s and a later minute."
        )
        form.addRow(
            self._lab(
                "Time windows (s):",
                "Only used when “Trim to time windows” is checked.",
            ),
            self.ed_windows,
        )

        self.chk_crop = QCheckBox("Crop frames")
        self.chk_crop.setToolTip(
            "If checked, keep only the rectangle defined by Left/Right/Top/Bottom "
            "(pixel coordinates after optional resize)."
        )
        form.addRow(self.chk_crop)
        crop = QHBoxLayout()
        self.spin_left = QSpinBox(); self.spin_left.setRange(0, 10000)
        self.spin_right = QSpinBox(); self.spin_right.setRange(0, 10000)
        self.spin_top = QSpinBox(); self.spin_top.setRange(0, 10000)
        self.spin_bottom = QSpinBox(); self.spin_bottom.setRange(0, 10000)
        for w, lab in (
            (self.spin_left, "L"),
            (self.spin_right, "R"),
            (self.spin_top, "T"),
            (self.spin_bottom, "B"),
        ):
            w.setToolTip(
                "Crop edge in pixels (after resize if resizing). L/R = horizontal, "
                "T/B = vertical. Only used when Crop frames is checked."
            )
            crop.addWidget(QLabel(lab))
            crop.addWidget(w)
        form.addRow(
            self._lab("Crop L/R/T/B:", "Pixel bounds of the crop window."),
            crop,
        )

        self.chk_bright = QCheckBox("Enhance brightness")
        self.spin_bright = QDoubleSpinBox()
        self.spin_bright.setRange(0.0, 10.0)
        self.spin_bright.setValue(1.0)
        tip_b = (
            "Brightness factor (PIL ImageEnhance). 1.0 = no change; >1 brighter; "
            "<1 darker. Only applied when the checkbox is on."
        )
        self.chk_bright.setToolTip(tip_b)
        self.spin_bright.setToolTip(tip_b)
        form.addRow(self.chk_bright, self.spin_bright)

        self.chk_contrast = QCheckBox("Enhance contrast")
        self.spin_contrast = QDoubleSpinBox()
        self.spin_contrast.setRange(0.0, 10.0)
        self.spin_contrast.setValue(1.0)
        tip_c = (
            "Contrast factor (PIL ImageEnhance). 1.0 = no change; >1 more contrast. "
            "Helps detection when illumination is flat. Only if checkbox is on."
        )
        self.chk_contrast.setToolTip(tip_c)
        self.spin_contrast.setToolTip(tip_c)
        form.addRow(self.chk_contrast, self.spin_contrast)

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(0, 240)
        self.spin_fps.setSpecialValueText("original")
        self.spin_fps.setValue(0)
        tip_fps = (
            "Downsample to this frames-per-second by dropping frames. "
            "“original”/0 keeps the source FPS. Lower FPS speeds later analysis "
            "if behavior dynamics allow it. Cannot raise FPS above the source."
        )
        self.spin_fps.setToolTip(tip_fps)
        form.addRow(self._lab("Target FPS:", tip_fps), self.spin_fps)
        layout.addWidget(opts)

        # Videos
        vbox = QGroupBox("Videos")
        vl = QVBoxLayout(vbox)
        row = QHBoxLayout()
        btn_ref = QPushButton("Refresh from project")
        btn_ref.clicked.connect(self.refresh_videos)
        btn_edit = QPushButton("Edit project…")
        btn_edit.clicked.connect(self.request_edit_project.emit)
        btn_add = QPushButton("Add extra video(s)…")
        btn_add.clicked.connect(self._add_extra)
        row.addWidget(btn_ref)
        row.addWidget(btn_edit)
        row.addWidget(btn_add)
        row.addStretch(1)
        vl.addLayout(row)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "Video", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setColumnWidth(0, 36)
        self.table.setColumnWidth(2, 100)
        vl.addWidget(self.table)
        layout.addWidget(vbox, 1)

        self.btn_run = QPushButton("Start preprocessing")
        self.btn_run.clicked.connect(self._run)
        layout.addWidget(self.btn_run)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        layout.addWidget(self.log)

        self.project.changed.connect(self.refresh_videos)
        self.project.project_replaced.connect(self.refresh_videos)
        self._extra: List[str] = []
        self._set_default_out()
        self.refresh_videos()

    @staticmethod
    def _lab(text: str, tip: str) -> QLabel:
        lab = QLabel(text)
        lab.setToolTip(tip)
        return lab

    def _set_default_out(self) -> None:
        p = self.project.project
        rel = p.paths.processed_root or "processed"
        if p.root_dir:
            self.ed_out.setText(str(p.resolve_path(rel)))
        else:
            self.ed_out.setText(rel)

    def _browse_out(self) -> None:
        set_line_edit_directory(self, self.ed_out, caption="Output folder")

    def refresh_videos(self) -> None:
        # project.changed during other workbench jobs must not thrash this table
        # while a preprocess batch is running.
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
        paths = [str(r) for _, r in choices] + list(self._extra)
        self.table.setRowCount(len(paths))
        self._row_by_path.clear()
        for r, path in enumerate(paths):
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
            self.table.setItem(r, 1, QTableWidgetItem(Path(path).name))
            status = self._status_by_path.get(path, "pending")
            self.table.setItem(r, 2, QTableWidgetItem(status))
            self._row_by_path[path] = r

    def _add_extra(self) -> None:
        if self._batch_active:
            QMessageBox.information(
                self, "Busy", "Finish the current batch before adding videos."
            )
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add videos",
            self.project.project.root_dir or "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.mpg *.mpeg);;All Files (*)",
        )
        for p in paths:
            if p not in self._extra:
                self._extra.append(p)
        self.refresh_videos()

    def _selected(self) -> List[str]:
        out = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it and it.checkState() == Qt.CheckState.Checked:
                out.append(str(it.data(Qt.ItemDataRole.UserRole)))
        return out

    def _parse_windows(self) -> List[List[float]]:
        text = self.ed_windows.text().strip()
        if not text:
            return [[0, 10]]
        windows = []
        for part in text.split(","):
            part = part.strip()
            if not part or "-" not in part:
                continue
            a, b = part.split("-", 1)
            windows.append([float(a), float(b)])
        return windows or [[0, 10]]

    def _run(self) -> None:
        if self._thread is not None or self._batch_active:
            QMessageBox.information(self, "Busy", "Preprocessing already running.")
            return
        videos = self._selected()
        out = self.ed_out.text().strip()
        if not videos:
            QMessageBox.warning(self, "Preprocess", "Select at least one video.")
            return
        if not out:
            QMessageBox.warning(self, "Preprocess", "Choose an output folder.")
            return
        fw = int(self.spin_width.value()) or None
        fps = int(self.spin_fps.value()) or None
        kwargs = dict(
            framewidth=fw,
            trim_video=self.chk_trim.isChecked(),
            time_windows=self._parse_windows(),
            enhance_brightness=self.chk_bright.isChecked(),
            enhance_contrast=self.chk_contrast.isChecked(),
            brightness=float(self.spin_bright.value()),
            contrast=float(self.spin_contrast.value()),
            crop_frame=self.chk_crop.isChecked(),
            left=int(self.spin_left.value()),
            right=int(self.spin_right.value()),
            top=int(self.spin_top.value()),
            bottom=int(self.spin_bottom.value()),
            fps_new=fps,
        )
        self._batch_active = True
        for path in videos:
            self._status_by_path[path] = "queued"
            row = self._row_by_path.get(path)
            if row is not None and self.table.item(row, 2):
                self.table.item(row, 2).setText("queued")
        self.btn_run.setEnabled(False)
        self.log.append(f"Starting {len(videos)} video(s) → {out}")
        self._thread = QThread(self)
        self._worker = _PreprocessWorker(videos, out, kwargs)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()

    def _on_progress(self, path: str, status: str, log_line: str) -> None:
        self._status_by_path[path] = status
        row = self._row_by_path.get(path)
        if row is not None:
            if self.table.item(row, 2):
                self.table.item(row, 2).setText(status)
            else:
                self.table.setItem(row, 2, QTableWidgetItem(status))
        self.log.append(log_line)

    def _cleanup(self) -> None:
        self._worker = None
        self._thread = None
        self._batch_active = False
        self.btn_run.setEnabled(True)

    def _on_done(self, ok: int, fail: int) -> None:
        self.log.append(f"Done. ok={ok} fail={fail}")
        self._batch_active = False
        self.refresh_videos()
        QMessageBox.information(
            self, "Preprocess", f"Finished.\nSucceeded: {ok}\nFailed: {fail}"
        )
