"""Preprocessing → Preprocess videos (PySide + tools.preprocess_video)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QKeySequence, QShortcut
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
from LabGym.gui_pyside.project.paths import list_project_video_choices
from LabGym.gui_pyside.widgets.path_browse import set_line_edit_directory
from LabGym.gui_pyside.workbenches.preprocessing.crop_geometry import (
    CropEdges,
    FrameSize,
    mismatch_lines,
    resized_frame_size,
)
from LabGym.gui_pyside.workbenches.preprocessing.crop_preview import (
    crop_kwargs_for_video,
    probe_video,
)
from LabGym.gui_pyside.workbenches.preprocessing.crop_window import CropWindow
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
                video_kwargs = crop_kwargs_for_video(path, self.kwargs)
                preprocess_video(path, self.out_dir, **video_kwargs)
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
        self._syncing_crop = False

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Drag on the picture to place the crop window. Options below it "
            "trim, resize, enhance, or reduce FPS. The video list is on the right."
        ))

        body = QHBoxLayout()
        left = QVBoxLayout()
        self._crop_editor = CropWindow(self)
        self._crop_editor.edges_edited.connect(self._on_editor_edges)
        self._crop_editor.crop_enabled_edited.connect(self._on_editor_crop_flag)
        left.addWidget(self._crop_editor, 1)
        undo = QShortcut(QKeySequence.StandardKey.Undo, self._crop_editor)
        undo.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        undo.activated.connect(self._crop_editor._on_undo)

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
        left.addLayout(out_row)

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
        self.spin_width.valueChanged.connect(self._on_frame_width_changed)
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
            "If checked, keep only the crop window: Left/Right/Top/Bottom "
            "pixel edges of the frame after the optional width resize."
        )
        self.chk_crop.toggled.connect(self._on_crop_toggled)
        form.addRow(self.chk_crop)
        crop = QHBoxLayout()
        self.spin_left = QSpinBox()
        self.spin_right = QSpinBox()
        self.spin_top = QSpinBox()
        self.spin_bottom = QSpinBox()
        for box, label, tip in (
            (self.spin_left, "L", "Left edge of the crop window, in resized-frame pixels."),
            (self.spin_right, "R", "Right edge of the crop window (exclusive), in resized-frame pixels."),
            (self.spin_top, "T", "Top edge of the crop window, in resized-frame pixels."),
            (self.spin_bottom, "B", "Bottom edge of the crop window (exclusive), in resized-frame pixels."),
        ):
            box.setRange(0, 10000)
            box.setToolTip(tip)
            box.valueChanged.connect(self._on_tab_crop_edges)
            crop.addWidget(QLabel(label))
            crop.addWidget(box)
            edge_undo = QShortcut(QKeySequence.StandardKey.Undo, box)
            edge_undo.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            edge_undo.activated.connect(self._crop_editor._on_undo)
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
        left.addWidget(opts)

        self.btn_run = QPushButton("Start preprocessing")
        self.btn_run.clicked.connect(self._run)
        left.addWidget(self.btn_run)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(80)
        left.addWidget(self.log)
        body.addLayout(left, 3)

        vbox = QGroupBox("Videos")
        vbox.setMinimumWidth(260)
        vl = QVBoxLayout(vbox)
        btn_ref = QPushButton("Refresh from project")
        btn_ref.clicked.connect(self.refresh_videos)
        btn_edit = QPushButton("Edit project…")
        btn_edit.clicked.connect(self.request_edit_project.emit)
        btn_add = QPushButton("Add extra video(s)…")
        btn_add.clicked.connect(self._add_extra)
        vl.addWidget(btn_ref)
        vl.addWidget(btn_edit)
        vl.addWidget(btn_add)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "Video", "Status"])
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_highlight_changed)
        self.table.setColumnWidth(0, 36)
        self.table.setColumnWidth(2, 100)
        vl.addWidget(self.table, 1)
        body.addWidget(vbox, 2)
        layout.addLayout(body, 1)

        # Full table rebuild only on project replace/open/edit — not every dirty/save.
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
        previous = self._crop_editor.current_path() or self._highlighted_path()

        choices = list_project_video_choices(self.project.project)
        paths = [str(r) for _, r in choices] + list(self._extra)
        self.table.blockSignals(True)
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
        self.table.blockSignals(False)
        self._focus_preview(previous)

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

    def _framewidth(self) -> Optional[int]:
        value = int(self.spin_width.value())
        return value or None

    def _crop_edges(self) -> CropEdges:
        return CropEdges(
            int(self.spin_left.value()),
            int(self.spin_top.value()),
            int(self.spin_right.value()),
            int(self.spin_bottom.value()),
        )

    def _path_at_row(self, row: int) -> str:
        item = self.table.item(row, 0)
        if item is None:
            return ""
        return str(item.data(Qt.ItemDataRole.UserRole) or "")

    def _highlighted_path(self) -> str:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return ""
        return self._path_at_row(selected[0].row())

    def _ensure_crop_editor(self) -> CropWindow:
        return self._crop_editor

    def _focus_preview(self, previous: str) -> None:
        """Show ``previous`` again, or the first checked video when it is gone."""
        if self.table.rowCount() == 0:
            self._crop_editor.show_empty()
            return
        path = previous if previous in self._row_by_path else ""
        if not path:
            checked = self._selected()
            path = checked[0] if checked else self._path_at_row(0)
        row = self._row_by_path.get(path, 0)
        self.table.blockSignals(True)
        self.table.selectRow(row)
        self.table.blockSignals(False)
        if self._crop_editor.current_path() != path:
            self._crop_editor.show_video(path, self._framewidth(), reset_frame=True)
            self._expand_crop_ranges()

    def _expand_crop_ranges(self) -> None:
        preview_w = 0
        preview_h = 0
        if self._crop_editor is not None:
            size = self._crop_editor.preview_size()
            if size is not None:
                preview_w = size.width
                preview_h = size.height
        for box, room in (
            (self.spin_left, max(self.spin_left.value(), preview_w)),
            (self.spin_right, max(self.spin_right.value(), preview_w)),
            (self.spin_top, max(self.spin_top.value(), preview_h)),
            (self.spin_bottom, max(self.spin_bottom.value(), preview_h)),
        ):
            box.setMaximum(max(10000, room, box.value()))

    def _on_highlight_changed(self) -> None:
        path = self._highlighted_path()
        if not path or path == (self._crop_editor.current_path() or ""):
            return
        self._crop_editor.show_video(path, self._framewidth(), reset_frame=True)
        self._expand_crop_ranges()

    def _on_frame_width_changed(self) -> None:
        editor = self._crop_editor
        editor.apply_frame_width(self._framewidth())
        if editor.current_path():
            editor.reload_frame()
            self._expand_crop_ranges()

    def _on_tab_crop_edges(self) -> None:
        if self._syncing_crop or self._crop_editor is None:
            return
        self._crop_editor.apply_typed_edges(self._crop_edges())

    def _on_crop_toggled(self, checked: bool) -> None:
        if self._syncing_crop or self._crop_editor is None:
            return
        self._crop_editor.set_crop_enabled(checked)

    def _on_editor_edges(self, left: int, top: int, right: int, bottom: int) -> None:
        self._syncing_crop = True
        try:
            for box, value in (
                (self.spin_left, left),
                (self.spin_top, top),
                (self.spin_right, right),
                (self.spin_bottom, bottom),
            ):
                box.setMaximum(max(box.maximum(), value, 10000))
                box.setValue(int(value))
        finally:
            self._syncing_crop = False

    def _on_editor_crop_flag(self, checked: bool) -> None:
        self._syncing_crop = True
        try:
            self.chk_crop.setChecked(checked)
        finally:
            self._syncing_crop = False

    def _confirm_size_mismatch(
        self,
        remembered: FrameSize,
        rows: List[tuple],
    ) -> bool:
        """Ask whether to run when checked videos do not match the crop's frame.

        Args:
            remembered: Resized size the crop window was set on.
            rows: ``(path, FrameSize)`` for each checked video that differs.

        Returns:
            True when the user continues. False leaves the crop window as it is.
        """
        lines = "\n".join(
            f"{Path(path).name} — {size.width}×{size.height}" for path, size in rows
        )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Crop window")
        box.setText(
            f"The crop window was set on a {remembered.width}×{remembered.height} frame.\n"
            "These videos resize to a different size:\n\n"
            f"{lines}\n\n"
            "Continue applies these pixel edges and clamps each video to its own frame."
        )
        continue_btn = box.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        return box.clickedButton() is continue_btn

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
        edges = self._crop_edges()
        if self.chk_crop.isChecked() and not edges.valid():
            QMessageBox.warning(
                self,
                "Crop window",
                "The crop window needs a positive width and height "
                "(Right greater than Left and Bottom greater than Top).",
            )
            return
        remembered = None if self._crop_editor is None else self._crop_editor.remembered
        if self.chk_crop.isChecked() and remembered is not None:
            sized = []
            for path in videos:
                info = probe_video(path)
                if info is None:
                    continue
                size = resized_frame_size(info.width, info.height, self._framewidth())
                sized.append((path, size))
            mismatches = mismatch_lines(remembered.resized, sized)
            if mismatches and not self._confirm_size_mismatch(remembered.resized, list(mismatches)):
                return
        fw = self._framewidth()
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
        self._start_batch(videos, out, kwargs)

    def _start_batch(self, videos: List[str], out: str, kwargs: dict) -> None:
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
