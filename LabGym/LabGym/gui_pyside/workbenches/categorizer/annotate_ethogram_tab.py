"""Categorizer → Generate training data → Annotate ethogram."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from LabGym.annotator.ui.main_window import MainWindow
from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.project.paths import list_project_video_choices


class AnnotateEthogramTab(QWidget):
    """Embedded Behavior Annotator wired to the workbench Project."""

    request_edit_project = Signal()

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._detached: Optional[MainWindow] = None
        self._block_combo = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Video:"))
        self.combo_video = QComboBox()
        self.combo_video.setMinimumWidth(220)
        self.combo_video.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.combo_video.currentIndexChanged.connect(self._on_video_combo)
        bar.addWidget(self.combo_video, 1)

        self.btn_load = QPushButton("Load into annotator")
        self.btn_load.setToolTip(
            "Open selected video + annotations + tracklets into the annotator"
        )
        self.btn_load.clicked.connect(self.load_current)
        bar.addWidget(self.btn_load)

        self.btn_reload = QPushButton("Refresh list")
        self.btn_reload.clicked.connect(self.refresh_video_list)
        bar.addWidget(self.btn_reload)

        self.btn_edit = QPushButton("Edit project…")
        self.btn_edit.clicked.connect(self.request_edit_project.emit)
        bar.addWidget(self.btn_edit)

        self.btn_detach = QPushButton("Separate window")
        self.btn_detach.setToolTip("Open annotator as a top-level window")
        self.btn_detach.clicked.connect(self.open_detached)
        bar.addWidget(self.btn_detach)

        layout.addLayout(bar)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(
            "QLabel { background: #2a2a2a; color: #eee; padding: 6px 8px; "
            "border-radius: 4px; }"
        )
        layout.addWidget(self.lbl_status)

        self.empty = QLabel(
            "No videos in this project. Use <b>Edit project…</b> to add a root "
            "folder and videos, then <b>Load into annotator</b>."
        )
        self.empty.setWordWrap(True)
        layout.addWidget(self.empty)

        self.annotator = MainWindow()
        self.annotator.setParent(self)
        self.annotator.setWindowFlags(Qt.WindowType.Widget)
        self.annotator.statusBar().showMessage(
            "Select a project video, then Load into annotator. Save ethogram with Ctrl+S."
        )
        layout.addWidget(self.annotator, 1)

        self.project.changed.connect(self._on_project_changed)
        self.project.project_replaced.connect(self._on_project_changed)
        self._on_project_changed()

    def _on_project_changed(self) -> None:
        self.refresh_video_list()
        self._update_status_banner()

    def refresh_video_list(self) -> None:
        choices = list_project_video_choices(self.project.project)
        self._block_combo = True
        self.combo_video.clear()
        cur = self.project.current_video_path()
        select_idx = 0
        for i, (label, resolved) in enumerate(choices):
            self.combo_video.addItem(label, resolved)
            try:
                if cur and Path(resolved).resolve() == Path(cur).resolve():
                    select_idx = i
            except OSError:
                if resolved == cur or label == cur:
                    select_idx = i
        if choices:
            self.combo_video.setCurrentIndex(select_idx)
            self.empty.hide()
            self.annotator.show()
            self.btn_load.setEnabled(True)
        else:
            self.empty.show()
            self.annotator.hide()
            self.btn_load.setEnabled(False)
        self._block_combo = False
        self._update_status_banner()

    def _on_video_combo(self, _index: int) -> None:
        if self._block_combo:
            return
        path = self.combo_video.currentData()
        if path:
            self.project.set_current_video(str(path), dirty=True)
        self._update_status_banner()

    def _selected_video_path(self) -> str:
        data = self.combo_video.currentData()
        if data:
            return str(data)
        return self.project.current_video_path()

    def _update_status_banner(self) -> None:
        ctx = self.project.resolve_context(self._selected_video_path() or None)
        lines = ctx.summary_lines()
        lines.append("Save ethogram with Ctrl+S (File → Save Annotations in annotator menu).")
        if not ctx.video_path:
            self.lbl_status.setText("No video selected.")
        elif not Path(ctx.video_path).is_file():
            self.lbl_status.setText(
                "Video path not found on disk:\n" + ctx.video_path
            )
        else:
            self.lbl_status.setText("  ·  ".join(lines))

    def load_current(self) -> bool:
        return self._apply_to(self.annotator)

    def open_detached(self) -> None:
        if self._detached is not None and self._detached.isVisible():
            self._detached.raise_()
            self._detached.activateWindow()
            self._apply_to(self._detached)
            return
        win = MainWindow()
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        win.destroyed.connect(self._on_detached_closed)
        self._detached = win
        win.show()
        win.raise_()
        self._apply_to(win)

    def _on_detached_closed(self, *_args) -> None:
        self._detached = None

    def _apply_to(self, window: MainWindow) -> bool:
        video = self._selected_video_path()
        if not video:
            QMessageBox.information(
                self,
                "Annotate ethogram",
                "Add videos to the project (Edit project…) first.",
            )
            self.request_edit_project.emit()
            return False
        if not Path(video).is_file():
            QMessageBox.warning(
                self,
                "Annotate ethogram",
                f"Video file not found:\n{video}",
            )
            return False

        ctx = self.project.resolve_context(video)
        self.project.set_current_video(video, dirty=True)

        ann = ctx.annotations_path if ctx.annotations_path else None
        # Only pass annotations_path if file exists; otherwise prefer sidecar search
        ann_arg = ann if (ann and Path(ann).is_file()) else None
        prefer_sidecar = ann_arg is None

        ok = window.load_video_from_path(
            video,
            annotations_path=ann_arg,
            tracklets_dir=ctx.tracklets_dir or None,
            behavior_mode=int(ctx.behavior_mode),
            exclusive_mode=bool(ctx.exclusive_mode),
            prefer_sidecar=prefer_sidecar,
        )
        if ok and ctx.tracklets_dir and window._loaded_tracklets is None:
            window.load_tracklets_from_path(ctx.tracklets_dir)
        if ok:
            note = ""
            if not ctx.tracklets_exists:
                note = "  ·  No tracklets found (ID dots optional; Detect/Review later)."
            window.statusBar().showMessage(
                f"Loaded project video{note}  ·  mode={ctx.behavior_mode}"
            )
            self._update_status_banner()
        return ok
