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

from LabGym.annotator.core.tracklets_bridge import LoadedTracklets
from LabGym.annotator.ui.main_window import MainWindow
from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.project.paths import (
    list_project_video_choices,
    paths_refer_to_same_video,
)
from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
    AnnotateSessionLoadThread,
    AnnotateSessionMetadata,
    loaded_tracklets_placeholder,
    overlay_subject_display,
)


def _annotator_alive(window: Optional[MainWindow]) -> bool:
    """False when the C++ QWidget has already been deleted."""
    if window is None:
        return False
    try:
        from shiboken6 import isValid
    except ImportError:
        return True
    return bool(isValid(window))


def _bgr_to_rgb(frame: object) -> object:
    """Convert OpenCV BGR preview to the annotator widget's RGB layout."""
    import numpy as np

    if frame is None or not isinstance(frame, np.ndarray):
        return frame
    if frame.ndim != 3 or frame.shape[2] != 3:
        return frame
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class AnnotateEthogramTab(QWidget):
    """Embedded Behavior Annotator wired to the workbench Project."""

    request_edit_project = Signal()

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._detached: Optional[MainWindow] = None
        self._block_combo = False
        self._session_thread: Optional[AnnotateSessionLoadThread] = None
        self._session_gen = 0
        self._session_window: Optional[MainWindow] = None
        self._session_meta: Optional[AnnotateSessionMetadata] = None

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

        self.btn_cancel_load = QPushButton("Cancel")
        self.btn_cancel_load.setEnabled(False)
        self.btn_cancel_load.setToolTip(
            "Abort the in-flight session load. Choosing another video also replaces it."
        )
        self.btn_cancel_load.clicked.connect(lambda: self._cancel_session_load())
        bar.addWidget(self.btn_cancel_load)

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
        self._wire_open_video(self.annotator)
        self.annotator.setParent(self)
        self.annotator.setWindowFlags(Qt.WindowType.Widget)
        self.annotator.statusBar().showMessage(
            "Select a project video, then Load into annotator. Save ethogram with Ctrl+S."
        )
        layout.addWidget(self.annotator, 1)

        # Video list + path resolve only when the project object is replaced.
        self.project.project_replaced.connect(self._on_project_replaced)
        self.project.changed.connect(self._update_status_banner)
        self._on_project_replaced()

    def _on_project_replaced(self) -> None:
        self.drop_session()
        self.refresh_video_list()

    def has_unsaved_ethogram(self) -> bool:
        """True when the embedded or detached annotator has an unsaved ethogram.

        Counts bouts, behaviors, exclusive mode, and open starts — not the
        ``*.labproj.json`` dirty flag.
        """
        if self.annotator.has_unsaved_ethogram():
            return True
        if self._detached is not None:
            return self._detached.has_unsaved_ethogram()
        return False

    def save_ethogram(self) -> bool:
        """Persist unsaved ethogram(s). Does not auto-save on **project swap**."""
        ok = True
        if self.annotator.has_unsaved_ethogram():
            ok = self.annotator.save_ethogram() and ok
        if self._detached is not None and self._detached.has_unsaved_ethogram():
            ok = self._detached.save_ethogram() and ok
        return ok

    def discard_unsaved_ethogram(self) -> None:
        """Revert in-memory ethogram edits; do not write."""
        self.annotator.discard_unsaved_ethogram()
        if self._detached is not None:
            self._detached.discard_unsaved_ethogram()

    def drop_session(self) -> None:
        """Release annotator capture and tracklets; do not **session load**."""
        self._cancel_session_load()
        self.annotator.drop_session()
        if self._detached is not None:
            self._detached.drop_session()
        self._session_meta = None
        self._session_window = None

    def refresh_video_list(self) -> None:
        """Fill the video combo from the project document (no disk probe)."""
        choices = list_project_video_choices(self.project.project)
        self._block_combo = True
        self.combo_video.clear()
        cur = self.project.current_video_path()
        select_idx = 0
        for i, (label, resolved) in enumerate(choices):
            self.combo_video.addItem(label, resolved)
            if cur and (
                paths_refer_to_same_video(str(resolved), cur)
                or paths_refer_to_same_video(label, cur)
            ):
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
        self.lbl_status.setText(
            "Select a project video, then Load into annotator."
        )

    def _on_video_combo(self, _index: int) -> None:
        if self._block_combo:
            return
        path = self.combo_video.currentData()
        if path:
            self.project.set_current_video(str(path), dirty=True)
        self._update_status_banner()
        if self._session_thread is not None:
            target = (
                self._session_window
                if _annotator_alive(self._session_window)
                else self.annotator
            )
            if not self._apply_to(target):
                self._cancel_session_load()

    def _selected_video_path(self) -> str:
        data = self.combo_video.currentData()
        if data:
            return str(data)
        return self.project.current_video_path()

    def _update_status_banner(self) -> None:
        if self._session_thread is not None:
            return
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
        """Start a session load of the selected video into the embedded annotator.

        Returns:
            False if no video is selected or the accepted-identities gate fails.
        """
        return self._apply_to(self.annotator)

    def open_detached(self) -> None:
        """Open or raise a top-level annotator window and session-load into it."""
        if self._detached is not None and self._detached.isVisible():
            self._detached.raise_()
            self._detached.activateWindow()
            self._apply_to(self._detached)
            return
        win = MainWindow()
        self._wire_open_video(win)
        win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        win.destroyed.connect(self._on_detached_closed)
        self._detached = win
        win.show()
        win.raise_()
        self._apply_to(win)

    def _on_detached_closed(self, *_args) -> None:
        closed = self._detached
        self._detached = None
        if self._session_window is closed:
            self._cancel_session_load()
            self._session_window = None

    def _wire_open_video(self, window: MainWindow) -> None:
        """File → Open must use the same accepted-identities gate as Load."""
        window.open_video_handler = lambda path, w=window: self._open_video_from_file(
            w, path
        )

    def _open_video_from_file(self, window: MainWindow, path: str) -> bool:
        """Start a **session load** for File → Open (gate runs off the UI thread).

        Returns True when the attempt was handled (started or refused).
        """
        if not path:
            QMessageBox.warning(
                window, "Annotate ethogram", "No video path given."
            )
            return True
        self._start_session_load(window, path)
        return True

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
        self._start_session_load(window, video)
        return True

    def _start_session_load(self, window: MainWindow, video_path: str) -> None:
        """Replace any in-flight session load with a new one.

        Hunt, accepted-identities gate, and npz/video I/O run on the worker.
        Does not mark the project dirty until metadata unlocks.

        Args:
            window: Embedded or detached annotator window.
            video_path: Selected project video (document path, not probed).
        """
        self._cancel_session_load()
        window.drop_session()
        gen = self._session_gen
        thread = AnnotateSessionLoadThread(
            "",
            video_path=str(video_path or ""),
            project=self.project.project,
            behavior_mode=int(self.project.project.defaults.behavior_mode),
            exclusive_mode=bool(self.project.project.defaults.exclusive_mode),
            parent=self,
        )
        thread.metadata_ready.connect(
            lambda meta, g=gen, w=window: self._on_session_metadata(w, meta, g)
        )
        thread.tracklets_ready.connect(
            lambda loaded, g=gen, w=window: self._on_session_tracklets(w, loaded, g)
        )
        thread.frame_ready.connect(
            lambda frame, g=gen, w=window: self._on_session_frame(w, frame, g)
        )
        thread.tracklets_error.connect(
            lambda msg, g=gen, w=window: self._on_session_tracklets_error(w, msg, g)
        )
        thread.video_error.connect(
            lambda msg, g=gen, w=window: self._on_session_video_error(w, msg, g)
        )
        thread.load_error.connect(
            lambda msg, g=gen, w=window: self._on_session_load_error(w, msg, g)
        )
        thread.gate_failed.connect(
            lambda msg, g=gen, w=window: self._on_session_gate_failed(w, msg, g)
        )
        thread.finished.connect(lambda g=gen: self._on_session_thread_finished(g))
        self._session_thread = thread
        self._session_window = window
        self.btn_cancel_load.setEnabled(True)
        self.lbl_status.setText("Loading package JSON…")
        thread.start()

    def _cancel_session_load(self) -> None:
        """Abort the in-flight session load without dropping loaded metadata."""
        self._session_gen += 1
        thread = self._session_thread
        if thread is not None:
            thread.cancel()
            thread.finished.connect(thread.deleteLater)
        self._session_thread = None
        self.btn_cancel_load.setEnabled(False)

    def _on_session_metadata(self, window: MainWindow, meta: object, gen: int) -> None:
        if gen != self._session_gen or not _annotator_alive(window):
            return
        if not isinstance(meta, AnnotateSessionMetadata):
            return
        self._apply_session_metadata(window, meta, gen)

    def _apply_session_metadata(
        self, window: MainWindow, meta: AnnotateSessionMetadata, gen: int
    ) -> None:
        """Unlock subjects / n_frames / ethogram from package JSON."""
        if gen != self._session_gen or not _annotator_alive(window):
            return
        self._session_meta = meta
        if meta.video_path:
            self.project.set_current_video(str(meta.video_path), dirty=True)
        sidecar = meta.annotations_data is not None
        placeholder = None
        if meta.subjects and not sidecar:
            placeholder = loaded_tracklets_placeholder(meta.review_dir, meta.subjects)
        window.begin_session_from_metadata(
            video_path=meta.video_path,
            n_frames=meta.n_frames,
            fps=meta.fps,
            subjects=meta.subjects,
            annotations_path=meta.annotations_path or None,
            annotations_data=meta.annotations_data,
            behavior_mode=meta.behavior_mode,
            exclusive_mode=meta.exclusive_mode,
            tracklets_placeholder=placeholder,
        )
        if sidecar and window.manager is not None:
            overlay_subject_display(window.manager.session, meta.subjects)
            if window.subject_panel:
                window.subject_panel.set_manager(window.manager)
        n = int(meta.n_frames or 0)
        self.lbl_status.setText(
            f"Frame 0/{max(0, n - 1)}  |  loading video and outlines…"
        )

    def _on_session_frame(self, window: MainWindow, frame: object, gen: int) -> None:
        if gen != self._session_gen or not _annotator_alive(window):
            return
        window.show_session_preview(_bgr_to_rgb(frame))

    def _on_session_tracklets(
        self, window: MainWindow, loaded: object, gen: int
    ) -> None:
        if gen != self._session_gen or not _annotator_alive(window):
            return
        if not isinstance(loaded, LoadedTracklets):
            return
        window.apply_loaded_tracklets(loaded)
        if window.manager:
            n = window.manager.session.total_frames
            self.lbl_status.setText(f"Loaded outlines  ·  {n} frames")

    def _on_session_tracklets_error(
        self, window: MainWindow, message: str, gen: int
    ) -> None:
        if gen != self._session_gen or not _annotator_alive(window):
            return
        if window.manager:
            window.manager.mark_saved()
        window.show_canvas_error(f"Could not load outlines: {message}")
        self.lbl_status.setText(f"Could not load outlines: {message}")

    def _on_session_video_error(
        self, window: MainWindow, message: str, gen: int
    ) -> None:
        if gen != self._session_gen or not _annotator_alive(window):
            return
        window.show_canvas_error(message)
        self.lbl_status.setText(message)

    def _on_session_load_error(
        self, window: MainWindow, message: str, gen: int
    ) -> None:
        if gen != self._session_gen or not _annotator_alive(window):
            return
        QMessageBox.warning(window, "Annotate ethogram", message)
        self.lbl_status.setText(message)

    def _on_session_gate_failed(
        self, window: MainWindow, message: str, gen: int
    ) -> None:
        if gen != self._session_gen or not _annotator_alive(window):
            return
        QMessageBox.warning(window, "Review IDs required", message)
        self.lbl_status.setText(message)

    def _on_session_thread_finished(self, gen: int) -> None:
        if gen != self._session_gen:
            return
        self._session_thread = None
        self.btn_cancel_load.setEnabled(False)
