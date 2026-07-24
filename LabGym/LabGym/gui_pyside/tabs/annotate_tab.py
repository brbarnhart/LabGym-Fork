"""Annotate tab: embedded Behavior Annotator + project load controls."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from LabGym.annotator.ui.main_window import MainWindow
from LabGym.gui_pyside.project_state import ProjectController


class AnnotateTab(QWidget):
    """Hosts the PySide ethogram annotator as an embedded workspace."""

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._detached: Optional[MainWindow] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        bar = QHBoxLayout()
        self.lbl = QLabel("Annotate ethograms for the active project.")
        self.lbl.setWordWrap(True)
        bar.addWidget(self.lbl, 1)

        self.btn_load = QPushButton("Load project into annotator")
        self.btn_load.setToolTip(
            "Open video + annotations + tracklets from Project settings"
        )
        self.btn_load.clicked.connect(self.load_project)
        bar.addWidget(self.btn_load)

        self.btn_detach = QPushButton("Open in separate window")
        self.btn_detach.setToolTip(
            "Detach a second annotator window (same process; project paths applied)"
        )
        self.btn_detach.clicked.connect(self.open_detached)
        bar.addWidget(self.btn_detach)

        layout.addLayout(bar)

        # Embed annotator MainWindow as a child widget (menus stay available)
        self.annotator = MainWindow()
        self.annotator.setParent(self)
        self.annotator.setWindowFlags(Qt.WindowType.Widget)
        # Slightly smaller default chrome when embedded
        self.annotator.statusBar().showMessage(
            "Embedded annotator — use Project settings, then “Load project into annotator”."
        )
        layout.addWidget(self.annotator, 1)

        self.project.changed.connect(self._on_project_changed)
        self._on_project_changed()

    def _on_project_changed(self) -> None:
        s = self.project.state
        self.lbl.setText(
            "Annotate ethograms  ·  "
            + s.status_summary()
            + "  ·  Save with Ctrl+S when finished."
        )

    def load_project(self) -> bool:
        """Apply Project settings to the embedded annotator."""
        return self._apply_to(self.annotator)

    def open_detached(self) -> None:
        """Open annotator as a top-level window with project context."""
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
        s = self.project.state
        if not s.video_path:
            QMessageBox.information(
                self,
                "Annotate",
                "Set a video path on the Project tab first.",
            )
            return False

        ann = s.inferred_annotations_path()
        ok = window.load_video_from_path(
            s.video_path,
            annotations_path=ann if ann else None,
            tracklets_dir=s.tracklets_dir or None,
            behavior_mode=int(s.behavior_mode),
            exclusive_mode=bool(s.exclusive_mode),
            prefer_sidecar=not bool(ann),
        )
        # If annotations path was set but file does not exist yet, load_video
        # already started a fresh session; mode/exclusive applied above.
        if ok and s.tracklets_dir and window._loaded_tracklets is None:
            window.load_tracklets_from_path(s.tracklets_dir)
        return ok

    def get_annotator(self) -> MainWindow:
        return self.annotator
