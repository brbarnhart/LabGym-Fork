"""PySide6 workbench shell for the ethogram-first UI migration.

Workbench icons across the top; each workbench owns its subtask tabs.
Projects are ``*.labproj.json`` (root folder + video list + defaults).

Phase 2: Categorizer → Generate training data (Annotate | Generate) is live.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.legacy_launch import launch_legacy_labgym
from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.project.editor_dialog import ProjectEditorDialog
from LabGym.gui_pyside.project.model import PROJECT_FILE_SUFFIX
from LabGym.gui_pyside.shell.workbench_bar import WorkbenchBar
from LabGym.gui_pyside.shell.workbench_host import WorkbenchHost
from LabGym.gui_pyside.workbenches.categorizer import CategorizerWorkbench
from LabGym.gui_pyside.workbenches.detector import DetectorWorkbench
from LabGym.gui_pyside.workbenches.preprocessing import PreprocessingWorkbench
from LabGym.gui_pyside.workbenches.results import ResultsWorkbench


class WorkbenchMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LabGym")
        self.resize(1280, 800)

        self.project = ProjectController(self)
        self._build_ui()
        self._build_menu()
        self._wire()
        self._update_title_and_status()

        # Restore last workbench; optionally offer last project via recent menu only
        last_wb = self.project.last_workbench_id()
        if last_wb in self.workbench_bar.ids():
            self.workbench_bar.set_current(last_wb)
        else:
            self.workbench_bar.set_current("categorizer")

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.workbench_bar = WorkbenchBar()
        layout.addWidget(self.workbench_bar)

        self.host = WorkbenchHost()
        self.wb_preprocessing = PreprocessingWorkbench(self.project)
        self.wb_detector = DetectorWorkbench(self.project)
        self.wb_categorizer = CategorizerWorkbench(self.project)
        self.wb_results = ResultsWorkbench(self.project)
        for wb in (
            self.wb_preprocessing,
            self.wb_detector,
            self.wb_categorizer,
            self.wb_results,
        ):
            self.host.add_workbench(wb)
            wb.connect_legacy(self._launch_legacy)
        self.wb_categorizer.connect_edit_project(self._edit_project)
        self.wb_detector.connect_edit_project(self._edit_project)
        self.wb_preprocessing.connect_edit_project(self._edit_project)

        layout.addWidget(self.host, 1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        act_new = QAction("&New Project…", self)
        act_new.setShortcut(QKeySequence.StandardKey.New)
        act_new.triggered.connect(self._new_project)
        file_menu.addAction(act_new)

        act_open = QAction("&Open Project…", self)
        act_open.setShortcut(QKeySequence.StandardKey.Open)
        act_open.triggered.connect(self._open_project)
        file_menu.addAction(act_open)

        self.recent_menu = file_menu.addMenu("Open &Recent")
        self._rebuild_recent_menu()

        file_menu.addSeparator()
        act_save = QAction("&Save Project", self)
        act_save.setShortcut(QKeySequence.StandardKey.Save)
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        act_save_as = QAction("Save Project &As…", self)
        act_save_as.setShortcut(QKeySequence.StandardKey.SaveAs)
        act_save_as.triggered.connect(self._save_project_as)
        file_menu.addAction(act_save_as)

        file_menu.addSeparator()
        act_quit = QAction("&Quit", self)
        act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        project_menu = menubar.addMenu("&Project")
        act_edit = QAction("&Edit Project…", self)
        act_edit.setShortcut(QKeySequence("Ctrl+E"))
        act_edit.triggered.connect(self._edit_project)
        project_menu.addAction(act_edit)

        act_annotate = QAction("Go to &Annotate ethogram", self)
        act_annotate.triggered.connect(self._goto_annotate_ethogram)
        project_menu.addAction(act_annotate)

        act_gen = QAction("Go to &Generate examples", self)
        act_gen.triggered.connect(self._goto_generate_examples)
        project_menu.addAction(act_gen)

        act_detect = QAction("Go to Detect + &track", self)
        act_detect.triggered.connect(self._goto_detect_track)
        project_menu.addAction(act_detect)

        act_review = QAction("Go to &Review IDs", self)
        act_review.triggered.connect(self._goto_review_ids)
        project_menu.addAction(act_review)

        tools_menu = menubar.addMenu("&Tools")
        act_legacy = QAction("Open LabGym (legacy wx)…", self)
        act_legacy.triggered.connect(self._launch_legacy)
        tools_menu.addAction(act_legacy)

        help_menu = menubar.addMenu("&Help")
        act_about = QAction("&About…", self)
        act_about.triggered.connect(self._about)
        help_menu.addAction(act_about)

    def _wire(self) -> None:
        self.workbench_bar.workbench_changed.connect(self._on_workbench_changed)
        self.project.changed.connect(self._update_title_and_status)

    def _on_workbench_changed(self, workbench_id: str) -> None:
        self.host.show_workbench(workbench_id)
        self.project.set_last_workbench_id(workbench_id)
        self._update_title_and_status()

    def goto(self, workbench_id: str, tab_id: Optional[str] = None) -> bool:
        """Navigate shell to a workbench and optional subtab."""
        self.workbench_bar.set_current(workbench_id)
        if tab_id:
            return self.host.goto_tab(workbench_id, tab_id)
        return True

    def _update_title_and_status(self) -> None:
        p = self.project.project
        dirty = " *" if self.project.dirty else ""
        name = p.display_name()
        self.setWindowTitle(f"LabGym — {name}{dirty}")
        self.statusBar().showMessage(self.project.status_summary())

    # --- project file ops ---

    def _confirm_discard(self) -> bool:
        if not self.project.dirty:
            return True
        r = QMessageBox.question(
            self,
            "Unsaved project",
            "The current project has unsaved changes. Discard them?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No
            | QMessageBox.StandardButton.Cancel,
        )
        if r == QMessageBox.StandardButton.Cancel:
            return False
        if r == QMessageBox.StandardButton.No:
            return self._save_project()
        return True

    def _new_project(self) -> None:
        if not self._confirm_discard():
            return
        self.project.new_project()
        self._edit_project()
        self._rebuild_recent_menu()

    def _open_project(self) -> None:
        if not self._confirm_discard():
            return
        start = self.project.last_project_path() or ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            start,
            f"LabGym Project (*{PROJECT_FILE_SUFFIX} *.json);;All Files (*)",
        )
        if not path:
            return
        try:
            self.project.load_from_path(path)
            self.project.set_last_project_path(path)
            self._rebuild_recent_menu()
        except Exception as exc:
            QMessageBox.critical(self, "Open Project", str(exc))

    def _open_recent(self, path: str) -> None:
        if not Path(path).is_file():
            QMessageBox.warning(self, "Open Recent", f"File not found:\n{path}")
            self._rebuild_recent_menu()
            return
        if not self._confirm_discard():
            return
        try:
            self.project.load_from_path(path)
            self.project.set_last_project_path(path)
            self._rebuild_recent_menu()
        except Exception as exc:
            QMessageBox.critical(self, "Open Project", str(exc))

    def _save_project(self) -> bool:
        if not self.project.project.file_path:
            return self._save_project_as()
        try:
            self.project.save()
            self.project.set_last_project_path(self.project.project.file_path)
            self._rebuild_recent_menu()
            self.statusBar().showMessage(
                "Saved  ·  " + self.project.status_summary(), 5000
            )
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save Project", str(exc))
            return False

    def _save_project_as(self) -> bool:
        start = self.project.project.file_path or (
            str(Path(self.project.project.root_dir) / f"{self.project.project.name}{PROJECT_FILE_SUFFIX}")
            if self.project.project.root_dir
            else f"{self.project.project.name}{PROJECT_FILE_SUFFIX}"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project As",
            start,
            f"LabGym Project (*{PROJECT_FILE_SUFFIX});;JSON (*.json)",
        )
        if not path:
            return False
        if not path.endswith(".json"):
            path = path + PROJECT_FILE_SUFFIX
        try:
            self.project.save(path)
            self.project.set_last_project_path(path)
            self._rebuild_recent_menu()
            self.statusBar().showMessage(
                "Saved  ·  " + self.project.status_summary(), 5000
            )
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save Project", str(exc))
            return False

    def _edit_project(self) -> None:
        dlg = ProjectEditorDialog(self.project, self)
        dlg.exec()
        self._update_title_and_status()

    def _goto_annotate_ethogram(self) -> None:
        self.goto("categorizer", "generate_training")
        self.wb_categorizer.generate_host.show_annotate()

    def _goto_generate_examples(self) -> None:
        self.goto("categorizer", "generate_training")
        self.wb_categorizer.generate_host.show_generate()

    def _goto_detect_track(self) -> None:
        self.goto("detector", "detect_track")

    def _goto_review_ids(self) -> None:
        self.goto("detector", "review_ids")

    def _rebuild_recent_menu(self) -> None:
        self.recent_menu.clear()
        paths = self.project.recent_paths()
        if not paths:
            act = QAction("(none)", self)
            act.setEnabled(False)
            self.recent_menu.addAction(act)
            return
        for path in paths:
            label = path
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, p=path: self._open_recent(p))
            self.recent_menu.addAction(act)
        self.recent_menu.addSeparator()
        act_clear = QAction("Clear recent", self)
        act_clear.triggered.connect(self._clear_recent)
        self.recent_menu.addAction(act_clear)

    def _clear_recent(self) -> None:
        self.project.clear_recent()
        self._rebuild_recent_menu()

    def _launch_legacy(self) -> None:
        try:
            launch_legacy_labgym()
            self.statusBar().showMessage(
                "Launched legacy LabGym (wx) in a separate process.", 5000
            )
        except Exception as exc:
            QMessageBox.critical(self, "Launch failed", str(exc))

    def _about(self) -> None:
        QMessageBox.information(
            self,
            "About LabGym",
            "LabGym workbench shell (PySide6)\n\n"
            "Ethogram-first workflow:\n"
            "Detect → Review IDs → Annotate ethogram → Generate pairs → "
            "Train → Process\n\n"
            "Phases 2–6 live: Generate training data, Review IDs,\n"
            "Detect + track, Preprocess / Draw markers,\n"
            "Train/Test detector & categorizer.\n"
            "Phase 7 (Process videos) still pending.\n"
            "See specifications.md and implementation-plan.md.",
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        event.accept()


# Back-compat alias used by older entry points / docs
WorkflowMainWindow = WorkbenchMainWindow


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("LabGym")
    app.setOrganizationName("LabGym")
    app.setStyle("Fusion")
    window = WorkbenchMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
