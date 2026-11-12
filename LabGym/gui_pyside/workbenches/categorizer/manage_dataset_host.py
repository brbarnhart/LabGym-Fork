"""Manage dataset host: Categories | Review examples | Evaluate."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.workbenches.categorizer.categories_tab import CategoriesTab
from LabGym.gui_pyside.workbenches.categorizer.evaluate_tab import EvaluateTab
from LabGym.gui_pyside.workbenches.categorizer.review_examples_tab import (
    ReviewExamplesTab,
)


class _ExampleStoreScanThread(QThread):
    """Walk the example store off the UI thread. Chrome is already painted."""

    scan_finished = Signal(object, str)  # rows or None, error

    def __init__(self, store_root: str, parent=None):
        super().__init__(parent)
        self._store_root = store_root

    def run(self) -> None:
        try:
            from LabGym.training.dataset_manifest import scan_example_store

            rows = scan_example_store(self._store_root)
            self.scan_finished.emit(rows, "")
        except Exception as exc:
            self.scan_finished.emit(None, str(exc))


class ManageDatasetHost(QWidget):
    """Nested host matching Generate training data: three Manage areas.

    Phase 3: Evaluate. Phase 4: Review. Phase 5: Categories + soft projection.
    """

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._scan_thread: Optional[_ExampleStoreScanThread] = None
        self._scan_needed = True
        self._scan_gen = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        intro = QLabel(
            "Manage dataset: taxonomy operations, example review "
            "(keep / exclude / recategorize), split tooling, and evaluation "
            "of already-trained categorizers. Dataset curation lives in the "
            "example store's dataset manifest; scores live under each model."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.inner = QTabWidget()
        self.categories_tab = CategoriesTab(project)
        self.review_tab = ReviewExamplesTab(project)
        self.evaluate_tab = EvaluateTab(project)

        self.inner.addTab(self.categories_tab, "Categories")
        self.inner.addTab(self.review_tab, "Review examples")
        self.inner.addTab(self.evaluate_tab, "Evaluate")
        layout.addWidget(self.inner, 1)

        self.project.project_replaced.connect(self._on_project_replaced)

    def show_evaluate(self) -> None:
        """Focus the Evaluate subtab (e.g. from other workbench actions)."""
        self.inner.setCurrentWidget(self.evaluate_tab)

    def show_review(self) -> None:
        """Focus the Review examples subtab."""
        self.inner.setCurrentWidget(self.review_tab)

    def show_categories(self) -> None:
        """Focus the Categories subtab."""
        self.inner.setCurrentWidget(self.categories_tab)

    def showEvent(self, event) -> None:  # noqa: N802 — Qt
        super().showEvent(event)
        self._maybe_scan_store()

    def hideEvent(self, event) -> None:  # noqa: N802 — Qt
        self._cancel_scan()
        super().hideEvent(event)

    def _on_project_replaced(self) -> None:
        self._scan_needed = True
        self._cancel_scan()
        QTimer.singleShot(0, self._maybe_scan_store)

    def _store_root(self) -> str:
        text = self.categories_tab.ed_store.text().strip()
        if text:
            return text
        try:
            p = self.project.project
            return str(p.resolve_path(p.paths.examples_root or "examples"))
        except Exception:
            return ""

    def _cancel_scan(self) -> None:
        self._scan_gen += 1
        thread = self._scan_thread
        if thread is not None:
            thread.requestInterruption()
        self._scan_thread = None

    def _maybe_scan_store(self) -> None:
        """Scan when this host is shown. Do not scan on project swap."""
        if not self._scan_needed or not self.isVisible():
            return
        store = self._store_root()
        self.categories_tab.begin_store_chrome(store)
        self.review_tab.begin_store_chrome(store)
        if not store or not Path(store).is_dir():
            self._scan_needed = False
            return
        self._scan_needed = False
        self._cancel_scan()
        gen = self._scan_gen
        thread = _ExampleStoreScanThread(store, parent=self)
        thread.scan_finished.connect(
            lambda rows, err, g=gen: self._on_scan_finished(rows, err, g)
        )
        self._scan_thread = thread
        thread.start()

    def _on_scan_finished(self, rows: object, error: str, gen: int) -> None:
        if gen != self._scan_gen:
            return
        self._scan_thread = None
        self.categories_tab.apply_store_scan(rows, error)
        self.review_tab.apply_store_scan(rows, error)
