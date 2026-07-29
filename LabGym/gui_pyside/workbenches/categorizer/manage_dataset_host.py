"""Manage dataset host: Categories | Review examples | Evaluate."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.workbenches.categorizer.evaluate_tab import EvaluateTab
from LabGym.gui_pyside.workbenches.categorizer.review_examples_tab import (
    ReviewExamplesTab,
)


class _PlaceholderSubtab(QWidget):
    """Temporary panel for Manage dataset areas not yet implemented."""

    def __init__(self, title: str, body: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        head = QLabel(f"<b>{title}</b>")
        head.setWordWrap(True)
        layout.addWidget(head)
        note = QLabel(body)
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)


class ManageDatasetHost(QWidget):
    """Nested host matching Generate training data: three Manage areas.

    Phase 3: Evaluate. Phase 4: Review examples + train effective view.
    Categories (taxonomy ops / soft projection) remains a later phase.
    """

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
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
        self.categories_tab = _PlaceholderSubtab(
            "Categories",
            "Coming next: merge / exclude categories (undoable), sealed-test "
            "and train/validation split tooling against dataset_manifest.json. "
            "Soft-label projection after taxonomy ops is also planned here.",
        )
        self.review_tab = ReviewExamplesTab(project)
        self.evaluate_tab = EvaluateTab(project)

        self.inner.addTab(self.categories_tab, "Categories")
        self.inner.addTab(self.review_tab, "Review examples")
        self.inner.addTab(self.evaluate_tab, "Evaluate")
        layout.addWidget(self.inner, 1)

    def show_evaluate(self) -> None:
        """Focus the Evaluate subtab (e.g. from other workbench actions)."""
        self.inner.setCurrentWidget(self.evaluate_tab)

    def show_review(self) -> None:
        """Focus the Review examples subtab."""
        self.inner.setCurrentWidget(self.review_tab)
