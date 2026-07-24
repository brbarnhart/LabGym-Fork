"""Legacy pipeline steps still running under the wx LabGym GUI."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.legacy_launch import launch_legacy_labgym


class PipelineTab(QWidget):
    """Detect, ID review, train, analyze — launch legacy until ported to PySide."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        intro = QLabel(
            "These steps still use the classic <b>wxPython LabGym</b> window. "
            "They open in a <b>separate process</b> so toolkits do not share an "
            "event loop. The ethogram-first core (Project → Annotate → Generate) "
            "is already PySide6."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        for title, body in (
            (
                "Detect & track",
                "Train or select a detector, run detection so tracklets / id_review "
                "artifacts are written next to your videos.",
            ),
            (
                "Fix ID swaps",
                "Open ID review, correct swaps at contacts, and save remapped "
                "tracklets. Point Project settings at that folder before annotating.",
            ),
            (
                "Train categorizer",
                "After Generate produces sorted behavior folders, train a categorizer "
                "on those folders. Optional hard_soft_aux with soft_labels.csv.",
            ),
            (
                "Analyze videos",
                "Run analysis with the trained categorizer on new videos.",
            ),
        ):
            box = QGroupBox(title)
            bl = QVBoxLayout(box)
            bl.addWidget(QLabel(body))
            btn = QPushButton("Open LabGym (legacy)")
            btn.clicked.connect(self._launch)
            bl.addWidget(btn)
            layout.addWidget(box)

        layout.addStretch(1)

    def _launch(self) -> None:
        try:
            launch_legacy_labgym()
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.critical(self, "Launch failed", str(exc))
