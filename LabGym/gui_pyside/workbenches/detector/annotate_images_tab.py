"""Detector → Generate training data → Annotate images (future placeholder)."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Official EZannot repo (LabGym docs recommend it for detector outlines).
_EZANNOT_URL = "https://github.com/yujiahu415/EZannot"


class AnnotateImagesTab(QWidget):
    """Placeholder until in-app (EZannot-like) detector annotation ships."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Annotate detector training images")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        body = QLabel(
            "In-app annotation for detector training images is a <b>planned future feature</b> "
            "(similar to tools such as EZannot: AI-assisted instance outlines and "
            "LabGym-oriented export).<br><br>"
            "Until that lands inside LabGym, generate frames with the "
            "<b>Generate images</b> subtab, then annotate them externally and export "
            "<b>COCO instance segmentation</b> JSON for <b>Train detector</b>."
        )
        body.setWordWrap(True)
        body.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(body)

        rec = QLabel(
            "<b>Recommended for now: EZannot</b><br>"
            "EZannot is free, runs privately on your machine, and is tailored to LabGym "
            "Detectors (one-click outlines, strong augmentation). "
            "Alternatives such as CVAT, Labelme, or Roboflow also work if you export "
            "COCO instance segmentation and avoid unwanted resize/auto-orient steps."
        )
        rec.setWordWrap(True)
        rec.setTextFormat(Qt.TextFormat.RichText)
        rec.setStyleSheet(
            "background: #1e2a36; color: #dce6f0; padding: 12px; border-radius: 6px;"
        )
        layout.addWidget(rec)

        note = QLabel(
            "Workflow: Generate images → annotate with EZannot (or similar) → "
            "Train detector with the image folder + COCO JSON."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #9ab; margin-top: 8px;")
        layout.addWidget(note)

        row = QHBoxLayout()
        btn = QPushButton("Open EZannot on GitHub…")
        btn.setToolTip(_EZANNOT_URL)
        btn.clicked.connect(self._open_ezannot)
        row.addWidget(btn)
        row.addStretch(1)
        layout.addLayout(row)

        layout.addStretch(1)

    def _open_ezannot(self) -> None:
        QDesktopServices.openUrl(QUrl(_EZANNOT_URL))
