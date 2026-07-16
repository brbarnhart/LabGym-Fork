"""VideoDisplayWidget: QLabel-based display with overlay for open + saved behaviors.

Two separate rows:
- Bottom row: completed/saved bout annotations at this frame
- Row above: currently toggled-on (live / recording) behaviors
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QFont, QPen
from PyQt6.QtWidgets import QLabel

from utils.helpers import numpy_to_qpixmap


class VideoDisplayWidget(QLabel):
    """Displays current video frame + colored behavior overlay bars + text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #1a1a1a; color: #ddd;")
        self._current_pixmap = None
        # Live (toggled on, not yet closed)
        self._open_behaviors: List[str] = []
        # Saved bouts covering the current frame
        self._annotated_behaviors: List[str] = []
        self._behavior_colors: dict[str, str] = {}  # name -> hex

    def set_behavior_colors(self, mapping: dict[str, str]) -> None:
        self._behavior_colors = dict(mapping)

    def show_frame(
        self,
        frame: np.ndarray,
        active_behaviors: Optional[List[str]] = None,
        *,
        open_behaviors: Optional[List[str]] = None,
        annotated_behaviors: Optional[List[str]] = None,
    ):
        """Show a frame with optional behavior overlays.

        Prefer open_behaviors + annotated_behaviors for the two-row layout.
        active_behaviors alone is treated as annotated (legacy single-list callers).
        """
        if open_behaviors is not None or annotated_behaviors is not None:
            self._open_behaviors = list(open_behaviors or [])
            self._annotated_behaviors = list(annotated_behaviors or [])
        elif active_behaviors is not None:
            # Legacy: single list → show as annotated only
            self._open_behaviors = []
            self._annotated_behaviors = list(active_behaviors)

        try:
            pix = numpy_to_qpixmap(frame)
            self._current_pixmap = pix
            self.setPixmap(pix)
        except Exception as e:
            self.setText(f"[frame error: {e}]")

        self.update()  # trigger paintEvent for overlay

    def clear(self):
        self._current_pixmap = None
        self._open_behaviors = []
        self._annotated_behaviors = []
        super().clear()  # safely clears any pixmap (QLabel.setPixmap does not accept None)
        self.setText("No video loaded")

    def _draw_behavior_row(
        self,
        painter: QPainter,
        y0: int,
        bar_height: int,
        label: str,
        names: List[str],
        label_color: QColor,
        *,
        live: bool = False,
    ) -> None:
        """Draw one labeled strip of behavior chips."""
        # Background strip
        painter.fillRect(0, y0, self.width(), bar_height + 6, QColor(20, 20, 20, 200))

        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        # Row label (Live / Saved)
        painter.setPen(label_color)
        label_w = 52
        painter.drawText(6, y0 + 16, label)

        x = 6 + label_w
        for name in names:
            color_hex = self._behavior_colors.get(name, "#FF5555")
            color = QColor(color_hex)
            bar_w = max(60, min(160, len(name) * 9 + 20))

            painter.setBrush(color)
            if live:
                # Outline to distinguish currently recording chips
                painter.setPen(QPen(QColor(255, 255, 255), 2))
            else:
                painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(x, y0 + 3, bar_w, 18)

            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x + 6, y0 + 16, name[:20])

            x += bar_w + 6
            if x > self.width() - 20:
                break

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._open_behaviors and not self._annotated_behaviors:
            return

        painter = QPainter(self)
        try:
            h = self.height()
            bar_height = 24
            gap = 2

            # Bottom row = saved annotations; row above = live toggled-on
            y_saved = h - bar_height - 4
            y_live = y_saved - bar_height - gap

            if self._annotated_behaviors:
                self._draw_behavior_row(
                    painter,
                    y_saved,
                    bar_height,
                    "Saved",
                    self._annotated_behaviors,
                    QColor(180, 180, 180),
                    live=False,
                )

            if self._open_behaviors:
                # If there are no saved labels, put Live on the bottom row
                y = y_live if self._annotated_behaviors else y_saved
                self._draw_behavior_row(
                    painter,
                    y,
                    bar_height,
                    "Live",
                    self._open_behaviors,
                    QColor(255, 200, 80),
                    live=True,
                )
        finally:
            painter.end()
