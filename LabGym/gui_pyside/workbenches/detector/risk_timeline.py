"""Risk / switch timeline for ID review (PySide)."""

from __future__ import annotations

from typing import List, Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget

from LabGym.id_review.types import ContactEvent, SwitchMarker


class RiskTimeline(QWidget):
    """Paint risk bands, switch markers, and playhead; click to seek."""

    seek_requested = Signal(int)

    def __init__(self, parent=None, height: int = 56):
        super().__init__(parent)
        self.setMinimumHeight(height)
        self.setMaximumHeight(height + 8)
        self.n_frames = 1
        self.frame = 0
        self.events: List[ContactEvent] = []
        self.markers: List[SwitchMarker] = []
        self.min_risk = 0.0
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_data(
        self,
        n_frames: int,
        frame: int,
        events: Sequence[ContactEvent],
        markers: Sequence[SwitchMarker],
        min_risk: float = 0.0,
    ) -> None:
        self.n_frames = max(1, int(n_frames))
        self.frame = int(max(0, min(frame, self.n_frames - 1)))
        self.events = list(events)
        self.markers = list(markers)
        self.min_risk = float(min_risk)
        self.update()

    def _x_to_frame(self, x: int) -> int:
        w = max(1, self.width())
        return int(max(0, min(round(x / w * (self.n_frames - 1)), self.n_frames - 1)))

    def _frame_to_x(self, f: int) -> int:
        w = max(1, self.width())
        if self.n_frames <= 1:
            return 0
        return int(round(f / (self.n_frames - 1) * (w - 1)))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.seek_requested.emit(self._x_to_frame(int(event.position().x())))
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 34))
        w, h = self.width(), self.height()

        # baseline
        p.setPen(QPen(QColor(80, 80, 90), 1))
        p.drawLine(0, h // 2, w, h // 2)

        for ev in self.events:
            if ev.risk_score < self.min_risk:
                continue
            x0 = self._frame_to_x(ev.start_frame)
            x1 = max(x0 + 2, self._frame_to_x(ev.end_frame))
            r = int(180 + 75 * min(1.0, ev.risk_score))
            g = int(120 * (1.0 - min(1.0, ev.risk_score)))
            b = 40
            if "possible_swap" in (ev.risk_flags or []):
                g = min(g, 60)
                r = 255
            p.fillRect(x0, 8, max(2, x1 - x0), h - 16, QColor(r, g, b, 180))

        p.setPen(QPen(QColor(80, 255, 120), 2))
        for m in self.markers:
            x = self._frame_to_x(m.frame)
            p.drawLine(x, 2, x, h - 2)
            p.setBrush(QColor(80, 255, 120))
            p.drawEllipse(x - 3, 7, 7, 7)

        px = self._frame_to_x(self.frame)
        p.setPen(QPen(QColor(255, 255, 0), 2))
        p.drawLine(px, 0, px, h)
        p.setPen(QColor(180, 180, 180))
        p.drawText(
            4,
            h - 4,
            "risk bands  |  green = switches  |  yellow = playhead",
        )
