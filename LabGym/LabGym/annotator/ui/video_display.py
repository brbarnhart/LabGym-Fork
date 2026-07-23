"""VideoDisplayWidget: frame display with behavior bars + track identity overlays.

Video and track overlays share one letterboxed transform so they stay aligned
when the window is resized.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRect, QRectF, QSize
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QPolygonF, QPixmap, QResizeEvent
from PySide6.QtWidgets import QLabel, QSizePolicy

from LabGym.annotator.utils.helpers import numpy_to_qpixmap


class VideoDisplayWidget(QLabel):
    """Displays current video frame + track overlays + behavior chips.

    Painting is done manually (not via QLabel scaledContents) so track geometry
    always uses the same scale/offset as the drawn frame.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #1a1a1a; color: #ddd;")
        # Do not let QLabel scale/stretch the pixmap independently of overlays
        self.setScaledContents(False)
        self._current_pixmap: Optional[QPixmap] = None
        self._frame_w = 0
        self._frame_h = 0
        # Live (toggled on, not yet closed)
        self._open_behaviors: List[str] = []
        # Saved bouts covering the current frame
        self._annotated_behaviors: List[str] = []
        self._behavior_colors: dict = {}  # name -> hex
        # Track overlays (image coords)
        self._track_overlays: List = []
        self._active_subject_id: Optional[int] = None
        self._show_tracks: bool = True

    def set_behavior_colors(self, mapping: dict) -> None:
        self._behavior_colors = dict(mapping)

    def set_active_subject_id(self, subject_id: Optional[int]) -> None:
        self._active_subject_id = subject_id
        self.update()

    def set_track_overlays(self, overlays: Sequence) -> None:
        self._track_overlays = list(overlays or [])
        self.update()

    def set_show_tracks(self, show: bool) -> None:
        self._show_tracks = bool(show)
        self.update()

    def show_frame(
        self,
        frame: np.ndarray,
        active_behaviors: Optional[List[str]] = None,
        *,
        open_behaviors: Optional[List[str]] = None,
        annotated_behaviors: Optional[List[str]] = None,
        track_overlays: Optional[Sequence] = None,
        active_subject_id: Optional[int] = None,
    ):
        """Show a frame with optional behavior + track overlays."""
        if open_behaviors is not None or annotated_behaviors is not None:
            self._open_behaviors = list(open_behaviors or [])
            self._annotated_behaviors = list(annotated_behaviors or [])
        elif active_behaviors is not None:
            self._open_behaviors = []
            self._annotated_behaviors = list(active_behaviors)

        if track_overlays is not None:
            self._track_overlays = list(track_overlays)
        if active_subject_id is not None:
            self._active_subject_id = active_subject_id

        try:
            if frame is not None and frame.ndim >= 2:
                self._frame_h, self._frame_w = int(frame.shape[0]), int(frame.shape[1])
            pix = numpy_to_qpixmap(frame)
            self._current_pixmap = pix
            # Clear QLabel's own pixmap so we fully control painting in paintEvent
            # (avoids double-draw / wrong scale from QLabel's default path).
            super().clear()
            self.setText("")
        except Exception as e:
            self._current_pixmap = None
            self.setText(f"[frame error: {e}]")

        self.update()

    def clear(self):
        self._current_pixmap = None
        self._frame_w = 0
        self._frame_h = 0
        self._open_behaviors = []
        self._annotated_behaviors = []
        self._track_overlays = []
        super().clear()
        self.setText("No video loaded")

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        # Recompute letterbox on every resize so overlays stay locked to video
        self.update()

    def _content_rect(self) -> QRectF:
        """Letterboxed destination rect for the video inside this widget.

        Always KeepAspectRatio fit of the *source frame* into the widget.
        Same transform is used for drawing the pixmap and mapping track coords.
        """
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return QRectF(0, 0, self.width(), self.height())

        # Prefer true video pixel size; fall back to pixmap size
        fw = self._frame_w or self._current_pixmap.width()
        fh = self._frame_h or self._current_pixmap.height()
        if fw <= 0 or fh <= 0:
            return QRectF(0, 0, self.width(), self.height())

        avail_w = max(1, self.width())
        avail_h = max(1, self.height())
        scale = min(avail_w / float(fw), avail_h / float(fh))
        dw = fw * scale
        dh = fh * scale
        x = (avail_w - dw) / 2.0
        y = (avail_h - dh) / 2.0
        return QRectF(x, y, dw, dh)

    def _image_to_widget(self, x: float, y: float) -> QPointF:
        """Map original image coordinates to widget coordinates."""
        rect = self._content_rect()
        fw = self._frame_w or (
            self._current_pixmap.width() if self._current_pixmap else 1
        )
        fh = self._frame_h or (
            self._current_pixmap.height() if self._current_pixmap else 1
        )
        if fw <= 0 or fh <= 0:
            return QPointF(x, y)
        sx = rect.width() / float(fw)
        sy = rect.height() / float(fh)
        return QPointF(rect.x() + x * sx, rect.y() + y * sy)

    def _draw_video(self, painter: QPainter) -> None:
        if self._current_pixmap is None or self._current_pixmap.isNull():
            return
        dest = self._content_rect()
        # Smooth scaled draw into letterbox rect
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        target = QRect(
            int(round(dest.x())),
            int(round(dest.y())),
            max(1, int(round(dest.width()))),
            max(1, int(round(dest.height()))),
        )
        painter.drawPixmap(target, self._current_pixmap)

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
        painter.fillRect(0, y0, self.width(), bar_height + 6, QColor(20, 20, 20, 200))

        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

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
                painter.setPen(QPen(QColor(255, 255, 255), 2))
            else:
                painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(x, y0 + 3, bar_w, 18)

            painter.setPen(QColor(255, 255, 255))
            painter.drawText(x + 6, y0 + 16, name[:20])

            x += bar_w + 6
            if x > self.width() - 20:
                break

    def _draw_tracks(self, painter: QPainter) -> None:
        if not self._show_tracks or not self._track_overlays:
            return
        if self._current_pixmap is None:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        # Scale pen width slightly with video scale so outlines stay visible
        rect = self._content_rect()
        fw = max(1, self._frame_w or self._current_pixmap.width())
        scale = rect.width() / float(fw)

        for ov in self._track_overlays:
            sid = getattr(ov, "subject_id", None)
            color = QColor(getattr(ov, "color", "#4FC3F7"))
            active = sid is not None and sid == self._active_subject_id
            width = (3.0 if active else 1.5) * max(scale, 0.5)
            alpha = 230 if active else 160
            color.setAlpha(alpha)

            pen = QPen(color, width)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            contour = getattr(ov, "contour", None)
            if contour is not None and getattr(ov, "valid", True):
                arr = np.asarray(contour).reshape(-1, 2)
                if arr.shape[0] >= 2:
                    poly = QPolygonF(
                        [
                            self._image_to_widget(float(p[0]), float(p[1]))
                            for p in arr
                        ]
                    )
                    painter.drawPolygon(poly)

            center = getattr(ov, "center", None)
            if center is not None and getattr(ov, "valid", True):
                pt = self._image_to_widget(float(center[0]), float(center[1]))
                r = (6 if active else 4) * max(scale, 0.5)
                painter.setBrush(color)
                painter.drawEllipse(pt, r, r)
                painter.setPen(QColor(255, 255, 255) if active else color)
                painter.drawText(pt + QPointF(8 * max(scale, 0.5), -4), str(sid))

    def paintEvent(self, event):
        # Background only from QLabel; we paint video ourselves
        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor(26, 26, 26))

            if self._current_pixmap is not None and not self._current_pixmap.isNull():
                self._draw_video(painter)
                self._draw_tracks(painter)
            else:
                # Placeholder text when no frame
                painter.setPen(QColor(200, 200, 200))
                painter.drawText(
                    self.rect(),
                    int(Qt.AlignmentFlag.AlignCenter),
                    self.text() or "No video loaded",
                )

            if not self._open_behaviors and not self._annotated_behaviors:
                if self._active_subject_id is not None:
                    painter.fillRect(0, 0, self.width(), 22, QColor(20, 20, 20, 180))
                    painter.setPen(QColor(200, 220, 255))
                    painter.drawText(
                        8, 16, f"Subject {self._active_subject_id}"
                    )
                return

            h = self.height()
            bar_height = 24
            gap = 2

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

            if self._active_subject_id is not None:
                painter.fillRect(0, 0, self.width(), 22, QColor(20, 20, 20, 180))
                painter.setPen(QColor(200, 220, 255))
                painter.drawText(8, 16, f"Subject {self._active_subject_id}")
        finally:
            painter.end()
