"""Timeline widget: bout overview, seek, and optional multi-range selection for export.

Selection mode is off by default so normal annotation only seeks.
When selection mode is on, left-drag adds additive time ranges used to filter
example-clip export to intersecting annotated bouts.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QColor, QMouseEvent, QKeyEvent, QPen, QCursor
from PyQt6.QtWidgets import QWidget, QMenu

from core.annotation_manager import AnnotationManager

# Pixel movement before a press is treated as a drag-select (not a seek)
_DRAG_PIXEL_THRESHOLD = 4


class TimelineWidget(QWidget):
    seek_requested = pyqtSignal(int)  # frame
    selection_changed = pyqtSignal(list)  # list[tuple[int, int]]
    selection_mode_changed = pyqtSignal(bool)

    def __init__(
        self,
        manager_getter: Callable[[], Optional[AnnotationManager]],
        video_getter: Callable,
        parent=None,
    ):
        """
        manager_getter: callable() -> AnnotationManager | None
        video_getter: callable() -> VideoHandler | None
        """
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.setMaximumHeight(52)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.manager_getter = manager_getter
        self.video_getter = video_getter
        self._current_frame = 0

        self._selection_mode = False
        self._selections: List[Tuple[int, int]] = []

        # Drag state
        self._press_x: Optional[float] = None
        self._press_frame: Optional[int] = None
        self._drag_current_frame: Optional[int] = None
        self._dragging = False
        self._active_button: Optional[Qt.MouseButton] = None

        self._update_style()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_current_frame(self, frame: int):
        self._current_frame = frame
        self.update()

    def is_selection_mode(self) -> bool:
        return self._selection_mode

    def set_selection_mode(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._selection_mode:
            return
        self._selection_mode = enabled
        self._reset_drag()
        self._update_style()
        self.selection_mode_changed.emit(enabled)
        self.update()

    def get_selections(self) -> List[Tuple[int, int]]:
        return list(self._selections)

    def clear_selections(self) -> None:
        if not self._selections and not self._dragging:
            return
        self._selections.clear()
        self._reset_drag()
        self.selection_changed.emit([])
        self.update()

    def set_selections(self, ranges: List[Tuple[int, int]]) -> None:
        self._selections = [self._normalize_range(a, b) for a, b in ranges]
        self.selection_changed.emit(self.get_selections())
        self.update()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_style(self) -> None:
        if self._selection_mode:
            self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            self.setToolTip(
                "Selection mode ON — drag to add export ranges (additive).\n"
                "Click to seek · Right-click a range to delete · Esc clears all"
            )
            self.setStyleSheet(
                "TimelineWidget { border: 2px solid #4a9eff; background: #222; }"
            )
        else:
            self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            self.setToolTip("Click to seek. Enable selection mode to drag-export ranges.")
            self.setStyleSheet("TimelineWidget { border: 1px solid #444; background: #1e1e1e; }")

    def _reset_drag(self) -> None:
        self._press_x = None
        self._press_frame = None
        self._drag_current_frame = None
        self._dragging = False
        self._active_button = None

    def _total_frames(self) -> int:
        vid = self.video_getter()
        if not vid or getattr(vid, "total_frames", 0) < 1:
            return 0
        return int(vid.total_frames)

    def _x_to_frame(self, x: float) -> int:
        total = self._total_frames()
        if total < 1:
            return 0
        w = max(1, self.width())
        frame = int((x / w) * total)
        return max(0, min(frame, total - 1))

    def _frame_to_x(self, frame: int) -> int:
        total = self._total_frames()
        if total < 1:
            return 0
        return int((frame / total) * self.width())

    def _normalize_range(self, a: int, b: int) -> Tuple[int, int]:
        total = self._total_frames()
        lo, hi = (a, b) if a <= b else (b, a)
        if total > 0:
            lo = max(0, min(lo, total - 1))
            hi = max(0, min(hi, total - 1))
        if hi < lo:
            hi = lo
        return lo, hi

    def _range_at_x(self, x: float) -> Optional[int]:
        """Return index of selection under x, or None."""
        frame = self._x_to_frame(x)
        # Prefer smallest range containing frame (most specific)
        hits: List[Tuple[int, int]] = []
        for i, (s, e) in enumerate(self._selections):
            if s <= frame <= e:
                hits.append((e - s, i))
        if not hits:
            return None
        hits.sort()
        return hits[0][1]

    def _emit_seek_at(self, x: float) -> None:
        if self._total_frames() < 1:
            return
        self.seek_requested.emit(self._x_to_frame(x))

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()

        # Background
        bg = QColor(30, 30, 30)
        if self._selection_mode:
            bg = QColor(28, 36, 48)
        painter.fillRect(0, 0, w, h, bg)

        mgr = self.manager_getter()
        vid = self.video_getter()
        if not mgr or not vid or vid.total_frames < 1:
            if self._selection_mode:
                painter.setPen(QColor(150, 180, 220))
                painter.drawText(8, h // 2 + 4, "Selection mode (load a video)")
            painter.end()
            return

        total = vid.total_frames
        scale = w / total

        # Draw bouts per behavior
        behaviors = mgr.session.behaviors
        n = max(1, len(behaviors))
        row_h = max(3, h // n)

        for idx, beh in enumerate(behaviors):
            y = idx * row_h
            color = QColor(beh.color)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)

            for bout in mgr.get_bouts_for_behavior(beh.name):
                x1 = int(bout.start_frame * scale)
                x2 = int((bout.end_frame + 1) * scale)
                painter.drawRect(x1, y, max(1, x2 - x1), row_h - 1)

        # Selection overlays (committed) — light translucent fill so bout colors remain visible
        for s, e in self._selections:
            x1 = int(s * scale)
            x2 = int((e + 1) * scale)
            rw = max(2, x2 - x1)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.fillRect(x1, 0, rw, h, QColor(80, 170, 255, 40))  # ~16% opacity
            painter.setPen(QPen(QColor(140, 210, 255, 220), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(x1, 0, rw - 1, h - 1)
            # Soft edge markers so the range is obvious even with a faint fill
            painter.fillRect(x1, 0, 2, h, QColor(140, 210, 255, 180))
            painter.fillRect(x1 + rw - 2, 0, 2, h, QColor(140, 210, 255, 180))

        # In-progress drag — also translucent
        if self._dragging and self._press_frame is not None and self._drag_current_frame is not None:
            s, e = self._normalize_range(self._press_frame, self._drag_current_frame)
            x1 = int(s * scale)
            x2 = int((e + 1) * scale)
            rw = max(2, x2 - x1)
            painter.fillRect(x1, 0, rw, h, QColor(255, 210, 80, 45))
            painter.setPen(QPen(QColor(255, 230, 120, 230), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(x1, 0, rw - 1, h - 1)

        # Current position marker
        x = int(self._current_frame * scale)
        painter.setPen(QColor(255, 255, 255))
        painter.drawLine(x, 0, x, h)

        # Mode badge
        if self._selection_mode:
            painter.setPen(QColor(180, 210, 255))
            label = f"SEL {len(self._selections)}"
            painter.drawText(6, 12, label)

        painter.end()

    # ------------------------------------------------------------------
    # Mouse
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent):
        if self._total_frames() < 1:
            return

        if event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(event.position().toPoint())
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        self._active_button = event.button()
        self._press_x = event.position().x()
        self._press_frame = self._x_to_frame(self._press_x)
        self._drag_current_frame = self._press_frame
        self._dragging = False

        if not self._selection_mode:
            # Annotation mode: seek immediately on press
            self._emit_seek_at(self._press_x)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._press_x is None or self._active_button != Qt.MouseButton.LeftButton:
            return
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return

        x = event.position().x()
        if self._selection_mode:
            if abs(x - self._press_x) >= _DRAG_PIXEL_THRESHOLD:
                self._dragging = True
                self._drag_current_frame = self._x_to_frame(x)
                self.update()
        else:
            # Optional scrub while dragging in seek mode
            self._emit_seek_at(x)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._press_x is None:
            return

        x = event.position().x()

        if self._selection_mode:
            if self._dragging and self._press_frame is not None:
                end_frame = self._x_to_frame(x)
                rng = self._normalize_range(self._press_frame, end_frame)
                # Require at least 1 frame of span (start != end after normalize is ok for single frame)
                # Single-frame ranges are allowed if user dragged enough pixels
                self._selections.append(rng)
                self.selection_changed.emit(self.get_selections())
            else:
                # Click without meaningful drag → seek
                self._emit_seek_at(x)
        # else: already sought on press / move

        self._reset_drag()
        self.update()

    def _show_context_menu(self, pos: QPoint) -> None:
        menu = QMenu(self)
        idx = self._range_at_x(pos.x())

        act_del = None
        if idx is not None:
            s, e = self._selections[idx]
            act_del = menu.addAction(f"Delete range [{s}–{e}]")
        act_clear = menu.addAction("Clear all ranges")
        if not self._selections:
            act_clear.setEnabled(False)

        chosen = menu.exec(self.mapToGlobal(pos))
        if chosen is None:
            return
        if act_del is not None and chosen == act_del and idx is not None:
            del self._selections[idx]
            self.selection_changed.emit(self.get_selections())
            self.update()
        elif chosen == act_clear:
            self.clear_selections()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Escape:
            if self._selections or self._dragging:
                self.clear_selections()
                event.accept()
                return
        super().keyPressEvent(event)
