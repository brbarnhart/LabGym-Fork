"""In-tab picture for placing the preprocess crop window.

The picture is one source frame after the Frame width resize. Dragging on
the picture writes Left / Right / Top / Bottom. Those edges stay shared
with the fields in the Preprocess videos options.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, QRectF, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.workbenches.preprocessing.crop_geometry import (
    CropEdges,
    CropSnapshot,
    CropWindowState,
    FrameSize,
    HitZone,
    ImageFit,
    RememberedFrame,
    drag_moved,
    edges_from_points,
    fit_image,
    hit_test,
    image_to_widget,
    intersect_edges,
    nudge_edges,
    resize_edges,
    shift_inside,
    widget_to_image,
    widget_to_image_clamped,
)
from LabGym.gui_pyside.workbenches.preprocessing.crop_preview import PreviewSession

_HANDLE = 4
_HIT_PX = 8.0
_DRAG_PX = 4.0


def _cursor_for(zone: HitZone) -> Qt.CursorShape:
    if zone in (HitZone.TOP_LEFT, HitZone.BOTTOM_RIGHT):
        return Qt.CursorShape.SizeFDiagCursor
    if zone in (HitZone.TOP_RIGHT, HitZone.BOTTOM_LEFT):
        return Qt.CursorShape.SizeBDiagCursor
    if zone in (HitZone.LEFT, HitZone.RIGHT):
        return Qt.CursorShape.SizeHorCursor
    if zone in (HitZone.TOP, HitZone.BOTTOM):
        return Qt.CursorShape.SizeVerCursor
    if zone == HitZone.INSIDE:
        return Qt.CursorShape.SizeAllCursor
    return Qt.CursorShape.CrossCursor


class CropCanvas(QWidget):
    """Picture of one resized frame with an adjustable crop rectangle."""

    edges_previewed = Signal(object)
    gesture_committed = Signal(object)
    gesture_cancelled = Signal()
    nudge_committed = Signal(int, int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(480, 270)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._frame: Optional[np.ndarray] = None
        self._pixmap: Optional[QPixmap] = None
        self._edges = CropEdges(0, 0, 0, 0)
        self._fit = ImageFit(1.0, 0.0, 0.0, 0.0, 0.0)
        self._message = "Open a video to place the crop window."
        self._drag: Optional[str] = None
        self._zone = HitZone.OUTSIDE
        self._press_widget = QPointF()
        self._press_image: Tuple[int, int] = (0, 0)
        self._origin = CropEdges(0, 0, 0, 0)

    def set_message(self, text: str) -> None:
        """Show ``text`` instead of a frame. Clears the current picture."""
        self._frame = None
        self._pixmap = None
        self._message = text
        self._drag = None
        self.update()

    def set_frame(self, bgr: np.ndarray) -> None:
        """Show ``bgr`` as the picture. The crop edges are not changed."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(image.copy())
        self._frame = bgr
        self._message = ""
        self.update()

    def set_edges(self, edges: CropEdges) -> None:
        """Draw ``edges``. Edges that hang off the picture are clipped visually."""
        self._edges = edges
        self.update()

    def frame_size(self) -> Optional[FrameSize]:
        """Return the shown frame size, or None when no picture is loaded."""
        if self._frame is None:
            return None
        height, width = self._frame.shape[:2]
        return FrameSize(width, height)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#222222"))
        size = self.frame_size()
        if self._pixmap is None or size is None:
            painter.setPen(QColor("#dddddd"))
            painter.drawText(
                self.rect(),
                int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
                self._message,
            )
            painter.end()
            return
        self._fit = fit_image(size.width, size.height, self.width(), self.height())
        target = QRectF(
            self._fit.offset_x,
            self._fit.offset_y,
            self._fit.display_width,
            self._fit.display_height,
        )
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._pixmap, QRectF(0, 0, size.width, size.height))
        visible = intersect_edges(self._edges, size)
        if visible is None:
            painter.end()
            return
        left, top = image_to_widget(visible.left, visible.top, self._fit)
        right, bottom = image_to_widget(visible.right, visible.bottom, self._fit)
        crop = QRectF(left, top, right - left, bottom - top)
        dim = QColor(0, 0, 0, 140)

        def _dim(x: float, y: float, w: float, h: float) -> None:
            if w > 0 and h > 0:
                painter.fillRect(QRectF(x, y, w, h), dim)

        _dim(target.left(), target.top(), target.width(), crop.top() - target.top())
        _dim(target.left(), crop.bottom(), target.width(), target.bottom() - crop.bottom())
        _dim(target.left(), crop.top(), crop.left() - target.left(), crop.height())
        _dim(crop.right(), crop.top(), target.right() - crop.right(), crop.height())
        painter.setPen(QPen(QColor("#ffe14a"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(crop)
        painter.setBrush(QColor("#ffe14a"))
        painter.setPen(Qt.PenStyle.NoPen)
        for cx, cy in (
            (left, top),
            ((left + right) / 2, top),
            (right, top),
            (left, (top + bottom) / 2),
            (right, (top + bottom) / 2),
            (left, bottom),
            ((left + right) / 2, bottom),
            (right, bottom),
        ):
            painter.drawRect(
                QRectF(cx - _HANDLE, cy - _HANDLE, _HANDLE * 2, _HANDLE * 2)
            )
        painter.end()

    def _sync_fit(self) -> None:
        size = self.frame_size()
        if size is None:
            return
        self._fit = fit_image(size.width, size.height, self.width(), self.height())

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self.frame_size() is None:
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._sync_fit()
        size = self.frame_size()
        assert size is not None
        point = widget_to_image(
            event.position().x(),
            event.position().y(),
            self._fit,
            size.width,
            size.height,
        )
        if point is None:
            return
        visible = intersect_edges(self._edges, size)
        zone = (
            hit_test(event.position().x(), event.position().y(), visible, self._fit, _HIT_PX)
            if visible is not None
            else HitZone.OUTSIDE
        )
        self._drag = "create" if zone == HitZone.OUTSIDE else (
            "move" if zone == HitZone.INSIDE else "resize"
        )
        self._zone = zone
        self._press_widget = QPointF(event.position())
        self._press_image = point
        self._origin = self._edges

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        size = self.frame_size()
        if size is None:
            return
        self._sync_fit()
        if self._drag is None:
            visible = intersect_edges(self._edges, size)
            zone = (
                hit_test(event.position().x(), event.position().y(), visible, self._fit, _HIT_PX)
                if visible is not None
                else HitZone.OUTSIDE
            )
            self.setCursor(_cursor_for(zone))
            return
        point = widget_to_image_clamped(
            event.position().x(),
            event.position().y(),
            self._fit,
            size.width,
            size.height,
        )
        if point is None:
            return
        self._apply_drag(point)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag is None:
            return
        self._sync_fit()
        size = self.frame_size()
        point = None
        if size is not None:
            point = widget_to_image_clamped(
                event.position().x(),
                event.position().y(),
                self._fit,
                size.width,
                size.height,
            )
        mode = self._drag
        self._drag = None
        moved = drag_moved(
            (self._press_widget.x(), self._press_widget.y()),
            (event.position().x(), event.position().y()),
            _DRAG_PX,
        )
        if size is None or point is None or not moved:
            self.gesture_cancelled.emit()
            return
        self._drag = mode
        live = self._live_edges(point, size)
        self._drag = None
        if live is None or not live.valid():
            self.gesture_cancelled.emit()
            return
        self.gesture_committed.emit(live)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key in (
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        ):
            step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
            dx = -step if key == Qt.Key.Key_Left else step if key == Qt.Key.Key_Right else 0
            dy = -step if key == Qt.Key.Key_Up else step if key == Qt.Key.Key_Down else 0
            self.nudge_committed.emit(dx, dy)
            event.accept()
            return
        super().keyPressEvent(event)

    def _apply_drag(self, point: Tuple[int, int]) -> None:
        size = self.frame_size()
        if size is None:
            return
        live = self._live_edges(point, size)
        if live is not None:
            self.edges_previewed.emit(live)

    def _live_edges(self, point: Tuple[int, int], size: FrameSize) -> Optional[CropEdges]:
        if self._drag == "create":
            return edges_from_points(
                self._press_image[0],
                self._press_image[1],
                point[0],
                point[1],
            )
        if self._drag == "move":
            dx = point[0] - self._press_image[0]
            dy = point[1] - self._press_image[1]
            base = intersect_edges(self._origin, size)
            if base is None:
                return None
            return shift_inside(
                CropEdges(base.left + dx, base.top + dy, base.right + dx, base.bottom + dy),
                size,
            )
        if self._drag == "resize":
            return resize_edges(self._origin, self._zone, point[0], point[1], size)
        return None


class CropWindow(QWidget):
    """Picture, frame slider, and crop rectangle embedded in Preprocess videos.

    ``edges_edited`` and ``crop_enabled_edited`` fire when a gesture changes
    the shared crop window. Field edits that originated in the options do not
    echo back.
    """

    edges_edited = Signal(int, int, int, int)
    crop_enabled_edited = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.state = CropWindowState()
        self._session = PreviewSession()
        self._framewidth: Optional[int] = None
        self._native: Optional[FrameSize] = None
        self._resized: Optional[FrameSize] = None
        self._current_index = 0
        self._gesture_before: Optional[CropSnapshot] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_video = QLabel("No video")
        self.lbl_video.setWordWrap(True)
        self.lbl_set_on = QLabel("Size not set")
        self.lbl_kept = QLabel(self._kept_text(self.state.edges))
        header = QHBoxLayout()
        header.addWidget(self.lbl_video, 1)
        header.addWidget(self.lbl_set_on)
        header.addWidget(self.lbl_kept)
        layout.addLayout(header)

        self.canvas = CropCanvas()
        self.canvas.edges_previewed.connect(self._on_preview)
        self.canvas.gesture_committed.connect(self._on_gesture)
        self.canvas.gesture_cancelled.connect(self._on_gesture_cancelled)
        self.canvas.nudge_committed.connect(self._on_nudge)
        layout.addWidget(self.canvas, 1)

        slider_row = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider)
        self.lbl_frame = QLabel("Frame 0")
        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(self.lbl_frame)
        self.btn_clear = QPushButton("Clear crop window")
        self.btn_clear.setToolTip(
            "Remove the rectangle, set the edges to 0, and uncheck Crop frames."
        )
        self.btn_clear.clicked.connect(self._on_clear)
        slider_row.addWidget(self.btn_clear)
        layout.addLayout(slider_row)

        hint = QLabel(
            "Drag on the picture to draw the crop window. Drag inside it to move it, "
            "or drag an edge or corner to resize it. Arrow keys nudge by 1 pixel "
            "(Shift+arrow by 10) when the picture is focused. Ctrl+Z undoes the last "
            "draw, move, resize, nudge, or clear."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

    def current_path(self) -> Optional[str]:
        """Return the video currently opened in the editor."""
        return self._session.path

    def preview_size(self) -> Optional[FrameSize]:
        """Return the resized size of the picture on screen, if one is loaded."""
        return self._resized

    @property
    def remembered(self) -> Optional[RememberedFrame]:
        """Resized frame the crop edges belong to, if one has been set."""
        return self.state.remembered

    def replace_edges_from_fields(self, edges: CropEdges, crop_enabled: bool) -> None:
        """Match the tab fields without treating the copy as an edit.

        Args:
            edges: Left, top, right, and bottom from the Preprocess tab.
            crop_enabled: Whether Crop frames is checked.
        """
        self.state.edges = edges
        self.state.crop_enabled = crop_enabled
        self._refresh_fields(emit=False)

    def apply_typed_edges(self, edges: CropEdges) -> None:
        """Store edges typed on the tab.

        When a picture is loaded, the crop window belongs to that picture
        afterward.
        """
        self.state.type_edges(edges, self._preview_pair())
        self._refresh_fields(emit=False)

    def set_crop_enabled(self, enabled: bool) -> None:
        """Follow the Crop frames checkbox without moving the rectangle."""
        self.state.set_crop_enabled(enabled)

    def apply_frame_width(self, framewidth: Optional[int]) -> None:
        """Scale the crop window for a new Frame width and remember the new size.

        Args:
            framewidth: Target width, or None to keep the native size.
        """
        self._framewidth = framewidth
        if self.state.apply_framewidth(framewidth):
            self._refresh_fields(emit=True)

    def show_video(self, path: str, framewidth: Optional[int], *, reset_frame: bool) -> None:
        """Load ``path`` and show a frame. Does not move an existing crop window.

        Args:
            path: Video to preview.
            framewidth: Target width, or None to keep the native size.
            reset_frame: When True, show the first frame. When False, keep
                the current slider index.
        """
        self._framewidth = framewidth
        if not self._session.open(path):
            self.canvas.set_message(self._session.error or "Could not read the video.")
            self.lbl_video.setText(Path(path).name)
            self.slider.setEnabled(False)
            self._native = None
            self._resized = None
            return
        info = self._session.info
        assert info is not None
        last = max(0, info.frame_count - 1) if info.frame_count else 0
        self.slider.setEnabled(True)
        self.slider.blockSignals(True)
        self.slider.setRange(0, last)
        if reset_frame:
            self._current_index = 0
        else:
            self._current_index = min(self._current_index, last)
        self.slider.setValue(self._current_index)
        self.slider.blockSignals(False)
        self.lbl_video.setText(Path(path).name)
        self._load_index(self._current_index)

    def reload_frame(self) -> None:
        """Redraw the current frame after Frame width changes."""
        if self._session.path is None:
            return
        self._load_index(self._current_index)

    def show_empty(self) -> None:
        """Clear the picture when the video list has nothing to show."""
        self._session.close()
        self._native = None
        self._resized = None
        self.canvas.set_message("Add a video to place the crop window.")
        self.lbl_video.setText("No video")
        self.lbl_frame.setText("Frame 0")
        self.slider.setEnabled(False)
        self.slider.setRange(0, 0)

    def _preview_pair(self) -> Optional[Tuple[FrameSize, FrameSize]]:
        if self._native is None or self._resized is None:
            return None
        return (self._native, self._resized)

    def _on_slider(self, value: int) -> None:
        self._current_index = int(value)
        self._load_index(self._current_index)

    def _load_index(self, index: int) -> None:
        shown, native, resized = self._session.read_resized(index, self._framewidth)
        info = self._session.info
        if shown is None or native is None or resized is None or info is None:
            self.canvas.set_message("Could not read that frame. The crop window is unchanged.")
            self._native = None
            self._resized = None
            return
        self._native = native
        self._resized = resized
        self.canvas.set_frame(shown)
        self.canvas.set_edges(self.state.edges)
        if self.state.note_preview(native, resized):
            self._refresh_size_label()
        fps = info.fps if info.fps > 0 else 0.0
        if fps:
            seconds = index / fps
            timing = f"  ({seconds:.2f} s)"
        else:
            timing = ""
        total = max(index, info.frame_count - 1) if info.frame_count else index
        self.lbl_frame.setText(f"Frame {index} / {total}{timing}")

    def _on_preview(self, edges: CropEdges) -> None:
        if self._gesture_before is None:
            self._gesture_before = self.state.snapshot()
        self.state.edges = edges
        self._refresh_fields(emit=True)

    def _on_gesture_cancelled(self) -> None:
        before = self._gesture_before
        self._gesture_before = None
        if before is None:
            return
        self.state.edges = before.edges
        self._refresh_fields(emit=True)

    def _on_gesture(self, edges: CropEdges) -> None:
        before = self._gesture_before
        self._gesture_before = None
        preview = self._preview_pair()
        if preview is None:
            if before is not None:
                self.state.edges = before.edges
                self._refresh_fields(emit=True)
            return
        native, resized = preview
        self.state.commit_gesture(edges, native, resized, before=before)
        self._refresh_fields(emit=True)
        self.crop_enabled_edited.emit(self.state.crop_enabled)

    def _on_nudge(self, dx: int, dy: int) -> None:
        preview = self._preview_pair()
        if preview is None:
            return
        _native, resized = preview
        moved = nudge_edges(self.state.edges, dx, dy, resized)
        if moved is None:
            return
        self._on_gesture(moved)

    def _on_clear(self) -> None:
        self.state.clear()
        self._refresh_fields(emit=True)
        self.crop_enabled_edited.emit(False)

    def _on_undo(self) -> None:
        if not self.state.undo():
            return
        self._refresh_fields(emit=True)
        self.crop_enabled_edited.emit(self.state.crop_enabled)

    def _refresh_fields(self, *, emit: bool) -> None:
        edges = self.state.edges
        self.canvas.set_edges(edges)
        self._refresh_size_label()
        if emit:
            self.edges_edited.emit(edges.left, edges.top, edges.right, edges.bottom)

    def _refresh_size_label(self) -> None:
        remembered = self.state.remembered
        if remembered is None:
            self.lbl_set_on.setText("Size not set")
        else:
            size = remembered.resized
            self.lbl_set_on.setText(f"Set on {size.width}×{size.height}")
        self.lbl_kept.setText(self._kept_text(self.state.edges))

    @staticmethod
    def _kept_text(edges: CropEdges) -> str:
        if not edges.valid():
            return "kept —"
        return f"kept {edges.width}×{edges.height}"

    def snapshot(self) -> CropSnapshot:
        """Return the editor state. Tests and undo use the same shape."""
        return self.state.snapshot()
