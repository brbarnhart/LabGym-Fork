"""Preprocessing → Draw markers (interactive canvas + burn onto videos)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPoint, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.project.paths import list_project_video_choices


class MarkerCanvas(QLabel):
    """Click-drag to draw lines or circles on a still frame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #222;")
        self._bgr: Optional[np.ndarray] = None
        self.draw_lines = False
        self.color = (255, 0, 0)  # BGR
        self.thickness = 2
        self.lines: List[dict] = []
        self.circles: List[dict] = []
        self._current: Optional[dict] = None
        self._scale = 1.0
        self._offset = (0, 0)

    def set_image(self, bgr: np.ndarray) -> None:
        self._bgr = bgr.copy()
        h, w = bgr.shape[:2]
        self.thickness = max(1, round((h + w) / 320))
        self.lines.clear()
        self.circles.clear()
        self._current = None
        self._repaint_image()

    def _repaint_image(self) -> None:
        if self._bgr is None:
            return
        img = self._bgr.copy()
        for line in self.lines:
            self._draw_line(img, line)
        if self._current and self.draw_lines:
            self._draw_line(img, self._current)
        for c in self.circles:
            self._draw_circle(img, c)
        if self._current and not self.draw_lines:
            self._draw_circle(img, self._current)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        # fit
        scaled = pix.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._scale = scaled.width() / max(1, w)
        self._offset = (
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._repaint_image()

    def _to_image(self, pos: QPoint) -> Tuple[int, int]:
        x = int((pos.x() - self._offset[0]) / max(1e-6, self._scale))
        y = int((pos.y() - self._offset[1]) / max(1e-6, self._scale))
        if self._bgr is not None:
            h, w = self._bgr.shape[:2]
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
        return x, y

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._bgr is None:
            return
        p = self._to_image(event.position().toPoint())
        self._current = {"start": p, "end": p, "color": self.color}
        self._repaint_image()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._current is None or self._bgr is None:
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._current["end"] = self._to_image(event.position().toPoint())
            self._repaint_image()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._current is None:
            return
        self._current["end"] = self._to_image(event.position().toPoint())
        if self.draw_lines:
            self.lines.append(self._current)
        else:
            self.circles.append(self._current)
        self._current = None
        self._repaint_image()

    def undo(self) -> None:
        if self.draw_lines and self.lines:
            self.lines.pop()
        elif not self.draw_lines and self.circles:
            self.circles.pop()
        self._repaint_image()

    @staticmethod
    def _draw_line(image, line) -> None:
        cv2.line(image, line["start"], line["end"], line["color"], max(1, 2))

    @staticmethod
    def _draw_circle(image, circle) -> None:
        s, e = circle["start"], circle["end"]
        r = int(((e[0] - s[0]) ** 2 + (e[1] - s[1]) ** 2) ** 0.5)
        cv2.circle(image, s, max(1, r), circle["color"], max(1, 2))

    def composite(self) -> Optional[np.ndarray]:
        if self._bgr is None:
            return None
        img = self._bgr.copy()
        for line in self.lines:
            self._draw_line(img, line)
        for c in self.circles:
            self._draw_circle(img, c)
        return img


class _BurnWorker(QObject):
    finished = Signal(int, int)
    progress = Signal(str)

    def __init__(
        self,
        videos: List[str],
        out_dir: str,
        lines: List[dict],
        circles: List[dict],
        framewidth: Optional[int],
        thickness: int,
    ):
        super().__init__()
        self.videos = videos
        self.out_dir = out_dir
        self.lines = lines
        self.circles = circles
        self.framewidth = framewidth
        self.thickness = thickness

    def run(self) -> None:
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        ok = fail = 0
        for path in self.videos:
            try:
                self.progress.emit(f"Drawing on {Path(path).name}…")
                self._burn_one(path)
                ok += 1
            except Exception as exc:
                fail += 1
                self.progress.emit(f"ERROR {Path(path).name}: {exc}")
        self.finished.emit(ok, fail)

    def _burn_one(self, path: str) -> None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")
        fps = round(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.framewidth:
            nw = int(self.framewidth)
            nh = int(h * nw / w)
        else:
            nw, nh = w, h
        name = Path(path).stem
        out_path = str(Path(self.out_dir) / f"{name}_markers.avi")
        writer = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"MJPG"), int(fps), (nw, nh), True
        )
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if self.framewidth:
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            for line in self.lines:
                cv2.line(
                    frame, line["start"], line["end"], line["color"], self.thickness
                )
            for c in self.circles:
                s, e = c["start"], c["end"]
                r = int(((e[0] - s[0]) ** 2 + (e[1] - s[1]) ** 2) ** 0.5)
                cv2.circle(frame, s, max(1, r), c["color"], self.thickness)
            writer.write(frame)
        cap.release()
        writer.release()


class DrawMarkersTab(QWidget):
    request_edit_project = Signal()

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None
        self._framewidth: Optional[int] = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Draw lines/circles on the first frame of a reference video, then burn "
            "the same markers onto selected videos."
        ))

        top = QHBoxLayout()
        self.combo_video = QComboBox()
        top.addWidget(QLabel("Reference video:"))
        top.addWidget(self.combo_video, 1)
        btn_load = QPushButton("Load frame")
        btn_load.clicked.connect(self._load_frame)
        top.addWidget(btn_load)
        btn_browse = QPushButton("Browse video…")
        btn_browse.clicked.connect(self._browse_video)
        top.addWidget(btn_browse)
        layout.addLayout(top)

        tools = QHBoxLayout()
        self.combo_shape = QComboBox()
        self.combo_shape.addItem("Circle", False)
        self.combo_shape.addItem("Line", True)
        self.combo_shape.currentIndexChanged.connect(self._on_shape)
        tools.addWidget(QLabel("Shape:"))
        tools.addWidget(self.combo_shape)
        btn_color = QPushButton("Color…")
        btn_color.clicked.connect(self._pick_color)
        tools.addWidget(btn_color)
        btn_undo = QPushButton("Undo")
        btn_undo.clicked.connect(lambda: self.canvas.undo())
        tools.addWidget(btn_undo)
        tools.addStretch(1)
        layout.addLayout(tools)

        self.canvas = MarkerCanvas()
        layout.addWidget(self.canvas, 1)

        out_row = QHBoxLayout()
        self.ed_out = QLineEdit()
        btn_out = QPushButton("Output folder…")
        btn_out.clicked.connect(self._browse_out)
        out_row.addWidget(QLabel("Output:"))
        out_row.addWidget(self.ed_out, 1)
        out_row.addWidget(btn_out)
        layout.addLayout(out_row)

        self.btn_burn = QPushButton("Burn markers onto project videos")
        self.btn_burn.clicked.connect(self._burn)
        layout.addWidget(self.btn_burn)
        self.log = QLabel("")
        self.log.setWordWrap(True)
        layout.addWidget(self.log)

        self.project.changed.connect(self._refresh_videos)
        self.project.project_replaced.connect(self._refresh_videos)
        self._refresh_videos()
        self._set_default_out()

    def _set_default_out(self) -> None:
        p = self.project.project
        rel = p.paths.processed_root or "processed"
        if p.root_dir:
            self.ed_out.setText(str(p.resolve_path(rel)))

    def _refresh_videos(self) -> None:
        self.combo_video.clear()
        for label, path in list_project_video_choices(self.project.project):
            self.combo_video.addItem(label, path)

    def _on_shape(self) -> None:
        self.canvas.draw_lines = bool(self.combo_shape.currentData())

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(255, 0, 0), self, "Marker color")
        if c.isValid():
            # store BGR for OpenCV
            self.canvas.color = (c.blue(), c.green(), c.red())

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Reference video",
            self.project.project.root_dir or "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if path:
            self.combo_video.addItem(Path(path).name, path)
            self.combo_video.setCurrentIndex(self.combo_video.count() - 1)
            self._load_frame()

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.ed_out.text())
        if d:
            self.ed_out.setText(d)

    def _load_frame(self) -> None:
        path = self.combo_video.currentData()
        if not path:
            QMessageBox.information(self, "Draw markers", "Select a video first.")
            return
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            QMessageBox.warning(self, "Draw markers", "Could not read a frame.")
            return
        self.canvas.set_image(frame)
        self.log.setText(f"Loaded frame from {path}")

    def _burn(self) -> None:
        if self.canvas._bgr is None:
            QMessageBox.information(self, "Draw markers", "Load a reference frame first.")
            return
        if not self.canvas.lines and not self.canvas.circles:
            QMessageBox.information(self, "Draw markers", "Draw at least one marker.")
            return
        out = self.ed_out.text().strip()
        if not out:
            QMessageBox.warning(self, "Draw markers", "Choose an output folder.")
            return
        videos = [p for _, p in list_project_video_choices(self.project.project)]
        if not videos:
            # fall back to reference only
            ref = self.combo_video.currentData()
            videos = [str(ref)] if ref else []
        if not videos:
            QMessageBox.warning(self, "Draw markers", "No videos to process.")
            return
        if self._thread is not None:
            return
        self.btn_burn.setEnabled(False)
        self._thread = QThread(self)
        worker = _BurnWorker(
            videos,
            out,
            list(self.canvas.lines),
            list(self.canvas.circles),
            self._framewidth,
            self.canvas.thickness,
        )
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(lambda m: self.log.setText(m))
        worker.finished.connect(self._on_done)
        worker.finished.connect(self._thread.quit)
        self._thread.finished.connect(lambda: setattr(self, "_thread", None))
        self._thread.finished.connect(lambda: self.btn_burn.setEnabled(True))
        self._worker = worker
        self._thread.start()

    def _on_done(self, ok: int, fail: int) -> None:
        self.log.setText(f"Finished. ok={ok} fail={fail}")
        QMessageBox.information(
            self, "Draw markers", f"Finished.\nSucceeded: {ok}\nFailed: {fail}"
        )
