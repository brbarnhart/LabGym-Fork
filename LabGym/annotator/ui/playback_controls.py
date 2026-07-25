"""Playback controls: play/pause, seek, speed, frame navigation."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSlider, QSpinBox, QComboBox, QLabel, QToolButton
)


class PlaybackControls(QWidget):
    # Signals (frame numbers are ints)
    play_pause_requested = Signal()
    seek_requested = Signal(int)          # absolute frame
    step_requested = Signal(int)          # +1 / -1 etc.
    speed_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._total_frames = 0
        self._fps = 30.0
        self._current_frame = 0
        self._is_playing = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Transport
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self._on_play_clicked)

        self.btn_step_back = QToolButton()
        self.btn_step_back.setText("◀◀")
        self.btn_step_back.clicked.connect(lambda: self.step_requested.emit(-1))

        self.btn_step_fwd = QToolButton()
        self.btn_step_fwd.setText("▶▶")
        self.btn_step_fwd.clicked.connect(lambda: self.step_requested.emit(1))

        self.btn_step_back10 = QToolButton()
        self.btn_step_back10.setText("◀ 10")
        self.btn_step_back10.clicked.connect(lambda: self.step_requested.emit(-10))

        self.btn_step_fwd10 = QToolButton()
        self.btn_step_fwd10.setText("10 ▶")
        self.btn_step_fwd10.clicked.connect(lambda: self.step_requested.emit(10))

        # Frame / time
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setMinimumWidth(140)

        self.spin_frame = QSpinBox()
        self.spin_frame.setRange(0, 0)
        self.spin_frame.setPrefix("f ")
        self.spin_frame.valueChanged.connect(self._on_spin_changed)

        # Seek slider (frame-based)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self._on_slider_moved)
        self.slider.setMinimumWidth(280)

        # Speed
        self.cmb_speed = QComboBox()
        for sp in [0.25, 0.5, 1.0, 1.5, 2.0, 4.0]:
            self.cmb_speed.addItem(f"{sp:.2f}x", sp)
        self.cmb_speed.setCurrentText("1.00x")
        self.cmb_speed.currentIndexChanged.connect(self._on_speed_changed)

        # Layout
        layout.addWidget(self.btn_play)
        layout.addWidget(self.btn_step_back10)
        layout.addWidget(self.btn_step_back)
        layout.addWidget(self.btn_step_fwd)
        layout.addWidget(self.btn_step_fwd10)
        layout.addSpacing(12)
        layout.addWidget(self.slider, 1)
        layout.addSpacing(8)
        layout.addWidget(self.spin_frame)
        layout.addSpacing(8)
        layout.addWidget(self.lbl_time)
        layout.addSpacing(12)
        layout.addWidget(QLabel("Speed:"))
        layout.addWidget(self.cmb_speed)

        self.setEnabled(False)  # enabled after video load

    def set_video_info(self, total_frames: int, fps: float):
        self._total_frames = max(1, int(total_frames))
        self._fps = max(1e-6, float(fps))

        self.slider.setRange(0, self._total_frames - 1)
        self.spin_frame.setRange(0, self._total_frames - 1)

        self.setEnabled(True)
        self.update_position(0)

    def update_position(self, frame: int, is_playing: bool | None = None):
        if is_playing is not None:
            self._is_playing = is_playing
            self.btn_play.setText("⏸ Pause" if is_playing else "▶ Play")
            self.btn_play.setChecked(is_playing)

        self._current_frame = max(0, min(frame, self._total_frames - 1))

        # Block signals to avoid feedback loops
        self.slider.blockSignals(True)
        self.spin_frame.blockSignals(True)
        self.slider.setValue(self._current_frame)
        self.spin_frame.setValue(self._current_frame)
        self.slider.blockSignals(False)
        self.spin_frame.blockSignals(False)

        self._update_time_label()

    def _update_time_label(self):
        # Simple mm:ss display (we can improve later with helpers)
        def fmt(f):
            sec = int(f / max(self._fps, 1e-6))
            m = sec // 60
            s = sec % 60
            return f"{m:02d}:{s:02d}"
        self.lbl_time.setText(f"{fmt(self._current_frame)} / {fmt(self._total_frames)}")

    def _on_play_clicked(self):
        self.play_pause_requested.emit()

    def _on_slider_moved(self, value: int):
        self.seek_requested.emit(value)

    def _on_spin_changed(self, value: int):
        self.seek_requested.emit(value)

    def _on_speed_changed(self):
        speed = self.cmb_speed.currentData()
        if speed:
            self.speed_changed.emit(float(speed))

    def current_speed(self) -> float:
        return float(self.cmb_speed.currentData() or 1.0)

    def set_playing(self, playing: bool):
        self.update_position(self._current_frame, playing)
