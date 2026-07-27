"""Detector → Review IDs & assign names/roles (PySide)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.widgets.path_browse import (
    browse_existing_directory,
    set_line_edit_directory,
)
from LabGym.gui_pyside.project.paths import (
    discover_tracklets_dir,
    list_project_video_choices,
)
from LabGym.gui_pyside.workbenches.detector.review_ids_hard_cases import (
    AnalysisFrameRange,
    default_output_dir,
    extract_hard_case_frames,
)
from LabGym.gui_pyside.workbenches.detector.review_ids_markers import MarkersTable
from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
    clone_markers,
    events_for_kind,
    load_review_package,
    resolve_video_path,
    save_review_package,
)
from LabGym.gui_pyside.workbenches.detector.review_ids_render import (
    VideoCaptureCache,
    bgr_to_qpixmap,
    compose_preview_frame,
    format_frame_status,
    read_preview_frame,
)
from LabGym.gui_pyside.workbenches.detector.risk_timeline import RiskTimeline
from LabGym.gui_pyside.workbenches.detector.subjects_table import SubjectsTable
from LabGym.id_review.dataset import make_swap_marker
from LabGym.id_review.types import ContactEvent, SwitchMarker


class _HardCaseExtractWorker(QObject):
    finished = Signal(object)  # ExtractHardCaseResult
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        video_path: str,
        out_path: str,
        ranges: List[AnalysisFrameRange],
        store_meta: dict,
        fps: float,
        skip: int,
        framewidth: Optional[int],
        n_frames: int,
    ):
        super().__init__()
        self.video_path = video_path
        self.out_path = out_path
        self.ranges = ranges
        self.store_meta = store_meta
        self.fps = fps
        self.skip = skip
        self.framewidth = framewidth
        self.n_frames = n_frames

    def run(self) -> None:
        try:
            result = extract_hard_case_frames(
                self.video_path,
                self.out_path,
                self.ranges,
                store_meta=self.store_meta,
                fps=self.fps,
                skip=self.skip,
                framewidth=self.framewidth,
                n_frames=self.n_frames,
                progress_callback=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


class ReviewIdsTab(QWidget):
    """Full-video ID review + subject names/roles for an identity package folder."""

    request_edit_project = Signal()
    package_saved = Signal(str)  # review_dir

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project

        self.review_dir: str = ""
        self.events: List[ContactEvent] = []
        self.markers: List[SwitchMarker] = []
        self._undo_stack: List[List[SwitchMarker]] = []
        self._stores: Dict[str, object] = {}
        self._baseline_stores: Dict[str, object] = {}
        self._cap = VideoCaptureCache()
        self._playing = False
        self._updating = False
        self.frame = 0
        self.n_frames = 1
        self.fps = 10.0
        self.animal_kind = "mouse"
        self.involved_ids: List[int] = [0, 1]
        self.min_risk = 0.0
        self._dirty = False
        self._already_corrected = False
        self._training_ranges: List[AnalysisFrameRange] = []
        self._range_start: Optional[int] = None
        self._extract_thread: Optional[QThread] = None

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        self._build_ui()
        self._bind_shortcuts()
        # Tracklet discovery per video is expensive — only on replace/open/edit.
        self.project.project_replaced.connect(self.refresh_video_list)
        self.project.project_replaced.connect(self._prefills_hard_case_out)
        self.refresh_video_list()
        self._prefills_hard_case_out()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        help_txt = QLabel(
            "Orange/red timeline = automatic contact risk. Green ticks = your switch markers. "
            "Cyan bands = periods marked for extra detector training images. "
            "Mark where IDs flip, assign experimental names/roles, then Save. "
            "Keys: ←/→ step, Space play, S mark swap, [ / ] range start/end, "
            "Delete remove, Ctrl+Z undo."
        )
        help_txt.setWordWrap(True)
        root.addWidget(help_txt)

        pick = QHBoxLayout()
        pick.addWidget(QLabel("Project video:"))
        self.combo_video = QComboBox()
        self.combo_video.currentIndexChanged.connect(self._on_video_combo)
        pick.addWidget(self.combo_video, 1)
        self.btn_open_pkg = QPushButton("Open package folder…")
        self.btn_open_pkg.setToolTip("Open an id_review / tracklets directory directly")
        self.btn_open_pkg.clicked.connect(self._browse_package)
        pick.addWidget(self.btn_open_pkg)
        self.btn_load = QPushButton("Load")
        self.btn_load.clicked.connect(self._load_selected_video_package)
        pick.addWidget(self.btn_load)
        self.btn_edit = QPushButton("Edit project…")
        self.btn_edit.clicked.connect(self.request_edit_project.emit)
        pick.addWidget(self.btn_edit)
        root.addLayout(pick)

        self.lbl_pkg = QLabel("No package loaded.")
        self.lbl_pkg.setStyleSheet(
            "QLabel { background: #2a2a2a; color: #eee; padding: 6px; border-radius: 4px; }"
        )
        root.addWidget(self.lbl_pkg)

        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(480, 320)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background: #222;")
        left_l.addWidget(self.video_label, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        left_l.addWidget(self.status)

        transport = QHBoxLayout()
        self.btn_back = QPushButton("◀ -1")
        self.btn_play = QPushButton("Play")
        self.btn_fwd = QPushButton("+1 ▶")
        self.btn_prev_risk = QPushButton("⟵ Risk")
        self.btn_next_risk = QPushButton("Risk ⟶")
        self.btn_mark = QPushButton("Mark swap (S)")
        self.btn_remove = QPushButton("Remove at frame")
        self.btn_del = QPushButton("Delete selected")
        self.btn_undo = QPushButton("Undo")
        self.btn_back.clicked.connect(lambda: self._nudge(-1))
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_fwd.clicked.connect(lambda: self._nudge(1))
        self.btn_prev_risk.clicked.connect(lambda: self._jump_risk(-1))
        self.btn_next_risk.clicked.connect(lambda: self._jump_risk(1))
        self.btn_mark.clicked.connect(self._mark_swap)
        self.btn_remove.clicked.connect(lambda: self._remove_at_current_frame())
        self.btn_del.clicked.connect(self._delete_selected_marker)
        self.btn_undo.clicked.connect(self._undo)
        for b in (
            self.btn_back,
            self.btn_play,
            self.btn_fwd,
            self.btn_prev_risk,
            self.btn_next_risk,
            self.btn_mark,
            self.btn_remove,
            self.btn_del,
            self.btn_undo,
        ):
            transport.addWidget(b)
        left_l.addLayout(transport)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        left_l.addWidget(self.slider)

        self.timeline = RiskTimeline()
        self.timeline.seek_requested.connect(self._seek)
        left_l.addWidget(self.timeline)

        filt = QHBoxLayout()
        lab_risk = QLabel("Min risk:")
        tip_risk = (
            "Hide contact-risk bands weaker than this score on the timeline "
            "(0–1). 0 shows all bands; raise to focus on stronger collision/swap risk."
        )
        lab_risk.setToolTip(tip_risk)
        filt.addWidget(lab_risk)
        self.spin_risk = QDoubleSpinBox()
        self.spin_risk.setRange(0.0, 1.0)
        self.spin_risk.setSingleStep(0.05)
        self.spin_risk.setDecimals(2)
        self.spin_risk.setToolTip(tip_risk)
        self.spin_risk.valueChanged.connect(self._on_risk_filter)
        filt.addWidget(self.spin_risk)
        lab_kind = QLabel("Kind:")
        tip_kind = "Animal category within the tracklet package (e.g. mouse) to review."
        lab_kind.setToolTip(tip_kind)
        filt.addWidget(lab_kind)
        self.kind_combo = QComboBox()
        self.kind_combo.setToolTip(tip_kind)
        self.kind_combo.currentTextChanged.connect(self._on_kind)
        filt.addWidget(self.kind_combo)
        lab_swap = QLabel("Swap IDs:")
        tip_swap = (
            "Which two track IDs to exchange when you press Mark swap. "
            "A swap at the current frame means from this frame onward those "
            "two identities are remapped."
        )
        lab_swap.setToolTip(tip_swap)
        filt.addWidget(lab_swap)
        self.id_a = QComboBox()
        self.id_b = QComboBox()
        self.id_a.setToolTip(tip_swap)
        self.id_b.setToolTip(tip_swap)
        filt.addWidget(self.id_a)
        filt.addWidget(self.id_b)
        filt.addStretch(1)
        left_l.addLayout(filt)

        split.addWidget(left)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)

        self.markers_panel = MarkersTable()
        self.marker_table = self.markers_panel.table  # used by setEnabled_controls
        self.marker_table.cellDoubleClicked.connect(self._on_marker_activated)
        right_l.addWidget(self.markers_panel, 1)

        subj_box = QGroupBox("Names / roles (saved as subjects.json)")
        subj_l = QVBoxLayout(subj_box)
        self.subjects_table = SubjectsTable()
        self.subjects_table.changed.connect(self._mark_dirty)
        subj_l.addWidget(self.subjects_table)
        right_l.addWidget(subj_box, 1)

        hard_box = QGroupBox("Hard cases → detector training images")
        hard_l = QVBoxLayout(hard_box)
        hard_help = QLabel(
            "Select time periods where the detector struggled. Frames from those "
            "ranges are written as JPGs you can annotate (EZannot) and use to "
            "retrain the detector."
        )
        hard_help.setWordWrap(True)
        hard_l.addWidget(hard_help)

        self.lbl_range_pending = QLabel("Range start: (not set)")
        hard_l.addWidget(self.lbl_range_pending)

        range_btns = QHBoxLayout()
        self.btn_range_start = QPushButton("Set start here ([)")
        self.btn_range_start.setToolTip(
            "Mark the current analysis frame as the start of a training range."
        )
        self.btn_range_start.clicked.connect(self._set_range_start)
        self.btn_range_end = QPushButton("Set end & add (])")
        self.btn_range_end.setToolTip(
            "Close the range at the current frame and add it to the list."
        )
        self.btn_range_end.clicked.connect(self._set_range_end_and_add)
        self.btn_add_risk = QPushButton("Add current risk band")
        self.btn_add_risk.setToolTip(
            "If the playhead is inside an automatic contact-risk band, add that "
            "whole band as a training range."
        )
        self.btn_add_risk.clicked.connect(self._add_current_risk_band)
        range_btns.addWidget(self.btn_range_start)
        range_btns.addWidget(self.btn_range_end)
        range_btns.addWidget(self.btn_add_risk)
        hard_l.addLayout(range_btns)

        self.list_ranges = QListWidget()
        self.list_ranges.setMinimumHeight(72)
        self.list_ranges.setToolTip("Analysis-frame ranges that will be exported.")
        hard_l.addWidget(self.list_ranges)

        range_edit = QHBoxLayout()
        self.btn_remove_range = QPushButton("Remove selected")
        self.btn_remove_range.clicked.connect(self._remove_selected_range)
        self.btn_clear_ranges = QPushButton("Clear all")
        self.btn_clear_ranges.clicked.connect(self._clear_ranges)
        range_edit.addWidget(self.btn_remove_range)
        range_edit.addWidget(self.btn_clear_ranges)
        range_edit.addStretch(1)
        hard_l.addLayout(range_edit)

        out_row = QHBoxLayout()
        self.ed_hard_out = QLineEdit()
        self.ed_hard_out.setToolTip(
            "Folder for extracted JPGs (same default as Generate images: "
            "project/detector_training_images)."
        )
        b_hard_out = QPushButton("Browse…")
        b_hard_out.clicked.connect(self._browse_hard_out)
        out_row.addWidget(QLabel("Output:"), 0)
        out_row.addWidget(self.ed_hard_out, 1)
        out_row.addWidget(b_hard_out, 0)
        hard_l.addLayout(out_row)

        opts = QHBoxLayout()
        self.spin_hard_skip = QSpinBox()
        self.spin_hard_skip.setRange(1, 1_000_000)
        self.spin_hard_skip.setValue(10)
        self.spin_hard_skip.setToolTip(
            "Write one image every N analysis frames within each range "
            "(start and end of each range are always kept). "
            "Smaller values = denser sampling of hard cases."
        )
        opts.addWidget(QLabel("Skip (frames):"))
        opts.addWidget(self.spin_hard_skip)
        self.chk_hard_resize = QCheckBox("Resize width")
        self.chk_hard_resize.setToolTip(
            "Optional proportional resize before writing JPGs."
        )
        self.spin_hard_width = QSpinBox()
        self.spin_hard_width.setRange(10, 10000)
        self.spin_hard_width.setValue(480)
        self.spin_hard_width.setEnabled(False)
        self.chk_hard_resize.toggled.connect(self.spin_hard_width.setEnabled)
        opts.addWidget(self.chk_hard_resize)
        opts.addWidget(self.spin_hard_width)
        opts.addStretch(1)
        hard_l.addLayout(opts)

        self.btn_gen_hard = QPushButton("Generate training images from ranges")
        self.btn_gen_hard.setToolTip(
            "Extract still frames from the source video for every listed range."
        )
        self.btn_gen_hard.clicked.connect(self._generate_hard_case_images)
        hard_l.addWidget(self.btn_gen_hard)

        self.lbl_hard_status = QLabel("")
        self.lbl_hard_status.setWordWrap(True)
        hard_l.addWidget(self.lbl_hard_status)

        right_l.addWidget(hard_box)

        save_row = QHBoxLayout()
        self.btn_save = QPushButton(
            "Save package (switches + remapped tracklets + subjects)"
        )
        self.btn_save.setToolTip(
            "Finalize switch markers, re-save corrected tracklets from original "
            "geometry, write subjects.json"
        )
        self.btn_save.clicked.connect(self.save_package)
        save_row.addWidget(self.btn_save)
        right_l.addLayout(save_row)

        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        self.setEnabled_controls(False)

    def setEnabled_controls(self, on: bool) -> None:
        for w in (
            self.btn_back,
            self.btn_play,
            self.btn_fwd,
            self.btn_prev_risk,
            self.btn_next_risk,
            self.btn_mark,
            self.btn_remove,
            self.btn_del,
            self.btn_undo,
            self.slider,
            self.btn_save,
            self.subjects_table,
            self.marker_table,
            self.btn_range_start,
            self.btn_range_end,
            self.btn_add_risk,
            self.btn_remove_range,
            self.btn_clear_ranges,
            self.btn_gen_hard,
            self.list_ranges,
            self.ed_hard_out,
            self.spin_hard_skip,
            self.chk_hard_resize,
        ):
            w.setEnabled(on)
        if on:
            self.spin_hard_width.setEnabled(self.chk_hard_resize.isChecked())
        else:
            self.spin_hard_width.setEnabled(False)

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._nudge(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._nudge(1))
        QShortcut(QKeySequence("S"), self, self._mark_swap)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self._delete_selected_marker)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("U"), self, self._undo)
        QShortcut(QKeySequence("R"), self, lambda: self._remove_at_current_frame())
        QShortcut(QKeySequence("["), self, self._set_range_start)
        QShortcut(QKeySequence("]"), self, self._set_range_end_and_add)

    # --- package load ---

    def refresh_video_list(self) -> None:
        # Preserve the selected video across rebuilds.
        current = self.combo_video.currentData()
        self.combo_video.blockSignals(True)
        self.combo_video.clear()
        self.combo_video.addItem("(open package folder…)", "")
        for label, resolved in list_project_video_choices(self.project.project):
            tracks = discover_tracklets_dir(self.project.project, resolved)
            flag = "✓" if tracks else "·"
            self.combo_video.addItem(f"{flag}  {label}", str(resolved))
        if current:
            for i in range(self.combo_video.count()):
                if self.combo_video.itemData(i) == current:
                    self.combo_video.setCurrentIndex(i)
                    break
        self.combo_video.blockSignals(False)

    def _on_video_combo(self, _i: int) -> None:
        pass

    def _browse_package(self) -> None:
        start = self.review_dir or self.project.project.root_dir or ""
        d = browse_existing_directory(
            self, start, "Select id_review / tracklets folder"
        )
        if d:
            self.load_package(d)

    def _load_selected_video_package(self) -> None:
        path = self.combo_video.currentData()
        if not path:
            self._browse_package()
            return
        tracks = discover_tracklets_dir(self.project.project, str(path))
        if not tracks:
            QMessageBox.warning(
                self,
                "Review IDs",
                "No tracklets / id_review folder found for this video.\n"
                "Run Detect + track first, or Open package folder…\n\n"
                f"Video:\n{path}",
            )
            return
        self.project.set_current_video(str(path), dirty=True)
        from LabGym.gui_pyside.project.paths import find_video_entry

        entry = find_video_entry(self.project.project, str(path))
        if entry is not None:
            try:
                rel = (
                    str(
                        Path(tracks)
                        .resolve()
                        .relative_to(Path(self.project.project.root_dir).resolve())
                    )
                    if self.project.project.root_dir
                    else tracks
                )
            except (ValueError, OSError):
                rel = tracks
            if entry.detection_dir != rel:
                entry.detection_dir = rel
                self.project.mark_dirty()
        self.load_package(tracks)

    def load_package(self, review_dir: str) -> bool:
        self._release_cap()
        try:
            pkg = load_review_package(review_dir)
        except (FileNotFoundError, ValueError) as exc:
            QMessageBox.warning(self, "Review IDs", str(exc))
            self.setEnabled_controls(False)
            return False

        self.review_dir = pkg.review_dir
        self.events = pkg.events
        self.markers = pkg.markers
        self._undo_stack.clear()
        self._stores = pkg.stores
        self._baseline_stores = pkg.baseline_stores
        self._already_corrected = pkg.already_corrected
        self.animal_kind = pkg.animal_kind
        self.n_frames = pkg.n_frames
        store = self._stores[self.animal_kind]
        self.involved_ids = list(store.ids)
        self.fps = pkg.fps
        self.frame = 0
        self._dirty = False
        self._training_ranges = []
        self._range_start = None
        self._refresh_range_list()
        self._prefills_hard_case_out()

        self.kind_combo.blockSignals(True)
        self.kind_combo.clear()
        self.kind_combo.addItems(sorted(self._stores.keys()))
        self.kind_combo.setCurrentText(self.animal_kind)
        self.kind_combo.blockSignals(False)
        self._refresh_id_combos()

        self.subjects_table.set_subjects(pkg.subjects)
        self.slider.setMaximum(max(0, self.n_frames - 1))
        self.setEnabled_controls(True)
        corr = "corrected" if self._already_corrected else "not yet remapped on disk"
        self.lbl_pkg.setText(f"Package: {self.review_dir}  ·  tracklets: {corr}")
        self._seek(0)
        self._refresh_marker_list()
        self._update_undo_button()
        return True

    def _refresh_id_combos(self) -> None:
        store = self._stores.get(self.animal_kind)
        ids = list(store.ids) if store else []
        self.id_a.blockSignals(True)
        self.id_b.blockSignals(True)
        self.id_a.clear()
        self.id_b.clear()
        for i in ids:
            self.id_a.addItem(str(i), int(i))
            self.id_b.addItem(str(i), int(i))
        if len(ids) >= 2:
            self.id_a.setCurrentIndex(0)
            self.id_b.setCurrentIndex(1)
        self.id_a.blockSignals(False)
        self.id_b.blockSignals(False)
        self.involved_ids = ids[:2] if len(ids) >= 2 else list(ids)

    def _selected_swap_ids(self) -> List[int]:
        a = self.id_a.currentData()
        b = self.id_b.currentData()
        if a is None or b is None:
            return list(self.involved_ids)[:2]
        return [int(a), int(b)]

    # --- seek / render ---

    def _primary_store(self):
        return self._stores.get(self.animal_kind)

    def _video_meta(self):
        store = self._primary_store()
        meta = dict(store.meta) if store else {}
        if self.events and self.events[0].video:
            meta.setdefault("video", self.events[0].video)
        video = resolve_video_path(
            self.review_dir,
            meta,
            self.events,
            self.project.current_video_path(),
        )
        fps = float(meta.get("fps") or self.fps or 10)
        return video, meta, fps

    def _release_cap(self) -> None:
        self._cap.release()

    def _stop_play(self) -> None:
        self._playing = False
        self._play_timer.stop()
        self.btn_play.setText("Play")

    def _seek(self, frame: int) -> None:
        self.frame = int(max(0, min(frame, max(0, self.n_frames - 1))))
        self._updating = True
        try:
            self.slider.setMaximum(max(0, self.n_frames - 1))
            self.slider.setValue(self.frame)
        finally:
            self._updating = False
        self._render()
        self._refresh_timeline()

    def _nudge(self, d: int) -> None:
        self._stop_play()
        self._seek(self.frame + d)

    def _on_slider(self, value: int) -> None:
        if self._updating:
            return
        self._stop_play()
        self._seek(int(value))

    def _toggle_play(self) -> None:
        if self._playing:
            self._stop_play()
            return
        self._playing = True
        self.btn_play.setText("Pause")
        interval = max(10, int(1000 / max(1.0, self.fps)))
        self._play_timer.start(interval)

    def _on_play_tick(self) -> None:
        if not self._playing:
            return
        if self.frame >= self.n_frames - 1:
            self._stop_play()
            return
        self._seek(self.frame + 1)

    def _render(self) -> None:
        video, meta, fps = self._video_meta()
        self.fps = fps
        t = self.frame / fps if fps else 0.0
        store = self._primary_store()

        frame, v_idx, err = read_preview_frame(
            self._cap,
            video_path=str(video) if video else "",
            store_meta=meta,
            analysis_frame=self.frame,
            fps=fps,
        )
        if frame is None:
            self.status.setText(
                f"Frame {self.frame}/{self.n_frames - 1}  t={t:.2f}s  |  {err or 'no video'}"
            )
            self.video_label.setText("(no video)")
            return

        frame, n_prev = compose_preview_frame(
            frame,
            store=store,
            markers=self.markers,
            animal_kind=self.animal_kind,
            analysis_frame=self.frame,
            already_corrected=self._already_corrected,
            highlight_ids=self._selected_swap_ids(),
        )
        n_risk = sum(
            1
            for e in events_for_kind(self.events, self.animal_kind)
            if e.start_frame <= self.frame <= e.end_frame
        )
        self.status.setText(
            format_frame_status(
                analysis_frame=self.frame,
                n_frames=self.n_frames,
                t_sec=t,
                video_idx=v_idx,
                n_markers=len(self.markers),
                in_risk=bool(n_risk),
                n_preview_markers=n_prev,
            )
        )
        self.video_label.setPixmap(bgr_to_qpixmap(frame))

    # --- markers ---

    def _events_for_kind(self) -> List[ContactEvent]:
        return events_for_kind(self.events, self.animal_kind)

    def _refresh_timeline(self) -> None:
        self.timeline.set_data(
            self.n_frames,
            self.frame,
            self._events_for_kind(),
            [m for m in self.markers if m.animal_kind == self.animal_kind],
            min_risk=self.min_risk,
            training_ranges=[
                (r.start_frame, r.end_frame) for r in self._training_ranges
            ],
            pending_start=self._range_start,
        )

    # --- hard-case training ranges ---

    def _prefills_hard_case_out(self) -> None:
        if self.ed_hard_out.text().strip():
            return
        root = self.project.project.root_dir or ""
        self.ed_hard_out.setText(default_output_dir(root or None))

    def _browse_hard_out(self) -> None:
        set_line_edit_directory(
            self, self.ed_hard_out, caption="Select folder for training images"
        )

    def _set_range_start(self) -> None:
        if not self.review_dir:
            return
        self._range_start = int(self.frame)
        self.lbl_range_pending.setText(f"Range start: analysis frame {self._range_start}")
        self._refresh_timeline()

    def _set_range_end_and_add(self) -> None:
        if not self.review_dir:
            return
        end = int(self.frame)
        start = self._range_start if self._range_start is not None else end
        self._add_range(AnalysisFrameRange(start, end))
        self._range_start = None
        self.lbl_range_pending.setText("Range start: (not set)")
        self._refresh_timeline()

    def _add_range(self, rng: AnalysisFrameRange) -> None:
        rng = AnalysisFrameRange(
            max(0, min(rng.start_frame, self.n_frames - 1)),
            max(0, min(rng.end_frame, self.n_frames - 1)),
            note=rng.note,
        )
        self._training_ranges.append(rng)
        self._refresh_range_list()
        self._refresh_timeline()

    def _add_current_risk_band(self) -> None:
        if not self.review_dir:
            return
        bands = [
            e
            for e in self._events_for_kind()
            if e.risk_score >= self.min_risk
            and e.start_frame <= self.frame <= e.end_frame
        ]
        if not bands:
            QMessageBox.information(
                self,
                "No risk band",
                "Playhead is not inside a contact-risk band (or all bands are "
                "hidden by the Min risk filter).",
            )
            return
        # Prefer the tightest band containing the playhead.
        bands.sort(key=lambda e: (e.end_frame - e.start_frame, e.start_frame))
        e = bands[0]
        self._add_range(
            AnalysisFrameRange(e.start_frame, e.end_frame, note="risk")
        )

    def _remove_selected_range(self) -> None:
        row = self.list_ranges.currentRow()
        if row < 0 or row >= len(self._training_ranges):
            return
        del self._training_ranges[row]
        self._refresh_range_list()
        self._refresh_timeline()

    def _clear_ranges(self) -> None:
        self._training_ranges.clear()
        self._range_start = None
        self.lbl_range_pending.setText("Range start: (not set)")
        self._refresh_range_list()
        self._refresh_timeline()

    def _refresh_range_list(self) -> None:
        self.list_ranges.clear()
        for r in self._training_ranges:
            self.list_ranges.addItem(r.label(self.fps))

    def _generate_hard_case_images(self) -> None:
        if self._extract_thread is not None:
            QMessageBox.warning(self, "Busy", "Image extraction is already running.")
            return
        if not self.review_dir:
            QMessageBox.information(self, "Generate images", "Load a package first.")
            return
        if not self._training_ranges:
            QMessageBox.warning(
                self,
                "No ranges",
                "Add at least one time range (Set start / Set end, or Add current "
                "risk band).",
            )
            return
        out = self.ed_hard_out.text().strip()
        if not out:
            QMessageBox.warning(
                self, "Missing output", "Choose an output folder for the images."
            )
            return

        video, meta, fps = self._video_meta()
        if not video:
            QMessageBox.warning(
                self,
                "No video",
                "Could not resolve the source video for this package.\n"
                "Select the project video or ensure the package meta points at a file.",
            )
            return

        n_est = 0
        from LabGym.gui_pyside.workbenches.detector.review_ids_hard_cases import (
            frames_to_extract,
        )

        n_est = len(
            frames_to_extract(
                self._training_ranges,
                skip=int(self.spin_hard_skip.value()),
                n_frames=self.n_frames,
            )
        )
        reply = QMessageBox.question(
            self,
            "Generate training images?",
            f"Extract ~{n_est} frame(s) from {len(self._training_ranges)} range(s)\n"
            f"Video:\n{video}\n\nInto:\n{out}",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Release preview capture so the worker can open the same file on Windows.
        self._stop_play()
        self._release_cap()

        framewidth = (
            int(self.spin_hard_width.value())
            if self.chk_hard_resize.isChecked()
            else None
        )
        worker = _HardCaseExtractWorker(
            video_path=str(video),
            out_path=out,
            ranges=list(self._training_ranges),
            store_meta=dict(meta),
            fps=float(fps),
            skip=int(self.spin_hard_skip.value()),
            framewidth=framewidth,
            n_frames=self.n_frames,
        )
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_hard_progress)
        worker.finished.connect(self._on_hard_finished)
        worker.error.connect(self._on_hard_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._clear_hard_thread)

        self._extract_thread = thread
        self.btn_gen_hard.setEnabled(False)
        self.lbl_hard_status.setText("Extracting frames…")
        thread.start()

    def _clear_hard_thread(self) -> None:
        self._extract_thread = None
        if self.review_dir:
            self.btn_gen_hard.setEnabled(True)
        self._render()

    def _on_hard_progress(self, msg: str) -> None:
        self.lbl_hard_status.setText(msg)

    def _on_hard_finished(self, result) -> None:
        out = self.ed_hard_out.text().strip()
        if result.error and not result.n_written:
            self.lbl_hard_status.setText(f"Failed: {result.error}")
            QMessageBox.critical(self, "Extraction failed", result.error)
            return
        msg = (
            f"Wrote {result.n_written} image(s)"
            + (f" ({result.n_failed} failed reads)" if result.n_failed else "")
            + f"\n→ {out}"
        )
        self.lbl_hard_status.setText(msg.replace("\n", "  "))
        QMessageBox.information(
            self,
            "Training images ready",
            f"{msg}\n\n"
            "Next: annotate with EZannot (or similar), export COCO JSON, then "
            "Train detector with this image folder.",
        )

    def _on_hard_error(self, err: str) -> None:
        self.lbl_hard_status.setText(f"Error: {err}")
        QMessageBox.critical(self, "Extraction failed", err)

    def _push_undo(self) -> None:
        self._undo_stack.append(clone_markers(self.markers))
        if len(self._undo_stack) > 50:
            self._undo_stack = self._undo_stack[-50:]
        self._update_undo_button()

    def _update_undo_button(self) -> None:
        self.btn_undo.setEnabled(bool(self._undo_stack))

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _apply_marker_change(self) -> None:
        self.markers.sort(key=lambda x: x.frame)
        self._dirty = True
        self._refresh_marker_list()
        self._refresh_timeline()
        self._render()
        self._update_undo_button()

    def _refresh_marker_list(self) -> None:
        self.markers_panel.set_markers(self.markers)

    def _mark_swap(self) -> None:
        if not self.review_dir:
            return
        if self._already_corrected:
            QMessageBox.information(
                self,
                "Tracklets already corrected",
                "This package’s tracklets were already remapped on disk.\n"
                "You can still edit names/roles and save subjects.json.\n\n"
                "To mark new ID swaps against raw tracks, re-export tracklets "
                "from Detect + track (or restore an uncorrected package).",
            )
            return
        ids = self._selected_swap_ids()
        if len(ids) != 2 or ids[0] == ids[1]:
            QMessageBox.warning(
                self, "Mark swap", "Select two different animal IDs to swap."
            )
            return
        self._push_undo()
        self.markers = [
            m
            for m in self.markers
            if not (m.frame == self.frame and m.animal_kind == self.animal_kind)
        ]
        try:
            m = make_swap_marker(
                self.frame,
                self.animal_kind,
                ids,
                fps=self.fps,
                marker_id=f"s{self.frame:06d}_{self.animal_kind}",
            )
        except ValueError as exc:
            if self._undo_stack:
                self.markers = self._undo_stack.pop()
            QMessageBox.warning(self, "Cannot mark", str(exc))
            self._update_undo_button()
            return
        for ev in self._events_for_kind():
            if ev.start_frame <= m.frame <= ev.end_frame + 5:
                m.linked_event_id = ev.event_id
                break
        self.markers.append(m)
        self._apply_marker_change()

    def _delete_selected_marker(self) -> None:
        mid = self.markers_panel.selected_marker_id()
        if not mid:
            if self._remove_at_current_frame(silent_if_none=True):
                return
            QMessageBox.information(
                self,
                "No marker selected",
                "Select a marker in the list, or move to a marked frame and remove.",
            )
            return
        self._push_undo()
        self.markers = [m for m in self.markers if m.marker_id != mid]
        self._apply_marker_change()

    def _remove_at_current_frame(self, silent_if_none: bool = False) -> bool:
        to_remove = [
            m
            for m in self.markers
            if m.frame == self.frame and m.animal_kind == self.animal_kind
        ]
        if not to_remove:
            if not silent_if_none:
                QMessageBox.information(
                    self, "Nothing to remove", f"No switch marker at frame {self.frame}."
                )
            return False
        self._push_undo()
        ids = {m.marker_id for m in to_remove}
        self.markers = [m for m in self.markers if m.marker_id not in ids]
        self._apply_marker_change()
        return True

    def _undo(self) -> None:
        if not self._undo_stack:
            QMessageBox.information(self, "Undo", "Nothing to undo.")
            return
        self.markers = self._undo_stack.pop()
        self._apply_marker_change()

    def _on_marker_activated(self, row: int, _col: int) -> None:
        f = self.markers_panel.frame_at_row(row)
        if f is None:
            return
        self._stop_play()
        self._seek(f)

    def _jump_risk(self, direction: int) -> None:
        bands = sorted(self._events_for_kind(), key=lambda e: e.start_frame)
        bands = [e for e in bands if e.risk_score >= self.min_risk]
        if not bands:
            return
        if direction > 0:
            for e in bands:
                if e.start_frame > self.frame:
                    self._stop_play()
                    self._seek(e.end_frame)
                    return
            self._seek(bands[0].start_frame)
        else:
            for e in reversed(bands):
                if e.end_frame < self.frame:
                    self._stop_play()
                    self._seek(e.end_frame)
                    return
            self._seek(bands[-1].start_frame)

    def _on_risk_filter(self, value: float) -> None:
        self.min_risk = float(value)
        self._refresh_timeline()

    def _on_kind(self, kind: str) -> None:
        if not kind or kind not in self._stores:
            return
        self.animal_kind = kind
        store = self._primary_store()
        if store:
            self.n_frames = store.n_frames
            self.involved_ids = list(store.ids)
            self.fps = float(store.meta.get("fps") or self.fps)
        self._refresh_id_combos()
        self._seek(min(self.frame, self.n_frames - 1))

    # --- save ---

    def save_package(self) -> None:
        if not self.review_dir:
            QMessageBox.information(self, "Save", "Load a package first.")
            return
        self._stop_play()
        result = save_review_package(
            self.review_dir,
            self.markers,
            self.events,
            self.subjects_table.get_subjects(),
            already_corrected=self._already_corrected,
            baseline_stores=self._baseline_stores,
        )
        if not result.ok:
            QMessageBox.critical(self, "Save failed", result.error)
            return
        self._already_corrected = result.already_corrected
        if result.stores:
            self._stores = result.stores
            self._baseline_stores = result.baseline_stores
        self._dirty = False
        self.lbl_pkg.setText(f"Package: {self.review_dir}  ·  tracklets: corrected")
        self.package_saved.emit(self.review_dir)
        QMessageBox.information(
            self,
            "Saved",
            f"Saved identity package:\n{self.review_dir}\n\n"
            f"Switches: {len(self.markers)}\n"
            f"{result.remap_note}"
            f"Subjects: {result.n_subjects}\n\n"
            "Annotate ethogram will pick up names/colors on next load.",
        )
        self._render()

    def closeEvent(self, event) -> None:
        self._stop_play()
        self._release_cap()
        super().closeEvent(event)
