"""Detector → Review IDs & assign names/roles (PySide)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.project.paths import (
    discover_tracklets_dir,
    list_project_video_choices,
)
from LabGym.gui_pyside.workbenches.detector.risk_timeline import RiskTimeline
from LabGym.gui_pyside.workbenches.detector.subjects_table import SubjectsTable
from LabGym.identity.package import (
    SubjectRecord,
    apply_decisions_and_save_tracklets,
    clone_store,
    load_subjects,
    save_subjects,
    subjects_from_track_ids,
)
from LabGym.id_review.dataset import (
    finalize_switch_annotations,
    load_events,
    load_switches,
    make_swap_marker,
)
from LabGym.id_review.samples import (
    analysis_frame_to_video_frame,
    detections_at_frame_after_markers,
    draw_detections_overlay,
)
from LabGym.id_review.tracklets import load_tracklets
from LabGym.id_review.types import ContactEvent, SwitchMarker


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
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_path: Optional[str] = None
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

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)

        self._build_ui()
        self._bind_shortcuts()
        self.project.changed.connect(self.refresh_video_list)
        self.project.project_replaced.connect(self.refresh_video_list)
        self.refresh_video_list()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        help_txt = QLabel(
            "Orange/red timeline = automatic contact risk. Green ticks = your switch markers. "
            "Mark where IDs flip, assign experimental names/roles, then Save. "
            "Keys: ←/→ step, Space play, S mark swap, Delete remove, Ctrl+Z undo."
        )
        help_txt.setWordWrap(True)
        root.addWidget(help_txt)

        # Package / video picker
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

        # --- left: video + transport ---
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
        filt.addWidget(QLabel("Min risk:"))
        self.spin_risk = QDoubleSpinBox()
        self.spin_risk.setRange(0.0, 1.0)
        self.spin_risk.setSingleStep(0.05)
        self.spin_risk.setDecimals(2)
        self.spin_risk.valueChanged.connect(self._on_risk_filter)
        filt.addWidget(self.spin_risk)
        filt.addWidget(QLabel("Kind:"))
        self.kind_combo = QComboBox()
        self.kind_combo.currentTextChanged.connect(self._on_kind)
        filt.addWidget(self.kind_combo)
        filt.addWidget(QLabel("Swap IDs:"))
        self.id_a = QComboBox()
        self.id_b = QComboBox()
        filt.addWidget(self.id_a)
        filt.addWidget(self.id_b)
        filt.addStretch(1)
        left_l.addLayout(filt)

        split.addWidget(left)

        # --- right: markers + subjects ---
        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)

        right_l.addWidget(QLabel("Switch markers"))
        self.marker_table = QTableWidget(0, 5)
        self.marker_table.setHorizontalHeaderLabels(
            ["ID", "Frame", "Time (s)", "IDs", "Linked risk"]
        )
        self.marker_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.marker_table.cellDoubleClicked.connect(self._on_marker_activated)
        right_l.addWidget(self.marker_table, 1)

        subj_box = QGroupBox("Names / roles (saved as subjects.json)")
        subj_l = QVBoxLayout(subj_box)
        self.subjects_table = SubjectsTable()
        self.subjects_table.changed.connect(self._mark_dirty)
        subj_l.addWidget(self.subjects_table)
        right_l.addWidget(subj_box, 1)

        save_row = QHBoxLayout()
        self.btn_save = QPushButton("Save package (switches + remapped tracklets + subjects)")
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
        ):
            w.setEnabled(on)

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._nudge(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._nudge(1))
        QShortcut(QKeySequence("S"), self, self._mark_swap)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, self._delete_selected_marker)
        QShortcut(QKeySequence("Ctrl+Z"), self, self._undo)
        QShortcut(QKeySequence("U"), self, self._undo)
        QShortcut(QKeySequence("R"), self, lambda: self._remove_at_current_frame())

    # --- package load ---

    def refresh_video_list(self) -> None:
        self.combo_video.blockSignals(True)
        self.combo_video.clear()
        self.combo_video.addItem("(open package folder…)", "")
        for label, resolved in list_project_video_choices(self.project.project):
            tracks = discover_tracklets_dir(self.project.project, resolved)
            flag = "✓" if tracks else "·"
            self.combo_video.addItem(f"{flag}  {label}", resolved)
        self.combo_video.blockSignals(False)

    def _on_video_combo(self, _i: int) -> None:
        pass

    def _browse_package(self) -> None:
        start = self.review_dir or self.project.project.root_dir or ""
        d = QFileDialog.getExistingDirectory(
            self, "Select id_review / tracklets folder", start
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
        # Remember detection dir on the video entry when possible
        entry = None
        from LabGym.gui_pyside.project.paths import find_video_entry

        entry = find_video_entry(self.project.project, str(path))
        if entry is not None:
            try:
                rel = str(Path(tracks).resolve().relative_to(
                    Path(self.project.project.root_dir).resolve()
                )) if self.project.project.root_dir else tracks
            except (ValueError, OSError):
                rel = tracks
            if entry.detection_dir != rel:
                entry.detection_dir = rel
                self.project.mark_dirty()
        self.load_package(tracks)

    def load_package(self, review_dir: str) -> bool:
        review_dir = str(Path(review_dir).resolve())
        if not Path(review_dir).is_dir():
            QMessageBox.warning(self, "Review IDs", f"Not a folder:\n{review_dir}")
            return False

        self._release_cap()
        self.review_dir = review_dir
        self.events = load_events(review_dir)
        self.markers = load_switches(review_dir)
        self._undo_stack.clear()
        self._stores.clear()
        self._baseline_stores.clear()

        # Load tracklets
        from LabGym.annotator.core.tracklets_bridge import discover_tracklet_kinds

        kinds = discover_tracklet_kinds(review_dir)
        if not kinds:
            QMessageBox.warning(
                self,
                "Review IDs",
                f"No *_tracklets_meta.json in:\n{review_dir}",
            )
            self.setEnabled_controls(False)
            return False

        from LabGym.id_review.apply import read_tracklets_identity_status

        status = read_tracklets_identity_status(review_dir)
        self._already_corrected = bool(status.get("corrected"))

        for kind in kinds:
            store = load_tracklets(review_dir, kind)
            self._stores[kind] = store
            self._baseline_stores[kind] = clone_store(store)

        # Prefer kind with most ids
        self.animal_kind = max(
            self._stores.keys(),
            key=lambda k: (len(self._stores[k].ids), self._stores[k].n_frames),
        )
        store = self._stores[self.animal_kind]
        self.n_frames = max(1, store.n_frames)
        self.involved_ids = list(store.ids)
        self.fps = float(
            store.meta.get("fps")
            or (self.events[0].fps if self.events else 10)
            or 10
        )
        self.frame = 0
        self._dirty = False

        self.kind_combo.blockSignals(True)
        self.kind_combo.clear()
        self.kind_combo.addItems(sorted(self._stores.keys()))
        self.kind_combo.setCurrentText(self.animal_kind)
        self.kind_combo.blockSignals(False)
        self._refresh_id_combos()

        # Subjects table
        recs = load_subjects(review_dir)
        if not recs:
            kind_ids = {k: list(s.ids) for k, s in self._stores.items()}
            recs = subjects_from_track_ids(kind_ids)
        self.subjects_table.set_subjects(recs)

        self.slider.setMaximum(max(0, self.n_frames - 1))
        self.setEnabled_controls(True)
        corr = "corrected" if self._already_corrected else "not yet remapped on disk"
        self.lbl_pkg.setText(f"Package: {review_dir}  ·  tracklets: {corr}")
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

    def _video_meta(self) -> Tuple[Optional[str], dict, float, Optional[int]]:
        store = self._primary_store()
        meta = dict(store.meta) if store else {}
        if self.events and self.events[0].video:
            meta.setdefault("video", self.events[0].video)
        video = meta.get("video")
        # Resolve relative video paths against review_dir / project
        if video and not Path(str(video)).is_file():
            candidates = [
                Path(self.review_dir) / video,
                Path(self.review_dir).parent / Path(video).name,
            ]
            cur = self.project.current_video_path()
            if cur:
                candidates.insert(0, Path(cur))
            for c in candidates:
                if c.is_file():
                    video = str(c)
                    break
        fps = float(meta.get("fps") or self.fps or 10)
        return video, meta, fps, meta.get("framewidth")

    def _ensure_cap(self, path: str) -> bool:
        if self._cap is not None and self._cap_path == path:
            return True
        self._release_cap()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
        self._cap = cap
        self._cap_path = path
        return True

    def _release_cap(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._cap_path = None

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
        video, meta, fps, fw = self._video_meta()
        self.fps = fps
        t = self.frame / fps if fps else 0.0
        store = self._primary_store()

        if not video or not self._ensure_cap(str(video)):
            self.status.setText(
                f"Frame {self.frame}/{self.n_frames - 1}  t={t:.2f}s  |  "
                f"video unavailable: {video}"
            )
            self.video_label.setText("(no video)")
            return

        v_idx = analysis_frame_to_video_frame(meta, self.frame, fps)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, v_idx)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self.status.setText(f"Failed to read video frame {v_idx}")
            return
        if fw is not None:
            try:
                fw = int(fw)
                h, w = frame.shape[:2]
                if w != fw and fw > 0:
                    frame = cv2.resize(
                        frame, (fw, int(h * fw / w)), interpolation=cv2.INTER_AREA
                    )
            except Exception:
                pass

        applied = [
            m
            for m in self.markers
            if m.frame <= self.frame and m.animal_kind == self.animal_kind
        ]
        if store is not None:
            # If tracklets already remapped on disk, do not re-apply markers for preview
            # (would double-swap). Otherwise preview remaps from markers.
            if self._already_corrected:
                dets = detections_at_frame_after_markers(store, self.frame, [])
            else:
                dets = detections_at_frame_after_markers(store, self.frame, applied)
            frame = draw_detections_overlay(
                frame,
                dets,
                highlight_ids=self._selected_swap_ids(),
                frame_idx=self.frame,
                n_markers_applied=0 if self._already_corrected else len(applied),
            )

        n_risk = sum(
            1 for e in self._events_for_kind() if e.start_frame <= self.frame <= e.end_frame
        )
        self.status.setText(
            f"Analysis frame {self.frame}/{self.n_frames - 1}  |  t={t:.2f}s  |  "
            f"video f={v_idx}  |  switches={len(self.markers)}  |  "
            f"in risk={bool(n_risk)}  |  preview markers={len(applied)}"
        )
        self._set_bgr_image(frame)

    def _set_bgr_image(self, arr: np.ndarray, max_w=900, max_h=540) -> None:
        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        rgb = np.ascontiguousarray(
            cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        )
        qimg = QImage(rgb.data, nw, nh, nw * 3, QImage.Format.Format_RGB888).copy()
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    # --- markers ---

    def _events_for_kind(self) -> List[ContactEvent]:
        return [e for e in self.events if e.animal_kind == self.animal_kind]

    def _refresh_timeline(self) -> None:
        self.timeline.set_data(
            self.n_frames,
            self.frame,
            self._events_for_kind(),
            [m for m in self.markers if m.animal_kind == self.animal_kind],
            min_risk=self.min_risk,
        )

    def _snapshot_markers(self) -> List[SwitchMarker]:
        return [SwitchMarker.from_dict(m.to_dict()) for m in self.markers]

    def _push_undo(self) -> None:
        self._undo_stack.append(self._snapshot_markers())
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
        self.marker_table.setRowCount(0)
        rows = sorted(self.markers, key=lambda x: x.frame)
        self.marker_table.setRowCount(len(rows))
        for i, m in enumerate(rows):
            self.marker_table.setItem(i, 0, QTableWidgetItem(m.marker_id))
            self.marker_table.setItem(i, 1, QTableWidgetItem(str(m.frame)))
            self.marker_table.setItem(
                i, 2, QTableWidgetItem(f"{m.time_sec:.2f}" if m.time_sec is not None else "")
            )
            self.marker_table.setItem(
                i, 3, QTableWidgetItem(",".join(str(x) for x in m.involved_ids))
            )
            self.marker_table.setItem(i, 4, QTableWidgetItem(m.linked_event_id or ""))

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
        rows = sorted({i.row() for i in self.marker_table.selectedIndexes()})
        if not rows:
            if self._remove_at_current_frame(silent_if_none=True):
                return
            QMessageBox.information(
                self,
                "No marker selected",
                "Select a marker in the list, or move to a marked frame and remove.",
            )
            return
        mid = self.marker_table.item(rows[0], 0).text()
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
        try:
            f = int(self.marker_table.item(row, 1).text())
        except (ValueError, AttributeError):
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
        try:
            decisions = finalize_switch_annotations(
                self.review_dir,
                self.markers,
                events=self.events,
                export_samples=True,
            )
            n = 0
            remap_note = ""
            if not self._already_corrected:
                n = apply_decisions_and_save_tracklets(
                    self.review_dir,
                    decisions,
                    baseline_stores=self._baseline_stores,
                    source="pyside_id_review",
                )
                self._already_corrected = True
                # After first remap, live stores match corrected disk; refresh clones
                for kind in list(self._stores.keys()):
                    self._stores[kind] = load_tracklets(self.review_dir, kind)
                    self._baseline_stores[kind] = clone_store(self._stores[kind])
                remap_note = f"Remap applications: {n}\n"
            else:
                remap_note = (
                    "Tracklets already corrected — skipped re-apply "
                    "(subjects + switches updated).\n"
                )
            subjects = self.subjects_table.get_subjects()
            save_subjects(self.review_dir, subjects)
            self._dirty = False
            self.lbl_pkg.setText(
                f"Package: {self.review_dir}  ·  tracklets: corrected"
            )
            self.package_saved.emit(self.review_dir)
            QMessageBox.information(
                self,
                "Saved",
                f"Saved identity package:\n{self.review_dir}\n\n"
                f"Switches: {len(self.markers)}\n"
                f"{remap_note}"
                f"Subjects: {len(subjects)}\n\n"
                "Annotate ethogram will pick up names/colors on next load.",
            )
            self._render()
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def closeEvent(self, event) -> None:
        self._stop_play()
        self._release_cap()
        super().closeEvent(event)
