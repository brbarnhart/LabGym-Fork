"""MainWindow: top-level container for Phase 1 player + annotation.

Wires:
- VideoDisplayWidget
- PlaybackControls
- BehaviorPalette
- VideoHandler + AnnotationManager
- Keyboard-driven annotation + playback
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox,
    QDialog, QCheckBox, QLabel, QInputDialog,
)

from LabGym.annotator.core.video_handler import VideoHandler
from LabGym.annotator.core.annotation_manager import AnnotationManager
from LabGym.annotator.core.data_models import AnnotationSession
from LabGym.annotator.core.tracklets_bridge import (
    LoadedTracklets,
    apply_subjects_to_session,
    load_tracklets_for_annotator,
    overlays_at_video_frame,
    try_autoload_id_review,
)
from LabGym.annotator.ui.video_display import VideoDisplayWidget
from LabGym.annotator.ui.playback_controls import PlaybackControls
from LabGym.annotator.ui.behavior_palette import BehaviorPalette
from LabGym.annotator.ui.subject_panel import SubjectPanel
from LabGym.annotator.ui.timeline_widget import TimelineWidget
from LabGym.annotator.ui.export_dialog import ExportDialog
from LabGym.annotator.ui.bout_list import BoutListWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LabGym Behavior Annotator")
        self.resize(1280, 720)

        self.video = VideoHandler()
        self._current_frame = 0
        self._is_playing = False
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._on_playback_tick)
        self._target_fps = 30.0
        self._playback_speed = 1.0

        self.manager: AnnotationManager | None = None
        self.palette: BehaviorPalette | None = None
        self.subject_panel: Optional[SubjectPanel] = None
        self.bout_list_dialog: Optional["BoutListWidget"] = None
        self._bout_list_window: Optional[QDialog] = None
        self._loaded_tracklets: Optional[LoadedTracklets] = None
        self._show_track_overlays = True

        self._build_ui()
        self._setup_shortcuts()
        self._settings = QSettings("LabGym", "annotator")

    def _build_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)

        # Menu
        self.menubar = self.menuBar()

        # Edit menu (Undo)
        edit_menu = self.menubar.addMenu("&Edit")
        self.act_undo = QAction("&Undo", self)
        self.act_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self.act_undo.triggered.connect(self.undo_last_action)
        self.act_undo.setEnabled(False)
        edit_menu.addAction(self.act_undo)

        file_menu = self.menubar.addMenu("&File")
        act_open = QAction("&Open Video...", self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self.open_video)
        file_menu.addAction(act_open)

        act_save = QAction("&Save Annotations...", self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self.save_annotations)
        file_menu.addAction(act_save)

        act_load = QAction("Load &Annotations...", self)
        act_load.triggered.connect(self.load_annotations)
        file_menu.addAction(act_load)

        behaviors_menu = self.menubar.addMenu("&Behaviors")
        act_save_tmpl = QAction("&Save Behavior Template...", self)
        act_save_tmpl.triggered.connect(self.save_behavior_template)
        behaviors_menu.addAction(act_save_tmpl)

        act_load_tmpl = QAction("&Load Behavior Template...", self)
        act_load_tmpl.triggered.connect(self.load_behavior_template)
        behaviors_menu.addAction(act_load_tmpl)

        tools_menu = self.menubar.addMenu("&Tools")
        act_metrics = QAction("Show &Metrics...", self)
        act_metrics.triggered.connect(self.show_metrics)
        tools_menu.addAction(act_metrics)

        act_examples = QAction("Generate &Examples for LabGym...", self)
        act_examples.triggered.connect(self.show_export_dialog)
        tools_menu.addAction(act_examples)

        self.act_selection_mode = QAction("Select timeline &regions for export", self)
        self.act_selection_mode.setCheckable(True)
        self.act_selection_mode.setChecked(False)
        self.act_selection_mode.setToolTip(
            "When on, drag on the timeline to mark ranges. Use them via the "
            "checkbox in Generate Examples for LabGym. "
            "When off, the timeline only seeks (normal annotation)."
        )
        self.act_selection_mode.toggled.connect(self._on_selection_mode_toggled)
        tools_menu.addAction(self.act_selection_mode)

        act_clear_sel = QAction("Clear timeline selections", self)
        act_clear_sel.triggered.connect(self._clear_timeline_selections)
        tools_menu.addAction(act_clear_sel)

        act_labels = QAction("Export &frame_labels.csv only...", self)
        act_labels.triggered.connect(self.export_frame_labels_only)
        tools_menu.addAction(act_labels)

        act_labels_all = QAction("Export frame_labels (all &subjects)...", self)
        act_labels_all.triggered.connect(self.export_frame_labels_all_subjects)
        tools_menu.addAction(act_labels_all)

        act_export_labgym = QAction("Export LabGym training tables…", self)
        act_export_labgym.triggered.connect(self.export_labgym_tables)
        tools_menu.addAction(act_export_labgym)

        act_soft = QAction("Export soft_labels.csv for examples folder…", self)
        act_soft.triggered.connect(self.export_soft_labels_for_examples)
        tools_menu.addAction(act_soft)

        act_bouts = QAction("Edit &Bouts...", self)
        act_bouts.triggered.connect(self.show_bout_list)
        tools_menu.addAction(act_bouts)

        tracks_menu = self.menubar.addMenu("&Tracks")
        act_load_tracks = QAction("&Load Tracklets…", self)
        act_load_tracks.setShortcut(QKeySequence("Ctrl+T"))
        act_load_tracks.triggered.connect(self.load_tracklets_dialog)
        tracks_menu.addAction(act_load_tracks)

        self.act_show_tracks = QAction("Show track &overlays", self)
        self.act_show_tracks.setCheckable(True)
        self.act_show_tracks.setChecked(True)
        self.act_show_tracks.toggled.connect(self._on_show_tracks_toggled)
        tracks_menu.addAction(self.act_show_tracks)

        # Central: video + subjects/palette on right
        center_layout = QHBoxLayout()

        self.video_widget = VideoDisplayWidget()
        center_layout.addWidget(self.video_widget, 3)

        self.right_panel = QWidget()
        right_l = QVBoxLayout(self.right_panel)
        right_l.setContentsMargins(0, 0, 0, 0)

        self.subject_panel = SubjectPanel(self.right_panel)
        self.subject_panel.subject_changed.connect(self._on_subject_changed)
        self.subject_panel.mode_changed.connect(self._on_behavior_mode_changed)
        self.subject_panel.load_tracklets_requested.connect(self.load_tracklets_dialog)
        right_l.addWidget(self.subject_panel)

        self.palette_container = QWidget()
        pal_l = QVBoxLayout(self.palette_container)
        from PySide6.QtWidgets import QLabel
        pal_l.addWidget(QLabel("Behaviors (click or double-click to toggle)"))
        right_l.addWidget(self.palette_container, 1)

        center_layout.addWidget(self.right_panel, 1)

        main_layout.addLayout(center_layout, 1)

        # Timeline + optional selection-mode control
        timeline_row = QHBoxLayout()
        self.chk_selection_mode = QCheckBox("Select regions for export")
        self.chk_selection_mode.setToolTip(
            "Toggle timeline drag-selection. Off = seek only (annotation). "
            "On = drag to add export ranges (additive)."
        )
        self.chk_selection_mode.toggled.connect(self._on_selection_mode_toggled)
        timeline_row.addWidget(self.chk_selection_mode)
        self.lbl_selection_status = QLabel("")
        self.lbl_selection_status.setStyleSheet("color: #9cf;")
        timeline_row.addWidget(self.lbl_selection_status, 1)
        main_layout.addLayout(timeline_row)

        self.timeline = TimelineWidget(
            manager_getter=lambda: self.manager,
            video_getter=lambda: self.video,
            parent=self
        )
        self.timeline.seek_requested.connect(self.seek_to)
        self.timeline.selection_changed.connect(self._on_timeline_selection_changed)
        self.timeline.selection_mode_changed.connect(self._sync_selection_mode_ui)
        main_layout.addWidget(self.timeline)

        # Bottom controls
        self.controls = PlaybackControls()
        self.controls.play_pause_requested.connect(self.toggle_play)
        self.controls.seek_requested.connect(self.seek_to)
        self.controls.step_requested.connect(self.step_frame)
        self.controls.speed_changed.connect(self.set_speed)
        main_layout.addWidget(self.controls)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready. Open a video to begin.")

    def _setup_shortcuts(self):
        # Playback
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self.toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self.step_frame(1))
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self.step_frame(-1))
        QShortcut(QKeySequence("Ctrl+Right"), self, lambda: self.step_frame(10))
        QShortcut(QKeySequence("Ctrl+Left"), self, lambda: self.step_frame(-10))

        # Subject cycle: [ previous, ] next
        QShortcut(QKeySequence(Qt.Key.Key_BracketLeft), self, self._cycle_subject_prev)
        QShortcut(QKeySequence(Qt.Key.Key_BracketRight), self, self._cycle_subject_next)

        # File already wired via actions + menu

    # --- Initialization of annotation session ---

    def _init_or_reset_manager(self, video_path: str, fps: float, total: int):
        """Create a fresh AnnotationSession + manager and wire palette."""
        sess = AnnotationSession(video_path=video_path, fps=fps, total_frames=total)
        self.manager = AnnotationManager(sess)

        # Clear old palette
        for i in reversed(range(self.palette_container.layout().count())):
            w = self.palette_container.layout().itemAt(i).widget()
            if w:
                w.setParent(None)

        self.palette = BehaviorPalette(self.manager, self.palette_container)
        self.palette_container.layout().addWidget(self.palette)

        self.palette.behavior_selected.connect(self._on_behavior_selected)
        self.palette.behavior_toggled.connect(self._toggle_bout_for_behavior)
        self.palette.behaviors_changed.connect(self._on_behaviors_changed)

        if self.subject_panel:
            self.subject_panel.set_manager(self.manager)

        # Add a couple of example behaviors for quick start (user can delete)
        if not self.manager.session.behaviors:
            self.manager.add_behavior("grooming", "#00AAFF", "1")
            self.manager.add_behavior("rearing", "#FFAA00", "2")
            self.palette.refresh()

        self._on_behaviors_changed()
        self._rebind_behavior_hotkeys()
        self._update_undo_action()
        if hasattr(self, "timeline"):
            self.timeline._adjust_height_for_subjects()

    def _on_behaviors_changed(self):
        if self.palette:
            self.video_widget.set_behavior_colors(self.palette.get_color_map())
            self._refresh_display_overlay()
            self._rebind_behavior_hotkeys()

    # Dynamic behavior hotkeys (simple implementation: recreate shortcuts on change)
    _behavior_shortcuts: list = []

    def _rebind_behavior_hotkeys(self):
        # Remove previous shortcuts
        for sc in getattr(self, "_behavior_shortcuts", []):
            try:
                sc.setEnabled(False)
                sc.deleteLater()
            except Exception:
                pass
        self._behavior_shortcuts = []

        if not self.manager:
            return

        for beh in self.manager.session.behaviors:
            if not beh.hotkey:
                continue
            key = getattr(Qt.Key, f"Key_{beh.hotkey.upper()}", None) or Qt.Key(ord(beh.hotkey.upper()))
            try:
                sc = QShortcut(QKeySequence(key), self)
                name = beh.name
                sc.activated.connect(lambda n=name: self._toggle_bout_for_behavior(n))
                self._behavior_shortcuts.append(sc)
            except Exception:
                pass  # non-critical if hotkey can't bind

    def _on_behavior_selected(self, name: str):
        self.statusBar().showMessage(f"Selected: {name}  (press hotkey or double-click to toggle bout)")

    def _toggle_bout_for_behavior(self, name: str):
        if not self.manager or not self.video.is_loaded:
            return
        try:
            partners = []
            if self.subject_panel:
                partners = self.subject_panel.get_partner_ids()
            action, bout = self.manager.toggle_bout(
                name, self._current_frame, partner_ids=partners
            )
            if self.palette:
                open_names = self.manager.get_open_behaviors_at_frame(self._current_frame)
                annotated = self.manager.get_annotated_behaviors_at_frame(self._current_frame)
                self.palette.update_active_indicators(open_names, annotated)
                self.palette.refresh()  # ensure list is consistent
            self._refresh_display_overlay()
            # Open/close can change live state without seeking; refresh bout list content
            if self.bout_list_dialog is not None and action == "closed":
                self.bout_list_dialog.refresh()
            self.statusBar().showMessage(f"{name}: {action} at frame {self._current_frame}")
            self._update_undo_action()
        except Exception as e:
            QMessageBox.warning(self, "Annotation error", str(e))

    def _current_track_overlays(self):
        if not self._loaded_tracklets or not self._show_track_overlays:
            return []
        colors = {}
        if self.manager:
            colors = {
                s.subject_id: s.color for s in self.manager.session.subjects
            }
        return overlays_at_video_frame(
            self._loaded_tracklets,
            self._current_frame,
            subject_colors=colors,
        )

    def _refresh_display_overlay(self):
        if not self.manager or not self.video.is_loaded:
            return
        open_names = self.manager.get_open_behaviors_at_frame(self._current_frame)
        annotated = self.manager.get_annotated_behaviors_at_frame(self._current_frame)
        if self.palette:
            self.palette.update_active_indicators(open_names, annotated)
        try:
            frame = self.video.get_frame(self._current_frame)
            self.video_widget.show_frame(
                frame,
                open_behaviors=open_names,
                annotated_behaviors=annotated,
                track_overlays=self._current_track_overlays(),
                active_subject_id=self.manager.session.active_subject_id,
            )
        except Exception:
            pass
        self._sync_bout_list_frame()
        if hasattr(self, "timeline"):
            self.timeline.update()

    def _on_subject_changed(self, subject_id: int):
        if self.manager:
            try:
                self.manager.set_active_subject(int(subject_id))
            except ValueError:
                return
        self._refresh_display_overlay()
        if self.bout_list_dialog is not None:
            self.bout_list_dialog.refresh()
        if hasattr(self, "timeline"):
            self.timeline.update()
        self.statusBar().showMessage(f"Active subject: {subject_id}")

    def _on_behavior_mode_changed(self, mode: int):
        if hasattr(self, "timeline"):
            self.timeline._adjust_height_for_subjects()
            self.timeline.update()
        self._refresh_display_overlay()
        self.statusBar().showMessage(f"Behavior mode set to {mode}")

    def _cycle_subject_prev(self):
        if self.subject_panel:
            self.subject_panel.select_previous()

    def _cycle_subject_next(self):
        if self.subject_panel:
            self.subject_panel.select_next()

    def _on_show_tracks_toggled(self, checked: bool):
        self._show_track_overlays = bool(checked)
        self.video_widget.set_show_tracks(checked)
        self._refresh_display_overlay()

    def load_tracklets_dialog(self):
        if not self.manager:
            QMessageBox.information(
                self, "Tracklets", "Load a video first, then load tracklets."
            )
            return
        start = ""
        if self.video.is_loaded:
            start = str(Path(self.video.metadata.path).parent)
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select id_review (or tracklets) folder",
            start,
        )
        if not directory:
            return
        try:
            self._apply_tracklets(
                load_tracklets_for_annotator(
                    directory,
                    video_total_frames=(
                        self.video.total_frames if self.video.is_loaded else None
                    ),
                )
            )
            self.statusBar().showMessage(f"Loaded tracklets from {directory}")
        except Exception as e:
            QMessageBox.critical(self, "Tracklets", f"Failed to load tracklets:\n{e}")

    def _apply_tracklets(self, loaded: LoadedTracklets) -> None:
        if not self.manager:
            return
        self._loaded_tracklets = loaded
        apply_subjects_to_session(self.manager.session, loaded)
        self.manager._ensure_active_maps()
        if self.subject_panel:
            self.subject_panel.set_manager(self.manager)
        if hasattr(self, "timeline"):
            self.timeline._adjust_height_for_subjects()
        self._refresh_display_overlay()

    def _sync_bout_list_frame(self) -> None:
        """Keep the Edit Bouts window highlight in sync with the current video frame."""
        if self.bout_list_dialog is not None:
            self.bout_list_dialog.set_current_frame(self._current_frame)

    # --- Video loading ---

    @staticmethod
    def sidecar_annotations_path(video_path: str | Path) -> Path:
        """Path for a sidecar annotations file next to the video.

        Inserts ``.annotations`` before the video extension and uses ``.json``:
        e.g. ``clip.avi`` → ``clip.annotations.json``.
        """
        return Path(video_path).with_suffix(".annotations.json")

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.mpg *.mpeg);;All Files (*)"
        )
        if not path:
            return

        try:
            meta = self.video.load(path)
            self._current_frame = 0
            self.controls.set_video_info(meta.total_frames, meta.fps)
            self._target_fps = meta.fps
            self.video_widget.clear()

            ann_path = self.sidecar_annotations_path(path)
            auto_loaded = False
            self._loaded_tracklets = None
            if ann_path.is_file():
                try:
                    self._apply_loaded_annotations(ann_path, warn_mismatch=True)
                    # Keep session video_path aligned with the video just opened
                    self.manager.session.video_path = path
                    auto_loaded = True
                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Annotations",
                        f"Found sidecar annotations but failed to load:\n{ann_path}\n\n{e}\n\n"
                        "Starting with a fresh annotation session instead.",
                    )
                    self._init_or_reset_manager(path, meta.fps, meta.total_frames)
            else:
                # No sidecar file — fresh session
                self._init_or_reset_manager(path, meta.fps, meta.total_frames)

            # Auto-discover id_review tracklets next to the video
            tracks_msg = ""
            try:
                auto_tracks = try_autoload_id_review(
                    path, video_total_frames=meta.total_frames
                )
                if auto_tracks is not None:
                    self._apply_tracklets(auto_tracks)
                    tracks_msg = f" | tracklets: {len(auto_tracks.subjects)} IDs"
            except Exception:
                pass

            # Show first frame (with overlays if tracklets loaded)
            self.controls.update_position(0)
            if hasattr(self, 'timeline'):
                self.timeline.set_current_frame(0)
                self.timeline.clear_selections()
                self.timeline.set_selection_mode(False)
                self._sync_selection_mode_ui(False)
            self._refresh_display_overlay()

            if auto_loaded:
                self.statusBar().showMessage(
                    f"Loaded: {Path(path).name}  |  {meta.total_frames} frames @ {meta.fps:.2f} fps  |  "
                    f"Auto-loaded annotations: {ann_path.name}{tracks_msg}"
                )
            else:
                self.statusBar().showMessage(
                    f"Loaded: {Path(path).name}  |  {meta.total_frames} frames @ {meta.fps:.2f} fps   |  "
                    f"Use hotkeys (1,2..) or double-click behaviors; [ ] cycle subjects{tracks_msg}"
                )
            self.setWindowTitle(f"LabGym Behavior Annotator — {Path(path).name}")
            self._update_undo_action()
            if self.bout_list_dialog is not None:
                self.bout_list_dialog.set_manager(self.manager)
                self.bout_list_dialog.set_current_frame(self._current_frame, scroll=True)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load video:\n{e}")

    # --- Annotation save/load ---

    def save_annotations(self):
        if not self.manager or not self.video.is_loaded:
            QMessageBox.information(self, "Save", "No video or annotations loaded.")
            return
        default = str(self.sidecar_annotations_path(self.manager.session.video_path))
        path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", default, "JSON (*.json)")
        if not path:
            return
        try:
            # Finalize any open bouts at the last viewed frame before saving
            self.manager.close_all_open_bouts(self._current_frame)
            self.manager.save_to_json(path)
            # Refresh UI in case indicators changed
            self._refresh_display_overlay()
            self.statusBar().showMessage(f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _apply_loaded_annotations(self, path: str | Path, *, warn_mismatch: bool = True) -> None:
        """Load annotations from JSON and rebuild the palette / UI bindings."""
        new_mgr = AnnotationManager.load_from_json(path)
        # Basic sanity: fps / frame count should roughly match the open video
        if self.video.is_loaded and warn_mismatch:
            if (
                abs(new_mgr.session.fps - self.video.fps) > 0.1
                or new_mgr.session.total_frames != self.video.total_frames
            ):
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Annotation file fps/total_frames differ from loaded video. Loading anyway.",
                )
        self.manager = new_mgr
        # Rebuild palette
        for i in reversed(range(self.palette_container.layout().count())):
            w = self.palette_container.layout().itemAt(i).widget()
            if w:
                w.setParent(None)
        self.palette = BehaviorPalette(self.manager, self.palette_container)
        self.palette_container.layout().addWidget(self.palette)
        self.palette.behavior_selected.connect(self._on_behavior_selected)
        self.palette.behavior_toggled.connect(self._toggle_bout_for_behavior)
        self.palette.behaviors_changed.connect(self._on_behaviors_changed)
        self.palette.sync_from_manager()
        if self.subject_panel:
            self.subject_panel.set_manager(self.manager)
        # Reload tracklets from tracks_ref if present
        tr = self.manager.session.tracks_ref
        if tr and tr.path:
            try:
                meta_parent = Path(tr.meta_path or tr.path).parent
                self._loaded_tracklets = load_tracklets_for_annotator(
                    meta_parent,
                    analysis_start_frame=tr.analysis_start_frame,
                    video_total_frames=(
                        self.video.total_frames if self.video.is_loaded else None
                    ),
                )
            except Exception:
                self._loaded_tracklets = None
        self._on_behaviors_changed()
        self._rebind_behavior_hotkeys()
        self._refresh_display_overlay()
        self._update_undo_action()
        if hasattr(self, "timeline"):
            self.timeline._adjust_height_for_subjects()

    def load_annotations(self):
        if not self.video.is_loaded or not self.manager:
            QMessageBox.information(self, "Load", "Load a video first.")
            return
        # Prefer the expected sidecar path as the dialog starting location
        start_dir = str(self.sidecar_annotations_path(self.video.metadata.path))
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Annotations", start_dir, "JSON (*.json)"
        )
        if not path:
            return
        try:
            self._apply_loaded_annotations(path, warn_mismatch=True)
            self.statusBar().showMessage(f"Loaded annotations: {path}")
            if self.bout_list_dialog is not None:
                self.bout_list_dialog.set_manager(self.manager)
                self.bout_list_dialog.set_current_frame(self._current_frame, scroll=True)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def show_metrics(self):
        if not self.manager or not self.video.is_loaded:
            QMessageBox.information(self, "Metrics", "Load a video and create some annotations first.")
            return
        from LabGym.annotator.core.metrics_calculator import MetricsCalculator
        from LabGym.annotator.ui.metrics_panel import MetricsDialog
        calc = MetricsCalculator(self.manager.session)
        dlg = MetricsDialog(calc, self)
        dlg.exec()

    def show_export_dialog(self):
        if not self.manager or not self.video.is_loaded:
            QMessageBox.information(self, "Export", "Load a video and annotate some behaviors first.")
            return
        ranges = self.timeline.get_selections() if hasattr(self, "timeline") else []
        dlg = ExportDialog(
            self.manager,
            self.video.metadata.path,
            self,
            selection_ranges=ranges,
        )
        dlg.exec()

    def _on_selection_mode_toggled(self, checked: bool):
        if not hasattr(self, "timeline"):
            return
        # Avoid feedback loops between action, checkbox, and timeline
        self.timeline.set_selection_mode(checked)
        self._sync_selection_mode_ui(checked)
        n = len(self.timeline.get_selections())
        if checked:
            self.statusBar().showMessage(
                f"Selection mode ON — drag on the timeline to add ranges ({n} selected). "
                "Then use Generate Examples → “Only export from selected timeline regions”."
            )
        else:
            self.statusBar().showMessage(
                f"Selection mode OFF — timeline seeks only ({n} range(s) kept)"
            )

    def _sync_selection_mode_ui(self, enabled: bool):
        """Keep menu action, checkbox, and enabled states aligned with the timeline."""
        for w in (getattr(self, "act_selection_mode", None), getattr(self, "chk_selection_mode", None)):
            if w is None:
                continue
            w.blockSignals(True)
            w.setChecked(enabled)
            w.blockSignals(False)

    def _on_timeline_selection_changed(self, ranges: list):
        n = len(ranges)
        if n == 0:
            self.lbl_selection_status.setText("")
        else:
            mode = "ON" if self.timeline.is_selection_mode() else "off"
            self.lbl_selection_status.setText(f"{n} range(s) selected · mode {mode}")
        if n > 0:
            self.statusBar().showMessage(
                f"{n} timeline range(s) selected — enable them in Generate Examples for LabGym"
            )

    def _clear_timeline_selections(self):
        if hasattr(self, "timeline"):
            self.timeline.clear_selections()
            self.statusBar().showMessage("Timeline selections cleared")

    def undo_last_action(self):
        if not self.manager:
            return
        if not self.manager.can_undo():
            QMessageBox.information(self, "Undo", "Nothing to undo.")
            return
        desc = self.manager.undo()
        # Refresh all dependent UI
        if self.palette:
            self.palette.sync_from_manager()
            self.palette.refresh()
        self._refresh_display_overlay()
        if hasattr(self, 'bout_list_dialog') and self.bout_list_dialog:
            self.bout_list_dialog.refresh()
            self.bout_list_dialog.set_current_frame(self._current_frame, scroll=True)
        self._update_undo_action()
        if hasattr(self, 'timeline'):
            self.timeline.update()
        self.statusBar().showMessage(desc)

    def show_bout_list(self):
        if not self.manager or not self.video.is_loaded:
            QMessageBox.information(self, "Bouts", "Load a video first.")
            return

        # If already open, bring it to the front and refresh (non-modal)
        if self._bout_list_window is not None and self._bout_list_window.isVisible():
            if self.bout_list_dialog:
                self.bout_list_dialog.set_manager(self.manager)
                self.bout_list_dialog.set_current_frame(self._current_frame, scroll=True)
            self._bout_list_window.raise_()
            self._bout_list_window.activateWindow()
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Bout List - Edit Type / Filter / Delete / Jump")
        dlg.resize(720, 420)
        dlg.setModal(False)
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        # Stay on top of the main window as a tool panel, but do not block it
        dlg.setWindowFlags(
            dlg.windowFlags()
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowCloseButtonHint
        )
        dlg.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        layout = QVBoxLayout(dlg)
        bout_widget = BoutListWidget(self.manager, dlg)
        bout_widget.jump_requested.connect(self.seek_to)
        bout_widget.bouts_changed.connect(self._refresh_after_bout_edit)
        bout_widget.set_current_frame(self._current_frame, scroll=True)
        layout.addWidget(bout_widget)

        self.bout_list_dialog = bout_widget
        self._bout_list_window = dlg
        dlg.finished.connect(self._on_bout_list_closed)
        dlg.show()

    def _on_bout_list_closed(self, _result: int = 0):
        self.bout_list_dialog = None
        self._bout_list_window = None

    def _refresh_after_bout_edit(self):
        """Called when bouts are deleted/edited from the bout list."""
        if self.palette:
            self.palette.refresh()
        self._refresh_display_overlay()
        self._update_undo_action()
        if hasattr(self, 'timeline'):
            self.timeline.update()
        # If the dialog is open, it will have refreshed itself via its own signal

    # --- Behavior Templates ---

    def save_behavior_template(self):
        if not self.manager:
            QMessageBox.information(self, "Save Template", "No session loaded.")
            return
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Behavior Template", "", "JSON Behavior Template (*.json)"
        )
        if not path:
            return
        try:
            self.manager.save_behavior_template(path)
            self.statusBar().showMessage(f"Saved behavior template: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def load_behavior_template(self):
        if not self.manager:
            QMessageBox.information(self, "Load Template", "Load a video first.")
            return
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Behavior Template", "", "JSON Behavior Template (*.json)"
        )
        if not path:
            return
        try:
            self.manager.load_behavior_template(path)
            if self.palette:
                self.palette.sync_from_manager()
                self.palette.refresh()
            self._on_behaviors_changed()
            self._refresh_display_overlay()
            self.statusBar().showMessage(f"Loaded behavior template: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _update_undo_action(self):
        if hasattr(self, 'act_undo') and self.act_undo:
            if self.manager and self.manager.can_undo():
                self.act_undo.setEnabled(True)
                # We don't know the top description without peeking, keep simple
            else:
                self.act_undo.setEnabled(False)


    def export_frame_labels_only(self):
        if not self.manager or not self.video.is_loaded:
            QMessageBox.information(self, "Export", "Load a video and annotate some behaviors first.")
            return
        from LabGym.annotator.core.example_generator import ExampleGenerator

        sid = self.manager.session.active_subject_id
        default = str(
            Path(self.video.metadata.path).with_name(f"frame_labels_subject{sid}.csv")
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Export frame_labels.csv (active subject)", default, "CSV (*.csv)"
        )
        if not path:
            return
        try:
            # Close any still-open bouts at the current frame so they are captured in labels
            self.manager.close_all_open_bouts(self._current_frame)
            gen = ExampleGenerator(
                self.manager.session,
                video_path=self.video.metadata.path,
                subject_id=sid,
            )
            open_starts = self.manager.get_open_starts()
            out = gen.export_frame_labels_csv(path, open_starts=open_starts)
            gen.close()
            self.statusBar().showMessage(f"Exported: {out}")
            QMessageBox.information(self, "Exported", f"frame_labels written to:\n{out}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def export_frame_labels_all_subjects(self):
        if not self.manager or not self.video.is_loaded:
            QMessageBox.information(self, "Export", "Load a video and annotate some behaviors first.")
            return
        from LabGym.annotator.core.example_generator import ExampleGenerator

        directory = QFileDialog.getExistingDirectory(
            self,
            "Folder for per-subject frame_labels CSVs",
            str(Path(self.video.metadata.path).parent),
        )
        if not directory:
            return
        try:
            self.manager.close_all_open_bouts(self._current_frame)
            gen = ExampleGenerator(
                self.manager.session, video_path=self.video.metadata.path
            )
            opens = {
                s.subject_id: self.manager.get_open_starts(subject_id=s.subject_id)
                for s in self.manager.session.subjects
            }
            paths = gen.export_frame_labels_all_subjects(
                directory, open_starts_by_subject=opens, combined=True
            )
            gen.close()
            self.statusBar().showMessage(f"Exported {len(paths)} label file(s)")
            QMessageBox.information(
                self,
                "Exported",
                "Wrote:\n" + "\n".join(str(p) for p in paths),
            )
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def export_labgym_tables(self):
        if not self.manager or not self.video.is_loaded:
            QMessageBox.information(self, "Export", "Load a video and annotate first.")
            return
        from LabGym.annotator.export_to_labgym import export_label_tables

        directory = QFileDialog.getExistingDirectory(
            self,
            "Folder for LabGym training tables",
            str(Path(self.video.metadata.path).parent),
        )
        if not directory:
            return
        try:
            self.manager.close_all_open_bouts(self._current_frame)
            paths = export_label_tables(
                self.manager.session,
                directory,
                video_path=self.video.metadata.path,
            )
            QMessageBox.information(
                self, "Exported", "Wrote:\n" + "\n".join(str(p) for p in paths)
            )
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    def export_soft_labels_for_examples(self):
        if not self.manager:
            QMessageBox.information(self, "Export", "No session loaded.")
            return
        from LabGym.annotator.export_to_labgym import export_soft_labels_for_examples

        directory = QFileDialog.getExistingDirectory(
            self,
            "Folder with LabGym examples (.avi/.jpg) to attach soft labels",
            str(Path(self.manager.session.video_path).parent)
            if self.manager.session.video_path
            else "",
        )
        if not directory:
            return
        length, ok = QInputDialog.getInt(
            self, "Window length", "time_step / animation length (frames):", 15, 1, 500
        )
        if not ok:
            return
        try:
            path = export_soft_labels_for_examples(
                self.manager.session, directory, window_len=int(length)
            )
            QMessageBox.information(self, "Exported", f"soft_labels.csv:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))

    # --- Playback ---

    def toggle_play(self):
        if not self.video.is_loaded:
            return

        self._is_playing = not self._is_playing
        self.controls.set_playing(self._is_playing)

        if self._is_playing:
            interval_ms = int(1000 / max(self._target_fps * self._playback_speed, 1))
            self._playback_timer.start(max(5, interval_ms))
        else:
            self._playback_timer.stop()

    def _on_playback_tick(self):
        if not self.video.is_loaded or not self._is_playing:
            return

        next_frame = self._current_frame + 1
        if next_frame >= self.video.total_frames:
            self._is_playing = False
            self.controls.set_playing(False)
            self._playback_timer.stop()
            return

        self._current_frame = next_frame
        self._display_current_frame()
        self.controls.update_position(self._current_frame)
        if hasattr(self, 'timeline'):
            self.timeline.set_current_frame(self._current_frame)

    def seek_to(self, frame: int):
        if not self.video.is_loaded:
            return

        frame = max(0, min(frame, self.video.total_frames - 1))
        was_playing = self._is_playing
        if was_playing:
            self._playback_timer.stop()

        self._current_frame = frame
        self._display_current_frame()
        self.controls.update_position(self._current_frame)
        if hasattr(self, 'timeline'):
            self.timeline.set_current_frame(self._current_frame)
        self._refresh_display_overlay()

        if was_playing:
            interval_ms = int(1000 / max(self._target_fps * self._playback_speed, 1))
            self._playback_timer.start(max(5, interval_ms))
            self._is_playing = True
            self.controls.set_playing(True)

    def step_frame(self, delta: int):
        if not self.video.is_loaded:
            return
        new_f = max(0, min(self._current_frame + delta, self.video.total_frames - 1))
        self.seek_to(new_f)

    def set_speed(self, speed: float):
        self._playback_speed = max(0.1, float(speed))
        if self._is_playing:
            # restart timer with new interval
            self._playback_timer.stop()
            interval_ms = int(1000 / max(self._target_fps * self._playback_speed, 1))
            self._playback_timer.start(max(5, interval_ms))

    def _display_current_frame(self):
        try:
            frame = self.video.get_frame(self._current_frame)
            open_names: list[str] = []
            annotated: list[str] = []
            active_sid = None
            if self.manager:
                open_names = self.manager.get_open_behaviors_at_frame(self._current_frame)
                annotated = self.manager.get_annotated_behaviors_at_frame(self._current_frame)
                active_sid = self.manager.session.active_subject_id
            if self.palette:
                self.palette.update_active_indicators(open_names, annotated)
            self.video_widget.show_frame(
                frame,
                open_behaviors=open_names,
                annotated_behaviors=annotated,
                track_overlays=self._current_track_overlays(),
                active_subject_id=active_sid,
            )
            self._sync_bout_list_frame()
        except Exception as e:
            self.statusBar().showMessage(f"Frame error: {e}")

    # --- Cleanup ---

    def closeEvent(self, event):
        self.video.release()
        super().closeEvent(event)


def main():
    # Allow running this module directly for quick testing
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
