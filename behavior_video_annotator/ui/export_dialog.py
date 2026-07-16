"""
ExportDialog: UI for generating curated behavior example clips.

Integrates with ExampleGenerator.
Runs export in a background QThread so the GUI stays responsive.

Optional: limit export to timeline-selected ranges (bout ∩ range), using the
same min clip length / min bout duration rules as full-bout mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QComboBox,
    QPushButton, QLineEdit, QFileDialog, QProgressBar, QMessageBox,
    QGroupBox, QFormLayout, QCheckBox,
)

from core.annotation_manager import AnnotationManager
from core.example_generator import ExampleGenerator


class ExportWorker(QObject):
    finished = pyqtSignal(list)          # list of written paths
    progress = pyqtSignal(int, int)      # done, total
    error = pyqtSignal(str)

    def __init__(
        self,
        generator: ExampleGenerator,
        output_dir: str,
        clip_length: int,
        mode: str,
        n_random: int,
        random_seed: int,
        open_starts: Optional[dict[str, int]] = None,
        min_bout_duration: int = 1,
        selection_ranges: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.generator = generator
        self.output_dir = output_dir
        self.clip_length = clip_length
        self.mode = mode
        self.n_random = n_random
        self.random_seed = random_seed
        self.open_starts = open_starts or {}
        self.min_bout_duration = min_bout_duration
        self.selection_ranges = list(selection_ranges) if selection_ranges else None

    def run(self):
        try:
            if self.selection_ranges:
                # Full-bout rules applied to bout ∩ selection intersections
                paths = self.generator.export_range_clips(
                    output_dir=self.output_dir,
                    ranges=self.selection_ranges,
                    min_bout_duration=self.min_bout_duration,
                    clip_length=self.clip_length,
                    progress_callback=self.progress.emit,
                )
            else:
                paths = self.generator.export_clips(
                    output_dir=self.output_dir,
                    clip_length=self.clip_length,
                    mode=self.mode,
                    n_random=self.n_random,
                    random_seed=self.random_seed,
                    progress_callback=self.progress.emit,
                    open_starts=self.open_starts,
                    min_bout_duration=self.min_bout_duration,
                    write_frame_labels=False,
                )
            self.finished.emit(paths)
        except Exception as e:
            self.error.emit(str(e))


class ExportDialog(QDialog):
    def __init__(
        self,
        manager: AnnotationManager,
        video_path: str,
        parent=None,
        selection_ranges: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Generate Examples for LabGym")
        self.resize(560, 460)

        self.manager = manager
        self.video_path = video_path
        self.selection_ranges: List[Tuple[int, int]] = list(selection_ranges or [])

        self._thread: Optional[QThread] = None
        self._worker: Optional[ExportWorker] = None

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # Output folder
        out_box = QGroupBox("Output")
        out_l = QHBoxLayout(out_box)
        self.out_edit = QLineEdit(str(Path(self.video_path).with_suffix("").parent / "labgym_examples"))
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_output)
        out_l.addWidget(QLabel("Folder:"))
        out_l.addWidget(self.out_edit, 1)
        out_l.addWidget(btn_browse)
        layout.addWidget(out_box)

        # Timeline selection filter
        sel_box = QGroupBox("Timeline selection")
        sel_l = QVBoxLayout(sel_box)
        n = len(self.selection_ranges)
        self.chk_use_selection = QCheckBox(
            f"Only export from selected timeline regions ({n} range(s))"
        )
        self.chk_use_selection.setToolTip(
            "When checked, only annotated bouts that overlap your timeline "
            "selection ranges are used. Intersections use the same min clip "
            "length / min bout duration rules as Full bout videos.\n\n"
            "Turn on “Select regions for export” on the main window and drag "
            "on the timeline to mark ranges first."
        )
        self.chk_use_selection.setEnabled(n > 0)
        if n == 0:
            self.chk_use_selection.setChecked(False)
            self.chk_use_selection.setText(
                "Only export from selected timeline regions (none selected)"
            )
        else:
            # Sensible default: if ranges exist, offer the option unchecked so user opts in
            self.chk_use_selection.setChecked(False)
        self.chk_use_selection.toggled.connect(self._update_mode_controls)
        sel_l.addWidget(self.chk_use_selection)

        self.sel_hint = QLabel(
            "Mark ranges on the timeline with “Select regions for export”, "
            "then re-open this dialog (or enable the checkbox above)."
            if n == 0
            else "Uses bout ∩ range intersections. Min clip length and min bout "
            "duration apply the same way as Full bout videos."
        )
        self.sel_hint.setWordWrap(True)
        self.sel_hint.setStyleSheet("color: #aaa;")
        sel_l.addWidget(self.sel_hint)
        layout.addWidget(sel_box)

        # Settings
        set_box = QGroupBox("Clip Settings")
        form = QFormLayout(set_box)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Full bout videos (one video per bout)", "full_bout")
        self.mode_combo.addItem("One centered clip per bout", "centered")
        self.mode_combo.addItem("N random clips per behavior", "random")
        self.mode_combo.addItem("All non-overlapping clips", "all")
        self.mode_combo.addItem(
            "Full bout coverage (centered short, strided+end for long)",
            "full_bout_coverage",
        )
        form.addRow("Sampling mode:", self.mode_combo)

        self.length_spin = QSpinBox()
        self.length_spin.setRange(1, 10000)
        self.length_spin.setValue(45)
        self.length_spin.setSuffix(" frames")
        self.length_spin.setToolTip(
            "Minimum clip length (full bout / selection modes). Short eligible "
            "bouts or intersections are padded to this length, centered."
        )
        self.length_label = QLabel("Clip length:")
        form.addRow(self.length_label, self.length_spin)

        self.min_bout_spin = QSpinBox()
        self.min_bout_spin.setRange(1, 10000)
        self.min_bout_spin.setValue(1)
        self.min_bout_spin.setSuffix(" frames")
        self.min_bout_spin.setToolTip(
            "Skip bouts (or bout∩range intersections) shorter than this."
        )
        self.min_bout_label = QLabel("Min bout duration:")
        form.addRow(self.min_bout_label, self.min_bout_spin)

        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 100)
        self.n_spin.setValue(3)
        form.addRow("N (for random mode):", self.n_spin)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(42)
        form.addRow("Random seed:", self.seed_spin)

        layout.addWidget(set_box)

        # Progress
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_generate = QPushButton("Generate Clips")
        self.btn_generate.clicked.connect(self.start_export)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        # Info label
        self.info = QLabel("")
        self.info.setWordWrap(True)
        layout.addWidget(self.info)

        self.mode_combo.currentIndexChanged.connect(self._update_mode_controls)
        self._update_mode_controls()

    def _use_selection(self) -> bool:
        return self.chk_use_selection.isChecked() and bool(self.selection_ranges)

    def _update_mode_controls(self):
        use_sel = self._use_selection()
        mode = self.mode_combo.currentData()
        is_full_bout = mode == "full_bout"
        is_random = mode == "random"

        # Selection path always uses full-bout min-length rules
        self.mode_combo.setEnabled(not use_sel)
        self.length_spin.setEnabled(True)
        self.min_bout_spin.setEnabled(use_sel or is_full_bout)
        self.min_bout_label.setEnabled(use_sel or is_full_bout)
        self.n_spin.setEnabled(is_random and not use_sel)
        self.seed_spin.setEnabled(is_random and not use_sel)

        if use_sel:
            self.length_label.setText("Min clip length:")
            n = len(self.selection_ranges)
            self.info.setText(
                f"<b>Timeline selection</b> ({n} range(s)): only bout∩range "
                "intersections are exported into behavior folders.\n"
                "• Intersection shorter than <b>min bout duration</b> → skipped\n"
                "• ≥ min bout duration but shorter than <b>min clip length</b> → "
                "padded clip of min clip length, centered on the intersection\n"
                "• ≥ min clip length → one video of the full intersection"
            )
        elif is_full_bout:
            self.length_label.setText("Min clip length:")
            self.info.setText(
                "Clips are written into subfolders named by behavior type.\n"
                "<b>Full bout videos</b>:\n"
                "• Bout shorter than <b>min bout duration</b> → skipped\n"
                "• Bout ≥ min bout duration but shorter than <b>min clip length</b> → "
                "one clip of min clip length, centered on the bout\n"
                "• Bout ≥ min clip length → one video of the full bout duration"
            )
        else:
            self.length_label.setText("Clip length:")
            self.info.setText(
                "Clips are written into subfolders named by behavior type."
            )

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", self.out_edit.text())
        if folder:
            self.out_edit.setText(folder)

    def start_export(self):
        out_dir = self.out_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "Output folder", "Please choose an output folder.")
            return

        clip_len = self.length_spin.value()
        min_bout = self.min_bout_spin.value()
        mode = self.mode_combo.currentData()
        n_rand = self.n_spin.value()
        seed = self.seed_spin.value()
        use_sel = self._use_selection()

        if use_sel and not self.selection_ranges:
            QMessageBox.warning(
                self,
                "No ranges",
                "No timeline ranges selected. Turn on selection mode and drag "
                "on the timeline, or uncheck the selection filter.",
            )
            return

        if (use_sel or mode == "full_bout") and min_bout > clip_len:
            reply = QMessageBox.question(
                self,
                "Thresholds",
                "Min bout duration is greater than min clip length.\n"
                "Short segments will only be skipped or exported full-length; "
                "no centered padding will occur.\n\nContinue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        if use_sel:
            # Preview so we can warn early about empty results
            gen_preview = ExampleGenerator(self.manager.session, video_path=self.video_path)
            preview = gen_preview.sample_clips_from_ranges(
                self.selection_ranges,
                min_bout_duration=min_bout,
                clip_length=clip_len,
            )
            if not preview:
                QMessageBox.information(
                    self,
                    "No clips",
                    "No annotated bouts intersect the selected ranges "
                    "(or all intersections were shorter than the min bout duration).",
                )
                return

        gen = ExampleGenerator(self.manager.session, video_path=self.video_path)

        self.btn_generate.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.progress.setMaximum(100)

        self._thread = QThread()
        open_starts = self.manager.get_open_starts() if hasattr(self.manager, "get_open_starts") else {}
        self._worker = ExportWorker(
            gen,
            out_dir,
            clip_len,
            mode,
            n_rand,
            seed,
            open_starts=open_starts,
            min_bout_duration=min_bout if (use_sel or mode == "full_bout") else 1,
            selection_ranges=self.selection_ranges if use_sel else None,
        )
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()

    def _on_progress(self, done: int, total: int):
        if total > 0:
            pct = int(done * 100 / total)
            self.progress.setMaximum(100)
            self.progress.setValue(pct)
        self.setWindowTitle(f"Generating... ({done}/{total})")

    def _on_finished(self, paths: list):
        self.progress.setValue(100)
        self.setWindowTitle("Generation complete")
        if self._use_selection():
            msg = (
                f"Exported {len(paths)} clip(s) from timeline selection "
                "(bout ∩ range), sorted into behavior folders."
            )
        elif self.mode_combo.currentData() == "full_bout":
            msg = f"Exported {len(paths)} full-bout video(s), sorted into behavior folders."
        else:
            msg = f"Exported {len(paths)} clip(s)."
        QMessageBox.information(self, "Done", msg)
        self.btn_generate.setEnabled(True)

    def _on_error(self, message: str):
        QMessageBox.critical(self, "Export Error", message)
        self.btn_generate.setEnabled(True)
        self.progress.setVisible(False)

    def _cleanup_thread(self):
        if self._thread:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            QMessageBox.information(self, "Export running", "Export is still running in background.")
        super().closeEvent(event)
