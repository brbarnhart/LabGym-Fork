"""Project settings: paths, modes, and generation defaults."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.project_state import ProjectController

MODE_LABELS = [
    (0, "0 — Non-interactive (per subject)"),
    (1, "1 — Interactive basic (group)"),
    (2, "2 — Interactive advanced (partners / costars)"),
]


class ProjectTab(QWidget):
    """Collect paths and defaults used by Annotate + Generate tabs."""

    apply_to_annotate = Signal()
    open_annotate = Signal()

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._block = False

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Configure this ethogram-first project once, then use the "
            "<b>Annotate</b> and <b>Generate</b> tabs. Settings are remembered "
            "between sessions."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # --- Paths ---
        paths = QGroupBox("Paths")
        form = QFormLayout(paths)

        self.ed_video = QLineEdit()
        btn_video = QPushButton("Browse…")
        btn_video.clicked.connect(self._browse_video)
        form.addRow("Video:", self._row(self.ed_video, btn_video))

        self.ed_tracklets = QLineEdit()
        btn_tr = QPushButton("Browse…")
        btn_tr.clicked.connect(self._browse_tracklets)
        form.addRow(
            "Tracklets (id_review):",
            self._row(self.ed_tracklets, btn_tr),
        )
        form.addRow(
            "",
            QLabel("Prefer the folder written after ID review (remapped IDs)."),
        )

        self.ed_ann = QLineEdit()
        self.ed_ann.setPlaceholderText("Default: video.annotations.json beside the video")
        btn_ann = QPushButton("Browse…")
        btn_ann.clicked.connect(self._browse_ann)
        form.addRow("Annotations JSON:", self._row(self.ed_ann, btn_ann))

        self.ed_out = QLineEdit()
        self.ed_out.setPlaceholderText("Default: <video_stem>_examples_from_ethogram")
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._browse_out)
        form.addRow("Examples output:", self._row(self.ed_out, btn_out))

        layout.addWidget(paths)

        # --- Ethogram ---
        eth = QGroupBox("Ethogram / annotation defaults")
        eth_form = QFormLayout(eth)

        self.combo_mode = QComboBox()
        for code, label in MODE_LABELS:
            self.combo_mode.addItem(label, code)
        eth_form.addRow("Behavior mode:", self.combo_mode)

        self.chk_exclusive = QCheckBox("Exclusive mode (new labels overwrite overlaps)")
        self.chk_exclusive.setChecked(True)
        eth_form.addRow(self.chk_exclusive)

        layout.addWidget(eth)

        # --- Generate defaults ---
        gen = QGroupBox("Example generation defaults (Stage C)")
        gen_form = QFormLayout(gen)

        self.spin_length = QSpinBox()
        self.spin_length.setRange(1, 500)
        self.spin_length.setValue(15)
        gen_form.addRow("Window length (frames):", self.spin_length)

        self.combo_sampling = QComboBox()
        for s in ("dense_in_bout", "bout_end", "bout_center", "coverage"):
            self.combo_sampling.addItem(s, s)
        gen_form.addRow("Sampling:", self.combo_sampling)

        self.spin_stride = QSpinBox()
        self.spin_stride.setRange(0, 500)
        self.spin_stride.setSpecialValueText("auto (length/3)")
        gen_form.addRow("Stride (dense):", self.spin_stride)

        self.spin_min_bout = QSpinBox()
        self.spin_min_bout.setRange(1, 500)
        self.spin_min_bout.setValue(1)
        gen_form.addRow("Min bout frames:", self.spin_min_bout)

        self.spin_social = QDoubleSpinBox()
        self.spin_social.setRange(0.0, 100.0)
        self.spin_social.setToolTip("Mode 2 only; 0 = include all other IDs")
        gen_form.addRow("Social distance (mode 2):", self.spin_social)

        self.chk_soft = QCheckBox("Write soft_labels.csv")
        self.chk_soft.setChecked(True)
        gen_form.addRow(self.chk_soft)

        self.chk_bg = QCheckBox("Background-free blobs")
        self.chk_bg.setChecked(True)
        gen_form.addRow(self.chk_bg)

        layout.addWidget(gen)

        notes_box = QGroupBox("Notes")
        notes_l = QVBoxLayout(notes_box)
        self.ed_notes = QTextEdit()
        self.ed_notes.setMaximumHeight(80)
        notes_l.addWidget(self.ed_notes)
        layout.addWidget(notes_box)

        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("Save project settings")
        self.btn_save.clicked.connect(self._commit_and_save)
        self.btn_apply = QPushButton("Apply to Annotate tab")
        self.btn_apply.setToolTip(
            "Push video / tracklets / annotations / mode into the Annotate tab"
        )
        self.btn_apply.clicked.connect(self._commit_and_apply)
        self.btn_annotate = QPushButton("Go to Annotate…")
        self.btn_annotate.clicked.connect(self._commit_and_open_annotate)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_apply)
        btn_row.addWidget(self.btn_annotate)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        layout.addStretch(1)

        self._wire_auto_commit()
        self.project.changed.connect(self._load_from_project)
        self._load_from_project()

    @staticmethod
    def _row(edit: QLineEdit, btn: QPushButton) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        h.addWidget(btn)
        return w

    def _wire_auto_commit(self) -> None:
        for w in (
            self.ed_video,
            self.ed_tracklets,
            self.ed_ann,
            self.ed_out,
        ):
            w.editingFinished.connect(self._commit_from_ui)
        self.combo_mode.currentIndexChanged.connect(self._commit_from_ui)
        self.chk_exclusive.toggled.connect(self._commit_from_ui)
        self.spin_length.valueChanged.connect(self._commit_from_ui)
        self.combo_sampling.currentIndexChanged.connect(self._commit_from_ui)
        self.spin_stride.valueChanged.connect(self._commit_from_ui)
        self.spin_min_bout.valueChanged.connect(self._commit_from_ui)
        self.spin_social.valueChanged.connect(self._commit_from_ui)
        self.chk_soft.toggled.connect(self._commit_from_ui)
        self.chk_bg.toggled.connect(self._commit_from_ui)
        self.ed_notes.textChanged.connect(self._commit_from_ui)

    def _load_from_project(self) -> None:
        if self._block:
            return
        self._block = True
        s = self.project.state
        self.ed_video.setText(s.video_path)
        self.ed_tracklets.setText(s.tracklets_dir)
        self.ed_ann.setText(s.annotations_path)
        self.ed_out.setText(s.examples_out_dir)
        idx = self.combo_mode.findData(int(s.behavior_mode))
        self.combo_mode.setCurrentIndex(max(0, idx))
        self.chk_exclusive.setChecked(bool(s.exclusive_mode))
        self.spin_length.setValue(int(s.window_length))
        sidx = self.combo_sampling.findData(s.sampling)
        self.combo_sampling.setCurrentIndex(max(0, sidx))
        self.spin_stride.setValue(int(s.stride))
        self.spin_min_bout.setValue(int(s.min_bout_frames))
        self.spin_social.setValue(float(s.social_distance))
        self.chk_soft.setChecked(bool(s.write_soft_labels))
        self.chk_bg.setChecked(bool(s.background_free))
        self.ed_notes.setPlainText(s.notes or "")
        self._block = False

    def _commit_from_ui(self) -> None:
        if self._block:
            return
        self.project.update(
            video_path=self.ed_video.text().strip(),
            tracklets_dir=self.ed_tracklets.text().strip(),
            annotations_path=self.ed_ann.text().strip(),
            examples_out_dir=self.ed_out.text().strip(),
            behavior_mode=int(self.combo_mode.currentData()),
            exclusive_mode=self.chk_exclusive.isChecked(),
            window_length=int(self.spin_length.value()),
            sampling=str(self.combo_sampling.currentData()),
            stride=int(self.spin_stride.value()),
            min_bout_frames=int(self.spin_min_bout.value()),
            social_distance=float(self.spin_social.value()),
            write_soft_labels=self.chk_soft.isChecked(),
            background_free=self.chk_bg.isChecked(),
            notes=self.ed_notes.toPlainText(),
        )

    def _commit_and_save(self) -> None:
        self._commit_from_ui()
        self.project.save_settings()

    def _commit_and_apply(self) -> None:
        self._commit_from_ui()
        self.project.save_settings()
        self.apply_to_annotate.emit()

    def _commit_and_open_annotate(self) -> None:
        self._commit_from_ui()
        self.project.save_settings()
        self.open_annotate.emit()

    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            self.ed_video.text() or "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.mpg *.mpeg);;All Files (*)",
        )
        if path:
            self.ed_video.setText(path)
            # Suggest defaults if empty
            if not self.ed_ann.text().strip():
                self.ed_ann.setPlaceholderText(
                    str(Path(path).with_suffix(".annotations.json"))
                )
            if not self.ed_out.text().strip():
                stem = Path(path).stem
                self.ed_out.setPlaceholderText(
                    str(Path(path).parent / f"{stem}_examples_from_ethogram")
                )
            self._commit_from_ui()

    def _browse_tracklets(self) -> None:
        start = self.ed_tracklets.text() or (
            str(Path(self.ed_video.text()).parent) if self.ed_video.text() else ""
        )
        d = QFileDialog.getExistingDirectory(self, "Tracklets / id_review folder", start)
        if d:
            self.ed_tracklets.setText(d)
            self._commit_from_ui()

    def _browse_ann(self) -> None:
        start = self.ed_ann.text() or self.project.state.inferred_annotations_path()
        path, _ = QFileDialog.getOpenFileName(
            self, "Annotations JSON", start, "JSON (*.json);;All Files (*)"
        )
        if path:
            self.ed_ann.setText(path)
            self._commit_from_ui()

    def _browse_out(self) -> None:
        start = self.ed_out.text() or self.project.state.inferred_examples_dir()
        d = QFileDialog.getExistingDirectory(self, "Examples output folder", start)
        if d:
            self.ed_out.setText(d)
            self._commit_from_ui()
