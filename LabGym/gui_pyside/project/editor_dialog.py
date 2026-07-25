"""Dialog to edit project root, video list, paths, and defaults."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.project.model import Project, ProjectDefaults, ProjectPaths, ProjectVideo

MODE_LABELS = [
    (0, "0 — Non-interactive"),
    (1, "1 — Interactive basic"),
    (2, "2 — Interactive advanced"),
]


class ProjectEditorDialog(QDialog):
    """Edit the open project in place; caller saves to disk via controller."""

    def __init__(self, controller: ProjectController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Edit Project")
        self.resize(640, 560)

        layout = QVBoxLayout(self)

        # Identity
        id_box = QGroupBox("Project")
        id_form = QFormLayout(id_box)
        self.ed_name = QLineEdit()
        id_form.addRow("Name:", self.ed_name)
        self.ed_root = QLineEdit()
        btn_root = QPushButton("Browse…")
        btn_root.clicked.connect(self._browse_root)
        id_form.addRow("Root folder:", self._row(self.ed_root, btn_root))
        layout.addWidget(id_box)

        # Videos
        vid_box = QGroupBox("Videos (explicit list)")
        vid_l = QVBoxLayout(vid_box)
        self.list_videos = QListWidget()
        vid_l.addWidget(self.list_videos)
        row = QHBoxLayout()
        btn_add = QPushButton("Add video(s)…")
        btn_add.clicked.connect(self._add_videos)
        btn_add_folder = QPushButton("Add folder…")
        btn_add_folder.clicked.connect(self._add_folder)
        btn_rm = QPushButton("Remove selected")
        btn_rm.clicked.connect(self._remove_selected)
        btn_toggle = QPushButton("Toggle enabled")
        btn_toggle.clicked.connect(self._toggle_enabled)
        btn_current = QPushButton("Set as current")
        btn_current.setToolTip("Use this video in Annotate / Generate tabs")
        btn_current.clicked.connect(self._set_current_video)
        row.addWidget(btn_add)
        row.addWidget(btn_add_folder)
        row.addWidget(btn_rm)
        row.addWidget(btn_toggle)
        row.addWidget(btn_current)
        row.addStretch(1)
        vid_l.addLayout(row)
        layout.addWidget(vid_box, 1)

        # Paths
        path_box = QGroupBox("Default relative paths (under root when relative)")
        path_form = QFormLayout(path_box)
        self.ed_detection = QLineEdit()
        self.ed_examples = QLineEdit()
        self.ed_models = QLineEdit()
        self.ed_processed = QLineEdit()
        self.ed_ann_root = QLineEdit()
        path_form.addRow("Detection output:", self.ed_detection)
        path_form.addRow("Examples:", self.ed_examples)
        path_form.addRow("Models:", self.ed_models)
        path_form.addRow("Processed:", self.ed_processed)
        path_form.addRow("Annotations root:", self.ed_ann_root)
        layout.addWidget(path_box)

        # Defaults
        def_box = QGroupBox("Defaults")
        def_form = QFormLayout(def_box)
        self.combo_mode = QComboBox()
        for code, label in MODE_LABELS:
            self.combo_mode.addItem(label, code)
        def_form.addRow("Behavior mode:", self.combo_mode)
        self.chk_exclusive = QCheckBox("Exclusive annotation mode")
        def_form.addRow(self.chk_exclusive)
        self.spin_length = QSpinBox()
        self.spin_length.setRange(1, 500)
        def_form.addRow("Window length:", self.spin_length)
        self.combo_sampling = QComboBox()
        for s in ("dense_in_bout", "bout_end", "bout_center", "coverage"):
            self.combo_sampling.addItem(s, s)
        def_form.addRow("Sampling:", self.combo_sampling)
        layout.addWidget(def_box)

        notes_box = QGroupBox("Notes")
        notes_l = QVBoxLayout(notes_box)
        self.ed_notes = QTextEdit()
        self.ed_notes.setMaximumHeight(72)
        notes_l.addWidget(self.ed_notes)
        layout.addWidget(notes_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._pending_current: str | None = None
        self._load_from_project()

    @staticmethod
    def _row(edit: QLineEdit, btn: QPushButton):
        from PySide6.QtWidgets import QWidget

        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        h.addWidget(btn)
        return w

    def _load_from_project(self) -> None:
        p = self.controller.project
        self.ed_name.setText(p.name)
        self.ed_root.setText(p.root_dir)
        self.list_videos.clear()
        for v in p.videos:
            self._add_list_item(v)
        self.ed_detection.setText(p.paths.detection_output_root)
        self.ed_examples.setText(p.paths.examples_root)
        self.ed_models.setText(p.paths.models_root)
        self.ed_processed.setText(p.paths.processed_root)
        self.ed_ann_root.setText(p.paths.annotations_root)
        idx = self.combo_mode.findData(int(p.defaults.behavior_mode))
        self.combo_mode.setCurrentIndex(max(0, idx))
        self.chk_exclusive.setChecked(bool(p.defaults.exclusive_mode))
        self.spin_length.setValue(int(p.defaults.window_length))
        sidx = self.combo_sampling.findData(p.defaults.sampling)
        self.combo_sampling.setCurrentIndex(max(0, sidx))
        self.ed_notes.setPlainText(p.notes or "")

    def _add_list_item(self, v: ProjectVideo) -> None:
        flag = "✓" if v.enabled else "·"
        item = QListWidgetItem(f"{flag}  {v.path}")
        item.setData(Qt.ItemDataRole.UserRole, v)
        if not v.enabled:
            item.setForeground(Qt.GlobalColor.gray)
        self.list_videos.addItem(item)

    def _browse_root(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Project root folder", self.ed_root.text() or ""
        )
        if d:
            self.ed_root.setText(d)

    def _add_videos(self) -> None:
        start = self.ed_root.text() or ""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add videos",
            start,
            "Video Files (*.mp4 *.avi *.mov *.mkv *.mpg *.mpeg);;All Files (*)",
        )
        root = self.ed_root.text().strip()
        for path in paths:
            rel = path
            if root:
                try:
                    rel = str(Path(path).resolve().relative_to(Path(root).resolve()))
                except ValueError:
                    rel = path
            # skip dups
            existing = {
                self.list_videos.item(i).data(Qt.ItemDataRole.UserRole).path
                for i in range(self.list_videos.count())
            }
            if rel in existing or path in existing:
                continue
            self._add_list_item(ProjectVideo(path=rel, enabled=True))

    def _add_folder(self) -> None:
        start = self.ed_root.text() or ""
        d = QFileDialog.getExistingDirectory(self, "Folder of videos", start)
        if not d:
            return
        exts = {".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg"}
        root = self.ed_root.text().strip()
        existing = {
            self.list_videos.item(i).data(Qt.ItemDataRole.UserRole).path
            for i in range(self.list_videos.count())
        }
        for p in sorted(Path(d).iterdir()):
            if p.suffix.lower() not in exts:
                continue
            path = str(p)
            rel = path
            if root:
                try:
                    rel = str(p.resolve().relative_to(Path(root).resolve()))
                except ValueError:
                    rel = path
            if rel in existing:
                continue
            self._add_list_item(ProjectVideo(path=rel, enabled=True))
            existing.add(rel)

    def _remove_selected(self) -> None:
        for item in self.list_videos.selectedItems():
            row = self.list_videos.row(item)
            self.list_videos.takeItem(row)

    def _toggle_enabled(self) -> None:
        for item in self.list_videos.selectedItems():
            v: ProjectVideo = item.data(Qt.ItemDataRole.UserRole)
            v.enabled = not v.enabled
            flag = "✓" if v.enabled else "·"
            item.setText(f"{flag}  {v.path}")
            item.setData(Qt.ItemDataRole.UserRole, v)
            item.setForeground(
                Qt.GlobalColor.black if v.enabled else Qt.GlobalColor.gray
            )

    def _set_current_video(self) -> None:
        items = self.list_videos.selectedItems()
        if not items:
            QMessageBox.information(self, "Current video", "Select a video first.")
            return
        v: ProjectVideo = items[0].data(Qt.ItemDataRole.UserRole)
        # Store relative path as project current; resolve on accept via defaults
        self._pending_current = v.path
        QMessageBox.information(
            self,
            "Current video",
            f"Will set current video to:\n{v.path}\n\n(Applied when you click OK.)",
        )

    def _accept(self) -> None:
        name = self.ed_name.text().strip() or "Untitled"
        root = self.ed_root.text().strip()
        videos: list[ProjectVideo] = []
        for i in range(self.list_videos.count()):
            v = self.list_videos.item(i).data(Qt.ItemDataRole.UserRole)
            if isinstance(v, ProjectVideo) and v.path:
                videos.append(v)
        paths = ProjectPaths(
            detection_output_root=self.ed_detection.text().strip() or "detection",
            examples_root=self.ed_examples.text().strip() or "examples",
            models_root=self.ed_models.text().strip() or "models",
            processed_root=self.ed_processed.text().strip() or "processed",
            annotations_root=self.ed_ann_root.text().strip(),
        )
        old = self.controller.project.defaults
        current = old.current_video
        if getattr(self, "_pending_current", None):
            current = self._pending_current
        defaults = ProjectDefaults(
            behavior_mode=int(self.combo_mode.currentData()),
            exclusive_mode=self.chk_exclusive.isChecked(),
            window_length=int(self.spin_length.value()),
            sampling=str(self.combo_sampling.currentData()),
            stride=old.stride,
            min_bout_frames=old.min_bout_frames,
            social_distance=old.social_distance,
            write_soft_labels=old.write_soft_labels,
            background_free=old.background_free,
            detector_name=old.detector_name,
            categorizer_name=old.categorizer_name,
            current_video=current,
        )
        proj = Project(
            name=name,
            root_dir=root,
            videos=videos,
            paths=paths,
            defaults=defaults,
            notes=self.ed_notes.toPlainText(),
            file_path=self.controller.project.file_path,
        )
        self.controller.replace(proj, dirty=True)
        self.accept()
