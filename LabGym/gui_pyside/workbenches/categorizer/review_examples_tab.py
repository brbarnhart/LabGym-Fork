"""Manage dataset → Review examples: keep / exclude / recategorize."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.widgets.path_browse import path_edit_row, set_line_edit_directory


class ReviewExamplesTab(QWidget):
    """Review queue from evaluation runs; decisions write the dataset manifest."""

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._queue: list = []
        self._manifest = None  # DatasetManifest | None
        self._session_done: set = set()  # example_ids reviewed this session

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Review high-loss and misclassified examples from evaluation runs. "
            "<b>Keep</b>, <b>Exclude</b>, and <b>Recategorize</b> apply immediately "
            "to the example store's <code>dataset_manifest.json</code> (with undo). "
            "Training uses this effective view by default."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(intro)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.ed_store = QLineEdit()
        self.ed_store.setToolTip(
            "Example store root (flat prepared folder or behavior subfolders). "
            "Manifest is stored here as dataset_manifest.json."
        )
        b_store = QPushButton("Browse…")
        b_store.clicked.connect(lambda: set_line_edit_directory(self, self.ed_store))
        form.addRow(
            self._lab("Example store:", "Where curation state is saved."),
            path_edit_row(self.ed_store, b_store),
        )

        self.ed_model = QLineEdit()
        self.ed_model.setToolTip(
            "Trained categorizer folder whose eval/ runs feed the review queue."
        )
        b_model = QPushButton("Browse…")
        b_model.clicked.connect(lambda: set_line_edit_directory(self, self.ed_model))
        form.addRow(
            self._lab("Categorizer folder:", "Used to list evaluation runs."),
            path_edit_row(self.ed_model, b_model),
        )
        layout.addLayout(form)

        mid = QHBoxLayout()
        runs_box = QGroupBox("Evaluation runs")
        runs_l = QVBoxLayout(runs_box)
        self.run_list = QListWidget()
        self.run_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.run_list.setMaximumHeight(120)
        runs_l.addWidget(self.run_list)
        run_btns = QHBoxLayout()
        btn_refresh_runs = QPushButton("Refresh runs")
        btn_refresh_runs.clicked.connect(self._refresh_runs)
        run_btns.addWidget(btn_refresh_runs)
        run_btns.addStretch(1)
        runs_l.addLayout(run_btns)
        mid.addWidget(runs_box, 2)

        src_box = QGroupBox("Queue sources")
        src_l = QVBoxLayout(src_box)
        self.chk_mis = QCheckBox("Misclassified")
        self.chk_mis.setChecked(True)
        self.chk_hl = QCheckBox("High-loss (train partition)")
        self.chk_hl.setChecked(True)
        self.chk_hide_done = QCheckBox("Hide reviewed this session")
        self.chk_hide_done.setChecked(True)
        src_l.addWidget(self.chk_mis)
        src_l.addWidget(self.chk_hl)
        src_l.addWidget(self.chk_hide_done)
        self.btn_build = QPushButton("Build review queue")
        self.btn_build.clicked.connect(self._build_queue)
        src_l.addWidget(self.btn_build)
        src_l.addStretch(1)
        mid.addWidget(src_box, 1)
        layout.addLayout(mid)

        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        self.queue_status = QLabel("Queue empty.")
        left_l.addWidget(self.queue_status)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["example_id", "sources", "true", "pred", "conf/loss", "status"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        left_l.addWidget(self.table, 1)
        split.addWidget(left)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)

        self.detail = QLabel("Select an example.")
        self.detail.setWordWrap(True)
        right_l.addWidget(self.detail)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(180)
        self.preview.setStyleSheet(
            "QLabel { background: #1e1e1e; color: #aaa; border: 1px solid #444; }"
        )
        self.preview.setText("No preview")
        right_l.addWidget(self.preview, 1)

        self.manifest_status = QLabel("Manifest: (not loaded)")
        self.manifest_status.setWordWrap(True)
        right_l.addWidget(self.manifest_status)

        act = QGroupBox("Review decision (writes manifest immediately)")
        act_l = QVBoxLayout(act)
        row1 = QHBoxLayout()
        self.btn_keep = QPushButton("Keep")
        self.btn_keep.setToolTip(
            "Leave in the effective training set with the current active label."
        )
        self.btn_keep.clicked.connect(self._do_keep)
        self.btn_exclude = QPushButton("Exclude")
        self.btn_exclude.setToolTip(
            "Remove from the effective training set without deleting the file."
        )
        self.btn_exclude.clicked.connect(self._do_exclude)
        row1.addWidget(self.btn_keep)
        row1.addWidget(self.btn_exclude)
        act_l.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Recategorize to:"))
        self.cmb_label = QComboBox()
        self.cmb_label.setEditable(True)
        self.cmb_label.setMinimumWidth(140)
        row2.addWidget(self.cmb_label, 1)
        self.btn_recat = QPushButton("Recategorize")
        self.btn_recat.clicked.connect(self._do_recategorize)
        row2.addWidget(self.btn_recat)
        act_l.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_undo = QPushButton("Undo last manifest change")
        self.btn_undo.clicked.connect(self._do_undo)
        self.btn_save = QPushButton("Save manifest")
        self.btn_save.setToolTip("Persist dataset_manifest.json (also auto-saved after actions).")
        self.btn_save.clicked.connect(self._save_manifest)
        row3.addWidget(self.btn_undo)
        row3.addWidget(self.btn_save)
        act_l.addLayout(row3)
        right_l.addWidget(act)

        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, 1)

        self.ed_model.editingFinished.connect(self._refresh_runs)
        self.ed_store.editingFinished.connect(self._on_store_changed)
        self.project.project_replaced.connect(self._apply_project_defaults)
        self._apply_project_defaults()

    @staticmethod
    def _lab(text: str, tip: str) -> QLabel:
        lab = QLabel(text)
        lab.setToolTip(tip)
        return lab

    def _apply_project_defaults(self) -> None:
        p = self.project.project
        name = (p.defaults.categorizer_name or "").strip()
        if name:
            self.ed_model.setText(name)
        try:
            ex = p.resolve_path(p.paths.examples_root or "examples")
            if Path(ex).is_dir():
                self.ed_store.setText(str(ex))
        except Exception:
            pass
        self._refresh_runs()
        self._on_store_changed()

    def _on_store_changed(self) -> None:
        store = self.ed_store.text().strip()
        if not store:
            self._manifest = None
            self.manifest_status.setText("Manifest: (no store)")
            return
        try:
            from LabGym.training.dataset_manifest import DatasetManifest
            from LabGym.training.review_queue import available_categories

            self._manifest = DatasetManifest.load_or_create(store)
            # Sync scan so review can act on all on-disk examples
            try:
                from LabGym.training.dataset_manifest import scan_example_store

                scanned = [
                    (eid, lab, hint)
                    for eid, lab, hint, _p in scan_example_store(store)
                ]
                self._manifest.sync_from_scan(scanned)
            except Exception:
                pass
            n = len(self._manifest.examples)
            path = self._manifest.path
            exists = path.is_file()
            self.manifest_status.setText(
                f"Manifest: {n} examples; "
                f"{'on disk' if exists else 'not yet saved'} → {path}"
            )
            cats = available_categories(store, manifest=self._manifest)
            self.cmb_label.clear()
            self.cmb_label.addItems(cats)
        except Exception as exc:
            self._manifest = None
            self.manifest_status.setText(f"Manifest error: {exc}")

    def _refresh_runs(self) -> None:
        self.run_list.clear()
        model = self.ed_model.text().strip()
        if not model:
            return
        try:
            from LabGym.training.evaluation import list_evaluation_runs

            runs = list_evaluation_runs(model)
        except Exception:
            return
        for info in runs:
            item = QListWidgetItem(info.display_label())
            item.setData(Qt.ItemDataRole.UserRole, info.run_dir)
            item.setToolTip(info.run_dir)
            self.run_list.addItem(item)
        # Select first run by default if none selected
        if self.run_list.count() and not self.run_list.selectedItems():
            self.run_list.item(0).setSelected(True)

    def _selected_run_dirs(self) -> List[str]:
        return [
            str(i.data(Qt.ItemDataRole.UserRole))
            for i in self.run_list.selectedItems()
            if i.data(Qt.ItemDataRole.UserRole)
        ]

    def _build_queue(self) -> None:
        store = self.ed_store.text().strip()
        runs = self._selected_run_dirs()
        if not runs:
            QMessageBox.warning(
                self, "Review", "Select at least one evaluation run (or refresh runs)."
            )
            return
        if not store or not Path(store).is_dir():
            QMessageBox.warning(self, "Review", "Set a valid example store folder.")
            return
        self._on_store_changed()
        try:
            from LabGym.training.review_queue import (
                build_review_queue,
                ensure_queue_in_manifest,
            )

            queue = build_review_queue(
                runs,
                include_misclassified=self.chk_mis.isChecked(),
                include_high_loss=self.chk_hl.isChecked(),
                dedupe=True,
                store_root=store,
            )
            if self._manifest is not None:
                ensure_queue_in_manifest(self._manifest, queue)
                self._manifest.save()
            self._queue = queue
            self._fill_table()
            self.queue_status.setText(
                f"Queue: {len(self._visible_indices())} shown "
                f"({len(queue)} total from {len(runs)} run(s))."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Review", str(exc))

    def _visible_indices(self) -> List[int]:
        hide = self.chk_hide_done.isChecked()
        out = []
        for i, it in enumerate(self._queue):
            if hide and it.example_id in self._session_done:
                continue
            out.append(i)
        return out

    def _fill_table(self) -> None:
        self.table.setRowCount(0)
        for qi in self._visible_indices():
            it = self._queue[qi]
            r = self.table.rowCount()
            self.table.insertRow(r)
            status = self._status_for(it.example_id)
            conf_loss = ""
            if it.loss is not None:
                conf_loss = f"L={it.loss:.4f}"
            elif it.confidence is not None:
                conf_loss = f"C={it.confidence:.3f}"
            vals = [
                it.example_id,
                "+".join(it.sources),
                it.true_label,
                it.pred_label,
                conf_loss,
                status,
            ]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                item.setData(Qt.ItemDataRole.UserRole, qi)
                self.table.setItem(r, c, item)
        if self.table.rowCount():
            self.table.selectRow(0)

    def _status_for(self, example_id: str) -> str:
        parts = []
        if example_id in self._session_done:
            parts.append("reviewed")
        if self._manifest is not None:
            rec = self._manifest.examples.get(example_id)
            if rec is not None:
                if rec.excluded:
                    parts.append("excluded")
                if rec.label_override:
                    parts.append(f"→{rec.label_override}")
        return ", ".join(parts) if parts else "pending"

    def _current_item(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        if not rows:
            return None
        r = rows[0].row()
        cell = self.table.item(r, 0)
        if cell is None:
            return None
        qi = cell.data(Qt.ItemDataRole.UserRole)
        if qi is None or qi < 0 or qi >= len(self._queue):
            return None
        return self._queue[qi]

    def _on_row_selected(self) -> None:
        it = self._current_item()
        if it is None:
            self.detail.setText("Select an example.")
            self.preview.setText("No preview")
            self.preview.setPixmap(QPixmap())
            return
        active = ""
        if self._manifest is not None:
            rec = self._manifest.examples.get(it.example_id)
            if rec is not None:
                active = (
                    f" active={rec.active_label} original={rec.original_label} "
                    f"excluded={rec.excluded} split={rec.split}"
                )
        self.detail.setText(
            f"<b>{it.example_id}</b><br/>"
            f"sources={'+'.join(it.sources)} run={it.run_id}<br/>"
            f"true={it.true_label} pred={it.pred_label} "
            f"conf={it.confidence} loss={it.loss}<br/>"
            f"media={it.media_path or '(not found)'}{active}"
        )
        self._show_preview(it.media_path)
        # Prefer true label as recategorize default suggestion: pred often wrong
        if it.pred_label:
            idx = self.cmb_label.findText(it.pred_label)
            if idx >= 0:
                self.cmb_label.setCurrentIndex(idx)
            else:
                self.cmb_label.setEditText(it.pred_label)

    def _show_preview(self, media_path: Optional[str]) -> None:
        self.preview.setPixmap(QPixmap())
        if not media_path:
            self.preview.setText("No media path")
            return
        p = Path(media_path)
        # Prefer sibling jpg for avi
        candidates = [p]
        if p.suffix.lower() == ".avi":
            candidates.insert(0, p.with_suffix(".jpg"))
        elif p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            candidates.insert(0, p.with_suffix(".jpg"))
        for c in candidates:
            if c.is_file() and c.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                pix = QPixmap(str(c))
                if not pix.isNull():
                    scaled = pix.scaled(
                        self.preview.size(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    self.preview.setPixmap(scaled)
                    return
        self.preview.setText(f"No image preview\n{p.name}")

    def _require_manifest_and_item(self):
        it = self._current_item()
        if it is None:
            QMessageBox.information(self, "Review", "Select an example in the queue.")
            return None, None
        if self._manifest is None:
            self._on_store_changed()
        if self._manifest is None:
            QMessageBox.warning(self, "Review", "Could not load/create dataset manifest.")
            return None, None
        # Ensure row exists
        from LabGym.training.review_queue import ensure_queue_in_manifest

        ensure_queue_in_manifest(self._manifest, [it])
        return self._manifest, it

    def _after_action(self, example_id: str) -> None:
        self._session_done.add(example_id)
        try:
            self._manifest.save()
        except Exception as exc:
            QMessageBox.warning(self, "Review", f"Saved decision in memory but disk write failed:\n{exc}")
        self.manifest_status.setText(
            f"Manifest saved → {self._manifest.path} "
            f"({len(self._manifest.examples)} examples, "
            f"undo depth={len(self._manifest.undo_stack)})"
        )
        # Advance selection
        row = self.table.currentRow()
        self._fill_table()
        if self.chk_hide_done.isChecked():
            if self.table.rowCount():
                self.table.selectRow(min(row, self.table.rowCount() - 1))
        else:
            if 0 <= row < self.table.rowCount():
                self.table.selectRow(row)

    def _do_keep(self) -> None:
        m, it = self._require_manifest_and_item()
        if m is None:
            return
        m.keep(it.example_id)
        self._after_action(it.example_id)

    def _do_exclude(self) -> None:
        m, it = self._require_manifest_and_item()
        if m is None:
            return
        m.exclude(it.example_id, excluded=True)
        self._after_action(it.example_id)

    def _do_recategorize(self) -> None:
        m, it = self._require_manifest_and_item()
        if m is None:
            return
        new_lab = self.cmb_label.currentText().strip()
        if not new_lab:
            QMessageBox.warning(self, "Recategorize", "Choose or type a behavior category.")
            return
        m.recategorize(it.example_id, new_lab)
        # Ensure category appears in combo
        if self.cmb_label.findText(new_lab) < 0:
            self.cmb_label.addItem(new_lab)
        self._after_action(it.example_id)

    def _do_undo(self) -> None:
        if self._manifest is None:
            self._on_store_changed()
        if self._manifest is None:
            return
        if not self._manifest.undo():
            QMessageBox.information(self, "Undo", "Nothing to undo.")
            return
        try:
            self._manifest.save()
        except Exception as exc:
            QMessageBox.warning(self, "Undo", f"Undo applied in memory; save failed:\n{exc}")
        self.manifest_status.setText(
            f"Undid last change → {self._manifest.path} "
            f"(undo depth={len(self._manifest.undo_stack)})"
        )
        self._fill_table()

    def _save_manifest(self) -> None:
        if self._manifest is None:
            self._on_store_changed()
        if self._manifest is None:
            QMessageBox.warning(self, "Save", "No manifest loaded.")
            return
        path = self._manifest.save()
        QMessageBox.information(self, "Save", f"Saved:\n{path}")
