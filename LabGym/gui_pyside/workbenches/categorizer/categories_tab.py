"""Manage dataset → Categories: taxonomy ops and split tooling."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
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
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.widgets.path_browse import path_edit_row, set_line_edit_directory


class CategoriesTab(QWidget):
    """Category merge/exclude and train/val/sealed-test split management."""

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._manifest = None

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Manage the active taxonomy and split membership for one example store. "
            "Merge and category exclude are undoable and drive <b>soft projection</b> "
            "at train time without rewriting <code>soft_labels.csv</code>. "
            "Sealed test never enters training or validation."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(intro)

        form = QFormLayout()
        self.ed_store = QLineEdit()
        self.ed_store.setToolTip(
            "Example store root. dataset_manifest.json is created/updated here."
        )
        b = QPushButton("Browse…")
        b.clicked.connect(lambda: set_line_edit_directory(self, self.ed_store))
        form.addRow(
            self._lab("Example store:", "Manifest + examples live here."),
            path_edit_row(self.ed_store, b),
        )
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_reload = QPushButton("Reload / sync store")
        self.btn_reload.clicked.connect(self._reload)
        self.btn_save = QPushButton("Save manifest")
        self.btn_save.clicked.connect(self._save)
        self.btn_undo = QPushButton("Undo last change")
        self.btn_undo.clicked.connect(self._undo)
        btn_row.addWidget(self.btn_reload)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_undo)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.status = QLabel("Load an example store.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        cat_box = QGroupBox("Categories (active labels)")
        cat_l = QVBoxLayout(cat_box)
        self.cat_table = QTableWidget(0, 7)
        self.cat_table.setHorizontalHeaderLabels(
            [
                "category",
                "active",
                "excluded",
                "train",
                "validation",
                "sealed_test",
                "unassigned",
            ]
        )
        self.cat_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.cat_table.horizontalHeader().setStretchLastSection(True)
        self.cat_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.cat_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.cat_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.cat_table.setAlternatingRowColors(True)
        cat_l.addWidget(self.cat_table)
        layout.addWidget(cat_box, 2)

        tax = QGroupBox("Taxonomy operations")
        tax_l = QHBoxLayout(tax)

        merge_box = QGroupBox("Merge into target")
        merge_l = QVBoxLayout(merge_box)
        merge_l.addWidget(
            QLabel("Select source categories in the table, set target name, then Merge.")
        )
        self.ed_merge_target = QLineEdit()
        self.ed_merge_target.setPlaceholderText("Target category name")
        merge_l.addWidget(self.ed_merge_target)
        self.btn_merge = QPushButton("Merge selected → target")
        self.btn_merge.clicked.connect(self._merge)
        merge_l.addWidget(self.btn_merge)
        tax_l.addWidget(merge_box)

        ex_box = QGroupBox("Category exclude")
        ex_l = QVBoxLayout(ex_box)
        ex_l.addWidget(QLabel("Exclude all examples of selected categories from training."))
        self.btn_exclude_cat = QPushButton("Exclude selected categories")
        self.btn_exclude_cat.clicked.connect(lambda: self._exclude_cats(True))
        self.btn_include_cat = QPushButton("Re-include selected categories")
        self.btn_include_cat.clicked.connect(lambda: self._exclude_cats(False))
        ex_l.addWidget(self.btn_exclude_cat)
        ex_l.addWidget(self.btn_include_cat)
        tax_l.addWidget(ex_box)

        ops_box = QGroupBox("Taxonomy op history")
        ops_l = QVBoxLayout(ops_box)
        self.ops_list = QListWidget()
        self.ops_list.setMaximumHeight(100)
        ops_l.addWidget(self.ops_list)
        tax_l.addWidget(ops_box)
        layout.addWidget(tax)

        layout.addWidget(self._build_split_group())

        self.ed_store.editingFinished.connect(self._reload)
        self.project.project_replaced.connect(self._apply_project_defaults)
        self._apply_project_defaults()

    def _build_split_group(self) -> QGroupBox:
        split = QGroupBox("Splits (train / validation / sealed test)")
        outer = QVBoxLayout(split)
        row = QHBoxLayout()

        tv = QGroupBox("Train / validation")
        tv_l = QFormLayout(tv)
        self.sp_val_frac = QDoubleSpinBox()
        self.sp_val_frac.setRange(0.05, 0.5)
        self.sp_val_frac.setSingleStep(0.05)
        self.sp_val_frac.setValue(0.2)
        tv_l.addRow("Val fraction:", self.sp_val_frac)
        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 10_000_000)
        self.sp_seed.setValue(42)
        tv_l.addRow("Seed:", self.sp_seed)
        self.btn_ensure_tv = QPushButton("Ensure train/val (stable)")
        self.btn_ensure_tv.setToolTip(
            "Assign unassigned examples; keep existing membership unless regenerate."
        )
        self.btn_ensure_tv.clicked.connect(lambda: self._ensure_tv(False))
        self.btn_regen_tv = QPushButton("Regenerate train/val")
        self.btn_regen_tv.setToolTip("Re-shuffle train/validation (sealed test untouched).")
        self.btn_regen_tv.clicked.connect(lambda: self._ensure_tv(True))
        tv_l.addRow(self.btn_ensure_tv)
        tv_l.addRow(self.btn_regen_tv)
        row.addWidget(tv)

        se = QGroupBox("Sealed test")
        se_l = QFormLayout(se)
        self.sp_sealed_frac = QDoubleSpinBox()
        self.sp_sealed_frac.setRange(0.05, 0.5)
        self.sp_sealed_frac.setSingleStep(0.05)
        self.sp_sealed_frac.setValue(0.1)
        se_l.addRow("Sealed fraction:", self.sp_sealed_frac)
        self.btn_assign_sealed = QPushButton("Assign sealed test (fraction)")
        self.btn_assign_sealed.setToolTip(
            "Stratified sample from train/val/unassigned → sealed_test."
        )
        self.btn_assign_sealed.clicked.connect(self._assign_sealed)
        self.btn_clear_sealed = QPushButton("Clear sealed → unassigned")
        self.btn_clear_sealed.clicked.connect(self._clear_sealed)
        se_l.addRow(self.btn_assign_sealed)
        se_l.addRow(self.btn_clear_sealed)
        row.addWidget(se)

        outer.addLayout(row)
        self.split_status = QLabel("")
        self.split_status.setWordWrap(True)
        outer.addWidget(self.split_status)
        return split

    @staticmethod
    def _lab(text: str, tip: str) -> QLabel:
        lab = QLabel(text)
        lab.setToolTip(tip)
        return lab

    def _apply_project_defaults(self) -> None:
        try:
            p = self.project.project
            ex = p.resolve_path(p.paths.examples_root or "examples")
            if Path(ex).is_dir():
                self.ed_store.setText(str(ex))
        except Exception as exc:
            self.status.setText(f"Could not resolve project examples path: {exc}")
        self._reload()

    def _require_manifest(self):
        if self._manifest is None:
            self._reload()
        if self._manifest is None:
            QMessageBox.warning(self, "Categories", "Set a valid example store first.")
            return None
        return self._manifest

    def _reload(self) -> None:
        store = self.ed_store.text().strip()
        if not store or not Path(store).is_dir():
            self._manifest = None
            self.status.setText("Set a valid example store folder.")
            self.cat_table.setRowCount(0)
            self.ops_list.clear()
            return
        try:
            from LabGym.training.dataset_manifest import DatasetManifest, scan_example_store

            m = DatasetManifest.load_or_create(store)
            scanned = [
                (eid, lab, hint) for eid, lab, hint, _p in scan_example_store(store)
            ]
            added = m.sync_from_scan(scanned)
            self._manifest = m
            self._refresh_views()
            extra = f" (+{len(added)} new)" if added else ""
            self.status.setText(
                f"Store: {store}  |  {len(m.examples)} examples{extra}  |  "
                f"undo depth={len(m.undo_stack)}  |  manifest → {m.path}"
            )
        except Exception as exc:
            self._manifest = None
            self.status.setText(f"Error: {exc}")
            QMessageBox.critical(self, "Categories", str(exc))

    def _refresh_views(self) -> None:
        m = self._manifest
        if m is None:
            return
        summary = m.category_summary()
        self.cat_table.setRowCount(0)
        for row in summary:
            r = self.cat_table.rowCount()
            self.cat_table.insertRow(r)
            vals = [
                row["category"],
                row["n_active"],
                row["n_excluded"],
                row["n_train"],
                row["n_validation"],
                row["n_sealed_test"],
                row["n_unassigned"],
            ]
            for c, v in enumerate(vals):
                self.cat_table.setItem(r, c, QTableWidgetItem(str(v)))

        self.ops_list.clear()
        for op in m.taxonomy_ops[-50:]:
            kind = op.get("op", "?")
            if kind == "merge":
                text = (
                    f"merge {op.get('sources')} → {op.get('target')} "
                    f"(n={op.get('n_examples')})"
                )
            elif kind in ("exclude_category", "include_category"):
                text = f"{kind}: {op.get('category')} (n={op.get('n_examples')})"
            else:
                text = str(op)
            self.ops_list.addItem(QListWidgetItem(text))

        counts = m.counts_by_split(include_excluded=False)
        excl = sum(1 for r in m.examples.values() if r.excluded)
        tax_excl = m.excluded_categories()
        self.split_status.setText(
            f"Active splits: train={counts.get('train', 0)}  "
            f"validation={counts.get('validation', 0)}  "
            f"sealed_test={counts.get('sealed_test', 0)}  "
            f"unassigned={counts.get('unassigned', 0)}  |  "
            f"excluded examples={excl}  |  "
            f"taxonomy-excluded categories={tax_excl or '—'}"
        )

    def _selected_categories(self) -> list:
        cats = []
        sel = self.cat_table.selectionModel()
        if sel is None:
            return cats
        for idx in sel.selectedRows():
            item = self.cat_table.item(idx.row(), 0)
            if item:
                cats.append(item.text())
        return cats

    def _save(self) -> None:
        m = self._require_manifest()
        if m is None:
            return
        path = m.save()
        self.status.setText(f"Saved → {path}")
        QMessageBox.information(self, "Save", f"Saved:\n{path}")

    def _undo(self) -> None:
        m = self._require_manifest()
        if m is None:
            return
        if not m.undo():
            QMessageBox.information(self, "Undo", "Nothing to undo.")
            return
        m.save()
        self._refresh_views()
        self.status.setText(f"Undid last change. undo depth={len(m.undo_stack)}")

    def _merge(self) -> None:
        m = self._require_manifest()
        if m is None:
            return
        sources = self._selected_categories()
        target = self.ed_merge_target.text().strip()
        if not sources:
            QMessageBox.warning(
                self, "Merge", "Select one or more source categories in the table."
            )
            return
        if not target:
            QMessageBox.warning(self, "Merge", "Enter a target category name.")
            return
        try:
            n = m.merge_categories(sources, target)
            m.save()
            self._refresh_views()
            QMessageBox.information(
                self,
                "Merge",
                f"Merged {sources} → {target}\nUpdated {n} example(s).",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Merge", str(exc))

    def _exclude_cats(self, excluded: bool) -> None:
        m = self._require_manifest()
        if m is None:
            return
        cats = self._selected_categories()
        if not cats:
            QMessageBox.warning(self, "Category exclude", "Select categories in the table.")
            return
        total = 0
        try:
            for c in cats:
                total += m.exclude_category(c, excluded=excluded)
            m.save()
            self._refresh_views()
            action = "Excluded" if excluded else "Re-included"
            QMessageBox.information(
                self,
                "Category exclude",
                f"{action} {len(cats)} categor(y/ies); {total} example(s).",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Category exclude", str(exc))

    def _ensure_tv(self, regenerate: bool) -> None:
        m = self._require_manifest()
        if m is None:
            return
        try:
            counts = m.ensure_train_val_split(
                val_fraction=float(self.sp_val_frac.value()),
                seed=int(self.sp_seed.value()),
                regenerate=regenerate,
                assign_new=True,
            )
            m.save()
            self._refresh_views()
            mode = "regenerated" if regenerate else "ensured"
            QMessageBox.information(
                self,
                "Train/val",
                f"Train/val {mode}.\n"
                f"train={counts.get('train', 0)} validation={counts.get('validation', 0)} "
                f"sealed={counts.get('sealed_test', 0)} unassigned={counts.get('unassigned', 0)}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Train/val", str(exc))

    def _assign_sealed(self) -> None:
        m = self._require_manifest()
        if m is None:
            return
        try:
            assigned = m.assign_sealed_test(
                fraction=float(self.sp_sealed_frac.value()),
                seed=int(self.sp_seed.value()),
            )
            m.save()
            self._refresh_views()
            QMessageBox.information(
                self,
                "Sealed test",
                f"Assigned {len(assigned)} example(s) to sealed_test.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Sealed test", str(exc))

    def _clear_sealed(self) -> None:
        m = self._require_manifest()
        if m is None:
            return
        try:
            from LabGym.training.dataset_manifest import SPLIT_UNASSIGNED

            n = m.clear_sealed_test(to_split=SPLIT_UNASSIGNED)
            m.save()
            self._refresh_views()
            QMessageBox.information(self, "Sealed test", f"Cleared {n} sealed example(s).")
        except Exception as exc:
            QMessageBox.critical(self, "Sealed test", str(exc))
