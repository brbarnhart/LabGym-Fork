"""BehaviorPalette: list of behaviors + add/rename/delete/color/hotkey controls.

For Phase 1 we provide a functional list + buttons. Full editing dialog can be added later.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QInputDialog, QLabel, QCheckBox
)

from LabGym.annotator.core.annotation_manager import AnnotationManager
from LabGym.annotator.core.data_models import Behavior


class BehaviorPalette(QWidget):
    behavior_selected = Signal(str)          # name
    behavior_toggled = Signal(str)           # name (request toggle bout)
    behaviors_changed = Signal()             # any structural change

    def __init__(self, manager: AnnotationManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self._selected_name: Optional[str] = None

        layout = QVBoxLayout(self)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(QLabel("Behaviors"))
        layout.addWidget(self.list_widget, 1)

        # Mode toggle (multi vs exclusive)
        self.mode_checkbox = QCheckBox("Exclusive mode (only one active at a time)")
        self.mode_checkbox.setToolTip(
            "Exclusive: only one behavior at a time.\n"
            "Starting a new bout overwrites any existing labels on those frames "
            "(splits/removes prior bouts) — useful for revising annotations.\n"
            "Pressing the active behavior's hotkey ends it.\n"
            "Multi (unchecked): behaviors are independent; no overwrite."
        )
        self.mode_checkbox.toggled.connect(self._on_mode_toggled)
        layout.addWidget(self.mode_checkbox)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_edit = QPushButton("Edit…")
        self.btn_edit.setToolTip(
            "Open a table to edit names, hotkeys, colors, and order of all behaviors"
        )
        self.btn_edit.clicked.connect(self.open_edit_dialog)
        self.btn_add = QPushButton("+ Add")
        self.btn_add.clicked.connect(self.add_behavior)
        self.btn_del = QPushButton("− Delete")
        self.btn_del.clicked.connect(self.delete_selected)
        btn_row.addWidget(self.btn_edit)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_del)
        layout.addLayout(btn_row)

        # Template save/load (reusable behavior sets with hotkeys + colors)
        tmpl_row = QHBoxLayout()
        self.btn_save_template = QPushButton("Save Template...")
        self.btn_save_template.clicked.connect(self.save_template)
        self.btn_save_template.setToolTip("Save current behaviors, colors, and hotkeys as a reusable template")
        self.btn_load_template = QPushButton("Load Template...")
        self.btn_load_template.clicked.connect(self.load_template)
        self.btn_load_template.setToolTip("Load a previously saved set of behaviors (preserves existing bout annotations for matching names)")
        tmpl_row.addWidget(self.btn_save_template)
        tmpl_row.addWidget(self.btn_load_template)
        layout.addLayout(tmpl_row)

        self.refresh()
        self._sync_mode_from_manager()

    def refresh(self):
        self.list_widget.clear()
        for beh in self.manager.session.behaviors:
            item = QListWidgetItem(f"{beh.name}  [{beh.hotkey or '–'}]")
            item.setData(Qt.ItemDataRole.UserRole, beh.name)
            # Store color for later display hints if wanted
            item.setForeground(QColor(beh.color))
            self.list_widget.addItem(item)

        # Re-select if possible
        if self._selected_name:
            for i in range(self.list_widget.count()):
                if self.list_widget.item(i).data(Qt.ItemDataRole.UserRole) == self._selected_name:
                    self.list_widget.setCurrentRow(i)
                    break

    def _on_item_clicked(self, item: QListWidgetItem):
        name = item.data(Qt.ItemDataRole.UserRole)
        self._selected_name = name
        self.behavior_selected.emit(name)

    def _on_item_double_clicked(self, item: QListWidgetItem):
        name = item.data(Qt.ItemDataRole.UserRole)
        self._selected_name = name
        self.behavior_toggled.emit(name)  # convenient double-click = toggle

    def selected_behavior(self) -> Optional[str]:
        return self._selected_name

    def select_behavior(self, name: str):
        self._selected_name = name
        for i in range(self.list_widget.count()):
            if self.list_widget.item(i).data(Qt.ItemDataRole.UserRole) == name:
                self.list_widget.setCurrentRow(i)
                self.behavior_selected.emit(name)
                return

    # --- Actions ---

    def open_edit_dialog(self):
        """Table editor for names, hotkeys, colors, and reorder."""
        from LabGym.annotator.ui.behavior_edit_dialog import BehaviorEditDialog

        dlg = BehaviorEditDialog(self.manager, self)
        if dlg.exec():
            # Selection may have been renamed
            if self._selected_name and self.manager.session.get_behavior(
                self._selected_name
            ) is None:
                self._selected_name = None
                if self.manager.session.behaviors:
                    self._selected_name = self.manager.session.behaviors[0].name
            self.refresh()
            self.behaviors_changed.emit()

    def add_behavior(self):
        name, ok = QInputDialog.getText(self, "Add Behavior", "Behavior name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        try:
            self.manager.add_behavior(name)
            self.refresh()
            self.behaviors_changed.emit()
            self.select_behavior(name)
        except ValueError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", str(e))

    def delete_selected(self):
        name = self.selected_behavior()
        if not name:
            return
        from PySide6.QtWidgets import QMessageBox
        if QMessageBox.question(self, "Delete", f"Delete behavior '{name}' and all its bouts?") != QMessageBox.StandardButton.Yes:
            return
        self.manager.remove_behavior(name)
        self._selected_name = None
        self.refresh()
        self.behaviors_changed.emit()

    def get_color_map(self) -> dict[str, str]:
        """Return name -> hex color for overlay use."""
        return {b.name: b.color for b in self.manager.session.behaviors}

    # --- Behavior Templates ---

    def save_template(self):
        if not self.manager or not self.manager.session.behaviors:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Save Template", "No behaviors to save.")
            return
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Behavior Template", "", "JSON Behavior Template (*.json)"
        )
        if not path:
            return
        try:
            self.manager.save_behavior_template(path)
            QMessageBox.information(self, "Saved", f"Behavior template saved to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    def load_template(self):
        if not self.manager:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Load Template", "Load a video first to apply a behavior template.")
            return
        from PySide6.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Behavior Template", "", "JSON Behavior Template (*.json)"
        )
        if not path:
            return
        try:
            self.manager.load_behavior_template(path)
            self.refresh()
            self.behaviors_changed.emit()
            QMessageBox.information(self, "Loaded", f"Behavior template loaded from:\n{path}\n\n"
                                                   "Bouts for matching behavior names were preserved.")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    # --- Mode handling ---

    def _sync_mode_from_manager(self):
        if hasattr(self, 'mode_checkbox'):
            self.mode_checkbox.blockSignals(True)
            self.mode_checkbox.setChecked(self.manager.exclusive_mode)
            self.mode_checkbox.blockSignals(False)

    def _on_mode_toggled(self, checked: bool):
        self.manager.set_exclusive_mode(checked)
        self.behaviors_changed.emit()  # in case UI wants to react
        # Optional: update status somewhere, but main_window listens via signals if needed

    def sync_from_manager(self):
        """Call this after loading a new session / manager to refresh list + mode."""
        self._sync_mode_from_manager()
        self.refresh()

    # --- Visual indication of active state ---

    def _prefix_for(self, name: str, open_set: set[str], annotated_set: set[str]) -> str:
        """▶ = currently toggled on; ● = saved bout at current frame; space = neither."""
        if name in open_set:
            return "▶ "
        if name in annotated_set:
            return "● "
        return "  "

    def refresh(self):
        self.list_widget.clear()
        # Open vs annotated-at-frame are refreshed properly via update_active_indicators
        # after seek; here mark only currently open (toggled) behaviors.
        for beh in self.manager.session.behaviors:
            is_open = self.manager.is_behavior_active(beh.name)
            prefix = "▶ " if is_open else "  "
            item = QListWidgetItem(f"{prefix}{beh.name}  [{beh.hotkey or '–'}]")
            item.setData(Qt.ItemDataRole.UserRole, beh.name)
            item.setForeground(QColor(beh.color))
            if is_open:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.list_widget.addItem(item)

        # Re-select if possible
        if self._selected_name:
            for i in range(self.list_widget.count()):
                if self.list_widget.item(i).data(Qt.ItemDataRole.UserRole) == self._selected_name:
                    self.list_widget.setCurrentRow(i)
                    break

    def update_active_indicators(
        self,
        open_names: list[str],
        annotated_names: Optional[list[str]] = None,
    ):
        """Update prefixes for open (▶) vs saved-at-frame (●) without full rebuild."""
        open_set = set(open_names)
        annotated_set = set(annotated_names or [])
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            name = item.data(Qt.ItemDataRole.UserRole)
            is_open = name in open_set
            prefix = self._prefix_for(name, open_set, annotated_set)
            beh = self.manager.session.get_behavior(name)
            hk = beh.hotkey or '–' if beh else '–'
            item.setText(f"{prefix}{name}  [{hk}]")
            font = item.font()
            font.setBold(is_open)
            item.setFont(font)
