"""Dialog: table editor for behavior names, hotkeys, colors, and order."""

from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from LabGym.annotator.core.annotation_manager import AnnotationManager

# Column indices
COL_NAME = 0
COL_HOTKEY = 1
COL_COLOR = 2

# Roles
ROLE_ORIGINAL = Qt.ItemDataRole.UserRole
ROLE_COLOR = Qt.ItemDataRole.UserRole + 1


class BehaviorEditDialog(QDialog):
    """Edit all behaviors in a table: name, hotkey, color, reorder."""

    def __init__(self, manager: AnnotationManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Edit Behaviors")
        self.resize(560, 420)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Edit name, hotkey, and color. Use ↑/↓ to reorder (list order is used "
                "in the palette and timeline). Double-click the color cell to pick a color."
            )
        )

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Hotkey", "Color"])
        self.table.horizontalHeader().setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            COL_HOTKEY, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            COL_COLOR, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        layout.addWidget(self.table, 1)

        row_btns = QHBoxLayout()
        self.btn_add = QPushButton("+ Add")
        self.btn_add.clicked.connect(self._add_row)
        self.btn_del = QPushButton("− Remove")
        self.btn_del.clicked.connect(self._remove_row)
        self.btn_up = QPushButton("↑ Move up")
        self.btn_up.clicked.connect(lambda: self._move_row(-1))
        self.btn_down = QPushButton("↓ Move down")
        self.btn_down.clicked.connect(lambda: self._move_row(1))
        self.btn_color = QPushButton("Pick color…")
        self.btn_color.clicked.connect(self._pick_color_for_selection)
        row_btns.addWidget(self.btn_add)
        row_btns.addWidget(self.btn_del)
        row_btns.addWidget(self.btn_up)
        row_btns.addWidget(self.btn_down)
        row_btns.addWidget(self.btn_color)
        row_btns.addStretch(1)
        layout.addLayout(row_btns)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._load_from_manager()

    def _load_from_manager(self) -> None:
        self.table.setRowCount(0)
        for beh in self.manager.session.behaviors:
            self._append_row(
                original=beh.name,
                name=beh.name,
                hotkey=beh.hotkey or "",
                color=beh.color or "#FF5555",
            )

    def _append_row(
        self,
        *,
        original: Optional[str],
        name: str,
        hotkey: str,
        color: str,
    ) -> int:
        r = self.table.rowCount()
        self.table.insertRow(r)

        name_item = QTableWidgetItem(name)
        name_item.setData(ROLE_ORIGINAL, original)
        self.table.setItem(r, COL_NAME, name_item)

        hk_item = QTableWidgetItem(hotkey)
        hk_item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
        self.table.setItem(r, COL_HOTKEY, hk_item)

        color_item = QTableWidgetItem(color)
        color_item.setData(ROLE_COLOR, color)
        color_item.setFlags(
            color_item.flags() & ~Qt.ItemFlag.ItemIsEditable
        )
        self._style_color_item(color_item, color)
        self.table.setItem(r, COL_COLOR, color_item)
        return r

    def _style_color_item(self, item: QTableWidgetItem, color: str) -> None:
        qc = QColor(color)
        if not qc.isValid():
            qc = QColor("#FF5555")
            color = qc.name()
        item.setText(color)
        item.setData(ROLE_COLOR, color)
        item.setBackground(qc)
        # Readable text on light/dark swatches
        luma = 0.299 * qc.red() + 0.587 * qc.green() + 0.114 * qc.blue()
        item.setForeground(QColor("#000000" if luma > 140 else "#FFFFFF"))
        item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))

    def _on_cell_double_clicked(self, row: int, col: int) -> None:
        if col == COL_COLOR:
            self._pick_color_row(row)

    def _pick_color_for_selection(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Color", "Select a row first.")
            return
        self._pick_color_row(row)

    def _pick_color_row(self, row: int) -> None:
        item = self.table.item(row, COL_COLOR)
        if item is None:
            return
        current = item.data(ROLE_COLOR) or item.text() or "#FF5555"
        color = QColorDialog.getColor(QColor(current), self, "Pick behavior color")
        if color.isValid():
            self._style_color_item(item, color.name())

    def _add_row(self) -> None:
        n = self.table.rowCount() + 1
        # Default palette of distinct colors
        defaults = [
            "#4FC3F7",
            "#FF8A65",
            "#81C784",
            "#CE93D8",
            "#FFD54F",
            "#F06292",
            "#4DB6AC",
            "#FF5555",
        ]
        color = defaults[(n - 1) % len(defaults)]
        row = self._append_row(
            original=None,
            name=f"behavior_{n}",
            hotkey=str(n) if n <= 9 else "",
            color=color,
        )
        self.table.setCurrentCell(row, COL_NAME)
        self.table.editItem(self.table.item(row, COL_NAME))

    def _remove_row(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        if self.table.rowCount() <= 1:
            QMessageBox.warning(
                self, "Remove", "Keep at least one behavior definition."
            )
            return
        name_item = self.table.item(row, COL_NAME)
        label = name_item.text() if name_item else "this behavior"
        if (
            QMessageBox.question(
                self,
                "Remove",
                f"Remove '{label}' from the list?\n"
                "(Existing bouts for that name will be deleted when you click OK.)",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self.table.removeRow(row)

    def _move_row(self, delta: int) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if new_row < 0 or new_row >= self.table.rowCount():
            return
        self._swap_rows(row, new_row)
        self.table.setCurrentCell(new_row, COL_NAME)

    def _swap_rows(self, a: int, b: int) -> None:
        """Swap two rows by exchanging item data."""
        for col in range(self.table.columnCount()):
            item_a = self.table.takeItem(a, col)
            item_b = self.table.takeItem(b, col)
            self.table.setItem(a, col, item_b)
            self.table.setItem(b, col, item_a)

    def _collect_rows(
        self,
    ) -> List[Tuple[Optional[str], str, str, Optional[str]]]:
        rows: List[Tuple[Optional[str], str, str, Optional[str]]] = []
        for r in range(self.table.rowCount()):
            name_item = self.table.item(r, COL_NAME)
            hk_item = self.table.item(r, COL_HOTKEY)
            color_item = self.table.item(r, COL_COLOR)
            if name_item is None:
                continue
            original = name_item.data(ROLE_ORIGINAL)
            if original is not None:
                original = str(original)
            name = name_item.text().strip()
            hk = hk_item.text().strip() if hk_item else ""
            color = (
                (color_item.data(ROLE_COLOR) if color_item else None)
                or (color_item.text() if color_item else "")
                or "#FF5555"
            )
            rows.append((original, name, str(color), hk or None))
        return rows

    def _accept(self) -> None:
        # End any open editors so cell text is committed
        self.table.setCurrentItem(None)

        rows = self._collect_rows()
        # Check duplicate hotkeys (warn only)
        hotkeys = [h for _, _, _, h in rows if h]
        if len(hotkeys) != len(set(hotkeys)):
            if (
                QMessageBox.question(
                    self,
                    "Duplicate hotkeys",
                    "Some hotkeys are used more than once. Continue anyway?",
                )
                != QMessageBox.StandardButton.Yes
            ):
                return
        try:
            self.manager.apply_behavior_table(rows)
        except Exception as e:
            QMessageBox.warning(self, "Invalid behaviors", str(e))
            return
        self.accept()
