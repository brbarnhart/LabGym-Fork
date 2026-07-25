"""Editable subject name / role / color table for identity package."""

from __future__ import annotations

from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from LabGym.identity.package import SubjectRecord

_COL_ID = 0
_COL_KIND = 1
_COL_NAME = 2
_COL_ROLE = 3
_COL_COLOR = 4


class SubjectsTable(QWidget):
    """Table of SubjectRecord rows."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Kind", "Display name", "Role", "Color"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            _COL_NAME, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            _COL_ROLE, QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.cellChanged.connect(self._on_cell)
        self.table.cellDoubleClicked.connect(self._on_double)
        layout.addWidget(self.table)

        row = QHBoxLayout()
        btn = QPushButton("Pick color…")
        btn.clicked.connect(self._pick_color_selected)
        row.addWidget(btn)
        row.addStretch(1)
        layout.addLayout(row)

        self._block = False

    def set_subjects(self, subjects: List[SubjectRecord]) -> None:
        self._block = True
        self.table.setRowCount(0)
        self.table.setRowCount(len(subjects))
        for r, s in enumerate(subjects):
            id_item = QTableWidgetItem(str(s.subject_id))
            id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, _COL_ID, id_item)

            kind_item = QTableWidgetItem(s.animal_kind)
            kind_item.setFlags(kind_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, _COL_KIND, kind_item)

            self.table.setItem(r, _COL_NAME, QTableWidgetItem(s.display_name))
            self.table.setItem(r, _COL_ROLE, QTableWidgetItem(s.role or ""))

            color_item = QTableWidgetItem(s.color or "")
            color_item.setBackground(QColor(s.color or "#888888"))
            self.table.setItem(r, _COL_COLOR, color_item)

            # stash track_id
            id_item.setData(Qt.ItemDataRole.UserRole, int(s.track_id or s.subject_id))
        self._block = False

    def get_subjects(self) -> List[SubjectRecord]:
        out: List[SubjectRecord] = []
        for r in range(self.table.rowCount()):
            sid = int(self.table.item(r, _COL_ID).text())
            kind = self.table.item(r, _COL_KIND).text()
            name = self.table.item(r, _COL_NAME).text() if self.table.item(r, _COL_NAME) else ""
            role = self.table.item(r, _COL_ROLE).text() if self.table.item(r, _COL_ROLE) else ""
            color = self.table.item(r, _COL_COLOR).text() if self.table.item(r, _COL_COLOR) else ""
            tid = self.table.item(r, _COL_ID).data(Qt.ItemDataRole.UserRole)
            out.append(
                SubjectRecord(
                    subject_id=sid,
                    animal_kind=kind,
                    display_name=name,
                    role=role,
                    color=color,
                    track_id=int(tid) if tid is not None else sid,
                )
            )
        return out

    def _on_cell(self, _r: int, _c: int) -> None:
        if not self._block:
            self.changed.emit()

    def _on_double(self, row: int, col: int) -> None:
        if col == _COL_COLOR:
            self._pick_color_row(row)

    def _pick_color_selected(self) -> None:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            return
        self._pick_color_row(rows[0])

    def _pick_color_row(self, row: int) -> None:
        item = self.table.item(row, _COL_COLOR)
        cur = item.text() if item else "#4FC3F7"
        color = QColorDialog.getColor(QColor(cur), self, "Subject color")
        if not color.isValid():
            return
        hexc = color.name()
        self._block = True
        item.setText(hexc)
        item.setBackground(color)
        self._block = False
        self.changed.emit()
