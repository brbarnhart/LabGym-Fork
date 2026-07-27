"""Switch-marker table widget for Review IDs."""

from __future__ import annotations

from typing import List, Optional, Sequence

from PySide6.QtWidgets import QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from LabGym.id_review.types import SwitchMarker


class MarkersTable(QWidget):
    """List of switch markers (ID, frame, time, IDs, linked risk)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Switch markers"))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Frame", "Time (s)", "IDs", "Linked risk"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)

    def set_markers(self, markers: Sequence[SwitchMarker]) -> None:
        rows = sorted(markers, key=lambda x: x.frame)
        self.table.setRowCount(0)
        self.table.setRowCount(len(rows))
        for i, m in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(m.marker_id))
            self.table.setItem(i, 1, QTableWidgetItem(str(m.frame)))
            self.table.setItem(
                i,
                2,
                QTableWidgetItem(
                    f"{m.time_sec:.2f}" if m.time_sec is not None else ""
                ),
            )
            self.table.setItem(
                i, 3, QTableWidgetItem(",".join(str(x) for x in m.involved_ids))
            )
            self.table.setItem(i, 4, QTableWidgetItem(m.linked_event_id or ""))

    def selected_marker_id(self) -> Optional[str]:
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            return None
        item = self.table.item(rows[0], 0)
        return item.text() if item else None

    def frame_at_row(self, row: int) -> Optional[int]:
        try:
            return int(self.table.item(row, 1).text())
        except (ValueError, AttributeError):
            return None
