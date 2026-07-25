"""BoutListWidget: Editable list of all annotated bouts.

Allows viewing, deleting, changing type, filtering by type, and jumping
to specific bouts. This provides the main way to correct mistakes in annotations.

Rows covering the main window's current frame are highlighted so active
annotations are easy to find while scrubbing or re-annotating.

Multi-select (Ctrl/Shift+click) supports bulk partner assignment for
interactive advanced (mode 2) bouts.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal, QTimer, QItemSelectionModel
from PySide6.QtGui import QColor, QBrush, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QMessageBox, QComboBox, QInputDialog, QHeaderView,
    QMenu, QAbstractItemView, QDialog, QDialogButtonBox, QListWidget,
    QListWidgetItem, QCheckBox,
)

from LabGym.annotator.core.annotation_manager import AnnotationManager
from LabGym.annotator.core.data_models import Bout, Subject

# Highlight colors for rows active at the current video frame
_HIGHLIGHT_BG = QColor(255, 220, 80, 90)       # soft gold for saved bout at frame
_HIGHLIGHT_LIVE_BG = QColor(255, 140, 40, 110)  # warmer for live open bout overlap
_HIGHLIGHT_FG = QColor(40, 30, 0)
_NORMAL_BG = QBrush()
_NORMAL_FG = QBrush()

# Table columns
_COL_MARKER = 0
_COL_BEHAVIOR = 1
_COL_START = 2
_COL_END = 3
_COL_PARTNERS = 4
_COL_DUR_F = 5
_COL_DUR_S = 6
_COL_ACTIONS = 7
_TEXT_COL_COUNT = 7  # marker..dur_s (not action buttons)


class PartnerPickDialog(QDialog):
    """Pick zero or more partner subjects for selected bout(s)."""

    def __init__(
        self,
        subjects: Sequence[Subject],
        *,
        active_subject_id: Optional[int] = None,
        initial_ids: Optional[Sequence[int]] = None,
        title: str = "Set Partners",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(360, 320)
        self._result_ids: List[int] = []

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Select interaction partner(s) for the selected bout(s).\n"
            "Leave all unchecked (or use Clear) for no partners."
        ))

        self.clear_box = QCheckBox("Clear partners (none)")
        self.clear_box.setToolTip(
            "When checked, selected bouts get empty partner_ids."
        )
        layout.addWidget(self.clear_box)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        initial = set(int(x) for x in (initial_ids or []))
        for subj in subjects:
            # Still allow selecting any subject; active is dimmed but usable
            # so multi-subject bulk edits stay flexible.
            label = f"{subj.display_name}  (id={subj.subject_id})"
            if active_subject_id is not None and subj.subject_id == active_subject_id:
                label += "  · active"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, int(subj.subject_id))
            item.setFlags(
                item.flags()
                | Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            item.setCheckState(
                Qt.CheckState.Checked
                if int(subj.subject_id) in initial
                else Qt.CheckState.Unchecked
            )
            self.list.addItem(item)
        layout.addWidget(self.list, 1)

        self.clear_box.toggled.connect(self._on_clear_toggled)
        if not initial:
            # Don't auto-check clear; user may want to pick partners
            pass

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_clear_toggled(self, checked: bool) -> None:
        self.list.setEnabled(not checked)
        if checked:
            for i in range(self.list.count()):
                self.list.item(i).setCheckState(Qt.CheckState.Unchecked)

    def selected_partner_ids(self) -> List[int]:
        if self.clear_box.isChecked():
            return []
        ids: List[int] = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                ids.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return ids


class BoutListWidget(QWidget):
    jump_requested = Signal(int)  # frame to seek to
    bouts_changed = Signal()      # notify parent to refresh everything

    def __init__(self, manager: AnnotationManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self._filter_behavior: Optional[str] = None  # None = show all
        self._current_frame: int = 0
        # row -> (name, bout_idx, start, end)
        self._row_meta: List[Tuple[str, int, int, int]] = []

        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel(
            "Bouts — multi-select rows · double-click Start/End/Partners · "
            "right-click for more"
        ))
        header.addStretch()

        header.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.setMinimumWidth(140)
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        header.addWidget(self.filter_combo)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh)
        header.addWidget(self.btn_refresh)
        layout.addLayout(header)

        # Shows which bouts are active at the main window's current frame
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "QLabel { background: #2a2a2a; color: #eee; padding: 6px 8px; "
            "border-radius: 4px; }"
        )
        layout.addWidget(self.status_label)

        self.table = QTableWidget()
        # Col 0 = active marker (fixed width); names stay in col 1 so they never get truncated by ▶/●
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(
            ["", "Behavior", "Start", "End", "Partners", "Dur (f)", "Dur (s)", "Actions"]
        )
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        # Ctrl/Shift multi-select for bulk partner edits
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # Editing is handled via double-click / context menu (validated)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context_menu)
        self.table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        header_view = self.table.horizontalHeader()
        header_view.setStretchLastSection(True)
        header_view.setSectionResizeMode(_COL_MARKER, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(_COL_MARKER, 28)
        # Allow background colors on items to show through selection styling
        self.table.setStyleSheet(
            "QTableWidget::item:selected { background: #3a6ea5; color: white; }"
        )
        layout.addWidget(self.table, 1)

        # Bottom buttons
        btn_row = QHBoxLayout()
        self.btn_edit_frames = QPushButton("Edit Start/End")
        self.btn_edit_frames.setToolTip("Edit start and end frames of the selected bout")
        self.btn_edit_frames.clicked.connect(self.edit_frames_selected)
        self.btn_set_partners = QPushButton("Set Partners…")
        self.btn_set_partners.setToolTip(
            "Set partner subject(s) on all selected bouts (Ctrl/Shift multi-select)"
        )
        self.btn_set_partners.clicked.connect(self.set_partners_selected)
        self.btn_clear_partners = QPushButton("Clear Partners")
        self.btn_clear_partners.setToolTip("Remove partners from all selected bouts")
        self.btn_clear_partners.clicked.connect(self.clear_partners_selected)
        self.btn_change_type = QPushButton("Change Type")
        self.btn_change_type.setToolTip("Change the behavior type of the selected bout")
        self.btn_change_type.clicked.connect(self.change_type_selected)
        self.btn_delete = QPushButton("Delete Selected")
        self.btn_delete.clicked.connect(self.delete_selected)
        self.btn_jump = QPushButton("Jump to Bout")
        self.btn_jump.clicked.connect(self.jump_to_selected)
        self.btn_jump_active = QPushButton("Jump to Active")
        self.btn_jump_active.setToolTip("Jump table selection to the first bout active at the current frame")
        self.btn_jump_active.clicked.connect(self._scroll_to_first_active)
        btn_row.addWidget(self.btn_edit_frames)
        btn_row.addWidget(self.btn_set_partners)
        btn_row.addWidget(self.btn_clear_partners)
        btn_row.addWidget(self.btn_change_type)
        btn_row.addWidget(self.btn_delete)
        btn_row.addWidget(self.btn_jump)
        btn_row.addWidget(self.btn_jump_active)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._rebuild_filter_combo()
        self.refresh()

    def _behavior_names(self) -> List[str]:
        return [b.name for b in self.manager.session.behaviors]

    def _rebuild_filter_combo(self) -> None:
        """Populate the filter dropdown while preserving the current selection if possible."""
        previous = self._filter_behavior
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All", None)
        for name in self._behavior_names():
            self.filter_combo.addItem(name, name)

        # Restore previous filter if still valid
        if previous is None:
            self.filter_combo.setCurrentIndex(0)
            self._filter_behavior = None
        else:
            idx = self.filter_combo.findData(previous)
            if idx >= 0:
                self.filter_combo.setCurrentIndex(idx)
                self._filter_behavior = previous
            else:
                self.filter_combo.setCurrentIndex(0)
                self._filter_behavior = None
        self.filter_combo.blockSignals(False)

    def _on_filter_changed(self, _index: int) -> None:
        self._filter_behavior = self.filter_combo.currentData()
        self.refresh()

    def _collect_bouts(self) -> List[Tuple[str, int, Bout]]:
        """Gather (behavior_name, index_in_behavior, bout), optionally filtered."""
        all_bouts: List[Tuple[str, int, Bout]] = []
        for beh in self.manager.session.behaviors:
            name = beh.name
            if self._filter_behavior is not None and name != self._filter_behavior:
                continue
            for idx, bout in enumerate(self.manager.get_bouts_for_behavior(name)):
                all_bouts.append((name, idx, bout))
        all_bouts.sort(key=lambda x: (x[2].start_frame, x[0]))
        return all_bouts

    def set_current_frame(self, frame: int, *, scroll: Optional[bool] = None) -> None:
        """Update which frame is considered 'active' and re-highlight rows.

        scroll: if None, auto-scroll only on larger frame jumps (not every play tick).
        """
        frame = max(0, int(frame))
        prev = self._current_frame
        if scroll is None:
            # During continuous playback (Δ=1) keep the list stable; jump on seeks
            scroll = abs(frame - prev) > 1
        self._current_frame = frame
        self._apply_active_highlights(scroll_to_active=scroll)

    def _active_row_indices(self) -> List[int]:
        """Rows whose bout interval contains the current frame."""
        f = self._current_frame
        return [
            row
            for row, (_n, _i, start, end) in enumerate(self._row_meta)
            if start <= f <= end
        ]

    def _open_at_frame(self) -> List[Tuple[str, int]]:
        """(behavior_name, start_frame) for live open bouts covering current frame."""
        result: List[Tuple[str, int]] = []
        opens = self.manager.get_open_starts()
        for name, start in opens.items():
            if start is not None and start <= self._current_frame:
                result.append((name, start))
        return result

    def _apply_active_highlights(self, scroll_to_active: bool = False) -> None:
        """Style rows active at the current frame; update status banner."""
        active_rows = set(self._active_row_indices())
        open_list = self._open_at_frame()
        open_names = {n for n, _ in open_list}

        bold_font = QFont()
        bold_font.setBold(True)
        normal_font = QFont()
        normal_font.setBold(False)

        highlight_brush = QBrush(_HIGHLIGHT_BG)
        live_brush = QBrush(_HIGHLIGHT_LIVE_BG)
        fg_brush = QBrush(_HIGHLIGHT_FG)

        for row in range(self.table.rowCount()):
            is_active = row in active_rows
            name = self._row_meta[row][0] if row < len(self._row_meta) else ""
            # If this saved bout is active AND same behavior is currently open/live, use live tint
            use_live = is_active and name in open_names
            bg = live_brush if use_live else (highlight_brush if is_active else _NORMAL_BG)
            fg = fg_brush if is_active else _NORMAL_FG
            font = bold_font if is_active else normal_font

            # Text columns (not action buttons)
            for col in range(_TEXT_COL_COUNT):
                item = self.table.item(row, col)
                if item is None:
                    continue
                item.setBackground(bg)
                item.setForeground(fg)
                item.setFont(font)

            # Marker in its own fixed column so behavior names keep full width
            marker_item = self.table.item(row, _COL_MARKER)
            if marker_item is not None:
                if is_active:
                    marker_item.setText("▶" if use_live else "●")
                    marker_item.setToolTip(
                        "Live (open) at current frame" if use_live
                        else "Saved bout active at current frame"
                    )
                else:
                    marker_item.setText("")
                    marker_item.setToolTip("")
                marker_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # Keep behavior name pure (never prepend markers)
            beh_item = self.table.item(row, _COL_BEHAVIOR)
            if beh_item is not None and row < len(self._row_meta):
                beh_item.setText(self._row_meta[row][0])

        # Status banner
        parts: List[str] = [f"Frame {self._current_frame}"]
        if active_rows:
            names = []
            for r in sorted(active_rows):
                n, _i, s, e = self._row_meta[r]
                names.append(f"{n} [{s}–{e}]")
            parts.append("Saved active: " + ", ".join(names))
        else:
            parts.append("Saved active: —")

        if open_list:
            live_bits = [f"{n} (from {st})" for n, st in open_list]
            parts.append("Live: " + ", ".join(live_bits))
        else:
            parts.append("Live: —")

        self.status_label.setText("  ·  ".join(parts))

        if scroll_to_active and active_rows:
            first = min(active_rows)
            # Don't steal selection from user constantly during play — only ensure visible
            self.table.scrollToItem(
                self.table.item(first, _COL_BEHAVIOR),
                self.table.ScrollHint.PositionAtCenter,
            )

    def _scroll_to_first_active(self) -> None:
        active = self._active_row_indices()
        if not active:
            open_list = self._open_at_frame()
            if open_list:
                QMessageBox.information(
                    self,
                    "Active Bout",
                    "There is a live (open) annotation, but no completed bout at this frame.\n"
                    f"Live: {', '.join(n for n, _ in open_list)}",
                )
            else:
                QMessageBox.information(
                    self, "Active Bout", "No bout is active at the current frame."
                )
            return
        row = min(active)
        self.table.selectRow(row)
        self.table.scrollToItem(
            self.table.item(row, _COL_BEHAVIOR),
            self.table.ScrollHint.PositionAtCenter,
        )
        self.table.setFocus()

    def _partner_label(self, partner_ids: Sequence[int]) -> str:
        if not partner_ids:
            return "—"
        names: List[str] = []
        for pid in partner_ids:
            subj = self.manager.session.get_subject(int(pid))
            if subj is not None:
                names.append(subj.display_name)
            else:
                names.append(str(int(pid)))
        return ", ".join(names)

    def refresh(self, *, preserve_view: bool = True):
        """Rebuild the table from current manager state (respecting filter).

        preserve_view: keep scroll position (and nearby selection) after rebuild so
        delete/type-change edits in the middle of a long list don't jump to the top.
        """
        # Keep filter options in sync with behaviors
        current_data = self.filter_combo.currentData() if self.filter_combo.count() else None
        names = self._behavior_names()
        combo_names = [
            self.filter_combo.itemData(i)
            for i in range(1, self.filter_combo.count())
        ]
        if names != combo_names or self.filter_combo.count() == 0:
            self._rebuild_filter_combo()
        elif current_data != self._filter_behavior:
            self._filter_behavior = self.filter_combo.currentData()

        vbar = self.table.verticalScrollBar()
        hbar = self.table.horizontalScrollBar()
        saved_v = vbar.value() if preserve_view else 0
        saved_h = hbar.value() if preserve_view else 0
        # Preserve multi-selection by bout identity (name, index) when possible
        saved_refs = self._selected_bout_refs() if preserve_view else []
        saved_row = self.table.currentRow() if preserve_view else -1

        self.table.setRowCount(0)
        self._row_meta = []
        all_bouts = self._collect_bouts()
        self.table.setRowCount(len(all_bouts))

        for row, (name, bout_idx, bout) in enumerate(all_bouts):
            marker_item = QTableWidgetItem("")
            marker_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(row, _COL_MARKER, marker_item)

            self.table.setItem(row, _COL_BEHAVIOR, QTableWidgetItem(name))

            start_item = QTableWidgetItem(str(bout.start_frame))
            start_item.setToolTip("Double-click to edit start frame")
            self.table.setItem(row, _COL_START, start_item)

            end_item = QTableWidgetItem(str(bout.end_frame))
            end_item.setToolTip("Double-click to edit end frame")
            self.table.setItem(row, _COL_END, end_item)

            partners_item = QTableWidgetItem(self._partner_label(bout.partner_ids))
            partners_item.setToolTip(
                "Double-click to set partners · multi-select rows for bulk edit"
            )
            partners_item.setData(
                Qt.ItemDataRole.UserRole + 1,
                [int(x) for x in (bout.partner_ids or [])],
            )
            self.table.setItem(row, _COL_PARTNERS, partners_item)

            dur_f = bout.duration_frames()
            self.table.setItem(row, _COL_DUR_F, QTableWidgetItem(str(dur_f)))

            fps = self.manager.session.fps or 30.0
            dur_s = round(dur_f / fps, 2)
            self.table.setItem(row, _COL_DUR_S, QTableWidgetItem(str(dur_s)))

            # Action buttons in last column
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(0, 0, 0, 0)

            type_btn = QPushButton("T")
            type_btn.setFixedWidth(28)
            type_btn.setToolTip("Change behavior type")
            type_btn.clicked.connect(
                lambda _, n=name, i=bout_idx: self._change_bout_type(n, i)
            )

            del_btn = QPushButton("✕")
            del_btn.setFixedWidth(28)
            del_btn.setToolTip("Delete this bout")
            del_btn.clicked.connect(lambda _, n=name, i=bout_idx: self._delete_bout(n, i))

            jump_btn = QPushButton("→")
            jump_btn.setFixedWidth(28)
            jump_btn.setToolTip("Jump to start of this bout")
            jump_btn.clicked.connect(lambda _, f=bout.start_frame: self.jump_requested.emit(f))

            btn_layout.addWidget(type_btn)
            btn_layout.addWidget(del_btn)
            btn_layout.addWidget(jump_btn)
            btn_layout.addStretch()

            self.table.setCellWidget(row, _COL_ACTIONS, btn_widget)

            # Store data for later (on Start column)
            start_item.setData(Qt.ItemDataRole.UserRole, (name, bout_idx))
            self._row_meta.append((name, bout_idx, bout.start_frame, bout.end_frame))

        self.table.resizeColumnsToContents()
        # Keep marker column narrow after resize-to-contents
        self.table.setColumnWidth(_COL_MARKER, 28)
        # Never auto-jump to the active bout on a full rebuild when preserving view
        self._apply_active_highlights(scroll_to_active=False)

        if preserve_view:
            def _restore_view():
                # Restore after layout settles; clamp if list got shorter after a delete
                vbar.setValue(min(saved_v, vbar.maximum()))
                hbar.setValue(min(saved_h, hbar.maximum()))
                self.table.clearSelection()
                sm = self.table.selectionModel()
                restored = 0
                if saved_refs and sm is not None:
                    flags = (
                        QItemSelectionModel.SelectionFlag.Select
                        | QItemSelectionModel.SelectionFlag.Rows
                    )
                    for r in range(self.table.rowCount()):
                        ref = self._bout_ref_at_row(r)
                        if ref is not None and ref in saved_refs:
                            idx = self.table.model().index(r, _COL_BEHAVIOR)
                            sm.select(idx, flags)
                            restored += 1
                if restored == 0 and saved_row >= 0 and self.table.rowCount() > 0:
                    target = min(saved_row, self.table.rowCount() - 1)
                    self.table.selectRow(target)

            # Immediate restore + one deferred pass (Qt often resets scroll during layout)
            _restore_view()
            QTimer.singleShot(0, _restore_view)

    def _selected_bout_ref(self) -> Optional[Tuple[str, int]]:
        """Return (behavior_name, bout_index) for the current selection, or None."""
        row = self.table.currentRow()
        return self._bout_ref_at_row(row)

    def _selected_rows(self) -> List[int]:
        rows = sorted({idx.row() for idx in self.table.selectedIndexes()})
        return [r for r in rows if 0 <= r < self.table.rowCount()]

    def _selected_bout_refs(self) -> List[Tuple[str, int]]:
        """Return unique (behavior_name, bout_index) for all selected rows."""
        refs: List[Tuple[str, int]] = []
        seen = set()
        for row in self._selected_rows():
            ref = self._bout_ref_at_row(row)
            if ref is None or ref in seen:
                continue
            seen.add(ref)
            refs.append(ref)
        return refs

    def _bout_ref_at_row(self, row: int) -> Optional[Tuple[str, int]]:
        if row < 0:
            return None
        item = self.table.item(row, _COL_START)  # Start column holds UserRole ref
        if not item:
            return None
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return None
        return data

    def _bout_frames_at_row(self, row: int) -> Optional[Tuple[str, int, int, int]]:
        """Return (name, bout_idx, start, end) for a row."""
        if row < 0 or row >= len(self._row_meta):
            return None
        return self._row_meta[row]

    def _frame_bounds(self) -> Tuple[int, int]:
        """Inclusive min/max valid frame indices for this session."""
        total = self.manager.session.total_frames
        if total and total > 0:
            return 0, total - 1
        return 0, 10**9

    def _apply_frame_edit(
        self,
        name: str,
        bout_index: int,
        start: int,
        end: int,
    ) -> bool:
        """Apply start/end change; returns True on success."""
        try:
            self.manager.update_bout_frames(name, bout_index, start, end)
        except (ValueError, IndexError) as e:
            QMessageBox.warning(self, "Edit Frames Failed", str(e))
            return False
        self.refresh(preserve_view=True)
        self.bouts_changed.emit()
        return True

    def _edit_single_frame_field(
        self,
        name: str,
        bout_index: int,
        current_start: int,
        current_end: int,
        field: str,
    ) -> None:
        """field is 'start' or 'end'."""
        lo, hi = self._frame_bounds()
        if field == "start":
            value, ok = QInputDialog.getInt(
                self,
                "Edit Start Frame",
                f"{name}: start frame (end stays {current_end}):",
                current_start,
                lo,
                hi,
            )
            if not ok:
                return
            new_start, new_end = value, current_end
        else:
            value, ok = QInputDialog.getInt(
                self,
                "Edit End Frame",
                f"{name}: end frame (start stays {current_start}):",
                current_end,
                lo,
                hi,
            )
            if not ok:
                return
            new_start, new_end = current_start, value

        if new_start > new_end:
            QMessageBox.warning(
                self,
                "Invalid Range",
                f"Start ({new_start}) cannot be after end ({new_end}).",
            )
            return
        self._apply_frame_edit(name, bout_index, new_start, new_end)

    def _edit_both_frames(
        self,
        name: str,
        bout_index: int,
        current_start: int,
        current_end: int,
    ) -> None:
        lo, hi = self._frame_bounds()
        start, ok = QInputDialog.getInt(
            self,
            "Edit Start Frame",
            f"{name}: start frame:",
            current_start,
            lo,
            hi,
        )
        if not ok:
            return
        end, ok = QInputDialog.getInt(
            self,
            "Edit End Frame",
            f"{name}: end frame (start = {start}):",
            max(current_end, start),
            lo,
            hi,
        )
        if not ok:
            return
        if start > end:
            QMessageBox.warning(
                self,
                "Invalid Range",
                f"Start ({start}) cannot be after end ({end}).",
            )
            return
        self._apply_frame_edit(name, bout_index, start, end)

    def _on_cell_double_clicked(self, row: int, column: int) -> None:
        meta = self._bout_frames_at_row(row)
        if meta is None:
            return
        name, bout_idx, start, end = meta
        if column == _COL_START:
            self._edit_single_frame_field(name, bout_idx, start, end, "start")
        elif column == _COL_END:
            self._edit_single_frame_field(name, bout_idx, start, end, "end")
        elif column == _COL_PARTNERS:
            # If this row is in a multi-selection, apply to all selected rows
            refs = self._selected_bout_refs()
            if not refs or (name, bout_idx) not in refs:
                refs = [(name, bout_idx)]
            self._set_partners_for_refs(refs)
        elif column == _COL_BEHAVIOR:
            # Double-click behavior name → edit both frames (handy shortcut)
            self._edit_both_frames(name, bout_idx, start, end)

    def _on_table_context_menu(self, pos) -> None:
        row = self.table.rowAt(pos.y())
        if row < 0:
            return
        # Keep multi-selection if right-click is inside it; otherwise select the row
        selected_rows = set(self._selected_rows())
        if row not in selected_rows:
            self.table.selectRow(row)
        meta = self._bout_frames_at_row(row)
        if meta is None:
            return
        name, bout_idx, start, end = meta
        n_sel = max(1, len(self._selected_bout_refs()))

        menu = QMenu(self)
        act_edit_start = menu.addAction("Edit start frame…")
        act_edit_end = menu.addAction("Edit end frame…")
        act_edit_both = menu.addAction("Edit start & end…")
        menu.addSeparator()
        act_set_start_cur = menu.addAction(
            f"Set start to current frame ({self._current_frame})"
        )
        act_set_end_cur = menu.addAction(
            f"Set end to current frame ({self._current_frame})"
        )
        menu.addSeparator()
        act_partners = menu.addAction(
            f"Set partners for {n_sel} selected bout(s)…"
        )
        act_clear_partners = menu.addAction(
            f"Clear partners for {n_sel} selected bout(s)"
        )
        menu.addSeparator()
        act_jump = menu.addAction("Jump to bout start")
        act_type = menu.addAction("Change type…")
        act_del = menu.addAction("Delete bout…")

        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == act_edit_start:
            self._edit_single_frame_field(name, bout_idx, start, end, "start")
        elif chosen == act_edit_end:
            self._edit_single_frame_field(name, bout_idx, start, end, "end")
        elif chosen == act_edit_both:
            self._edit_both_frames(name, bout_idx, start, end)
        elif chosen == act_set_start_cur:
            self._apply_frame_edit(name, bout_idx, self._current_frame, end)
        elif chosen == act_set_end_cur:
            self._apply_frame_edit(name, bout_idx, start, self._current_frame)
        elif chosen == act_partners:
            self.set_partners_selected()
        elif chosen == act_clear_partners:
            self.clear_partners_selected()
        elif chosen == act_jump:
            self.jump_requested.emit(start)
        elif chosen == act_type:
            self._change_bout_type(name, bout_idx)
        elif chosen == act_del:
            self._delete_bout(name, bout_idx)
    def edit_frames_selected(self) -> None:
        row = self.table.currentRow()
        meta = self._bout_frames_at_row(row)
        if meta is None:
            QMessageBox.information(self, "Edit Frames", "Select a bout first.")
            return
        name, bout_idx, start, end = meta
        self._edit_both_frames(name, bout_idx, start, end)

    def _initial_partners_for_refs(
        self, refs: Sequence[Tuple[str, int]]
    ) -> List[int]:
        """If all selected bouts share the same partners, pre-check those."""
        if not refs:
            return []
        common: Optional[List[int]] = None
        for name, index in refs:
            bouts = self.manager.get_bouts_for_behavior(name)
            if not (0 <= index < len(bouts)):
                continue
            pids = [int(x) for x in (bouts[index].partner_ids or [])]
            if common is None:
                common = pids
            elif common != pids:
                return []
        return list(common or [])

    def _set_partners_for_refs(
        self,
        refs: Sequence[Tuple[str, int]],
        *,
        clear: bool = False,
    ) -> None:
        if not refs:
            QMessageBox.information(
                self, "Set Partners", "Select one or more bouts first."
            )
            return
        subjects = list(self.manager.session.subjects)
        if not subjects and not clear:
            QMessageBox.warning(
                self,
                "Set Partners",
                "No subjects loaded. Load tracklets / subjects first.",
            )
            return

        if clear:
            partner_ids: List[int] = []
        else:
            dlg = PartnerPickDialog(
                subjects,
                active_subject_id=self.manager.session.active_subject_id,
                initial_ids=self._initial_partners_for_refs(refs),
                title=f"Set Partners ({len(refs)} bout(s))",
                parent=self,
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            partner_ids = dlg.selected_partner_ids()

        try:
            n = self.manager.update_bouts_partners(list(refs), partner_ids)
        except (ValueError, IndexError) as e:
            QMessageBox.warning(self, "Set Partners Failed", str(e))
            return
        self.refresh(preserve_view=True)
        self.bouts_changed.emit()
        if n == 0:
            self.status_label.setText(
                self.status_label.text() + "  ·  Partners unchanged"
            )

    def set_partners_selected(self) -> None:
        refs = self._selected_bout_refs()
        self._set_partners_for_refs(refs, clear=False)

    def clear_partners_selected(self) -> None:
        refs = self._selected_bout_refs()
        if not refs:
            QMessageBox.information(
                self, "Clear Partners", "Select one or more bouts first."
            )
            return
        if QMessageBox.question(
            self,
            "Clear Partners",
            f"Remove partners from {len(refs)} selected bout(s)?",
        ) != QMessageBox.StandardButton.Yes:
            return
        self._set_partners_for_refs(refs, clear=True)

    def _pick_new_type(self, current_name: str) -> Optional[str]:
        """Show a dialog to pick a new behavior type. Returns name or None if cancelled."""
        names = self._behavior_names()
        if not names:
            QMessageBox.warning(self, "Change Type", "No behaviors defined.")
            return None
        try:
            default_idx = names.index(current_name)
        except ValueError:
            default_idx = 0

        new_name, ok = QInputDialog.getItem(
            self,
            "Change Bout Type",
            f"Current type: {current_name}\nSelect new behavior type:",
            names,
            default_idx,
            False,
        )
        if not ok or not new_name:
            return None
        if new_name == current_name:
            QMessageBox.information(
                self, "Change Type", "That is already the current type."
            )
            return None
        return new_name

    def _change_bout_type(self, name: str, bout_index: int):
        new_name = self._pick_new_type(name)
        if new_name is None:
            return
        try:
            self.manager.change_bout_type(name, bout_index, new_name)
        except (ValueError, IndexError) as e:
            QMessageBox.warning(self, "Change Type Failed", str(e))
            return
        self.refresh()
        self.bouts_changed.emit()

    def change_type_selected(self):
        ref = self._selected_bout_ref()
        if ref is None:
            QMessageBox.information(self, "Change Type", "Select a bout first.")
            return
        name, idx = ref
        self._change_bout_type(name, idx)

    def _delete_bout(self, name: str, bout_index: int):
        if QMessageBox.question(
            self, "Delete Bout",
            f"Delete bout #{bout_index} for '{name}'?\n"
            "This cannot be undone from here (use Undo in menu)."
        ) != QMessageBox.StandardButton.Yes:
            return

        self.manager.delete_bout(name, bout_index)
        self.refresh()
        self.bouts_changed.emit()

    def delete_selected(self):
        refs = self._selected_bout_refs()
        if not refs:
            return
        if len(refs) == 1:
            name, idx = refs[0]
            self._delete_bout(name, idx)
            return
        if QMessageBox.question(
            self,
            "Delete Bouts",
            f"Delete {len(refs)} selected bouts?\n"
            "This cannot be undone from here (use Undo in menu).",
        ) != QMessageBox.StandardButton.Yes:
            return
        # Delete high indices first per behavior so remaining indices stay valid
        ordered = sorted(refs, key=lambda t: (t[0], -t[1]))
        for name, idx in ordered:
            try:
                self.manager.delete_bout(name, idx)
            except (ValueError, IndexError):
                continue
        self.refresh()
        self.bouts_changed.emit()

    def jump_to_selected(self):
        row = self.table.currentRow()
        if row < 0:
            return
        item = self.table.item(row, _COL_START)
        if item is None:
            return
        start = int(item.text())
        self.jump_requested.emit(start)

    def set_manager(self, manager: AnnotationManager):
        self.manager = manager
        self._filter_behavior = None
        self._rebuild_filter_combo()
        self.refresh()
