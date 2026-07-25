"""Subject selector + behavior-mode controls for multi-animal annotation."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
)

from LabGym.annotator.core.annotation_manager import AnnotationManager
from LabGym.annotator.core.data_models import (
    BEHAVIOR_MODE_INTERACTIVE_ADVANCED,
    BEHAVIOR_MODE_INTERACTIVE_BASIC,
    BEHAVIOR_MODE_NON_INTERACTIVE,
    Subject,
)

# re-export for mode checks

MODE_LABELS = [
    (BEHAVIOR_MODE_NON_INTERACTIVE, "Non-interactive (per subject)"),
    (BEHAVIOR_MODE_INTERACTIVE_BASIC, "Interactive basic (group)"),
    (BEHAVIOR_MODE_INTERACTIVE_ADVANCED, "Interactive advanced (roles)"),
]


class SubjectPanel(QWidget):
    """Lists subjects, allows selection, and sets LabGym behavior mode."""

    subject_changed = Signal(int)  # subject_id
    mode_changed = Signal(int)  # behavior_mode
    load_tracklets_requested = Signal()
    clear_tracklets_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager: Optional[AnnotationManager] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("Subjects / Tracks")
        box_l = QVBoxLayout(box)

        self.lbl_status = QLabel("No tracklets loaded — single subject mode")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #aaa; font-size: 11px;")
        box_l.addWidget(self.lbl_status)

        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(120)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        box_l.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load tracklets…")
        self.btn_load.setToolTip(
            "Load LabGym id_review tracklets (*_tracklets.npz) to enable multi-subject annotation and ID overlays"
        )
        self.btn_load.clicked.connect(self.load_tracklets_requested.emit)
        self.btn_prev = QPushButton("[")
        self.btn_prev.setFixedWidth(28)
        self.btn_prev.setToolTip("Previous subject ([)")
        self.btn_prev.clicked.connect(self.select_previous)
        self.btn_next = QPushButton("]")
        self.btn_next.setFixedWidth(28)
        self.btn_next.setToolTip("Next subject (])")
        self.btn_next.clicked.connect(self.select_next)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_prev)
        btn_row.addWidget(self.btn_next)
        box_l.addLayout(btn_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        for code, label in MODE_LABELS:
            self.mode_combo.addItem(label, code)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_index)
        mode_row.addWidget(self.mode_combo, 1)
        box_l.addLayout(mode_row)

        partner_row = QHBoxLayout()
        partner_row.addWidget(QLabel("Partner(s):"))
        self.partner_combo = QComboBox()
        self.partner_combo.setToolTip(
            "Interactive advanced: optional partner subject for the next bout "
            "(stored as partner_ids on the bout)."
        )
        self.partner_combo.addItem("(none)", None)
        partner_row.addWidget(self.partner_combo, 1)
        box_l.addLayout(partner_row)

        layout.addWidget(box)

        self.partner_ids_changed = None  # unused; use get_partner_ids()

    def set_manager(self, manager: Optional[AnnotationManager]) -> None:
        self.manager = manager
        self.refresh()

    def refresh(self) -> None:
        self.list_widget.clear()
        if not self.manager:
            self.lbl_status.setText("No session")
            return

        subjects: List[Subject] = list(self.manager.session.subjects)
        active = self.manager.session.active_subject_id
        tracks = self.manager.session.tracks_ref
        if tracks and tracks.path:
            self.lbl_status.setText(
                f"{len(subjects)} subject(s) · analysis offset "
                f"{tracks.analysis_start_frame} · {Path_short(tracks.path)}"
            )
        else:
            self.lbl_status.setText(
                f"{len(subjects)} subject(s) — load tracklets for ID overlays"
            )

        for subj in subjects:
            item = QListWidgetItem(f"{subj.display_name}  (id={subj.subject_id})")
            item.setData(Qt.ItemDataRole.UserRole, int(subj.subject_id))
            item.setForeground(QColor(subj.color))
            self.list_widget.addItem(item)
            if subj.subject_id == active:
                self.list_widget.setCurrentItem(item)

        # Partner combo (exclude active subject)
        prev = self.partner_combo.currentData()
        self.partner_combo.blockSignals(True)
        self.partner_combo.clear()
        self.partner_combo.addItem("(none)", None)
        for subj in subjects:
            if subj.subject_id == active:
                continue
            self.partner_combo.addItem(subj.display_name, int(subj.subject_id))
        # restore previous partner if still valid
        if prev is not None:
            pi = self.partner_combo.findData(prev)
            if pi >= 0:
                self.partner_combo.setCurrentIndex(pi)
        self.partner_combo.blockSignals(False)

        # Mode combo
        mode = int(self.manager.session.behavior_mode)
        idx = self.mode_combo.findData(mode)
        if idx < 0:
            idx = 0
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(idx)
        self.mode_combo.blockSignals(False)

        # Group mode: dim per-subject list hint
        if mode == BEHAVIOR_MODE_INTERACTIVE_BASIC:
            self.lbl_status.setText(
                self.lbl_status.text()
                + "\nInteractive basic: annotate group-level behaviors "
                "(stored under interaction_bouts)."
            )
        elif mode == BEHAVIOR_MODE_INTERACTIVE_ADVANCED:
            self.lbl_status.setText(
                self.lbl_status.text()
                + "\nInteractive advanced: set partner(s) for role bouts."
            )
        advanced = mode == BEHAVIOR_MODE_INTERACTIVE_ADVANCED
        self.partner_combo.setEnabled(advanced)

    def get_partner_ids(self) -> List[int]:
        data = self.partner_combo.currentData()
        if data is None:
            return []
        return [int(data)]

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        sid = int(item.data(Qt.ItemDataRole.UserRole))
        if self.manager:
            try:
                self.manager.set_active_subject(sid)
            except ValueError:
                return
        self.subject_changed.emit(sid)

    def _on_mode_index(self, _index: int) -> None:
        code = self.mode_combo.currentData()
        if code is None or not self.manager:
            return
        self.manager.session.behavior_mode = int(code)
        self.mode_changed.emit(int(code))
        self.refresh()

    def select_previous(self) -> None:
        self._cycle(-1)

    def select_next(self) -> None:
        self._cycle(1)

    def _cycle(self, delta: int) -> None:
        if not self.manager or not self.manager.session.subjects:
            return
        ids = [s.subject_id for s in self.manager.session.subjects]
        try:
            i = ids.index(self.manager.session.active_subject_id)
        except ValueError:
            i = 0
        i = (i + delta) % len(ids)
        sid = ids[i]
        self.manager.set_active_subject(sid)
        self.refresh()
        self.subject_changed.emit(sid)


def Path_short(path: str, max_len: int = 48) -> str:
    p = str(path)
    if len(p) <= max_len:
        return p
    return "…" + p[-(max_len - 1) :]
