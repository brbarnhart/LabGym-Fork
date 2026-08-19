"""Modeless dialog: extract detector training images from Review IDs ranges."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class HardCaseExtractDialog(QDialog):
    """Form for selecting ranges and writing detector training JPGs.

    Modeless so the Review IDs timeline can still be scrubbed while open.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Hard cases → detector training images")
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setMinimumWidth(440)
        self.resize(480, 520)

        root = QVBoxLayout(self)

        hard_help = QLabel(
            "Select time periods where the detector struggled. Frames from those "
            "ranges are written as JPGs you can annotate (EZannot) and use to "
            "retrain the detector."
        )
        hard_help.setWordWrap(True)
        root.addWidget(hard_help)

        self.lbl_range_pending = QLabel("Range start: (not set)")
        root.addWidget(self.lbl_range_pending)

        range_btns = QHBoxLayout()
        self.btn_range_start = QPushButton("Set start here ([)")
        self.btn_range_start.setToolTip(
            "Mark the current analysis frame as the start of a training range."
        )
        self.btn_range_end = QPushButton("Set end & add (])")
        self.btn_range_end.setToolTip(
            "Close the range at the current frame and add it to the list."
        )
        self.btn_add_risk = QPushButton("Add current risk band")
        self.btn_add_risk.setToolTip(
            "If the playhead is inside an automatic contact-risk band, add that "
            "whole band as a training range."
        )
        range_btns.addWidget(self.btn_range_start)
        range_btns.addWidget(self.btn_range_end)
        range_btns.addWidget(self.btn_add_risk)
        root.addLayout(range_btns)

        self.list_ranges = QListWidget()
        self.list_ranges.setObjectName("list_ranges")
        self.list_ranges.setMinimumHeight(72)
        self.list_ranges.setToolTip("Analysis-frame ranges that will be exported.")
        root.addWidget(self.list_ranges)

        range_edit = QHBoxLayout()
        self.btn_remove_range = QPushButton("Remove selected")
        self.btn_clear_ranges = QPushButton("Clear all")
        range_edit.addWidget(self.btn_remove_range)
        range_edit.addWidget(self.btn_clear_ranges)
        range_edit.addStretch(1)
        root.addLayout(range_edit)

        out_row = QHBoxLayout()
        self.ed_hard_out = QLineEdit()
        self.ed_hard_out.setObjectName("ed_hard_out")
        self.ed_hard_out.setToolTip(
            "Folder for extracted JPGs (same default as Generate images: "
            "project/detector_training_images)."
        )
        self.btn_hard_out = QPushButton("Browse…")
        out_row.addWidget(QLabel("Output:"), 0)
        out_row.addWidget(self.ed_hard_out, 1)
        out_row.addWidget(self.btn_hard_out, 0)
        root.addLayout(out_row)

        opts = QHBoxLayout()
        self.spin_hard_skip = QSpinBox()
        self.spin_hard_skip.setObjectName("spin_hard_skip")
        self.spin_hard_skip.setRange(1, 1_000_000)
        self.spin_hard_skip.setValue(10)
        self.spin_hard_skip.setToolTip(
            "Write one image every N analysis frames within each range "
            "(start and end of each range are always kept). "
            "Smaller values = denser sampling of hard cases."
        )
        opts.addWidget(QLabel("Skip (frames):"))
        opts.addWidget(self.spin_hard_skip)
        self.chk_hard_resize = QCheckBox("Resize width")
        self.chk_hard_resize.setToolTip(
            "Optional proportional resize before writing JPGs."
        )
        self.spin_hard_width = QSpinBox()
        self.spin_hard_width.setRange(10, 10000)
        self.spin_hard_width.setValue(480)
        self.spin_hard_width.setEnabled(False)
        self.chk_hard_resize.toggled.connect(self.spin_hard_width.setEnabled)
        opts.addWidget(self.chk_hard_resize)
        opts.addWidget(self.spin_hard_width)
        opts.addStretch(1)
        root.addLayout(opts)

        self.btn_gen_hard = QPushButton("Generate training images from ranges")
        self.btn_gen_hard.setObjectName("btn_gen_hard")
        self.btn_gen_hard.setToolTip(
            "Extract still frames from the source video for every listed range."
        )
        root.addWidget(self.btn_gen_hard)

        self.lbl_hard_status = QLabel("")
        self.lbl_hard_status.setWordWrap(True)
        root.addWidget(self.lbl_hard_status)

    def set_extract_enabled(self, on: bool) -> None:
        """Enable or disable extract form controls.

        Args:
            on: True when a review package is loaded.
        """
        for w in (
            self.btn_range_start,
            self.btn_range_end,
            self.btn_add_risk,
            self.btn_remove_range,
            self.btn_clear_ranges,
            self.btn_gen_hard,
            self.list_ranges,
            self.ed_hard_out,
            self.btn_hard_out,
            self.spin_hard_skip,
            self.chk_hard_resize,
        ):
            w.setEnabled(on)
        if on:
            self.spin_hard_width.setEnabled(self.chk_hard_resize.isChecked())
        else:
            self.spin_hard_width.setEnabled(False)
