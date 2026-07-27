"""Tools → Dense generate + sort examples (classic LabGym pipeline pop-out)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.tools_windows.dense_backend import (
    DenseGenerateConfig,
    load_detector_animal_kinds,
    run_dense_generate,
    run_manual_sort,
)
from LabGym.gui_pyside.widgets.path_browse import (
    browse_existing_directory,
    set_line_edit_directory,
)

_VIDEO_FILTER = (
    "Video files (*.avi *.mpg *.mpeg *.mp4 *.mkv *.m4v *.mov *.wmv);;All files (*.*)"
)


class _Worker(QObject):
    finished = Signal(str)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            def _prog(msg: str) -> None:
                self.progress.emit(msg)

            self.kwargs["progress"] = _prog
            result = self.fn(*self.args, **self.kwargs)
            self.finished.emit("" if result is None else str(result))
        except Exception as exc:
            self.error.emit(str(exc))


def _row(line: QLineEdit, button: QPushButton) -> QWidget:
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(line, 1)
    h.addWidget(button)
    return w


class DenseGenerateTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._videos: List[str] = []
        self._thread: Optional[QThread] = None
        self._animal_kinds: List[str] = []

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Classic dense generation: write <b>unsorted</b> animation+pattern pairs "
                "for later sorting. Prefer ethogram-first under Categorizer when possible."
            )
        )

        form = QFormLayout()
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItem("Non-interactive (mode 0)", 0)
        self.cmb_mode.addItem("Interactive basic (mode 1)", 1)
        self.cmb_mode.addItem("Interactive advanced (mode 2)", 2)
        self.cmb_mode.addItem("Static images (mode 3)", 3)
        form.addRow("Behavior mode:", self.cmb_mode)

        self.ed_videos = QLineEdit()
        self.ed_videos.setReadOnly(True)
        b_v = QPushButton("Browse…")
        b_v.clicked.connect(self._browse_videos)
        form.addRow("Videos / images:", _row(self.ed_videos, b_v))

        self.ed_out = QLineEdit()
        b_o = QPushButton("Browse…")
        b_o.clicked.connect(lambda: self._browse_dir(self.ed_out))
        form.addRow("Output folder:", _row(self.ed_out, b_o))

        self.chk_detector = QCheckBox("Use detector (recommended)")
        self.chk_detector.setChecked(True)
        form.addRow("", self.chk_detector)

        self.ed_detector = QLineEdit()
        b_d = QPushButton("Browse…")
        b_d.clicked.connect(self._browse_detector)
        form.addRow("Detector folder:", _row(self.ed_detector, b_d))
        self.lbl_kinds = QLabel("Animal kinds: (none)")
        form.addRow("", self.lbl_kinds)

        self.spin_animals = QSpinBox()
        self.spin_animals.setRange(1, 1000)
        self.spin_animals.setValue(1)
        self.spin_animals.setToolTip(
            "Animals per category when using a detector (same count applied to each kind)."
        )
        form.addRow("Animals per kind:", self.spin_animals)

        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0, 1e9)
        self.spin_start.setDecimals(2)
        self.spin_start.setSuffix(" s")
        form.addRow("Start time:", self.spin_start)

        self.spin_duration = QDoubleSpinBox()
        self.spin_duration.setRange(0, 1e9)
        self.spin_duration.setDecimals(2)
        self.spin_duration.setSuffix(" s")
        self.spin_duration.setToolTip("0 = to end of video.")
        form.addRow("Duration (0=end):", self.spin_duration)

        self.spin_length = QSpinBox()
        self.spin_length.setRange(3, 1000)
        self.spin_length.setValue(15)
        form.addRow("Example length (frames):", self.spin_length)

        self.spin_skip = QSpinBox()
        self.spin_skip.setRange(0, 1_000_000)
        self.spin_skip.setValue(1)
        form.addRow("Skip frames between examples:", self.spin_skip)

        self.spin_social = QDoubleSpinBox()
        self.spin_social.setRange(0, 1e6)
        self.spin_social.setDecimals(2)
        self.spin_social.setToolTip("Mode 2 interaction distance (0 = infinite).")
        form.addRow("Social distance (mode 2):", self.spin_social)

        self.chk_bg_free = QCheckBox("Background-free animations")
        self.chk_bg_free.setChecked(True)
        form.addRow("", self.chk_bg_free)

        self.chk_black_bg = QCheckBox("Black background when background-free")
        self.chk_black_bg.setChecked(True)
        form.addRow("", self.chk_black_bg)

        self.chk_bodyparts = QCheckBox("Include body parts in pattern images")
        form.addRow("", self.chk_bodyparts)

        self.spin_std = QSpinBox()
        self.spin_std.setRange(0, 255)
        self.spin_std.setValue(0)
        form.addRow("Body-part STD:", self.spin_std)

        self.chk_resize = QCheckBox("Resize frame width")
        self.spin_width = QSpinBox()
        self.spin_width.setRange(10, 10000)
        self.spin_width.setValue(480)
        self.spin_width.setEnabled(False)
        self.chk_resize.toggled.connect(self.spin_width.setEnabled)
        rw = QHBoxLayout()
        rw.addWidget(self.chk_resize)
        rw.addWidget(self.spin_width)
        rw.addStretch(1)
        form.addRow("Resize:", rw)

        layout.addLayout(form)

        self.btn = QPushButton("Start dense generation")
        self.btn.clicked.connect(self._run)
        layout.addWidget(self.btn)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

    def _browse_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select videos / images", "", _VIDEO_FILTER)
        if paths:
            self._videos = paths
            self.ed_videos.setText(f"{len(paths)} file(s)")

    def _browse_dir(self, edit: QLineEdit) -> None:
        set_line_edit_directory(self, edit, caption="Select folder")

    def _browse_detector(self) -> None:
        d = browse_existing_directory(
            self, self.ed_detector.text(), "Select detector folder"
        )
        if not d:
            return
        self.ed_detector.setText(d)
        kinds = load_detector_animal_kinds(d)
        self._animal_kinds = kinds
        self.lbl_kinds.setText("Animal kinds: " + (str(kinds) if kinds else "(none found)"))

    def _run(self) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "Busy", "Generation already running.")
            return
        if not self._videos or not self.ed_out.text().strip():
            QMessageBox.warning(self, "Missing paths", "Select videos and an output folder.")
            return
        use_det = self.chk_detector.isChecked()
        det = self.ed_detector.text().strip()
        if use_det and not det:
            QMessageBox.warning(self, "Detector", "Select a detector folder or uncheck Use detector.")
            return
        if use_det and not self._animal_kinds:
            self._animal_kinds = load_detector_animal_kinds(det)
        if use_det and not self._animal_kinds:
            QMessageBox.warning(
                self,
                "Detector",
                "Could not read animal_names from model_parameters.txt.",
            )
            return

        animal_number: int | Dict[str, int]
        if use_det:
            n = int(self.spin_animals.value())
            animal_number = {k: n for k in self._animal_kinds}
        else:
            animal_number = int(self.spin_animals.value())

        reply = QMessageBox.question(
            self,
            "Start dense generation?",
            f"Generate unsorted examples from {len(self._videos)} file(s)?",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        cfg = DenseGenerateConfig(
            path_to_videos=list(self._videos),
            result_path=self.ed_out.text().strip(),
            behavior_mode=int(self.cmb_mode.currentData()),
            use_detector=use_det,
            path_to_detector=det or None,
            animal_kinds=list(self._animal_kinds),
            animal_number=animal_number,
            framewidth=self.spin_width.value() if self.chk_resize.isChecked() else None,
            t=float(self.spin_start.value()),
            duration=float(self.spin_duration.value()),
            length=int(self.spin_length.value()),
            skip_redundant=int(self.spin_skip.value()),
            social_distance=float(self.spin_social.value()),
            include_bodyparts=self.chk_bodyparts.isChecked(),
            std=int(self.spin_std.value()),
            background_free=self.chk_bg_free.isChecked(),
            black_background=self.chk_black_bg.isChecked(),
        )
        self._start_worker(run_dense_generate, cfg)

    def _start_worker(self, fn, *args) -> None:
        worker = _Worker(fn, *args)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.log.append)
        worker.finished.connect(self._on_done)
        worker.error.connect(self._on_err)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._clear)
        self._thread = thread
        self.btn.setEnabled(False)
        thread.start()

    def _clear(self) -> None:
        self._thread = None
        self.btn.setEnabled(True)

    def _on_done(self, _msg: str) -> None:
        self.log.append("Done.")
        QMessageBox.information(
            self,
            "Generation complete",
            "Unsorted examples are ready. Use the Sort tabs to organize them.",
        )

    def _on_err(self, err: str) -> None:
        self.log.append(f"Error: {err}")
        QMessageBox.critical(self, "Generation failed", err)


class DenseSortManualTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Interactive sort with OpenCV (same keys as classic LabGym): "
                "<b>o</b> prev, <b>p</b> next, <b>q</b> quit, <b>u</b> undo, "
                "and your behavior keys."
            )
        )
        form = QFormLayout()
        self.ed_in = QLineEdit()
        b_in = QPushButton("Browse…")
        b_in.clicked.connect(lambda: self._dir(self.ed_in))
        form.addRow("Unsorted examples:", _row(self.ed_in, b_in))
        self.ed_out = QLineEdit()
        b_out = QPushButton("Browse…")
        b_out.clicked.connect(lambda: self._dir(self.ed_out))
        form.addRow("Sorted output:", _row(self.ed_out, b_out))
        self.ed_keys = QLineEdit()
        self.ed_keys.setPlaceholderText("a-walk,b-rear,c-groom")
        self.ed_keys.setToolTip('Format: key-behavior,key-behavior,…  Reserved: o p q u')
        form.addRow("Key-behavior pairs:", self.ed_keys)
        layout.addLayout(form)
        self.btn = QPushButton("Start interactive sort")
        self.btn.clicked.connect(self._run)
        layout.addWidget(self.btn)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

    def _dir(self, edit: QLineEdit) -> None:
        set_line_edit_directory(self, edit, caption="Select folder")

    def _parse_keys(self) -> Dict[str, str]:
        text = self.ed_keys.text().strip()
        if not text:
            raise ValueError("Enter key-behavior pairs.")
        out: Dict[str, str] = {}
        for pair in text.split(","):
            pair = pair.strip()
            if not pair:
                continue
            key, name = pair.split("-", 1)
            key = key.strip()
            name = name.strip()
            if len(key) != 1:
                raise ValueError(f"Key must be one character: {key!r}")
            out[key] = name
        if not out:
            raise ValueError("No valid key-behavior pairs.")
        return out

    def _run(self) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "Busy", "Sort already running.")
            return
        inp = self.ed_in.text().strip()
        out = self.ed_out.text().strip()
        if not inp or not out:
            QMessageBox.warning(self, "Missing paths", "Select input and output folders.")
            return
        try:
            keys = self._parse_keys()
        except Exception as exc:
            QMessageBox.warning(self, "Keys", str(exc))
            return

        worker = _Worker(run_manual_sort, inp, out, keys)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.log.append)
        worker.finished.connect(lambda _m: self._done())
        worker.error.connect(self._err)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._clear)
        self._thread = thread
        self.btn.setEnabled(False)
        thread.start()

    def _clear(self) -> None:
        self._thread = None
        self.btn.setEnabled(True)

    def _done(self) -> None:
        self.log.append("Interactive sort finished.")
        QMessageBox.information(self, "Sort complete", "Manual sorting session ended.")

    def _err(self, err: str) -> None:
        self.log.append(f"Error: {err}")
        QMessageBox.critical(self, "Sort failed", err)


class DenseSortFileTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Sort unsorted LabGym examples using a CSV of frame labels, a "
                "subject-aware CSV, or a Behavior Annotator <code>.annotations.json</code>."
            )
        )
        form = QFormLayout()
        self.ed_in = QLineEdit()
        b_in = QPushButton("Browse…")
        b_in.clicked.connect(lambda: self._dir(self.ed_in))
        form.addRow("Unsorted examples folder:", _row(self.ed_in, b_in))
        self.ed_out = QLineEdit()
        b_out = QPushButton("Browse…")
        b_out.clicked.connect(lambda: self._dir(self.ed_out))
        form.addRow("Sorted output folder:", _row(self.ed_out, b_out))
        self.ed_ann = QLineEdit()
        b_ann = QPushButton("Browse…")
        b_ann.clicked.connect(self._browse_ann)
        form.addRow("Annotations JSON (optional):", _row(self.ed_ann, b_ann))
        layout.addLayout(form)

        row = QHBoxLayout()
        self.btn_csv = QPushButton("Sort from CSV in folder")
        self.btn_csv.setToolTip("CSV must sit with the unsorted examples.")
        self.btn_csv.clicked.connect(self._sort_csv)
        self.btn_subj = QPushButton("Sort subject-aware CSV")
        self.btn_subj.clicked.connect(self._sort_subject)
        self.btn_ann = QPushButton("Sort from annotations JSON")
        self.btn_ann.clicked.connect(self._sort_ann)
        row.addWidget(self.btn_csv)
        row.addWidget(self.btn_subj)
        row.addWidget(self.btn_ann)
        layout.addLayout(row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log, 1)

    def _dir(self, edit: QLineEdit) -> None:
        set_line_edit_directory(self, edit, caption="Select folder")

    def _browse_ann(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select annotations JSON", "", "JSON (*.json);;All (*.*)"
        )
        if p:
            self.ed_ann.setText(p)

    def _paths(self):
        inp = self.ed_in.text().strip()
        out = self.ed_out.text().strip()
        if not inp or not out:
            QMessageBox.warning(self, "Missing paths", "Select input and output folders.")
            return None, None
        return inp, out

    def _start(self, fn, *args) -> None:
        if self._thread is not None:
            QMessageBox.warning(self, "Busy", "Sort already running.")
            return
        worker = _Worker(fn, *args)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.log.append)
        worker.finished.connect(self._done)
        worker.error.connect(self._err)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._clear)
        self._thread = thread
        for b in (self.btn_csv, self.btn_subj, self.btn_ann):
            b.setEnabled(False)
        thread.start()

    def _clear(self) -> None:
        self._thread = None
        for b in (self.btn_csv, self.btn_subj, self.btn_ann):
            b.setEnabled(True)

    def _done(self, msg: str) -> None:
        self.log.append("Done. " + (msg or ""))
        QMessageBox.information(self, "Sort complete", msg or "Sort finished.")

    def _err(self, err: str) -> None:
        self.log.append(f"Error: {err}")
        QMessageBox.critical(self, "Sort failed", err)

    def _sort_csv(self) -> None:
        inp, out = self._paths()
        if not inp:
            return

        def job(progress=None):
            from LabGym.tools import sort_examples_from_csv

            if progress:
                progress("Sorting from CSV…")
            sort_examples_from_csv(inp, out)
            return out

        self._start(job)

    def _sort_subject(self) -> None:
        inp, out = self._paths()
        if not inp:
            return

        def job(progress=None):
            from LabGym.tools import sort_examples_from_csv_subject_aware

            if progress:
                progress("Subject-aware CSV sort…")
            counts = sort_examples_from_csv_subject_aware(inp, out)
            return str(counts)

        self._start(job)

    def _sort_ann(self) -> None:
        inp, out = self._paths()
        if not inp:
            return
        ann = self.ed_ann.text().strip()
        if not ann:
            QMessageBox.warning(self, "Annotations", "Select an annotations JSON file.")
            return

        def job(progress=None):
            from LabGym.tools import sort_examples_from_annotations

            if progress:
                progress("Sorting from annotations…")
            counts = sort_examples_from_annotations(ann, inp, out)
            return str(counts)

        self._start(job)


class DenseGenerateSortWindow(QMainWindow):
    """Pop-out window for classic dense generate + sort."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dense generate + sort examples (classic LabGym)")
        self.resize(820, 720)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        intro = QLabel(
            "Power-user / classic pipeline. The main Categorizer workflow remains "
            "<b>ethogram-first</b> (Annotate → Generate sorted pairs)."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        tabs = QTabWidget()
        tabs.addTab(DenseGenerateTab(), "Generate unsorted")
        tabs.addTab(DenseSortManualTab(), "Sort (manual keys)")
        tabs.addTab(DenseSortFileTab(), "Sort (CSV / annotations)")
        layout.addWidget(tabs, 1)
