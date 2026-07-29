"""Manage dataset → Evaluate: run metrics, browse eval runs, light compare."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.widgets.path_browse import path_edit_row, set_line_edit_directory


class _EvalWorker(QObject):
    """Run ``Categorizers.test_categorizer`` off the UI thread."""

    finished = Signal(str)  # run_dir or empty
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, groundtruth: str, model: str, result: Optional[str] = None):
        super().__init__()
        self.groundtruth = groundtruth
        self.model = model
        self.result = result

    def run(self) -> None:
        try:
            from LabGym.categorizer import Categorizers

            self.progress.emit("Scoring categorizer (shared evaluation engine)…")
            ca = Categorizers()
            run_dir = ca.test_categorizer(
                self.groundtruth, self.model, result_path=self.result
            )
            self.finished.emit(str(run_dir or ""))
        except Exception as exc:
            self.error.emit(str(exc))


class _CompareWorker(QObject):
    """Score several already-trained models on one ground-truth folder."""

    finished = Signal(list)  # list of (model, run_dir|None, error|None)
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, groundtruth: str, models: List[str]):
        super().__init__()
        self.groundtruth = groundtruth
        self.models = models

    def run(self) -> None:
        try:
            from LabGym.categorizer import Categorizers

            ca = Categorizers()
            results: List[tuple] = []
            for i, model in enumerate(self.models, start=1):
                self.progress.emit(
                    f"[{i}/{len(self.models)}] Evaluating {Path(model).name}…"
                )
                try:
                    run_dir = ca.test_categorizer(self.groundtruth, model)
                    results.append((model, str(run_dir or ""), None))
                except Exception as exc:
                    results.append((model, "", str(exc)))
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


class EvaluateTab(QWidget):
    """Run and browse categorizer evaluation artifacts under model/eval/."""

    def __init__(self, project: ProjectController, parent=None):
        super().__init__(parent)
        self.project = project
        self._thread: Optional[QThread] = None
        self._loaded: Dict[str, Any] = {}
        self._compare_rows: List[Dict[str, Any]] = []

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Evaluate already-trained categorizers on a declared ground-truth "
            "folder (prefer sealed test or a dedicated test store). Metrics use "
            "the same engine as Test categorizer. Artifacts live under "
            "<code>model/eval/&lt;run_id&gt;/</code>."
        )
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(intro)

        self.drift_banner = QLabel("")
        self.drift_banner.setWordWrap(True)
        self.drift_banner.setStyleSheet(
            "QLabel { background: #fff3cd; color: #664d03; padding: 6px; "
            "border: 1px solid #ffecb5; border-radius: 3px; }"
        )
        self.drift_banner.hide()
        layout.addWidget(self.drift_banner)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.ed_model = QLineEdit()
        self.ed_model.setToolTip(
            "Trained categorizer folder (must include model_parameters.txt)."
        )
        b_model = QPushButton("Browse…")
        b_model.clicked.connect(lambda: set_line_edit_directory(self, self.ed_model))
        form.addRow(
            self._lab("Categorizer folder:", "Primary model for run / browse."),
            path_edit_row(self.ed_model, b_model),
        )

        self.ed_gt = QLineEdit()
        self.ed_gt.setToolTip(
            "Ground-truth examples with one subfolder per behavior category "
            "(same layout as training examples)."
        )
        b_gt = QPushButton("Browse…")
        b_gt.clicked.connect(lambda: set_line_edit_directory(self, self.ed_gt))
        form.addRow(
            self._lab("Ground-truth examples:", "Folder used for scoring."),
            path_edit_row(self.ed_gt, b_gt),
        )
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run evaluation")
        self.btn_run.setToolTip("Score the categorizer and write a new eval run.")
        self.btn_run.clicked.connect(self._run_evaluation)
        self.btn_refresh = QPushButton("Refresh runs")
        self.btn_refresh.setToolTip("Reload eval/ directory for the selected model.")
        self.btn_refresh.clicked.connect(self._refresh_runs)
        self.btn_open = QPushButton("Open run folder…")
        self.btn_open.clicked.connect(self._open_run_folder)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_refresh)
        btn_row.addWidget(self.btn_open)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(0, 0, 0, 0)
        left_l.addWidget(QLabel("Evaluation runs (newest first)"))
        self.run_list = QListWidget()
        self.run_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.run_list.currentItemChanged.connect(self._on_run_selected)
        left_l.addWidget(self.run_list, 1)

        compare_box = QGroupBox("Compare already-trained models")
        compare_box.setToolTip(
            "Score multiple trained categorizers on the same ground-truth folder. "
            "Not a hyperparameter sweep."
        )
        cb_l = QVBoxLayout(compare_box)
        self.compare_models = QListWidget()
        self.compare_models.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self.compare_models.setMaximumHeight(100)
        cb_l.addWidget(self.compare_models)
        cb_btns = QHBoxLayout()
        btn_scan = QPushButton("Scan project models")
        btn_scan.clicked.connect(self._scan_models_for_compare)
        self.btn_compare = QPushButton("Compare selected")
        self.btn_compare.clicked.connect(self._run_compare)
        cb_btns.addWidget(btn_scan)
        cb_btns.addWidget(self.btn_compare)
        cb_l.addLayout(cb_btns)
        left_l.addWidget(compare_box)

        split.addWidget(left)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(0, 0, 0, 0)

        self.summary_label = QLabel("Select a run to view metrics.")
        self.summary_label.setWordWrap(True)
        right_l.addWidget(self.summary_label)

        self.detail_tabs = QTabWidget()
        self.tbl_f1 = self._make_table(["Behavior", "F1"])
        self.tbl_pairs = self._make_table(["True", "Predicted", "Count"])
        self.tbl_confusion = QTableWidget(0, 0)
        self.tbl_confusion.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self.tbl_report = self._make_table(["Label", "precision", "recall", "f1-score", "support"])
        self.tbl_pred = self._make_table(
            ["example_id", "true_label", "pred_label", "confidence", "misclassified"]
        )
        self.chk_mis_only = QCheckBox("Misclassified only")
        self.chk_mis_only.setChecked(True)
        self.chk_mis_only.toggled.connect(self._reload_predictions_view)
        pred_wrap = QWidget()
        pred_l = QVBoxLayout(pred_wrap)
        pred_l.setContentsMargins(0, 0, 0, 0)
        pred_l.addWidget(self.chk_mis_only)
        pred_l.addWidget(self.tbl_pred, 1)

        self.tbl_high_loss = self._make_table(
            ["rank", "example_id", "loss", "true_label", "pred_label"]
        )
        self.tbl_settings = self._make_table(["Setting", "Value"])
        self.tbl_compare = self._make_table(
            [
                "Model",
                "macro_f1",
                "n_examples",
                "n_misclassified",
                "time_step",
                "network",
                "label_mode",
                "run_id",
            ]
        )
        self.log = QTextEdit()
        self.log.setReadOnly(True)

        self.detail_tabs.addTab(self.tbl_f1, "Per-class F1")
        self.detail_tabs.addTab(self.tbl_pairs, "Confused pairs")
        self.detail_tabs.addTab(self.tbl_confusion, "Confusion")
        self.detail_tabs.addTab(self.tbl_report, "Report")
        self.detail_tabs.addTab(pred_wrap, "Predictions")
        self.detail_tabs.addTab(self.tbl_high_loss, "High-loss")
        self.detail_tabs.addTab(self.tbl_settings, "Settings")
        self.detail_tabs.addTab(self.tbl_compare, "Compare")
        self.detail_tabs.addTab(self.log, "Log")
        right_l.addWidget(self.detail_tabs, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        layout.addWidget(split, 1)

        self.ed_model.editingFinished.connect(self._on_model_path_changed)
        self.ed_gt.editingFinished.connect(self._update_drift_banner)
        self.project.project_replaced.connect(self._apply_project_defaults)
        self._apply_project_defaults()

    @staticmethod
    def _lab(text: str, tip: str) -> QLabel:
        lab = QLabel(text)
        lab.setToolTip(tip)
        return lab

    @staticmethod
    def _make_table(headers: List[str]) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        t.horizontalHeader().setStretchLastSection(True)
        t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        t.setAlternatingRowColors(True)
        return t

    def _apply_project_defaults(self) -> None:
        p = self.project.project
        name = (p.defaults.categorizer_name or "").strip()
        if name:
            self.ed_model.setText(name)
        try:
            ex = p.resolve_path(p.paths.examples_root or "examples")
            if not self.ed_gt.text().strip() and Path(ex).is_dir():
                self.ed_gt.setText(str(ex))
        except Exception:
            pass
        self._on_model_path_changed()
        self._scan_models_for_compare()

    def _on_model_path_changed(self) -> None:
        self._refresh_runs()
        self._update_drift_banner()

    def _update_drift_banner(self) -> None:
        model = self.ed_model.text().strip()
        gt = self.ed_gt.text().strip()
        if not model or not gt:
            self.drift_banner.hide()
            return
        try:
            from LabGym.training.evaluation import (
                format_taxonomy_drift_message,
                model_classnames_from_parameters,
                store_behavior_categories,
                taxonomy_drift,
            )

            drift = taxonomy_drift(
                model_classnames_from_parameters(model),
                store_behavior_categories(gt),
            )
            msg = format_taxonomy_drift_message(drift)
            if msg:
                self.drift_banner.setText(msg)
                self.drift_banner.show()
            else:
                self.drift_banner.hide()
        except Exception:
            self.drift_banner.hide()

    def _set_busy(self, busy: bool) -> None:
        self.btn_run.setEnabled(not busy)
        self.btn_compare.setEnabled(not busy)
        self.btn_refresh.setEnabled(not busy)

    def _run_evaluation(self) -> None:
        if self._thread is not None:
            return
        model = self.ed_model.text().strip()
        gt = self.ed_gt.text().strip()
        if not model or not gt:
            QMessageBox.warning(
                self, "Evaluate", "Set categorizer folder and ground-truth examples."
            )
            return
        if not Path(model).is_dir():
            QMessageBox.warning(self, "Evaluate", f"Model folder not found:\n{model}")
            return
        if not Path(gt).is_dir():
            QMessageBox.warning(self, "Evaluate", f"Ground-truth folder not found:\n{gt}")
            return
        self._update_drift_banner()
        self._set_busy(True)
        self.log.append(f"Starting evaluation…\n  model={model}\n  gt={gt}")
        self._thread = QThread(self)
        worker = _EvalWorker(gt, model)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(lambda m: self.log.append(m))
        worker.finished.connect(self._eval_done)
        worker.error.connect(self._eval_err)
        worker.finished.connect(self._thread.quit)
        worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._worker = worker
        self._thread.start()

    def _cleanup_thread(self) -> None:
        self._thread = None
        self._set_busy(False)

    def _eval_done(self, run_dir: str) -> None:
        if run_dir:
            self.log.append(f"Done → {run_dir}")
            self._refresh_runs()
            self._select_run_dir(run_dir)
            QMessageBox.information(self, "Evaluate", f"Evaluation run saved:\n{run_dir}")
        else:
            self.log.append(
                "Evaluation finished without a run directory "
                "(category names may not match the model)."
            )
            QMessageBox.warning(
                self,
                "Evaluate",
                "No evaluation run was written. Check that ground-truth folder "
                "names match the categorizer class list (see Log / console).",
            )

    def _eval_err(self, msg: str) -> None:
        self.log.append(f"ERROR: {msg}")
        QMessageBox.critical(self, "Evaluate failed", msg)

    def _refresh_runs(self) -> None:
        self.run_list.clear()
        model = self.ed_model.text().strip()
        if not model:
            return
        try:
            from LabGym.training.evaluation import list_evaluation_runs

            runs = list_evaluation_runs(model)
        except Exception as exc:
            self.log.append(f"Could not list runs: {exc}")
            return
        for info in runs:
            item = QListWidgetItem(info.display_label())
            item.setData(Qt.ItemDataRole.UserRole, info.run_dir)
            item.setToolTip(info.run_dir)
            self.run_list.addItem(item)
        if self.run_list.count() and self.run_list.currentRow() < 0:
            self.run_list.setCurrentRow(0)

    def _select_run_dir(self, run_dir: str) -> None:
        target = str(Path(run_dir))
        for i in range(self.run_list.count()):
            item = self.run_list.item(i)
            if item and str(Path(str(item.data(Qt.ItemDataRole.UserRole)))) == target:
                self.run_list.setCurrentRow(i)
                return

    def _on_run_selected(
        self, current: Optional[QListWidgetItem], _previous: Optional[QListWidgetItem]
    ) -> None:
        if current is None:
            return
        run_dir = current.data(Qt.ItemDataRole.UserRole)
        if not run_dir:
            return
        self._load_run(str(run_dir))

    def _load_run(self, run_dir: str) -> None:
        try:
            from LabGym.training.evaluation import load_evaluation_run

            loaded = load_evaluation_run(run_dir)
        except Exception as exc:
            self.log.append(f"Failed to load {run_dir}: {exc}")
            return
        self._loaded = loaded
        meta = loaded.get("run_meta") or {}
        summary = loaded.get("metrics_summary") or {}
        macro = summary.get("macro_f1", meta.get("macro_f1"))
        n = summary.get("n_examples", meta.get("n_examples"))
        n_mis = summary.get("n_misclassified", meta.get("n_misclassified"))
        src = meta.get("source", "")
        gt = (meta.get("ground_truth_snapshot") or {}).get("path", "")
        self.summary_label.setText(
            f"<b>{meta.get('run_id', Path(run_dir).name)}</b>  "
            f"source={src}  macro F1={macro}  n={n}  misclassified={n_mis}"
            + (f"<br/>GT: {gt}" if gt else "")
            + f"<br/><code>{run_dir}</code>"
        )

        # Per-class F1 worst-first
        ranked = summary.get("per_class_f1_worst_first") or []
        self.tbl_f1.setRowCount(0)
        for row in ranked:
            if isinstance(row, dict):
                lab, f1 = row.get("label", ""), row.get("f1", "")
            else:
                lab, f1 = row[0], row[1]
            r = self.tbl_f1.rowCount()
            self.tbl_f1.insertRow(r)
            self.tbl_f1.setItem(r, 0, QTableWidgetItem(str(lab)))
            self.tbl_f1.setItem(r, 1, QTableWidgetItem(f"{float(f1):.4f}" if f1 != "" else ""))

        pairs = summary.get("top_confused_pairs") or []
        self.tbl_pairs.setRowCount(0)
        for row in pairs:
            if isinstance(row, dict):
                a, b, c = row.get("true", ""), row.get("pred", ""), row.get("count", 0)
            else:
                a, b, c = row[0], row[1], row[2]
            r = self.tbl_pairs.rowCount()
            self.tbl_pairs.insertRow(r)
            self.tbl_pairs.setItem(r, 0, QTableWidgetItem(str(a)))
            self.tbl_pairs.setItem(r, 1, QTableWidgetItem(str(b)))
            self.tbl_pairs.setItem(r, 2, QTableWidgetItem(str(c)))

        self._fill_confusion(loaded.get("confusion_counts") or {})
        self._fill_report(loaded.get("classification_report") or {})
        self._reload_predictions_view()
        self._fill_high_loss(loaded.get("high_loss"))
        self._fill_settings(meta.get("model_settings") or {})

    def _fill_confusion(self, payload: Dict[str, Any]) -> None:
        names = [str(c) for c in (payload.get("classnames") or [])]
        matrix = payload.get("matrix") or []
        n = len(names)
        self.tbl_confusion.clear()
        self.tbl_confusion.setRowCount(n)
        self.tbl_confusion.setColumnCount(n)
        self.tbl_confusion.setHorizontalHeaderLabels(names)
        self.tbl_confusion.setVerticalHeaderLabels(names)
        for i in range(n):
            row = matrix[i] if i < len(matrix) else []
            for j in range(n):
                val = row[j] if j < len(row) else 0
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if i == j:
                    item.setBackground(QColor(220, 245, 220))
                elif int(val) > 0:
                    item.setBackground(QColor(255, 230, 230))
                self.tbl_confusion.setItem(i, j, item)

    def _fill_report(self, report: Dict[str, Any]) -> None:
        self.tbl_report.setRowCount(0)
        if not report:
            return
        # Prefer class rows then aggregates
        keys = [k for k in report.keys() if k not in ("accuracy",)]
        # Put accuracy-like last
        ordered = [k for k in keys if k not in ("macro avg", "weighted avg")]
        for k in ("macro avg", "weighted avg"):
            if k in report:
                ordered.append(k)
        for key in ordered:
            val = report[key]
            if not isinstance(val, dict):
                r = self.tbl_report.rowCount()
                self.tbl_report.insertRow(r)
                self.tbl_report.setItem(r, 0, QTableWidgetItem(str(key)))
                self.tbl_report.setItem(r, 1, QTableWidgetItem(str(val)))
                continue
            r = self.tbl_report.rowCount()
            self.tbl_report.insertRow(r)
            self.tbl_report.setItem(r, 0, QTableWidgetItem(str(key)))
            for col, field in enumerate(
                ("precision", "recall", "f1-score", "support"), start=1
            ):
                v = val.get(field, "")
                if isinstance(v, float):
                    text = f"{v:.4f}"
                else:
                    text = str(v)
                self.tbl_report.setItem(r, col, QTableWidgetItem(text))
        if "accuracy" in report:
            r = self.tbl_report.rowCount()
            self.tbl_report.insertRow(r)
            self.tbl_report.setItem(r, 0, QTableWidgetItem("accuracy"))
            self.tbl_report.setItem(
                r, 3, QTableWidgetItem(f"{float(report['accuracy']):.4f}")
            )

    def _reload_predictions_view(self) -> None:
        preds = self._loaded.get("predictions")
        self.tbl_pred.setRowCount(0)
        if preds is None:
            return
        try:
            import pandas as pd

            df = preds if isinstance(preds, pd.DataFrame) else pd.DataFrame(preds)
        except Exception:
            return
        if self.chk_mis_only.isChecked() and "misclassified" in df.columns:
            df = df[df["misclassified"].astype(int) != 0]
        cols = ["example_id", "true_label", "pred_label", "confidence", "misclassified"]
        for _, row in df.iterrows():
            r = self.tbl_pred.rowCount()
            self.tbl_pred.insertRow(r)
            for c, col in enumerate(cols):
                val = row[col] if col in row.index else ""
                if col == "confidence" and val != "" and val is not None:
                    try:
                        val = f"{float(val):.4f}"
                    except (TypeError, ValueError):
                        pass
                self.tbl_pred.setItem(r, c, QTableWidgetItem(str(val)))

    def _fill_high_loss(self, hl: Any) -> None:
        self.tbl_high_loss.setRowCount(0)
        if hl is None:
            return
        try:
            import pandas as pd

            df = hl if isinstance(hl, pd.DataFrame) else pd.DataFrame(hl)
        except Exception:
            return
        cols = ["rank", "example_id", "loss", "true_label", "pred_label"]
        for _, row in df.iterrows():
            r = self.tbl_high_loss.rowCount()
            self.tbl_high_loss.insertRow(r)
            for c, col in enumerate(cols):
                val = row[col] if col in row.index else ""
                if col == "loss" and val != "" and val is not None:
                    try:
                        val = f"{float(val):.6f}"
                    except (TypeError, ValueError):
                        pass
                self.tbl_high_loss.setItem(r, c, QTableWidgetItem(str(val)))

    def _fill_settings(self, settings: Dict[str, Any]) -> None:
        self.tbl_settings.setRowCount(0)
        # Prefer key settings first
        preferred = [
            "time_step",
            "network",
            "behavior_kind",
            "label_mode",
            "lambda_soft",
            "level_tconv",
            "level_conv",
            "dim_tconv",
            "dim_conv",
            "channel",
            "classnames",
        ]
        keys = list(preferred)
        for k in settings:
            if k not in keys:
                keys.append(k)
        for key in keys:
            if key not in settings:
                continue
            r = self.tbl_settings.rowCount()
            self.tbl_settings.insertRow(r)
            self.tbl_settings.setItem(r, 0, QTableWidgetItem(str(key)))
            val = settings[key]
            if isinstance(val, list):
                text = ", ".join(str(x) for x in val)
            else:
                text = str(val)
            self.tbl_settings.setItem(r, 1, QTableWidgetItem(text))

    def _open_run_folder(self) -> None:
        item = self.run_list.currentItem()
        path = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not path:
            model = self.ed_model.text().strip()
            path = str(Path(model) / "eval") if model else ""
        if not path or not Path(path).exists():
            QMessageBox.information(self, "Open folder", "No run folder to open.")
            return
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))

    def _scan_models_for_compare(self) -> None:
        self.compare_models.clear()
        paths: List[str] = []
        try:
            from LabGym.gui_pyside.model_paths import scan_categorizer_paths

            paths = scan_categorizer_paths(self.project.project)
        except Exception:
            paths = []
        # Always include current model field if set
        current = self.ed_model.text().strip()
        if current and current not in paths:
            paths = [current] + paths
        for p in paths:
            item = QListWidgetItem(Path(p).name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setToolTip(p)
            self.compare_models.addItem(item)

    def _run_compare(self) -> None:
        if self._thread is not None:
            return
        gt = self.ed_gt.text().strip()
        if not gt or not Path(gt).is_dir():
            QMessageBox.warning(self, "Compare", "Set a valid ground-truth examples folder.")
            return
        selected = self.compare_models.selectedItems()
        models = [str(i.data(Qt.ItemDataRole.UserRole)) for i in selected if i.data(Qt.ItemDataRole.UserRole)]
        if len(models) < 2:
            QMessageBox.warning(
                self,
                "Compare",
                "Select at least two already-trained categorizers in the compare list.",
            )
            return
        self._set_busy(True)
        self.log.append(f"Comparing {len(models)} models on {gt}…")
        self.detail_tabs.setCurrentWidget(self.tbl_compare)
        self._thread = QThread(self)
        worker = _CompareWorker(gt, models)
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.progress.connect(lambda m: self.log.append(m))
        worker.finished.connect(self._compare_done)
        worker.error.connect(self._eval_err)
        worker.finished.connect(self._thread.quit)
        worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        self._worker = worker
        self._thread.start()

    def _compare_done(self, results: list) -> None:
        from LabGym.training.evaluation import load_evaluation_run

        self.tbl_compare.setRowCount(0)
        self._compare_rows = []
        for model, run_dir, err in results:
            row_data: Dict[str, Any] = {
                "model": Path(model).name,
                "model_path": model,
                "run_id": "",
                "macro_f1": "",
                "n_examples": "",
                "n_misclassified": "",
                "time_step": "",
                "network": "",
                "label_mode": "",
                "error": err or "",
            }
            if run_dir and not err:
                try:
                    loaded = load_evaluation_run(run_dir)
                    meta = loaded.get("run_meta") or {}
                    summary = loaded.get("metrics_summary") or {}
                    settings = meta.get("model_settings") or {}
                    row_data["run_id"] = meta.get("run_id", Path(run_dir).name)
                    row_data["macro_f1"] = summary.get(
                        "macro_f1", meta.get("macro_f1", "")
                    )
                    row_data["n_examples"] = summary.get(
                        "n_examples", meta.get("n_examples", "")
                    )
                    row_data["n_misclassified"] = summary.get(
                        "n_misclassified", meta.get("n_misclassified", "")
                    )
                    row_data["time_step"] = settings.get("time_step", "")
                    row_data["network"] = settings.get("network", "")
                    row_data["label_mode"] = settings.get("label_mode", "")
                    self.log.append(f"  {Path(model).name} → {run_dir}")
                except Exception as exc:
                    row_data["error"] = str(exc)
            elif err:
                self.log.append(f"  {Path(model).name} ERROR: {err}")
            self._compare_rows.append(row_data)
            r = self.tbl_compare.rowCount()
            self.tbl_compare.insertRow(r)
            vals = [
                row_data["model"],
                self._fmt_f1(row_data["macro_f1"]),
                str(row_data["n_examples"]),
                str(row_data["n_misclassified"]),
                str(row_data["time_step"]),
                str(row_data["network"]),
                str(row_data["label_mode"]),
                str(row_data["run_id"] or row_data["error"]),
            ]
            for c, v in enumerate(vals):
                self.tbl_compare.setItem(r, c, QTableWidgetItem(v))
        # Sort display by macro_f1 descending when possible
        self.tbl_compare.sortItems(1, Qt.SortOrder.DescendingOrder)
        self.log.append("Compare finished.")
        QMessageBox.information(
            self,
            "Compare",
            f"Compared {len(results)} model(s). See the Compare tab.",
        )

    @staticmethod
    def _fmt_f1(val: Any) -> str:
        if val == "" or val is None:
            return ""
        try:
            return f"{float(val):.4f}"
        except (TypeError, ValueError):
            return str(val)
