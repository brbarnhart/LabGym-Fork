"""Simple metrics display (Phase 1/3 stub).

Shows a table + export button. Full integration in next pass.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QFileDialog, QMessageBox
)
import pandas as pd

from LabGym.annotator.core.metrics_calculator import MetricsCalculator


class MetricsDialog(QDialog):
    def __init__(self, calculator: MetricsCalculator, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Behavior Metrics")
        self.resize(900, 300)
        self.calculator = calculator

        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self._populate()
        layout.addWidget(self.table)

        btn_export = QPushButton("Export to Excel (.xlsx)")
        btn_export.clicked.connect(self.export_xlsx)
        layout.addWidget(btn_export)

    def _populate(self):
        df = self.calculator.to_dataframe()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(list(df.columns))

        for r, row in df.iterrows():
            for c, col in enumerate(df.columns):
                val = row[col]
                self.table.setItem(r, c, QTableWidgetItem(str(val)))

        self.table.resizeColumnsToContents()

    def export_xlsx(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Metrics", "metrics.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        try:
            df = self.calculator.to_dataframe()
            # Also add a second sheet idea in future
            df.to_excel(path, index=False, engine="openpyxl")
            QMessageBox.information(self, "Exported", f"Saved to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export failed", str(e))
