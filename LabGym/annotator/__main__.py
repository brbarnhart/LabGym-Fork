"""Entry point for the LabGym Behavior Annotator (PySide6)."""

from __future__ import annotations

import sys


def main() -> None:
    from PySide6.QtWidgets import QApplication

    from LabGym.annotator.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("LabGym Behavior Annotator")
    app.setOrganizationName("LabGym")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
