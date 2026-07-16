"""Entry point for the Mouse Behavior Video Annotator (Phase 1)."""

import sys
from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    # Basic Fusion style (dark theme added in later polish)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
