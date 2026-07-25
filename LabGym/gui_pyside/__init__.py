"""PySide6 ethogram-first workbench shell (default LabGym UI).

Launch:
  LabGym
  LabGym-workflow
  python -m LabGym.gui_pyside
  python -m LabGym
"""

from .main_window import main

__all__ = ["main"]
