"""Unit tests for shared path browse helpers (no real folder dialogs)."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton


def _app():
    return QApplication.instance() or QApplication(sys.argv)


def test_path_edit_row_layout():
    _app()
    from LabGym.gui_pyside.widgets.path_browse import path_edit_row

    edit = QLineEdit()
    btn = QPushButton("Browse…")
    row = path_edit_row(edit, btn)
    assert row.layout() is not None
    assert row.layout().count() == 2
    assert row.layout().itemAt(0).widget() is edit
    assert row.layout().itemAt(1).widget() is btn


def test_set_line_edit_directory_accept(monkeypatch):
    _app()
    from LabGym.gui_pyside.widgets import path_browse

    edit = QLineEdit("C:/old")
    monkeypatch.setattr(
        path_browse.QFileDialog,
        "getExistingDirectory",
        lambda *a, **k: "C:/chosen",
    )
    ok = path_browse.set_line_edit_directory(None, edit, caption="Pick")
    assert ok is True
    assert edit.text() == "C:/chosen"


def test_set_line_edit_directory_cancel(monkeypatch):
    _app()
    from LabGym.gui_pyside.widgets import path_browse

    edit = QLineEdit("C:/old")
    monkeypatch.setattr(
        path_browse.QFileDialog,
        "getExistingDirectory",
        lambda *a, **k: "",
    )
    ok = path_browse.set_line_edit_directory(None, edit)
    assert ok is False
    assert edit.text() == "C:/old"


def test_browse_existing_directory_none(monkeypatch):
    _app()
    from LabGym.gui_pyside.widgets import path_browse

    monkeypatch.setattr(
        path_browse.QFileDialog,
        "getExistingDirectory",
        lambda *a, **k: "",
    )
    assert path_browse.browse_existing_directory() is None
