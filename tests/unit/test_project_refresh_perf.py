"""High-ROI project refresh: signals + tracklets discovery cache."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


def _app():
    return QApplication.instance() or QApplication(sys.argv)


def _ctrl_with_videos(tmp_path: Path, n: int = 3):
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.project.model import Project

    ctrl = ProjectController()
    proj = Project.new(name="perf", root_dir=str(tmp_path))
    for i in range(n):
        v = tmp_path / f"v{i}.avi"
        v.write_bytes(b"x")
        proj.add_video(str(v))
    ctrl.replace(proj, dirty=False)
    return ctrl


def test_load_and_replace_emit_only_project_replaced(tmp_path: Path):
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.project.model import Project

    ctrl = ProjectController()
    replaced = []
    changed = []
    ctrl.project_replaced.connect(lambda: replaced.append(1))
    ctrl.changed.connect(lambda: changed.append(1))

    proj = Project.new(name="a", root_dir=str(tmp_path))
    path = tmp_path / "a.labproj.json"
    proj.save(str(path))

    replaced.clear()
    changed.clear()
    ctrl.load_from_path(path)
    assert len(replaced) == 1
    assert len(changed) == 0

    replaced.clear()
    changed.clear()
    ctrl.save()
    assert len(replaced) == 0
    assert len(changed) == 1  # dirty asterisk only

    replaced.clear()
    changed.clear()
    ctrl.replace(Project.new(name="b", root_dir=str(tmp_path)), dirty=True)
    assert len(replaced) == 1
    assert len(changed) == 0


def test_mark_dirty_does_not_rebuild_detect_table(tmp_path: Path):
    _app()
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab

    ctrl = _ctrl_with_videos(tmp_path)
    tab = DetectTrackTab(ctrl)
    path0 = str(tab.table.item(0, 0).data(Qt.ItemDataRole.UserRole))
    tab._set_status(0, path0, "done", "note")
    calls = {"n": 0}
    orig = tab.refresh_videos

    def wrapped():
        calls["n"] += 1
        return orig()

    tab.refresh_videos = wrapped  # type: ignore[method-assign]
    ctrl.mark_dirty()
    assert calls["n"] == 0
    assert tab.table.item(0, 2).text() == "done"


def test_project_replaced_rebuilds_detect_table(tmp_path: Path):
    _app()
    from LabGym.gui_pyside.project.model import Project
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab

    ctrl = _ctrl_with_videos(tmp_path, n=2)
    tab = DetectTrackTab(ctrl)
    assert tab.table.rowCount() == 2

    proj = Project.new(name="more", root_dir=str(tmp_path))
    for name in ("x.avi", "y.avi", "z.avi"):
        p = tmp_path / name
        p.write_bytes(b"x")
        proj.add_video(str(p))
    ctrl.replace(proj, dirty=True)
    assert tab.table.rowCount() == 3


def test_discover_tracklets_cache(tmp_path: Path, monkeypatch):
    from LabGym.gui_pyside.project import paths as paths_mod
    from LabGym.gui_pyside.project.model import Project
    from LabGym.gui_pyside.project.paths import (
        clear_tracklets_discovery_cache,
        discover_tracklets_dir,
    )

    clear_tracklets_discovery_cache()
    proj = Project.new(name="c", root_dir=str(tmp_path))
    vid = tmp_path / "clip.avi"
    vid.write_bytes(b"x")
    proj.add_video(str(vid))
    # Create a real id_review package under detection/clip/
    pkg = tmp_path / "detection" / "clip" / "id_review"
    pkg.mkdir(parents=True)
    (pkg / "meta.json").write_text("{}", encoding="utf-8")

    calls = {"n": 0}
    real = paths_mod._looks_like_tracklets_dir

    def counting(directory):
        calls["n"] += 1
        return real(directory)

    monkeypatch.setattr(paths_mod, "_looks_like_tracklets_dir", counting)

    r1 = discover_tracklets_dir(proj, str(vid.resolve()))
    n_after_first = calls["n"]
    assert r1
    r2 = discover_tracklets_dir(proj, str(vid.resolve()))
    assert r2 == r1
    assert calls["n"] == n_after_first  # cache hit, no extra probes

    clear_tracklets_discovery_cache()
    discover_tracklets_dir(proj, str(vid.resolve()))
    assert calls["n"] > n_after_first
