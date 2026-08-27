"""Seam tests: video disk-status probes + ProjectController load/replace."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from LabGym.gui_pyside.project.controller import ProjectController
from LabGym.gui_pyside.project.model import Project
from LabGym.gui_pyside.project.video_disk_status import (
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_PENDING,
    STATUS_UNKNOWN,
    apply_probe_to_project,
    combo_item_label,
    probe_video,
    probe_videos,
    run_blocked_unknown_paths,
    status_label,
)


def _project_with_videos(tmp_path: Path, names=("a.avi", "b.avi")) -> Project:
    proj = Project.new(name="status", root_dir=str(tmp_path))
    for name in names:
        v = tmp_path / name
        v.write_bytes(b"x")
        proj.add_video(str(v))
    return proj


def _package_at(tmp_path: Path, stem: str) -> Path:
    pkg = tmp_path / "detection" / stem / "id_review"
    pkg.mkdir(parents=True)
    (pkg / "meta.json").write_text("{}", encoding="utf-8")
    return pkg


def test_status_unknown_label():
    assert status_label(STATUS_UNKNOWN) == "…"
    assert status_label(STATUS_PENDING) == "pending"


def test_probe_done_when_package_exists(tmp_path: Path):
    proj = _project_with_videos(tmp_path, names=("clip.avi",))
    pkg = _package_at(tmp_path, "clip")
    vid = str((tmp_path / "clip.avi").resolve())
    result = probe_video(proj, vid)
    assert result.status == STATUS_DONE
    assert Path(result.tracklets_dir) == pkg
    assert result.status != STATUS_UNKNOWN


def test_probe_pending_when_no_package(tmp_path: Path):
    proj = _project_with_videos(tmp_path, names=("clip.avi",))
    vid = str((tmp_path / "clip.avi").resolve())
    result = probe_video(proj, vid)
    assert result.status == STATUS_PENDING
    assert result.tracklets_dir == ""


def test_stored_detection_dir_is_hint(tmp_path: Path):
    proj = _project_with_videos(tmp_path, names=("clip.avi",))
    custom = tmp_path / "elsewhere" / "id_review"
    custom.mkdir(parents=True)
    (custom / "meta.json").write_text("{}", encoding="utf-8")
    vid = str((tmp_path / "clip.avi").resolve())
    entry = proj.videos[0]
    entry.detection_dir = str(custom)
    result = probe_video(proj, vid)
    assert result.status == STATUS_DONE
    assert Path(result.tracklets_dir) == custom


def test_successful_probe_fills_detection_dir_without_dirty(tmp_path: Path):
    proj = _project_with_videos(tmp_path, names=("clip.avi",))
    _package_at(tmp_path, "clip")
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    assert ctrl.dirty is False
    vid = str((tmp_path / "clip.avi").resolve())
    result = probe_video(ctrl.project, vid)
    apply_probe_to_project(ctrl.project, result)
    assert ctrl.dirty is False
    assert ctrl.project.videos[0].detection_dir
    from LabGym.gui_pyside.project.paths import discover_tracklets_dir

    cached = discover_tracklets_dir(ctrl.project, vid)
    assert cached == result.tracklets_dir


def test_timeout_errors_one_video_and_others_continue(tmp_path: Path, monkeypatch):
    proj = _project_with_videos(tmp_path, names=("slow.avi", "ok.avi"))
    from LabGym.gui_pyside.project import paths as paths_mod

    real = paths_mod.discover_tracklets_dir

    def slow_or_real(project, video_path):
        if Path(video_path).name == "slow.avi":
            time.sleep(2.0)
            return ""
        return real(project, video_path)

    monkeypatch.setattr(paths_mod, "discover_tracklets_dir", slow_or_real)
    paths = [
        str((tmp_path / "slow.avi").resolve()),
        str((tmp_path / "ok.avi").resolve()),
    ]
    results = probe_videos(proj, paths, timeout_s=0.2)
    assert len(results) == 2
    assert results[0].status == STATUS_ERROR
    assert results[0].timed_out is True
    assert results[1].status == STATUS_PENDING


def test_cancel_stops_remaining_probes(tmp_path: Path):
    proj = _project_with_videos(tmp_path, names=("a.avi", "b.avi", "c.avi"))
    ev = threading.Event()
    seen = []

    def on_result(result):
        seen.append(result.video_path)
        ev.set()

    paths = [str((tmp_path / n).resolve()) for n in ("a.avi", "b.avi", "c.avi")]
    probe_videos(proj, paths, cancel_event=ev, on_result=on_result)
    assert len(seen) == 1


def test_controller_load_replace_does_not_probe(tmp_path: Path, monkeypatch):
    from LabGym.gui_pyside.project import video_disk_status as vds
    from LabGym.gui_pyside.project import paths as paths_mod

    hits = []
    monkeypatch.setattr(
        vds,
        "probe_video",
        lambda *a, **k: hits.append("probe")
        or vds.ProbeResult(video_path="", status=STATUS_UNKNOWN),
    )
    monkeypatch.setattr(
        paths_mod,
        "discover_tracklets_dir",
        lambda *a, **k: hits.append("discover") or "",
    )

    proj = _project_with_videos(tmp_path)
    path = tmp_path / "p.labproj.json"
    proj.save(str(path))

    ctrl = ProjectController()
    ctrl.load_from_path(path)
    assert hits == []

    ctrl.replace(Project.new(name="x", root_dir=str(tmp_path)), dirty=True)
    assert hits == []


def test_detect_track_paints_unknown_without_probing(tmp_path: Path, monkeypatch):
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.project import paths as paths_mod
    from LabGym.gui_pyside.workbenches.detector import detect_track_tab as dtt
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab

    hits = []
    monkeypatch.setattr(
        paths_mod, "discover_tracklets_dir", lambda *a, **k: hits.append("discover") or ""
    )
    monkeypatch.setattr(
        dtt, "scan_detector_paths", lambda *a, **k: hits.append("scan") or []
    )

    proj = _project_with_videos(tmp_path)
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    tab = DetectTrackTab(ctrl)
    assert hits == []
    assert tab.table.rowCount() == 2
    assert tab.table.item(0, 2).text() == "…"
    assert tab.table.item(1, 2).text() == "…"
    assert tab.lbl_probe.text() == "0 / 2 checked"


def test_detect_track_probes_fill_pending_without_faking_done(
    tmp_path: Path, monkeypatch
):
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.project import video_disk_status as vds
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab

    def fake_probe(project, video_path, **kwargs):
        return vds.ProbeResult(video_path=video_path, status=STATUS_PENDING)

    monkeypatch.setattr(vds, "probe_video", fake_probe)

    proj = _project_with_videos(tmp_path)
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    tab = DetectTrackTab(ctrl)
    assert tab.table.item(0, 2).text() == "…"
    tab._start_status_probes()
    thread = tab._probe_thread
    assert thread is not None
    thread.wait(5000)
    app = QApplication.instance()
    assert app is not None
    app.processEvents()
    assert tab.table.item(0, 2).text() == "pending"
    assert tab.table.item(1, 2).text() == "pending"
    assert tab.lbl_probe.text() == "2 / 2 checked"
    assert ctrl.dirty is False


def test_project_swap_does_not_discover_or_scan(tmp_path: Path, monkeypatch):
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.project import paths as paths_mod
    from LabGym.gui_pyside.workbenches.detector import detect_track_tab as dtt
    from LabGym.training import dataset_manifest as dm

    hits = []
    monkeypatch.setattr(
        paths_mod, "discover_tracklets_dir", lambda *a, **k: hits.append("discover") or ""
    )
    monkeypatch.setattr(
        dtt, "scan_detector_paths", lambda *a, **k: hits.append("scan_det") or []
    )
    monkeypatch.setattr(
        dm, "scan_example_store", lambda *a, **k: hits.append("scan_ex") or []
    )

    from LabGym.gui_pyside.main_window import WorkbenchMainWindow

    proj = _project_with_videos(tmp_path)
    path = tmp_path / "p.labproj.json"
    proj.save(str(path))

    w = WorkbenchMainWindow()
    hits.clear()
    w.project.load_from_path(path)
    assert "discover" not in hits
    assert "scan_det" not in hits
    assert "scan_ex" not in hits
    w.close()


def test_probe_never_returns_pending_or_done_on_timeout(tmp_path: Path, monkeypatch):
    proj = _project_with_videos(tmp_path, names=("clip.avi",))
    from LabGym.gui_pyside.project import paths as paths_mod

    def hang(project, video_path):
        time.sleep(2.0)
        return str(tmp_path / "detection" / "clip" / "id_review")

    monkeypatch.setattr(paths_mod, "discover_tracklets_dir", hang)
    result = probe_video(
        proj, str((tmp_path / "clip.avi").resolve()), timeout_s=0.15
    )
    assert result.status not in (STATUS_PENDING, STATUS_DONE)
    assert result.status == STATUS_ERROR
    assert result.timed_out is True


def test_combo_item_label_never_fakes_done_while_unknown():
    assert "✓" not in combo_item_label("clip.avi", STATUS_UNKNOWN)
    assert "…" in combo_item_label("clip.avi", STATUS_UNKNOWN)
    assert combo_item_label("clip.avi", STATUS_DONE).endswith("✓")
    assert "✓" not in combo_item_label("clip.avi", STATUS_PENDING)


def test_run_blocked_unknown_paths_treats_missing_as_unknown():
    checked = ["a.avi", "b.avi", "c.avi"]
    status = {
        "a.avi": (STATUS_PENDING, ""),
        "c.avi": (STATUS_DONE, "/pkg"),
    }
    blocked = run_blocked_unknown_paths(checked, status)
    assert blocked == ["b.avi"]
    assert run_blocked_unknown_paths(["a.avi", "c.avi"], status) == []


def _app():
    import sys

    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication(sys.argv)


def test_process_videos_paints_unknown_without_probing(tmp_path: Path, monkeypatch):
    _app()
    from LabGym.gui_pyside.project import paths as paths_mod
    from LabGym.gui_pyside.workbenches.categorizer import process_videos_tab as pvt
    from LabGym.gui_pyside.workbenches.categorizer.process_videos_tab import (
        ProcessVideosTab,
    )

    hits = []
    monkeypatch.setattr(
        paths_mod, "discover_tracklets_dir", lambda *a, **k: hits.append("discover") or ""
    )
    monkeypatch.setattr(
        pvt, "scan_categorizer_paths", lambda *a, **k: hits.append("scan_cat") or []
    )

    proj = _project_with_videos(tmp_path)
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    tab = ProcessVideosTab(ctrl)
    assert hits == []
    assert tab.table.rowCount() == 2
    assert tab.table.item(0, 2).text() == "…"
    assert tab.table.item(1, 2).text() == "…"
    assert tab.lbl_probe.text() == "0 / 2 checked"


def test_process_videos_run_refuses_unknown(tmp_path: Path, monkeypatch):
    _app()
    from LabGym.gui_pyside.workbenches.categorizer.process_videos_tab import (
        ProcessVideosTab,
    )

    proj = _project_with_videos(tmp_path)
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    tab = ProcessVideosTab(ctrl)
    tab.ed_categorizer.setText(str(tmp_path))
    tab.ed_out.setText(str(tmp_path / "analysis"))
    warned = []
    monkeypatch.setattr(
        "LabGym.gui_pyside.workbenches.categorizer.process_videos_tab.QMessageBox.warning",
        lambda *a, **k: warned.append(str(a[2] if len(a) > 2 else a)),
    )
    tab._start()
    assert tab._batch_active is False
    assert warned
    assert "unknown" in warned[0].lower()


def test_detect_track_run_refuses_unknown(tmp_path: Path, monkeypatch):
    _app()
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab

    proj = _project_with_videos(tmp_path)
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    tab = DetectTrackTab(ctrl)
    warned = []
    monkeypatch.setattr(
        "LabGym.gui_pyside.workbenches.detector.detect_track_tab.QMessageBox.warning",
        lambda *a, **k: warned.append(str(a[2] if len(a) > 2 else a)),
    )
    monkeypatch.setattr(
        "LabGym.gui_pyside.workbenches.detector.detect_track_tab.QMessageBox.information",
        lambda *a, **k: None,
    )
    tab._start_batch()
    assert tab._batch_active is False
    assert warned
    assert "unknown" in warned[0].lower()


def test_review_ids_combo_unknown_marks_not_fake_check(tmp_path: Path, monkeypatch):
    _app()
    from LabGym.gui_pyside.project import paths as paths_mod
    from LabGym.gui_pyside.workbenches.detector.review_ids_tab import ReviewIdsTab

    hits = []
    monkeypatch.setattr(
        paths_mod, "discover_tracklets_dir", lambda *a, **k: hits.append("discover") or ""
    )
    proj = _project_with_videos(tmp_path)
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    tab = ReviewIdsTab(ctrl)
    assert hits == []
    texts = [
        tab.combo_video.itemText(i) for i in range(1, tab.combo_video.count())
    ]
    assert texts
    for text in texts:
        assert "✓" not in text
        assert "…" in text


def test_review_ids_load_unknown_starts_session_load(tmp_path: Path, monkeypatch):
    _app()
    from LabGym.gui_pyside.project import paths as paths_mod
    from LabGym.gui_pyside.workbenches.detector.review_ids_tab import ReviewIdsTab

    pkg = _package_at(tmp_path, "a")
    monkeypatch.setattr(paths_mod, "discover_tracklets_dir", lambda *a, **k: str(pkg))
    proj = _project_with_videos(tmp_path)
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    tab = ReviewIdsTab(ctrl)
    loads = []
    tab.load_package = lambda d: loads.append(d) or True
    tab.combo_video.setCurrentIndex(1)
    tab._load_selected_video_package()
    assert loads == [str(pkg)]


def test_manage_dataset_does_not_scan_on_swap(tmp_path: Path, monkeypatch):
    _app()
    from LabGym.training import dataset_manifest as dm
    from LabGym.gui_pyside.workbenches.categorizer.manage_dataset_host import (
        ManageDatasetHost,
    )

    hits = []
    monkeypatch.setattr(
        dm, "scan_example_store", lambda *a, **k: hits.append("scan_ex") or []
    )
    proj = _project_with_videos(tmp_path)
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    host = ManageDatasetHost(ctrl)
    assert hits == []
    ctrl.replace(Project.new(name="other", root_dir=str(tmp_path)), dirty=False)
    assert hits == []


def test_manage_dataset_scans_when_shown(tmp_path: Path, monkeypatch):
    _app()
    from LabGym.training import dataset_manifest as dm
    from LabGym.gui_pyside.workbenches.categorizer.manage_dataset_host import (
        ManageDatasetHost,
    )

    hits = []
    monkeypatch.setattr(
        dm, "scan_example_store", lambda *a, **k: hits.append("scan_ex") or []
    )
    examples = tmp_path / "examples"
    examples.mkdir()
    proj = _project_with_videos(tmp_path)
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    host = ManageDatasetHost(ctrl)
    host.show()
    host._maybe_scan_store()
    thread = host._scan_thread
    if thread is not None:
        thread.wait(5000)
    app = _app()
    app.processEvents()
    assert "scan_ex" in hits


def test_detect_track_scans_detectors_on_show_not_construct(
    tmp_path: Path, monkeypatch
):
    _app()
    from LabGym.gui_pyside.workbenches.detector import detect_track_tab as dtt
    from LabGym.gui_pyside.workbenches.detector.detect_track_tab import DetectTrackTab

    hits = []
    monkeypatch.setattr(
        dtt, "scan_detector_paths", lambda *a, **k: hits.append("scan_det") or []
    )
    proj = _project_with_videos(tmp_path)
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    tab = DetectTrackTab(ctrl)
    assert hits == []
    tab.show()
    tab._maybe_scan_models()
    assert "scan_det" in hits


def test_process_videos_scans_categorizers_on_show_not_construct(
    tmp_path: Path, monkeypatch
):
    _app()
    from LabGym.gui_pyside.workbenches.categorizer import process_videos_tab as pvt
    from LabGym.gui_pyside.workbenches.categorizer.process_videos_tab import (
        ProcessVideosTab,
    )

    hits = []
    monkeypatch.setattr(
        pvt, "scan_categorizer_paths", lambda *a, **k: hits.append("scan_cat") or []
    )
    proj = _project_with_videos(tmp_path)
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    tab = ProcessVideosTab(ctrl)
    assert hits == []
    tab.show()
    tab._maybe_scan_models()
    assert "scan_cat" in hits
