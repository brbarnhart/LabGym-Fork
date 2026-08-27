"""Project swap: unsaved switch markers / ethogram gate and session drop."""

from __future__ import annotations

import json
from pathlib import Path

from LabGym.annotator.core.annotation_manager import AnnotationManager
from LabGym.annotator.core.data_models import AnnotationSession
from LabGym.gui_pyside.session_swap import (
    CHOICE_CANCEL,
    CHOICE_DISCARD,
    CHOICE_SAVE,
    KIND_ETHOGRAM,
    KIND_SWITCH_MARKERS,
    drop_sessions,
    inspect_sessions,
    resolve_unsaved_session,
    switch_markers_signature,
)
from LabGym.id_review.types import SwitchMarker


class _FakeReview:
    def __init__(self, *, unsaved: bool = False) -> None:
        self.unsaved = unsaved
        self.saved = False
        self.discarded = False
        self.dropped = False
        self.stores = {"mouse": [1, 2, 3]}
        self.wrote = False
        self.save_ok = True

    def has_unsaved_switch_markers(self) -> bool:
        return self.unsaved

    def save_switch_markers(self) -> bool:
        if not self.save_ok:
            return False
        self.saved = True
        self.wrote = True
        self.unsaved = False
        return True

    def discard_unsaved_switch_markers(self) -> None:
        self.discarded = True
        self.unsaved = False

    def drop_session(self) -> None:
        self.dropped = True
        self.stores = {}


class _FakeEthogram:
    def __init__(self, *, unsaved: bool = False) -> None:
        self.unsaved = unsaved
        self.saved = False
        self.discarded = False
        self.dropped = False
        self.tracklets = object()
        self.wrote = False
        self.save_ok = True

    def has_unsaved_ethogram(self) -> bool:
        return self.unsaved

    def save_ethogram(self) -> bool:
        if not self.save_ok:
            return False
        self.saved = True
        self.wrote = True
        self.unsaved = False
        return True

    def discard_unsaved_ethogram(self) -> None:
        self.discarded = True
        self.unsaved = False

    def drop_session(self) -> None:
        self.dropped = True
        self.tracklets = None


def _marker(frame: int = 4) -> SwitchMarker:
    return SwitchMarker(
        marker_id=f"s{frame:06d}_mouse",
        frame=frame,
        animal_kind="mouse",
        involved_ids=[0, 1],
        mapping={0: 1, 1: 0},
    )


def test_inspect_ignores_project_dirty_flag() -> None:
    from LabGym.gui_pyside.project.controller import ProjectController

    ctrl = ProjectController()
    ctrl.mark_dirty()
    assert ctrl.dirty is True
    review = _FakeReview(unsaved=False)
    ethogram = _FakeEthogram(unsaved=False)
    report = inspect_sessions(review_ids=review, ethogram=ethogram)
    assert report.is_dirty is False
    assert report.kinds() == ()


def test_inspect_reports_switch_markers_and_ethogram_independently() -> None:
    review = _FakeReview(unsaved=True)
    ethogram = _FakeEthogram(unsaved=False)
    report = inspect_sessions(review_ids=review, ethogram=ethogram)
    assert report.unsaved_switch_markers is True
    assert report.unsaved_ethogram is False
    assert report.kinds() == (KIND_SWITCH_MARKERS,)

    report2 = inspect_sessions(
        review_ids=_FakeReview(unsaved=False),
        ethogram=_FakeEthogram(unsaved=True),
    )
    assert report2.kinds() == (KIND_ETHOGRAM,)


def test_clean_session_resolves_without_prompt_and_drop_releases() -> None:
    review = _FakeReview(unsaved=False)
    ethogram = _FakeEthogram(unsaved=False)
    prompts: list = []

    def prompt(report):
        prompts.append(report)
        return CHOICE_CANCEL

    assert resolve_unsaved_session(
        review_ids=review, ethogram=ethogram, prompt=prompt
    )
    assert prompts == []
    drop_sessions(review_ids=review, ethogram=ethogram)
    assert review.dropped is True
    assert review.stores == {}
    assert ethogram.dropped is True
    assert ethogram.tracklets is None
    assert review.saved is False
    assert review.discarded is False


def test_unsaved_switch_markers_block_until_save_or_discard() -> None:
    review = _FakeReview(unsaved=True)
    assert (
        resolve_unsaved_session(
            review_ids=review, ethogram=_FakeEthogram(), prompt=None
        )
        is False
    )
    assert review.unsaved is True
    assert review.dropped is False

    assert (
        resolve_unsaved_session(
            review_ids=review,
            ethogram=_FakeEthogram(),
            prompt=lambda _r: CHOICE_CANCEL,
        )
        is False
    )
    assert review.unsaved is True
    assert review.wrote is False


def test_save_then_swap_writes_and_drop_does_not_auto_load() -> None:
    review = _FakeReview(unsaved=True)
    ethogram = _FakeEthogram(unsaved=False)
    loaded = []

    def prompt(_report):
        return CHOICE_SAVE

    assert resolve_unsaved_session(
        review_ids=review, ethogram=ethogram, prompt=prompt
    )
    assert review.saved is True
    assert review.wrote is True
    assert review.unsaved is False
    drop_sessions(review_ids=review, ethogram=ethogram)
    assert review.dropped is True
    assert review.stores == {}
    assert loaded == []


def test_discard_then_swap_does_not_write() -> None:
    review = _FakeReview(unsaved=True)
    ethogram = _FakeEthogram(unsaved=True)

    assert resolve_unsaved_session(
        review_ids=review,
        ethogram=ethogram,
        prompt=lambda _r: CHOICE_DISCARD,
    )
    assert review.discarded is True
    assert review.wrote is False
    assert ethogram.discarded is True
    assert ethogram.wrote is False
    drop_sessions(review_ids=review, ethogram=ethogram)
    assert review.dropped is True
    assert ethogram.dropped is True


def test_failed_save_aborts_without_drop() -> None:
    review = _FakeReview(unsaved=True)
    review.save_ok = False
    assert (
        resolve_unsaved_session(
            review_ids=review,
            ethogram=_FakeEthogram(),
            prompt=lambda _r: CHOICE_SAVE,
        )
        is False
    )
    assert review.dropped is False
    assert review.unsaved is True


def test_combined_save_second_failure_does_not_drop() -> None:
    review = _FakeReview(unsaved=True)
    ethogram = _FakeEthogram(unsaved=True)
    ethogram.save_ok = False
    assert (
        resolve_unsaved_session(
            review_ids=review,
            ethogram=ethogram,
            prompt=lambda _r: CHOICE_SAVE,
        )
        is False
    )
    assert review.saved is True
    assert review.wrote is True
    assert ethogram.saved is False
    assert ethogram.wrote is False
    assert review.dropped is False
    assert ethogram.dropped is False
    drop_sessions(review_ids=review, ethogram=ethogram)
    assert review.dropped is True


def test_switch_markers_signature_ignores_object_identity() -> None:
    a = [_marker(9)]
    b = [_marker(9)]
    assert switch_markers_signature(a) == switch_markers_signature(b)
    assert switch_markers_signature(a) != switch_markers_signature([_marker(10)])


def test_ethogram_unsaved_is_independent_of_project_file(tmp_path: Path) -> None:
    sess = AnnotationSession(video_path="clip.avi", fps=10.0, total_frames=20)
    mgr = AnnotationManager(sess)
    mgr.add_behavior("grooming", "#00AAFF", "1")
    mgr.mark_saved()
    assert mgr.has_unsaved_changes() is False
    mgr.add_bout("grooming", 1, 5)
    assert mgr.has_unsaved_changes() is True
    path = tmp_path / "clip.annotations.json"
    mgr.save_to_json(path)
    assert mgr.has_unsaved_changes() is False
    assert json.loads(path.read_text(encoding="utf-8"))["bouts"]
    mgr.add_bout("grooming", 8, 10)
    assert mgr.has_unsaved_changes() is True
    mgr.discard_unsaved_changes()
    assert mgr.has_unsaved_changes() is False
    assert mgr.get_bouts_for_behavior("grooming")[0].end_frame == 5


def test_review_ids_subjects_dirt_is_not_unsaved_switch_markers() -> None:
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.workbenches.detector.review_ids_tab import ReviewIdsTab

    tab = ReviewIdsTab(ProjectController())
    tab.markers = [_marker(2)]
    tab._saved_markers = [_marker(2)]
    tab._dirty = True
    assert tab.has_unsaved_switch_markers() is False
    tab.markers = [_marker(3)]
    assert tab.has_unsaved_switch_markers() is True
    tab.discard_unsaved_switch_markers()
    assert tab.has_unsaved_switch_markers() is False
    assert tab.markers[0].frame == 2


def test_project_replaced_drops_review_ids_and_does_not_session_load() -> None:
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.project.model import Project
    from LabGym.gui_pyside.workbenches.detector.review_ids_tab import ReviewIdsTab

    ctrl = ProjectController()
    tab = ReviewIdsTab(ctrl)
    loads: list[str] = []
    tab.load_package = lambda d: loads.append(d) or True  # type: ignore[method-assign]
    tab._stores["mouse"] = object()
    tab.review_dir = "pkg"
    tab.markers = [_marker(1)]
    tab._saved_markers = [_marker(1)]
    tab._cap._path = "clip.avi"
    ctrl.replace(Project.new(), dirty=False)
    assert tab.review_dir == ""
    assert tab._stores == {}
    assert tab._cap.path is None
    assert tab.has_unsaved_switch_markers() is False
    assert loads == []


def test_open_and_edit_blocked_when_session_unresolved(
    monkeypatch, tmp_path: Path
) -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QFileDialog

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.main_window import WorkbenchMainWindow

    recent = tmp_path / "recent.labproj.json"
    recent.write_text("{}", encoding="utf-8")

    w = WorkbenchMainWindow()
    gates: list[str] = []

    def refuse(**_kwargs) -> bool:
        gates.append("session")
        return False

    monkeypatch.setattr(
        "LabGym.gui_pyside.main_window.resolve_unsaved_session", refuse
    )
    monkeypatch.setattr(w, "_confirm_discard", lambda: True)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: ("should-not-open.json", "")
    )
    loads: list[str] = []
    monkeypatch.setattr(w.project, "load_from_path", lambda p: loads.append(p))
    editors: list[int] = []

    class _NoDialog:
        def __init__(self, *a, **k):
            editors.append(1)

        def exec(self):
            return 0

    monkeypatch.setattr(
        "LabGym.gui_pyside.main_window.ProjectEditorDialog", _NoDialog
    )
    news: list[int] = []
    monkeypatch.setattr(w.project, "new_project", lambda *a, **k: news.append(1))
    w._open_project()
    w._open_recent(str(recent))
    w._edit_project()
    w._new_project()
    assert gates == ["session", "session", "session", "session"]
    assert loads == []
    assert editors == []
    assert news == []
    w.close()


def test_project_file_cancel_keeps_session_loaded(monkeypatch) -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QFileDialog

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.main_window import WorkbenchMainWindow

    w = WorkbenchMainWindow()
    review = w.wb_detector.review_tab
    review._stores["mouse"] = object()
    review.review_dir = "pkg"
    monkeypatch.setattr(
        "LabGym.gui_pyside.main_window.resolve_unsaved_session",
        lambda **_k: True,
    )
    monkeypatch.setattr(w, "_confirm_discard", lambda: False)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: ("should-not-open.json", "")
    )
    loads: list[str] = []
    monkeypatch.setattr(w.project, "load_from_path", lambda p: loads.append(p))
    w._open_project()
    assert review.review_dir == "pkg"
    assert "mouse" in review._stores
    assert loads == []
    w.close()


def test_open_and_edit_cancel_keep_session_loaded(monkeypatch) -> None:
    import sys

    from PySide6.QtWidgets import QApplication, QFileDialog

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.main_window import WorkbenchMainWindow

    w = WorkbenchMainWindow()
    review = w.wb_detector.review_tab
    review._stores["mouse"] = object()
    review.review_dir = "pkg"

    monkeypatch.setattr(
        "LabGym.gui_pyside.main_window.resolve_unsaved_session",
        lambda **_k: True,
    )
    monkeypatch.setattr(w, "_confirm_discard", lambda: True)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    loads: list[str] = []
    monkeypatch.setattr(w.project, "load_from_path", lambda p: loads.append(p))

    class _CancelDialog:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            return 0

    monkeypatch.setattr(
        "LabGym.gui_pyside.main_window.ProjectEditorDialog", _CancelDialog
    )
    w._open_project()
    assert review.review_dir == "pkg"
    assert "mouse" in review._stores
    assert loads == []
    w._edit_project()
    assert review.review_dir == "pkg"
    assert "mouse" in review._stores
    w.close()


def test_clean_committed_swap_drops_session_without_prompt(monkeypatch) -> None:
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.main_window import WorkbenchMainWindow
    from LabGym.gui_pyside.project.model import Project

    w = WorkbenchMainWindow()
    review = w.wb_detector.review_tab
    review._stores["mouse"] = object()
    review.review_dir = "pkg"
    review._cap._path = "clip.avi"
    prompts: list = []
    monkeypatch.setattr(
        w, "_prompt_unsaved_session", lambda r: prompts.append(r) or CHOICE_CANCEL
    )
    assert w._confirm_project_swap() is True
    assert prompts == []
    assert review.review_dir == "pkg"
    w.project.replace(Project.new(), dirty=False)
    assert review.review_dir == ""
    assert review._stores == {}
    assert review._cap.path is None
    assert w.wb_categorizer.generate_host.annotate_tab.annotator.manager is None
    w.close()


def test_project_replaced_drops_ethogram_and_does_not_session_load(tmp_path: Path) -> None:
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.project.model import Project
    from LabGym.gui_pyside.workbenches.categorizer.annotate_ethogram_tab import (
        AnnotateEthogramTab,
    )

    vid = tmp_path / "clip.avi"
    vid.write_bytes(b"x")
    proj = Project.new(name="p", root_dir=str(tmp_path))
    proj.add_video(str(vid))
    proj.defaults.current_video = str(vid)
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    tab = AnnotateEthogramTab(ctrl)
    loads: list[str] = []
    tab.load_current = lambda: loads.append("load") or True  # type: ignore[method-assign]
    tab.annotator._loaded_tracklets = object()
    tab.annotator.manager = AnnotationManager(
        AnnotationSession(video_path=str(vid), fps=10.0, total_frames=4)
    )
    other = Project.new(name="q", root_dir=str(tmp_path))
    other.add_video(str(vid))
    other.defaults.current_video = str(vid)
    ctrl.replace(other, dirty=False)
    assert tab.annotator._loaded_tracklets is None
    assert tab.annotator.manager is None
    assert loads == []
    assert tab.has_unsaved_ethogram() is False
