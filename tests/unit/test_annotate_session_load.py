"""Annotate session load: metadata first, all overlay kinds, cancel/replace."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from LabGym.id_review.apply import write_tracklets_identity_status
from LabGym.id_review.tracklets import save_tracklets
from LabGym.id_review.types import SCHEMA_VERSION, TrackletStore


def _square(x: int, y: int, size: int = 4) -> np.ndarray:
    return np.array(
        [
            [[x, y]],
            [[x + size, y]],
            [[x + size, y + size]],
            [[x, y + size]],
        ],
        dtype=np.int32,
    )


def _store(*, n_frames: int = 8, kind: str = "mouse", ids=(0, 1)) -> TrackletStore:
    ids = [int(i) for i in ids]
    n_ids = len(ids)
    centers = np.zeros((n_ids, n_frames, 2), dtype=np.float64)
    valid = np.ones((n_ids, n_frames), dtype=bool)
    heights = np.full((n_ids, n_frames), 10.0)
    contours = []
    for row, tid in enumerate(ids):
        row_c = []
        for f in range(n_frames):
            centers[row, f] = [tid * 40.0 + f, float(tid)]
            x, y = int(centers[row, f, 0]), int(centers[row, f, 1])
            row_c.append(_square(x, y))
        contours.append(row_c)
    return TrackletStore(
        schema_version=SCHEMA_VERSION,
        animal_kind=kind,
        ids=ids,
        n_frames=n_frames,
        centers=centers,
        valid=valid,
        heights=heights,
        contours=contours,
        meta={"video": "clip.avi", "fps": 10.0},
    )


def _package_two_kinds_accepted(tmp_path: Path) -> Path:
    """Published remapped tracklets (Annotate), not Review IDs raw stores."""
    review = tmp_path / "id_review"
    review.mkdir()
    save_tracklets(_store(kind="mouse", ids=(0, 1), n_frames=8), str(review))
    save_tracklets(_store(kind="object", ids=(0,), n_frames=8), str(review))
    write_tracklets_identity_status(
        str(review), corrected=True, accepted=True, has_raw=False
    )
    (review / "subjects.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "subjects": [
                    {
                        "subject_id": 0,
                        "animal_kind": "mouse",
                        "display_name": "resident",
                        "role": "R",
                        "color": "#4FC3F7",
                        "track_id": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "clip.avi").write_bytes(b"fake")
    return review


def test_metadata_from_json_without_opening_npz(tmp_path: Path, monkeypatch):
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        load_annotate_session_metadata,
    )
    from LabGym.id_review import tracklets as tracklets_mod

    review = _package_two_kinds_accepted(tmp_path)

    def boom(*_a, **_k):
        raise AssertionError("session metadata must not open npz")

    monkeypatch.setattr(tracklets_mod.np, "load", boom)

    meta = load_annotate_session_metadata(str(review), video_hint=str(tmp_path / "clip.avi"))
    assert meta.n_frames == 8
    assert set(meta.kinds) == {"mouse", "object"}
    assert [s.display_name for s in meta.subjects] == ["resident"]
    assert meta.accepted is True
    assert meta.video_path.endswith("clip.avi")


def test_metadata_survives_missing_npz(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        load_annotate_session_metadata,
    )

    review = _package_two_kinds_accepted(tmp_path)
    (review / "mouse_tracklets.npz").unlink()
    meta = load_annotate_session_metadata(str(review))
    assert meta.n_frames == 8
    assert "mouse" in meta.kinds


def test_session_load_emits_metadata_before_tracklets_and_frame(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        run_annotate_session_load,
    )

    review = _package_two_kinds_accepted(tmp_path)
    order: list[str] = []

    def on_metadata(_meta):
        order.append("metadata")

    def on_tracklets(loaded):
        order.append("tracklets")
        assert set(loaded.stores) == {"mouse", "object"}
        assert loaded.stores["mouse"].contours is None

    def on_frame(_frame):
        order.append("frame")

    state = run_annotate_session_load(
        str(review),
        video_path=str(tmp_path / "clip.avi"),
        on_metadata=on_metadata,
        on_tracklets=on_tracklets,
        on_frame=on_frame,
        read_frame=lambda **_k: (np.zeros((4, 4, 3), dtype=np.uint8), None),
    )
    assert state == "ok"
    assert order[0] == "metadata"
    assert "tracklets" in order
    assert "frame" in order


def test_session_load_loads_every_overlay_kind(tmp_path: Path, monkeypatch):
    from LabGym.annotator.core.tracklets_bridge import load_tracklets_for_annotator
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        run_annotate_session_load,
    )
    from LabGym.id_review.tracklets import load_tracklets as real_load

    review = _package_two_kinds_accepted(tmp_path)
    opened: list[str] = []

    def wrapped(directory, animal_kind, prefix=None):
        opened.append(animal_kind)
        return real_load(directory, animal_kind, prefix=prefix)

    monkeypatch.setattr("LabGym.id_review.tracklets.load_tracklets", wrapped)
    got = {"kinds": None}

    state = run_annotate_session_load(
        str(review),
        on_tracklets=lambda loaded: got.__setitem__("kinds", set(loaded.stores)),
        read_frame=lambda **_k: (np.zeros((2, 2, 3), dtype=np.uint8), None),
        load_tracklets=lambda d, **k: load_tracklets_for_annotator(d),
    )
    assert state == "ok"
    assert got["kinds"] == {"mouse", "object"}
    assert set(opened) == {"mouse", "object"}


def test_session_load_uses_published_tracklets_not_raw(tmp_path: Path, monkeypatch):
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        run_annotate_session_load,
    )

    review = _package_two_kinds_accepted(tmp_path)

    def boom(*_a, **_k):
        raise AssertionError("Annotate must not load raw tracklets")

    monkeypatch.setattr("LabGym.id_review.raw_store.load_tracklets", boom)
    got = {"loaded": None}
    state = run_annotate_session_load(
        str(review),
        skip_video=True,
        on_tracklets=lambda loaded: got.__setitem__("loaded", loaded),
    )
    assert state == "ok"
    assert got["loaded"] is not None
    assert set(got["loaded"].stores) == {"mouse", "object"}


def test_session_load_frame_and_tracklets_are_independent(tmp_path: Path):
    from LabGym.annotator.core.tracklets_bridge import load_tracklets_for_annotator
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        run_annotate_session_load,
    )

    review = _package_two_kinds_accepted(tmp_path)
    tracks_started = threading.Event()
    frame_started = threading.Event()
    order: list[str] = []

    def slow_tracks(review_dir, **_k):
        order.append("tracks_start")
        tracks_started.set()
        assert frame_started.wait(2.0)
        return load_tracklets_for_annotator(review_dir)

    def slow_frame(**_k):
        order.append("frame_start")
        frame_started.set()
        assert tracks_started.wait(2.0)
        return np.zeros((2, 2, 3), dtype=np.uint8), None

    state = run_annotate_session_load(
        str(review),
        on_metadata=lambda _m: order.append("metadata"),
        on_tracklets=lambda _l: order.append("tracklets"),
        on_frame=lambda _f: order.append("frame"),
        load_tracklets=slow_tracks,
        read_frame=slow_frame,
    )
    assert state == "ok"
    assert order[0] == "metadata"
    assert "tracks_start" in order and "frame_start" in order


def test_cancel_after_metadata_skips_tracklets_and_frame(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        run_annotate_session_load,
    )

    review = _package_two_kinds_accepted(tmp_path)
    cancel = threading.Event()
    later: list[str] = []

    def on_metadata(_meta):
        later.append("metadata")
        cancel.set()

    state = run_annotate_session_load(
        str(review),
        cancel_event=cancel,
        on_metadata=on_metadata,
        on_tracklets=lambda *_a: later.append("tracklets"),
        on_frame=lambda *_a: later.append("frame"),
        read_frame=lambda **_k: (np.zeros((2, 2, 3), dtype=np.uint8), None),
    )
    assert state == "cancelled"
    assert later == ["metadata"]


def test_corrupt_npz_keeps_metadata(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        run_annotate_session_load,
    )

    review = _package_two_kinds_accepted(tmp_path)
    (review / "mouse_tracklets.npz").write_bytes(b"not-an-npz")
    got = {"meta": None, "tracks_err": None, "tracks": None}

    state = run_annotate_session_load(
        str(review),
        on_metadata=lambda m: got.__setitem__("meta", m),
        on_tracklets=lambda l: got.__setitem__("tracks", l),
        on_tracklets_error=lambda m: got.__setitem__("tracks_err", m),
        read_frame=lambda **_k: (None, "video unavailable: clip.avi"),
        on_video_error=lambda m: got.__setitem__("video_err", m),
    )
    assert state == "ok"
    assert got["meta"] is not None
    assert got["meta"].n_frames == 8
    assert got["tracks"] is None
    assert got["tracks_err"]


def test_missing_video_does_not_drop_metadata(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        run_annotate_session_load,
    )

    review = _package_two_kinds_accepted(tmp_path)
    got = {"meta": None, "video_err": None, "tracks": None}

    state = run_annotate_session_load(
        str(review),
        on_metadata=lambda m: got.__setitem__("meta", m),
        on_tracklets=lambda l: got.__setitem__("tracks", l),
        on_video_error=lambda m: got.__setitem__("video_err", m),
        read_frame=lambda **_k: (None, "video unavailable: missing.avi"),
    )
    assert state == "ok"
    assert got["meta"] is not None
    assert got["tracks"] is not None
    assert "video unavailable" in (got["video_err"] or "")


def test_no_package_metadata_does_not_invent_n_frames():
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        run_annotate_session_load,
    )

    got = {"meta": None}
    state = run_annotate_session_load(
        "",
        video_path="clip.avi",
        skip_video=True,
        on_metadata=lambda m: got.__setitem__("meta", m),
    )
    assert state == "ok"
    assert got["meta"] is not None
    assert got["meta"].n_frames is None
    assert got["meta"].kinds == []


def test_sidecar_fills_n_frames_when_no_package(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        run_annotate_session_load,
    )

    sidecar = tmp_path / "clip.annotations.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "video_path": "clip.avi",
                "fps": 12.0,
                "total_frames": 40,
                "behaviors": [],
                "subjects": [{"subject_id": 0, "animal_kind": "mouse"}],
                "bouts": {"0": {}},
            }
        ),
        encoding="utf-8",
    )
    got = {"meta": None}
    state = run_annotate_session_load(
        "",
        video_path=str(tmp_path / "clip.avi"),
        annotations_path=str(sidecar),
        skip_video=True,
        on_metadata=lambda m: got.__setitem__("meta", m),
    )
    assert state == "ok"
    assert got["meta"].n_frames == 40
    assert got["meta"].fps == 12.0
    assert got["meta"].annotations_data is not None


def test_worker_does_not_pass_package_length_as_video_length(tmp_path: Path):
    from LabGym.annotator.core.tracklets_bridge import load_tracklets_for_annotator
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        run_annotate_session_load,
    )

    review = _package_two_kinds_accepted(tmp_path)
    seen: dict = {}

    def spy(directory, **kwargs):
        seen.update(kwargs)
        return load_tracklets_for_annotator(directory)

    state = run_annotate_session_load(
        str(review), skip_video=True, load_tracklets=spy
    )
    assert state == "ok"
    assert seen.get("video_total_frames") is None


def test_runner_source_does_not_use_analysis_job_queue():
    import ast
    from pathlib import Path as P

    from LabGym.gui_pyside.workbenches.categorizer import annotate_session_load as sl

    tree = ast.parse(P(sl.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("gui_pyside.jobs" in name for name in imported)
    assert "SequentialJobQueue" not in sl.__dict__


def _make_tab(tmp_path: Path, *, accepted: bool = True):
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.project.model import Project
    from LabGym.gui_pyside.workbenches.categorizer.annotate_ethogram_tab import (
        AnnotateEthogramTab,
    )
    from LabGym.id_review.raw_store import save_raw_tracklets

    vid = tmp_path / "clip.avi"
    vid.write_bytes(b"fake")
    if accepted:
        _package_two_kinds_accepted(tmp_path)
    else:
        review = tmp_path / "id_review"
        review.mkdir()
        save_raw_tracklets(
            review, {"mouse": _store(kind="mouse", ids=(0, 1), n_frames=8)}
        )
        write_tracklets_identity_status(
            str(review), corrected=False, accepted=False, has_raw=True
        )
    proj = Project.new(name="p", root_dir=str(tmp_path))
    proj.add_video(str(vid))
    proj.defaults.current_video = str(vid)
    proj.defaults.behavior_mode = 0
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)
    return AnnotateEthogramTab(ctrl)


@pytest.mark.gui
def test_annotate_load_does_not_enqueue_job_queue(tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from LabGym.gui_pyside.jobs.sequential_queue import SequentialJobQueue

    hits: list[str] = []
    monkeypatch.setattr(
        SequentialJobQueue, "start", lambda *a, **k: hits.append("start")
    )
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: 0)

    def boom_load(self, path):
        raise AssertionError("session load must not open VideoHandler on the UI thread")

    monkeypatch.setattr(
        "LabGym.annotator.core.video_handler.VideoHandler.load", boom_load
    )

    tab = _make_tab(tmp_path)
    started = tab.load_current()
    assert started is True
    thread = tab._session_thread
    app = QApplication.instance()
    assert app is not None
    if thread is not None:
        thread.wait(8000)
    for _ in range(20):
        app.processEvents()
        if tab.annotator._loaded_tracklets is not None and getattr(
            tab.annotator._loaded_tracklets, "stores", None
        ):
            break
    assert hits == []
    assert tab.annotator.manager is not None
    assert tab.annotator.manager.session.total_frames == 8
    names = [s.display_name for s in tab.annotator.manager.session.subjects]
    assert "resident" in names
    loaded = tab.annotator._loaded_tracklets
    assert loaded is not None
    assert set(loaded.stores) == {"mouse", "object"}
    assert tab.annotator.video.is_loaded is False
    assert tab.annotator.has_unsaved_ethogram() is False
    tab.drop_session()


@pytest.mark.gui
def test_annotate_cancel_aborts_in_flight_load(tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from LabGym.gui_pyside.workbenches.categorizer import annotate_session_load as sl
    from LabGym.gui_pyside.workbenches.categorizer.annotate_ethogram_tab import (
        AnnotateEthogramTab,
    )

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: 0)
    gate = threading.Event()
    real_run = sl.run_annotate_session_load

    def gated_run(*args, **kwargs):
        kwargs = dict(kwargs)
        orig = kwargs.get("load_tracklets")

        def blocked(review_dir, **kw):
            gate.wait(5.0)
            if orig is not None:
                return orig(review_dir, **kw)
            from LabGym.annotator.core.tracklets_bridge import (
                load_tracklets_for_annotator,
            )

            return load_tracklets_for_annotator(review_dir)

        kwargs["load_tracklets"] = blocked
        kwargs["skip_video"] = True
        return real_run(*args, **kwargs)

    monkeypatch.setattr(sl, "run_annotate_session_load", gated_run)

    tab = _make_tab(tmp_path)
    assert tab.load_current() is True
    thread = tab._session_thread
    assert thread is not None
    tab._cancel_session_load()
    gate.set()
    thread.wait(5000)
    app = QApplication.instance()
    assert app is not None
    app.processEvents()
    assert tab._session_thread is None
    tab.drop_session()


@pytest.mark.gui
def test_video_combo_during_load_replaces(tmp_path: Path, monkeypatch):
    import sys

    from PySide6.QtWidgets import QApplication, QMessageBox

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.project.model import Project
    from LabGym.gui_pyside.workbenches.categorizer import annotate_session_load as sl
    from LabGym.gui_pyside.workbenches.categorizer.annotate_ethogram_tab import (
        AnnotateEthogramTab,
    )
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        AnnotateSessionMetadata,
    )

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: 0)
    monkeypatch.setattr(
        "LabGym.gui_pyside.workbenches.categorizer.annotate_ethogram_tab.may_use_downstream",
        lambda *a, **k: True,
    )

    vid_a = tmp_path / "a.avi"
    vid_b = tmp_path / "b.avi"
    vid_a.write_bytes(b"x")
    vid_b.write_bytes(b"y")
    proj = Project.new(name="p", root_dir=str(tmp_path))
    proj.add_video(str(vid_a))
    proj.add_video(str(vid_b))
    proj.defaults.current_video = str(vid_a)
    ctrl = ProjectController()
    ctrl.replace(proj, dirty=False)

    started: list[str] = []
    gens_seen: list[int] = []
    block = threading.Event()

    def gated_run(*args, **kwargs):
        started.append(str(kwargs.get("video_path") or args[0] if args else ""))
        cancel = kwargs.get("cancel_event")
        on_metadata = kwargs.get("on_metadata")
        if on_metadata is not None:
            on_metadata(
                AnnotateSessionMetadata(
                    video_path=str(kwargs.get("video_path") or ""),
                    annotations_path="",
                    review_dir="",
                    n_frames=8,
                    fps=10.0,
                    kinds=[],
                )
            )
        for _ in range(50):
            if cancel is not None and cancel.is_set():
                return "cancelled"
            if block.wait(0.05):
                break
        return "ok"

    monkeypatch.setattr(sl, "run_annotate_session_load", gated_run)

    tab = AnnotateEthogramTab(ctrl)
    orig_start = tab._start_session_load

    def tracking_start(window, ctx):
        orig_start(window, ctx)
        gens_seen.append(tab._session_gen)

    monkeypatch.setattr(tab, "_start_session_load", tracking_start)
    app = QApplication.instance()
    assert app is not None
    try:
        assert tab.load_current() is True
        assert tab._session_thread is not None
        first_target = tab._session_window
        other = None
        for i in range(tab.combo_video.count()):
            data = str(tab.combo_video.itemData(i) or "")
            if data.endswith("b.avi"):
                other = i
                break
        assert other is not None
        tab.combo_video.setCurrentIndex(int(other))
        for _ in range(40):
            app.processEvents()
            if len(started) >= 2:
                break
            time.sleep(0.05)
        assert [p.endswith("a.avi") for p in started].count(True) >= 1
        assert [p.endswith("b.avi") for p in started].count(True) >= 1
        assert tab._session_window is first_target
        assert len(gens_seen) >= 2
        assert gens_seen[-1] != gens_seen[0]
    finally:
        block.set()
        thread = tab._session_thread
        if thread is not None:
            thread.wait(5000)
        tab.drop_session()


@pytest.mark.gui
def test_accepted_identities_gate_does_not_open_video(tmp_path: Path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QMessageBox

    from LabGym.annotator.ui.main_window import MainWindow

    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: 0)
    opened: list[str] = []
    monkeypatch.setattr(
        MainWindow, "open_session_video", lambda self, path: opened.append(path)
    )
    monkeypatch.setattr(
        MainWindow,
        "begin_session_from_metadata",
        lambda self, **k: opened.append("meta"),
    )

    tab = _make_tab(tmp_path, accepted=False)
    ok = tab.load_current()
    app = QApplication.instance()
    assert app is not None
    app.processEvents()
    assert ok is False
    assert opened == []
    assert tab._session_thread is None
    assert tab.annotator.video.is_loaded is False
    assert tab.annotator.manager is None
    tab.drop_session()


@pytest.mark.gui
def test_sidecar_bouts_survive_subset_subjects_and_stay_saved(tmp_path: Path):
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.annotator.core.tracklets_bridge import LoadedTracklets
    from LabGym.annotator.ui.main_window import MainWindow
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        loaded_tracklets_placeholder,
        overlay_subject_display,
    )
    from LabGym.identity.package import SubjectRecord
    from LabGym.id_review.types import SCHEMA_VERSION, TrackletStore

    sidecar = {
        "schema_version": 2,
        "video_path": "clip.avi",
        "fps": 10.0,
        "total_frames": 8,
        "behaviors": [{"name": "grooming", "color": "#00AAFF", "hotkey": "1"}],
        "subjects": [
            {
                "subject_id": 0,
                "animal_kind": "mouse",
                "display_name": "a",
                "color": "#4FC3F7",
            },
            {
                "subject_id": 1,
                "animal_kind": "mouse",
                "display_name": "b",
                "color": "#FF8A65",
            },
        ],
        "bouts": {
            "0": {"grooming": [{"start_frame": 1, "end_frame": 2}]},
            "1": {"grooming": [{"start_frame": 3, "end_frame": 4}]},
        },
        "interaction_bouts": {},
        "exclusive_mode": False,
        "behavior_mode": 0,
    }
    recs = [
        SubjectRecord(
            subject_id=0,
            animal_kind="mouse",
            display_name="resident",
            track_id=0,
        )
    ]
    win = MainWindow()
    win.begin_session_from_metadata(
        video_path="clip.avi",
        n_frames=8,
        fps=10.0,
        subjects=recs,
        annotations_data=sidecar,
        tracklets_placeholder=loaded_tracklets_placeholder(".", recs),
    )
    overlay_subject_display(win.manager.session, recs)
    assert "1" in win.manager.session.bouts
    assert win.manager.session.bouts["1"]["grooming"][0].start_frame == 3
    assert win.has_unsaved_ethogram() is False

    n = 8
    store = TrackletStore(
        schema_version=SCHEMA_VERSION,
        animal_kind="mouse",
        ids=[0, 1],
        n_frames=n,
        centers=np.zeros((2, n, 2)),
        valid=np.ones((2, n), dtype=bool),
        heights=np.ones((2, n)),
        contours=None,
        meta={"fps": 10.0},
    )
    loaded = LoadedTracklets(
        directory=".",
        stores={"mouse": store},
        subjects=list(win.manager.session.subjects),
        subject_to_track={0: ("mouse", 0), 1: ("mouse", 1)},
    )
    win.apply_loaded_tracklets(loaded)
    assert win.manager.session.bouts["1"]["grooming"][0].start_frame == 3
    assert win.has_unsaved_ethogram() is False
    win.drop_session()


@pytest.mark.gui
def test_timeline_length_without_live_decoder():
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.annotator.ui.main_window import MainWindow

    win = MainWindow()
    win.begin_session_from_metadata(
        video_path="clip.avi", n_frames=8, fps=10.0
    )
    assert win.video.is_loaded is False
    assert win.timeline._total_frames() == 8
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QPaintEvent

    win.timeline.resize(400, 48)
    win.timeline.paintEvent(QPaintEvent(QRect(0, 0, 400, 48)))
    win.drop_session()


@pytest.mark.gui
def test_step_frame_opens_live_decoder(monkeypatch):
    import sys

    from PySide6.QtWidgets import QApplication, QMessageBox

    QApplication.instance() or QApplication(sys.argv)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: 0)

    from LabGym.annotator.ui.main_window import MainWindow

    win = MainWindow()
    win.begin_session_from_metadata(
        video_path="clip.avi", n_frames=8, fps=10.0
    )
    opened: list[str] = []
    monkeypatch.setattr(
        win, "ensure_live_video", lambda: opened.append("open") or "unavailable"
    )
    win.step_frame(1)
    assert opened == ["open"]
    assert win.video.is_loaded is False
    win.drop_session()


@pytest.mark.gui
def test_bout_toggle_and_save_after_metadata_without_play(
    tmp_path: Path, monkeypatch
):
    import sys

    from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

    QApplication.instance() or QApplication(sys.argv)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: 0)

    from LabGym.annotator.ui.main_window import MainWindow

    sidecar = {
        "schema_version": 2,
        "video_path": str(tmp_path / "clip.avi"),
        "fps": 10.0,
        "total_frames": 8,
        "behaviors": [{"name": "grooming", "color": "#00AAFF", "hotkey": "1"}],
        "subjects": [
            {
                "subject_id": 0,
                "animal_kind": "mouse",
                "display_name": "a",
                "color": "#4FC3F7",
            }
        ],
        "bouts": {"0": {"grooming": []}},
        "interaction_bouts": {},
        "exclusive_mode": False,
        "behavior_mode": 0,
    }
    win = MainWindow()
    win.begin_session_from_metadata(
        video_path=str(tmp_path / "clip.avi"),
        n_frames=8,
        fps=10.0,
        annotations_data=sidecar,
        annotations_path=str(tmp_path / "clip.annotations.json"),
    )
    assert win.video.is_loaded is False
    win._toggle_bout_for_behavior("grooming")
    assert win.has_unsaved_ethogram() is True
    out = tmp_path / "saved.json"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), "JSON (*.json)")
    )
    win.save_annotations()
    assert out.is_file()
    assert win.video.is_loaded is False
    win.drop_session()


def test_in_flight_tracklets_load_is_cancellable(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.categorizer.annotate_session_load import (
        run_annotate_session_load,
    )

    review = _package_two_kinds_accepted(tmp_path)
    cancel = threading.Event()
    started = threading.Event()
    called: list[str] = []

    def blocked_tracks(_review_dir, **_k):
        started.set()
        time.sleep(0.05)
        if cancel.is_set():
            return None
        called.append("finished")
        raise AssertionError("tracklets load should not finish")

    def worker():
        run_annotate_session_load(
            str(review),
            skip_video=True,
            cancel_event=cancel,
            load_tracklets=blocked_tracks,
            on_tracklets=lambda *_a: called.append("delivered"),
        )

    t = threading.Thread(target=worker)
    t.start()
    assert started.wait(2.0)
    cancel.set()
    t.join(2.0)
    assert t.is_alive() is False
    assert "delivered" not in called
