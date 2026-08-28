"""Review IDs session load: metadata first, visible kind, cancel/replace."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from LabGym.id_review.dataset import dumps_jsonl_line
from LabGym.id_review.raw_store import save_raw_tracklets
from LabGym.id_review.tracklets import save_tracklets
from LabGym.id_review.types import ContactEvent, TrackletStore

from tests.unit.tracklet_fixtures import make_tracklet_store


def _store(*, n_frames: int = 8, kind: str = "mouse", ids=(0, 1)) -> TrackletStore:
    return make_tracklet_store(
        n_frames=n_frames,
        kind=kind,
        ids=ids,
        size=4,
        x_scale=40.0,
        y_scale=1.0,
        video="clip.avi",
        fps=10.0,
    )


def _write_event(review: Path, *, kind: str = "mouse") -> None:
    ev = ContactEvent(
        event_id="e000001",
        animal_kind=kind,
        involved_ids=[0, 1],
        start_frame=1,
        end_frame=3,
        pre_frame=0,
        post_frame=4,
        risk_score=0.8,
        risk_flags=["close_contact"],
        video="clip.avi",
        fps=10.0,
    )
    (review / "events.jsonl").write_text(
        dumps_jsonl_line(ev.to_dict()) + "\n", encoding="utf-8"
    )


def _package_two_kinds(tmp_path: Path) -> Path:
    review = tmp_path / "id_review"
    review.mkdir()
    save_raw_tracklets(
        review,
        {
            "mouse": _store(kind="mouse", ids=(0, 1), n_frames=8),
            "object": _store(kind="object", ids=(0,), n_frames=8),
        },
    )
    _write_event(review)
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
    from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
        load_review_package_metadata,
    )
    from LabGym.id_review import tracklets as tracklets_mod

    review = _package_two_kinds(tmp_path)

    def boom(*_a, **_k):
        raise AssertionError("session metadata must not open npz")

    monkeypatch.setattr(tracklets_mod.np, "load", boom)

    meta = load_review_package_metadata(str(review))
    assert meta.n_frames == 8
    assert meta.animal_kind == "mouse"
    assert meta.kinds == ["mouse", "object"]
    assert [s.display_name for s in meta.subjects] == ["resident"]
    assert len(meta.events) == 1
    assert meta.events[0].event_id == "e000001"
    assert meta.has_raw is True
    assert meta.kind_info["mouse"].ids == [0, 1]


def test_metadata_survives_missing_npz(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
        load_review_package_metadata,
    )

    review = _package_two_kinds(tmp_path)
    (review / "raw" / "mouse_tracklets.npz").unlink()
    meta = load_review_package_metadata(str(review))
    assert meta.n_frames == 8
    assert "mouse" in meta.kinds


def test_visible_kind_load_does_not_open_other_kinds(tmp_path: Path, monkeypatch):
    from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
        load_review_kind_store,
    )
    from LabGym.id_review.tracklets import load_tracklets as real_load

    review = _package_two_kinds(tmp_path)
    opened: list[str] = []

    def wrapped(directory, animal_kind, prefix=None):
        opened.append(animal_kind)
        return real_load(directory, animal_kind, prefix=prefix)

    monkeypatch.setattr(
        "LabGym.id_review.raw_store.load_tracklets", wrapped
    )
    store = load_review_kind_store(str(review), "mouse")
    assert store.animal_kind == "mouse"
    assert store.contours is None
    assert opened == ["mouse"]


def test_session_load_emits_metadata_before_kind_and_frame(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.detector.review_ids_session_load import (
        run_review_session_load,
    )

    review = _package_two_kinds(tmp_path)
    order: list[str] = []

    def on_metadata(_meta):
        order.append("metadata")

    def on_kind(kind, store):
        order.append(f"kind:{kind}")
        assert store.contours is None

    def on_frame(_frame):
        order.append("frame")

    state = run_review_session_load(
        str(review),
        on_metadata=on_metadata,
        on_kind=on_kind,
        on_frame=on_frame,
        read_frame=lambda **_k: (np.zeros((4, 4, 3), dtype=np.uint8), None),
    )
    assert state == "ok"
    assert order[0] == "metadata"
    assert "kind:mouse" in order
    assert "kind:object" not in order
    assert "frame" in order


def test_session_load_frame_and_kind_are_independent(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.detector.review_ids_session_load import (
        run_review_session_load,
    )

    review = _package_two_kinds(tmp_path)
    kind_started = threading.Event()
    frame_started = threading.Event()
    order: list[str] = []

    def slow_kind(review_dir, kind):
        order.append("kind_start")
        kind_started.set()
        assert frame_started.wait(2.0)
        from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
            load_review_kind_store,
        )

        return load_review_kind_store(review_dir, kind)

    def slow_frame(**_k):
        order.append("frame_start")
        frame_started.set()
        assert kind_started.wait(2.0)
        return np.zeros((2, 2, 3), dtype=np.uint8), None

    state = run_review_session_load(
        str(review),
        on_metadata=lambda _m: order.append("metadata"),
        on_kind=lambda k, _s: order.append(f"kind:{k}"),
        on_frame=lambda _f: order.append("frame"),
        load_kind=slow_kind,
        read_frame=slow_frame,
    )
    assert state == "ok"
    assert order[0] == "metadata"
    assert "kind_start" in order and "frame_start" in order


def test_cancel_after_metadata_skips_kind_and_frame(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.detector.review_ids_session_load import (
        run_review_session_load,
    )

    review = _package_two_kinds(tmp_path)
    cancel = threading.Event()
    later: list[str] = []

    def on_metadata(_meta):
        later.append("metadata")
        cancel.set()

    state = run_review_session_load(
        str(review),
        cancel_event=cancel,
        on_metadata=on_metadata,
        on_kind=lambda *_a: later.append("kind"),
        on_frame=lambda *_a: later.append("frame"),
        read_frame=lambda **_k: (np.zeros((2, 2, 3), dtype=np.uint8), None),
    )
    assert state == "cancelled"
    assert later == ["metadata"]


def test_corrupt_npz_keeps_metadata(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.detector.review_ids_session_load import (
        run_review_session_load,
    )

    review = _package_two_kinds(tmp_path)
    (review / "raw" / "mouse_tracklets.npz").write_bytes(b"not-an-npz")
    got = {"meta": None, "kind_err": None, "kind": None}

    def on_metadata(meta):
        got["meta"] = meta

    def on_kind_error(kind, message):
        got["kind_err"] = (kind, message)

    state = run_review_session_load(
        str(review),
        on_metadata=on_metadata,
        on_kind=lambda k, s: got.__setitem__("kind", k),
        on_kind_error=on_kind_error,
        read_frame=lambda **_k: (None, "video unavailable: clip.avi"),
        on_video_error=lambda m: got.__setitem__("video_err", m),
    )
    assert state == "ok"
    assert got["meta"] is not None
    assert got["meta"].n_frames == 8
    assert got["kind"] is None
    assert got["kind_err"] is not None
    assert got["kind_err"][0] == "mouse"


def test_missing_video_does_not_drop_metadata(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.detector.review_ids_session_load import (
        run_review_session_load,
    )

    review = _package_two_kinds(tmp_path)
    got = {"meta": None, "video_err": None, "kind": None}

    state = run_review_session_load(
        str(review),
        on_metadata=lambda m: got.__setitem__("meta", m),
        on_kind=lambda k, _s: got.__setitem__("kind", k),
        on_video_error=lambda m: got.__setitem__("video_err", m),
        read_frame=lambda **_k: (None, "video unavailable: missing.avi"),
    )
    assert state == "ok"
    assert got["meta"] is not None
    assert got["kind"] == "mouse"
    assert "video unavailable" in (got["video_err"] or "")


def test_kind_only_load_is_cancellable(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.detector.review_ids_session_load import (
        run_review_session_load,
    )

    review = _package_two_kinds(tmp_path)
    cancel = threading.Event()
    started = threading.Event()
    called: list[str] = []

    def blocked_kind(_review_dir, kind):
        started.set()
        time.sleep(0.05)
        if cancel.is_set():
            raise RuntimeError("should be skipped after cancel")
        called.append(kind)
        raise AssertionError("kind load should not finish")

    def worker():
        run_review_session_load(
            str(review),
            visible_kind="object",
            skip_metadata=True,
            skip_video=True,
            cancel_event=cancel,
            load_kind=blocked_kind,
            on_kind=lambda *_a: called.append("delivered"),
        )

    t = threading.Thread(target=worker)
    t.start()
    assert started.wait(2.0)
    cancel.set()
    t.join(2.0)
    assert t.is_alive() is False
    assert "delivered" not in called


def test_runner_source_does_not_use_analysis_job_queue():
    import ast
    from pathlib import Path

    from LabGym.gui_pyside.workbenches.detector import review_ids_session_load as sl

    tree = ast.parse(Path(sl.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any("gui_pyside.jobs" in name for name in imported)
    assert "SequentialJobQueue" not in sl.__dict__


def test_load_review_package_still_loads_packed_kinds(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
        load_review_package,
    )

    review = _package_two_kinds(tmp_path)
    pkg = load_review_package(str(review))
    assert set(pkg.stores) == {"mouse", "object"}
    assert pkg.stores["mouse"] is pkg.baseline_stores["mouse"]
    assert pkg.stores["mouse"].contours is None


@pytest.mark.gui
def test_review_ids_load_does_not_enqueue_job_queue(tmp_path: Path, monkeypatch):
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.jobs.sequential_queue import SequentialJobQueue
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.workbenches.detector.review_ids_tab import ReviewIdsTab

    hits: list[str] = []
    monkeypatch.setattr(
        SequentialJobQueue, "start", lambda *a, **k: hits.append("start")
    )
    review = _package_two_kinds(tmp_path)
    tab = ReviewIdsTab(ProjectController())
    started = tab.load_package(str(review))
    assert started is True
    thread = tab._session_thread
    if thread is not None:
        thread.wait(8000)
    app = QApplication.instance()
    assert app is not None
    app.processEvents()
    assert hits == []
    assert tab.n_frames == 8
    assert tab.animal_kind == "mouse"
    assert "mouse" in tab._stores
    assert "object" not in tab._stores
    tab.drop_session()


@pytest.mark.gui
def test_review_ids_cancel_aborts_in_flight_load(tmp_path: Path, monkeypatch):
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.workbenches.detector import review_ids_session_load as sl
    from LabGym.gui_pyside.workbenches.detector.review_ids_tab import ReviewIdsTab

    review = _package_two_kinds(tmp_path)
    gate = threading.Event()
    real_run = sl.run_review_session_load

    def gated_run(*args, **kwargs):
        kwargs = dict(kwargs)
        orig_kind = kwargs.get("load_kind")

        def blocked(review_dir, kind):
            gate.wait(5.0)
            if orig_kind is not None:
                return orig_kind(review_dir, kind)
            from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
                load_review_kind_store,
            )

            return load_review_kind_store(review_dir, kind)

        kwargs["load_kind"] = blocked
        return real_run(*args, **kwargs)

    monkeypatch.setattr(sl, "run_review_session_load", gated_run)

    tab = ReviewIdsTab(ProjectController())
    assert tab.load_package(str(review)) is True
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
def test_kind_combo_starts_cancellable_kind_load(tmp_path: Path, monkeypatch):
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
        load_review_kind_store,
        load_review_package_metadata,
    )
    from LabGym.gui_pyside.workbenches.detector.review_ids_tab import ReviewIdsTab

    review = _package_two_kinds(tmp_path)
    tab = ReviewIdsTab(ProjectController())
    meta = load_review_package_metadata(str(review))
    tab._apply_session_metadata(meta, gen=tab._session_gen)
    tab._stores["mouse"] = load_review_kind_store(str(review), "mouse")
    tab.animal_kind = "mouse"
    started: list[str] = []

    def fake_start(self_tab, review_dir, **kwargs):
        started.append(kwargs.get("visible_kind") or "")

    monkeypatch.setattr(ReviewIdsTab, "_start_session_load", fake_start)
    tab.kind_combo.blockSignals(True)
    tab.kind_combo.clear()
    tab.kind_combo.addItems(["mouse", "object"])
    tab.kind_combo.blockSignals(False)
    tab.kind_combo.setCurrentText("object")
    assert started == ["object"]


@pytest.mark.gui
def test_first_switch_edit_clones_packed_stores(tmp_path: Path):
    import sys

    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication(sys.argv)

    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
        load_review_kind_store,
        load_review_package_metadata,
    )
    from LabGym.gui_pyside.workbenches.detector.review_ids_tab import ReviewIdsTab

    review = _package_two_kinds(tmp_path)
    tab = ReviewIdsTab(ProjectController())
    meta = load_review_package_metadata(str(review))
    tab._apply_session_metadata(meta, gen=tab._session_gen)
    loaded = load_review_kind_store(str(review), "mouse")
    tab._on_session_kind("mouse", loaded, tab._session_gen)
    assert tab._stores["mouse"] is tab._baseline_stores["mouse"]
    tab._ensure_mutable_stores()
    assert tab._stores["mouse"] is not tab._baseline_stores["mouse"]
    assert tab._stores["mouse"].contours is None
    tab._ensure_mutable_stores()
    cloned = tab._stores["mouse"]
    tab._ensure_mutable_stores()
    assert tab._stores["mouse"] is cloned
    tab.drop_session()
