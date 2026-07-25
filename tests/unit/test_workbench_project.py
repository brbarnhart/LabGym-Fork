"""Unit tests for workbench Project model (Phase 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from LabGym.gui_pyside.project.model import (
    PROJECT_SCHEMA_VERSION,
    Project,
    ProjectDefaults,
    ProjectPaths,
    ProjectVideo,
)


def test_project_roundtrip(tmp_path: Path):
    proj = Project.new(name="Exp1", root_dir=str(tmp_path))
    proj.add_video(str(tmp_path / "a.avi"))
    proj.add_video(str(tmp_path / "b.avi"))
    proj.defaults.behavior_mode = 2
    proj.defaults.window_length = 21
    proj.notes = "hello"

    path = tmp_path / "Exp1.labproj.json"
    proj.save(str(path))
    assert path.is_file()

    loaded = Project.load(path)
    assert loaded.schema_version == PROJECT_SCHEMA_VERSION
    assert loaded.name == "Exp1"
    assert loaded.root_dir == str(tmp_path)
    assert len(loaded.videos) == 2
    assert loaded.defaults.behavior_mode == 2
    assert loaded.defaults.window_length == 21
    assert loaded.notes == "hello"
    assert loaded.file_path == str(path.resolve())


def test_add_video_relative_to_root(tmp_path: Path):
    vid = tmp_path / "clips" / "x.avi"
    vid.parent.mkdir()
    vid.write_bytes(b"")
    proj = Project.new(root_dir=str(tmp_path))
    entry = proj.add_video(str(vid))
    # Prefer relative path under root (OS-specific separators)
    assert Path(entry.path) == Path("clips") / "x.avi"
    assert entry.resolved_path(proj.root_dir).resolve() == vid.resolve()


def test_add_video_dedupes(tmp_path: Path):
    vid = tmp_path / "a.avi"
    vid.write_bytes(b"")
    proj = Project.new(root_dir=str(tmp_path))
    proj.add_video(str(vid))
    proj.add_video(str(vid))
    assert len(proj.videos) == 1


def test_newer_schema_rejected(tmp_path: Path):
    path = tmp_path / "p.labproj.json"
    path.write_text(
        json.dumps({"schema_version": 999, "name": "x", "videos": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="newer than supported"):
        Project.load(path)


def test_paths_and_defaults_defaults():
    p = ProjectPaths.from_dict({})
    assert p.detection_output_root == "detection"
    d = ProjectDefaults.from_dict({})
    assert d.exclusive_mode is True
    assert d.sampling == "dense_in_bout"


def test_enabled_videos_filter():
    proj = Project.new()
    proj.videos = [
        ProjectVideo("a.avi", enabled=True),
        ProjectVideo("b.avi", enabled=False),
    ]
    assert len(proj.enabled_videos()) == 1
    assert proj.enabled_videos()[0].path == "a.avi"


def test_resolve_video_context_annotations_and_examples(tmp_path: Path):
    from LabGym.gui_pyside.project.paths import (
        resolve_video_context,
        set_current_video,
    )

    vid = tmp_path / "clip.avi"
    vid.write_bytes(b"")
    ann = tmp_path / "clip.annotations.json"
    ann.write_text("{}", encoding="utf-8")
    tracks = tmp_path / "id_review"
    tracks.mkdir()
    (tracks / "mouse_tracklets.npz").write_bytes(b"")

    proj = Project.new(name="p", root_dir=str(tmp_path))
    proj.add_video(str(vid))
    set_current_video(proj, str(vid))
    ctx = resolve_video_context(proj)
    assert Path(ctx.video_path).resolve() == vid.resolve()
    assert Path(ctx.annotations_path).resolve() == ann.resolve()
    assert ctx.annotations_exists
    assert Path(ctx.tracklets_dir).resolve() == tracks.resolve()
    assert ctx.tracklets_exists
    assert "clip_examples_from_ethogram" in ctx.examples_out_dir


def test_resolve_detection_dir_override(tmp_path: Path):
    from LabGym.gui_pyside.project.paths import resolve_video_context

    vid = tmp_path / "v.avi"
    vid.write_bytes(b"")
    det = tmp_path / "custom_det"
    det.mkdir()
    (det / "x_tracklets.npz").write_bytes(b"")

    proj = Project.new(root_dir=str(tmp_path))
    entry = proj.add_video(str(vid))
    entry.detection_dir = "custom_det"
    proj.defaults.current_video = entry.path
    ctx = resolve_video_context(proj)
    assert Path(ctx.tracklets_dir).resolve() == det.resolve()
