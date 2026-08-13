"""Downstream ethogram / example-store check used before Review IDs save."""

from __future__ import annotations

from pathlib import Path

from LabGym.gui_pyside.project.model import Project
from LabGym.gui_pyside.project.paths import set_current_video
from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
    check_downstream_artifacts,
)


class _RaisingPath:
    """Path-like that fails when the filesystem lookup starts."""

    def __fspath__(self) -> str:
        raise OSError("simulated lookup failure")


def test_lookup_raise_is_failed_check_not_all_clear():
    result = check_downstream_artifacts(annotations_path=_RaisingPath())
    assert result.check_failed is True
    assert result.requires_confirm is True
    assert result.ethogram_path is None
    assert result.examples_path is None
    assert result.error


def test_project_lookup_raise_is_failed_check(monkeypatch):
    from LabGym.gui_pyside.workbenches.detector import review_ids_package as pkg

    def _boom(_project):
        raise RuntimeError("cannot resolve current video")

    monkeypatch.setattr(pkg, "current_video_path", _boom)
    result = check_downstream_artifacts(Project.new())
    assert result.check_failed is True
    assert result.requires_confirm is True
    assert result.ethogram_path is None
    assert result.examples_path is None
    assert "cannot resolve current video" in result.error


def test_existing_ethogram_requires_confirm_and_is_not_failed(tmp_path: Path):
    vid = tmp_path / "clip.avi"
    vid.write_bytes(b"")
    ann = tmp_path / "clip.annotations.json"
    ann.write_text("{}", encoding="utf-8")
    proj = Project.new(name="p", root_dir=str(tmp_path))
    proj.add_video(str(vid))
    set_current_video(proj, str(vid))

    result = check_downstream_artifacts(proj)
    assert result.check_failed is False
    assert result.requires_confirm is True
    assert Path(result.ethogram_path).resolve() == ann.resolve()
    assert result.examples_path is None


def test_existing_example_store_requires_confirm(tmp_path: Path):
    vid = tmp_path / "clip.avi"
    vid.write_bytes(b"")
    examples = tmp_path / "examples" / "clip_examples_from_ethogram"
    examples.mkdir(parents=True)
    (examples / "walk").mkdir()
    (examples / "walk" / "ex1.avi").write_bytes(b"")
    proj = Project.new(name="p", root_dir=str(tmp_path))
    proj.add_video(str(vid))
    set_current_video(proj, str(vid))

    result = check_downstream_artifacts(proj)
    assert result.check_failed is False
    assert result.requires_confirm is True
    assert result.ethogram_path is None
    assert Path(result.examples_path).resolve() == examples.resolve()


def test_missing_ethogram_and_examples_is_all_clear(tmp_path: Path):
    vid = tmp_path / "clip.avi"
    vid.write_bytes(b"")
    proj = Project.new(name="p", root_dir=str(tmp_path))
    proj.add_video(str(vid))
    set_current_video(proj, str(vid))

    result = check_downstream_artifacts(proj)
    assert result.check_failed is False
    assert result.requires_confirm is False
    assert result.ethogram_path is None
    assert result.examples_path is None
    assert result.error == ""


def test_empty_example_store_is_not_stale(tmp_path: Path):
    vid = tmp_path / "clip.avi"
    vid.write_bytes(b"")
    examples = tmp_path / "examples" / "clip_examples_from_ethogram"
    examples.mkdir(parents=True)
    proj = Project.new(name="p", root_dir=str(tmp_path))
    proj.add_video(str(vid))
    set_current_video(proj, str(vid))

    result = check_downstream_artifacts(proj)
    assert result.check_failed is False
    assert result.requires_confirm is False
    assert result.examples_path is None
