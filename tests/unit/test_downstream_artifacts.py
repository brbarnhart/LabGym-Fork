"""Downstream ethogram / example-store check used before Review IDs save."""

from __future__ import annotations

from pathlib import Path

from LabGym.gui_pyside.project.model import Project
from LabGym.gui_pyside.project.paths import set_current_video
from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
    DownstreamArtifactCheck,
    check_downstream_artifacts,
    should_confirm_stale_downstream,
    video_path_for_review_package,
)
from LabGym.identity.package import DETECT_JOB_FILENAME


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


def test_missing_video_path_is_failed_check_not_current_video(tmp_path: Path):
    other = tmp_path / "other.avi"
    other.write_bytes(b"")
    (tmp_path / "other.annotations.json").write_text("{}", encoding="utf-8")
    proj = Project.new(name="p", root_dir=str(tmp_path))
    proj.add_video(str(other))
    set_current_video(proj, str(other))

    result = check_downstream_artifacts(proj)
    assert result.check_failed is True
    assert result.requires_confirm is True
    assert result.ethogram_path is None
    assert "identify which video" in result.error


def test_existing_ethogram_requires_confirm_and_is_not_failed(tmp_path: Path):
    vid = tmp_path / "clip.avi"
    vid.write_bytes(b"")
    ann = tmp_path / "clip.annotations.json"
    ann.write_text("{}", encoding="utf-8")
    proj = Project.new(name="p", root_dir=str(tmp_path))
    proj.add_video(str(vid))
    set_current_video(proj, str(vid))

    result = check_downstream_artifacts(proj, video_path=str(vid))
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

    result = check_downstream_artifacts(proj, video_path=str(vid))
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

    result = check_downstream_artifacts(proj, video_path=str(vid))
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

    result = check_downstream_artifacts(proj, video_path=str(vid))
    assert result.check_failed is False
    assert result.requires_confirm is False
    assert result.examples_path is None


def test_check_uses_explicit_video_not_project_current(tmp_path: Path):
    current = tmp_path / "current.avi"
    current.write_bytes(b"")
    (tmp_path / "current.annotations.json").write_text("{}", encoding="utf-8")
    target = tmp_path / "target.avi"
    target.write_bytes(b"")
    target_ann = tmp_path / "target.annotations.json"
    target_ann.write_text("{}", encoding="utf-8")
    proj = Project.new(name="p", root_dir=str(tmp_path))
    proj.add_video(str(current))
    proj.add_video(str(target))
    set_current_video(proj, str(current))

    result = check_downstream_artifacts(proj, video_path=str(target))
    assert result.check_failed is False
    assert Path(result.ethogram_path).resolve() == target_ann.resolve()


def test_should_confirm_only_when_rebuild_and_artifacts_or_failed():
    hit = DownstreamArtifactCheck(check_failed=False, ethogram_path="a.json")
    failed = DownstreamArtifactCheck(check_failed=True, error="boom")
    clear = DownstreamArtifactCheck(check_failed=False)
    assert should_confirm_stale_downstream(will_rebuild=True, check=hit) is True
    assert should_confirm_stale_downstream(will_rebuild=True, check=failed) is True
    assert should_confirm_stale_downstream(will_rebuild=True, check=clear) is False
    assert should_confirm_stale_downstream(will_rebuild=False, check=hit) is False
    assert should_confirm_stale_downstream(will_rebuild=False, check=failed) is False


def test_video_path_for_package_prefers_detect_job_over_current(tmp_path: Path):
    current = tmp_path / "current.avi"
    current.write_bytes(b"")
    recorded = tmp_path / "recorded.avi"
    recorded.write_bytes(b"")
    review = tmp_path / "id_review"
    review.mkdir()
    (review / DETECT_JOB_FILENAME).write_text(
        '{"video_path": "%s", "behavior_mode": 0}'
        % recorded.as_posix(),
        encoding="utf-8",
    )
    proj = Project.new(name="p", root_dir=str(tmp_path))
    proj.add_video(str(current))
    set_current_video(proj, str(current))

    got = video_path_for_review_package(str(review), project=proj)
    assert Path(got).resolve() == recorded.resolve()


def test_video_path_for_package_matches_project_folder(tmp_path: Path):
    vid = tmp_path / "clip.avi"
    vid.write_bytes(b"")
    review = tmp_path / "id_review"
    review.mkdir()
    (review / "mouse_tracklets.npz").write_bytes(b"")
    (review / "mouse_tracklets_meta.json").write_text("{}", encoding="utf-8")
    proj = Project.new(name="p", root_dir=str(tmp_path))
    proj.add_video(str(vid))

    got = video_path_for_review_package(str(review), project=proj)
    assert Path(got).resolve() == vid.resolve()


def test_video_path_for_package_uses_hint_first(tmp_path: Path):
    hinted = tmp_path / "hinted.avi"
    hinted.write_bytes(b"")
    recorded = tmp_path / "recorded.avi"
    recorded.write_bytes(b"")
    review = tmp_path / "id_review"
    review.mkdir()
    (review / DETECT_JOB_FILENAME).write_text(
        '{"video_path": "%s"}' % recorded.as_posix(), encoding="utf-8"
    )
    got = video_path_for_review_package(str(review), hinted_video=str(hinted))
    assert Path(got).resolve() == hinted.resolve()
