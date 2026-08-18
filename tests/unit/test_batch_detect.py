"""Unit tests for headless detect+track config / listing (no GPU run)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from LabGym.detection.batch_detect import (
    DetectTrackConfig,
    DetectTrackResult,
    detect_and_track_video,
    is_detector_folder,
    list_detectors,
    load_detector_animal_kinds,
    validate_detector_folder,
)


def _write_detector(
    folder: Path,
    *,
    names=None,
    with_weights: bool = True,
    params_text: str | None = None,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    if params_text is not None:
        (folder / "model_parameters.txt").write_text(params_text, encoding="utf-8")
    else:
        names = names or ["mouse"]
        (folder / "model_parameters.txt").write_text(
            json.dumps(
                {
                    "animal_names": list(names),
                    "animal_mapping": {str(i): n for i, n in enumerate(names)},
                    "inferencing_framesize": 480,
                }
            ),
            encoding="utf-8",
        )
    if with_weights:
        (folder / "model_final.pth").write_bytes(b"fake")
        (folder / "config.yaml").write_text("MODEL: {}\n", encoding="utf-8")
    return folder


def test_load_detector_animal_kinds(tmp_path: Path):
    det = _write_detector(
        tmp_path / "my_det", names=["mouse", "object"], with_weights=False
    )
    assert load_detector_animal_kinds(det) == ["mouse", "object"]


def test_list_detectors(tmp_path: Path):
    a = _write_detector(tmp_path / "a", names=["mouse"])
    b = _write_detector(tmp_path / "nested" / "b", names=["mouse"])
    # Incomplete / empty params must not be listed
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "model_parameters.txt").write_text("{}", encoding="utf-8")
    found = list_detectors(tmp_path)
    assert a in found
    assert b in found
    assert bad not in found


def test_rejects_categorizer_folder(tmp_path: Path):
    cat = tmp_path / "annotator-lv_2"
    cat.mkdir()
    (cat / "model_parameters.txt").write_text(
        "classnames,dim_tconv,dim_conv,channel,time_step,network\n"
        "Follow,32,32,3,10,2\n",
        encoding="utf-8",
    )
    (cat / "model.keras").write_bytes(b"fake")
    with pytest.raises(ValueError, match="categorizer"):
        load_detector_animal_kinds(cat)
    with pytest.raises(ValueError, match="categorizer"):
        validate_detector_folder(cat)
    assert is_detector_folder(cat) is False
    assert list_detectors(tmp_path) == []


def test_validate_requires_weights(tmp_path: Path):
    det = _write_detector(tmp_path / "det", with_weights=False)
    with pytest.raises(ValueError, match="incomplete"):
        validate_detector_folder(det, require_weights=True)
    # Kinds still readable without weights
    assert load_detector_animal_kinds(det) == ["mouse"]


def test_resolved_animal_number_defaults():
    cfg = DetectTrackConfig(
        video_path="x.avi",
        detector_path="d",
        results_root="out",
        animal_kinds=["mouse", "object"],
        animal_number={},
    )
    assert cfg.resolved_animal_number() == {"mouse": 1, "object": 1}
    # single entry applies to all kinds (UI "animals per kind")
    cfg.animal_number = {"mouse": 3}
    assert cfg.resolved_animal_number() == {"mouse": 3, "object": 3}
    # per-kind map when multiple keys present
    cfg.animal_number = {"mouse": 2, "object": 5}
    assert cfg.resolved_animal_number() == {"mouse": 2, "object": 5}


def test_detect_missing_video(tmp_path: Path):
    det = tmp_path / "det"
    det.mkdir()
    (det / "model_parameters.txt").write_text(
        json.dumps({"animal_names": ["mouse"], "inferencing_framesize": 100}),
        encoding="utf-8",
    )
    cfg = DetectTrackConfig(
        video_path=str(tmp_path / "missing.avi"),
        detector_path=str(det),
        results_root=str(tmp_path / "out"),
        animal_kinds=["mouse"],
    )
    result = detect_and_track_video(cfg)
    assert result.ok is False
    assert "not found" in result.error.lower()


def test_detect_and_track_mocked(tmp_path: Path):
    import sys
    import types

    video = tmp_path / "clip.avi"
    video.write_bytes(b"fake")
    det = _write_detector(tmp_path / "det", names=["mouse"])
    out = tmp_path / "detection"
    out.mkdir()
    id_review = out / "clip" / "id_review"
    id_review.mkdir(parents=True)

    fake_aad = MagicMock()
    fake_aad.results_path = str(out / "clip")
    fake_aad.prepare_analysis = MagicMock()
    fake_aad.acquire_information = MagicMock()
    fake_aad.craft_data = MagicMock()

    # Avoid importing Detectron2 via analyzebehavior_dt
    fake_mod = types.ModuleType("LabGym.analyzebehavior_dt")
    fake_mod.AnalyzeAnimalDetector = MagicMock(return_value=fake_aad)
    prev = sys.modules.get("LabGym.analyzebehavior_dt")
    sys.modules["LabGym.analyzebehavior_dt"] = fake_mod
    try:
        with patch(
            "LabGym.id_review.dataset.export_review_pack",
            return_value=(str(id_review), []),
        ), patch(
            "LabGym.id_review.apply.write_tracklets_identity_status"
        ), patch(
            "LabGym.annotator.core.tracklets_bridge.discover_tracklet_kinds",
            return_value=[],
        ):
            cfg = DetectTrackConfig(
                video_path=str(video),
                detector_path=str(det),
                results_root=str(out),
                animal_kinds=["mouse"],
                animal_number={"mouse": 2},
                export_id_review=True,
                write_default_subjects=False,
            )
            result = detect_and_track_video(cfg)
    finally:
        if prev is None:
            sys.modules.pop("LabGym.analyzebehavior_dt", None)
        else:
            sys.modules["LabGym.analyzebehavior_dt"] = prev

    assert result.ok is True, result.error
    assert result.id_review_dir == str(id_review)
    fake_aad.prepare_analysis.assert_called_once()
    fake_aad.acquire_information.assert_called_once()
    fake_aad.craft_data.assert_called_once()


def _analyzer_with_tracks(results_path: Path, kind: str = "mouse", n: int = 6):
    from types import SimpleNamespace

    centers = {0: [(0.0, 0.0)] * n, 1: [(10.0, 0.0)] * n}
    heights = {0: [8.0] * n, 1: [8.0] * n}
    contours = {0: [None] * n, 1: [None] * n}
    return SimpleNamespace(
        results_path=str(results_path),
        animal_kinds=[kind],
        animal_centers={kind: centers},
        animal_heights={kind: heights},
        animal_contours={kind: contours},
        animal_area={kind: 10.0},
        fps=10,
        t=0,
        length=0,
        path_to_video="clip.avi",
        framewidth=None,
        frameheight=None,
        duration=0,
        all_time=list(range(n)),
        prepare_analysis=MagicMock(),
        acquire_information=MagicMock(),
        acquire_information_interact_basic=MagicMock(),
        craft_data=MagicMock(),
        detector=None,
    )


def _run_detect_with_stub(
    tmp_path: Path,
    *,
    behavior_mode: int,
    export_id_review: bool,
):
    """Run detect_and_track_video with a stub analyzer (no GPU acquire)."""
    import sys
    import types

    video = tmp_path / "clip.avi"
    video.write_bytes(b"fake")
    det = _write_detector(tmp_path / "det", names=["mouse"])
    out = tmp_path / "detection"
    out.mkdir()
    results_path = out / "clip"
    results_path.mkdir()
    fake_aad = _analyzer_with_tracks(results_path)

    fake_mod = types.ModuleType("LabGym.analyzebehavior_dt")
    fake_mod.AnalyzeAnimalDetector = MagicMock(return_value=fake_aad)
    prev = sys.modules.get("LabGym.analyzebehavior_dt")
    sys.modules["LabGym.analyzebehavior_dt"] = fake_mod
    try:
        cfg = DetectTrackConfig(
            video_path=str(video),
            detector_path=str(det),
            results_root=str(out),
            animal_kinds=["mouse"],
            animal_number={"mouse": 2},
            behavior_mode=behavior_mode,
            export_id_review=export_id_review,
            write_default_subjects=False,
            extract_contact_samples=False,
        )
        result = detect_and_track_video(cfg)
    finally:
        if prev is None:
            sys.modules.pop("LabGym.analyzebehavior_dt", None)
        else:
            sys.modules["LabGym.analyzebehavior_dt"] = prev
    return result, fake_aad, results_path / "id_review"


@pytest.mark.parametrize("mode", [0, 2])
def test_detect_writes_raw_even_if_export_disabled(tmp_path: Path, mode: int):
    from LabGym.annotator.core.tracklets_bridge import discover_tracklet_kinds
    from LabGym.id_review.raw_store import has_accepted_identities, has_raw_snapshot

    result, fake_aad, id_review = _run_detect_with_stub(
        tmp_path, behavior_mode=mode, export_id_review=False
    )
    assert result.ok is True, result.error
    assert has_raw_snapshot(id_review) is True
    assert discover_tracklet_kinds(id_review) == []
    assert has_accepted_identities(id_review) is False
    assert (id_review / "raw" / "mouse_tracklets.npz").is_file()
    fake_aad.acquire_information.assert_called_once()
    fake_aad.craft_data.assert_called_once()


def test_detect_interactive_basic_does_not_write_identity_package(tmp_path: Path):
    from LabGym.id_review.raw_store import has_accepted_identities, has_raw_snapshot

    result, fake_aad, id_review = _run_detect_with_stub(
        tmp_path, behavior_mode=1, export_id_review=True
    )
    assert result.ok is True, result.error
    assert result.id_review_dir == ""
    assert has_raw_snapshot(id_review) is False
    assert has_accepted_identities(id_review) is False
    assert not id_review.is_dir()
    assert not (id_review / "raw" / "mouse_tracklets.npz").is_file()
    fake_aad.acquire_information_interact_basic.assert_called_once()
    fake_aad.acquire_information.assert_not_called()
    fake_aad.craft_data.assert_not_called()
