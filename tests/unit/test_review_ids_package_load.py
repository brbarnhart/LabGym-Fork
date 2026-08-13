"""load_review_package with a minimal on-disk identity package fixture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from LabGym.id_review.tracklets import save_tracklets
from LabGym.id_review.types import SCHEMA_VERSION, TrackletStore


def _minimal_store(n_frames: int = 8) -> TrackletStore:
    ids = [0, 1]
    centers = np.zeros((2, n_frames, 2), dtype=np.float64)
    valid = np.ones((2, n_frames), dtype=bool)
    heights = np.full((2, n_frames), 10.0)
    contours = []
    for row, tid in enumerate(ids):
        row_c = []
        for f in range(n_frames):
            centers[row, f] = [tid * 50.0 + f, float(tid)]
            x, y = centers[row, f]
            cnt = np.array(
                [
                    [[int(x), int(y)]],
                    [[int(x) + 4, int(y)]],
                    [[int(x) + 4, int(y) + 4]],
                    [[int(x), int(y) + 4]],
                ],
                dtype=np.int32,
            )
            row_c.append(cnt)
        contours.append(row_c)
    return TrackletStore(
        schema_version=SCHEMA_VERSION,
        animal_kind="mouse",
        ids=ids,
        n_frames=n_frames,
        centers=centers,
        valid=valid,
        heights=heights,
        contours=contours,
        meta={"video": "clip.avi", "fps": 10.0},
    )


def test_load_review_package_happy(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
        load_review_package,
    )

    review = tmp_path / "id_review"
    review.mkdir()
    save_tracklets(_minimal_store(), str(review))
    # empty events/switches optional files — load_events/load_switches tolerate missing
    (review / "events.json").write_text("[]", encoding="utf-8")
    (review / "switches.json").write_text("[]", encoding="utf-8")
    (tmp_path / "clip.avi").write_bytes(b"fake")

    pkg = load_review_package(str(review))
    assert pkg.review_dir == str(review.resolve())
    assert "mouse" in pkg.stores
    assert pkg.animal_kind == "mouse"
    assert pkg.n_frames == 8
    assert pkg.fps == 10.0
    assert pkg.already_corrected is False
    assert pkg.has_raw is True
    assert pkg.accepted is False
    assert len(pkg.subjects) >= 1
    assert not (review / "mouse_tracklets.npz").is_file()
    assert (review / "raw" / "mouse_tracklets.npz").is_file()


def test_load_review_package_missing_dir():
    from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
        load_review_package,
    )

    with pytest.raises(FileNotFoundError):
        load_review_package("Z:/definitely/not/a/real/package")


def test_load_review_package_no_tracklets(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
        load_review_package,
    )

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="tracklets"):
        load_review_package(str(empty))


def test_save_review_package_publishes_from_raw(tmp_path: Path):
    from LabGym.gui_pyside.workbenches.detector.review_ids_package import (
        load_review_package,
        save_review_package,
    )
    from LabGym.id_review.dataset import make_swap_marker
    from LabGym.id_review.raw_store import has_accepted_identities, save_raw_tracklets
    from LabGym.id_review.tracklets import load_tracklets

    review = tmp_path / "id_review"
    review.mkdir()
    raw = _minimal_store(n_frames=10)
    save_raw_tracklets(review, {"mouse": raw})
    pkg = load_review_package(str(review))
    marker = make_swap_marker(4, "mouse", [0, 1], fps=10.0)
    result = save_review_package(
        str(review),
        [marker],
        pkg.events,
        pkg.subjects,
        already_corrected=False,
        baseline_stores=pkg.baseline_stores,
        has_raw=True,
    )
    assert result.ok, result.error
    assert result.accepted is True
    assert has_accepted_identities(review) is True
    published = load_tracklets(str(review), "mouse")
    assert published.centers[0, 4, 0] == raw.centers[1, 4, 0]
    # second save must not invert
    result2 = save_review_package(
        str(review),
        [marker],
        pkg.events,
        pkg.subjects,
        already_corrected=False,
        baseline_stores=pkg.baseline_stores,
        has_raw=True,
    )
    assert result2.ok, result2.error
    published2 = load_tracklets(str(review), "mouse")
    assert published2.centers[0, 4, 0] == raw.centers[1, 4, 0]
