"""Fixtures for LabGym integration / smoke tests (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """Session-scoped QApplication for GUI smoke tests."""
    app = QApplication.instance() or QApplication(sys.argv)
    return app


@pytest.fixture
def project_controller_with_videos(tmp_path: Path):
    """ProjectController with a root dir and two fake video files."""
    from LabGym.gui_pyside.project.controller import ProjectController
    from LabGym.gui_pyside.project.model import Project

    ctrl = ProjectController()
    proj = Project.new(name="smoke", root_dir=str(tmp_path))
    for name in ("clip_a.avi", "clip_b.avi"):
        p = tmp_path / name
        p.write_bytes(b"fake-video")
        proj.add_video(str(p))
    ctrl.replace(proj, dirty=False)
    return ctrl


@pytest.fixture
def minimal_id_review_package(tmp_path: Path) -> Path:
    """On-disk id_review folder with tracklets + empty events/switches."""
    import numpy as np

    from LabGym.id_review.tracklets import save_tracklets
    from LabGym.id_review.types import SCHEMA_VERSION, TrackletStore

    review = tmp_path / "detection" / "clip" / "id_review"
    review.mkdir(parents=True)
    n_frames = 6
    ids = [0, 1]
    centers = np.zeros((2, n_frames, 2), dtype=np.float64)
    valid = np.ones((2, n_frames), dtype=bool)
    heights = np.full((2, n_frames), 10.0)
    contours = []
    for row, tid in enumerate(ids):
        row_c = []
        for f in range(n_frames):
            centers[row, f] = [float(tid * 40 + f), float(tid)]
            x, y = centers[row, f]
            cnt = np.array(
                [
                    [[int(x), int(y)]],
                    [[int(x) + 3, int(y)]],
                    [[int(x) + 3, int(y) + 3]],
                    [[int(x), int(y) + 3]],
                ],
                dtype=np.int32,
            )
            row_c.append(cnt)
        contours.append(row_c)
    store = TrackletStore(
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
    save_tracklets(store, str(review))
    (review / "events.json").write_text("[]", encoding="utf-8")
    (review / "switches.json").write_text("[]", encoding="utf-8")
    (tmp_path / "clip.avi").write_bytes(b"fake")
    return review
