"""Hydrate analyzer geometry from remapped tracklets."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from LabGym.analysis.hydrate_from_tracklets import fill_geometry_from_stores
from LabGym.id_review.types import SCHEMA_VERSION, TrackletStore


def test_fill_geometry_copies_centers_and_contours():
    n = 5
    centers = np.zeros((1, n, 2), dtype=np.float64)
    valid = np.ones((1, n), dtype=bool)
    heights = np.full((1, n), 12.0)
    contours = []
    row = []
    for f in range(n):
        centers[0, f] = [float(f), 1.0]
        row.append(
            np.array([[[f, 1]], [[f + 1, 1]], [[f + 1, 2]], [[f, 2]]], dtype=np.int32)
        )
    contours.append(row)
    store = TrackletStore(
        schema_version=SCHEMA_VERSION,
        animal_kind="mouse",
        ids=[0],
        n_frames=n,
        centers=centers,
        valid=valid,
        heights=heights,
        contours=contours,
        meta={},
    )
    analyzer = SimpleNamespace(
        total_analysis_framecount=n,
        animal_kinds=["mouse"],
        animal_centers={"mouse": {0: [None] * n}},
        animal_contours={"mouse": {0: [None] * n}},
        animal_heights={"mouse": {0: [None] * n}},
        animal_existingcenters={"mouse": {0: (-10000, -10000)}},
        register_counts={"mouse": {0: None}},
        animation_analyzer=False,
        pattern_images={"mouse": {0: [None] * n}},
        dim_conv=8,
    )
    fill_geometry_from_stores(analyzer, {"mouse": store})
    assert analyzer.animal_centers["mouse"][0][3] == (3.0, 1.0)
    assert analyzer.animal_contours["mouse"][0][3] is not None
    assert analyzer.animal_heights["mouse"][0][3] == 12.0
