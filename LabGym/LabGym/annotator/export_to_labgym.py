"""Export annotator sessions into LabGym training artifacts.

- Per-subject / combined frame_labels CSV
- Soft-label sidecar for windows matching LabGym example length
- Subject-aware sort of unsorted LabGym examples
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from LabGym.annotator.core.annotation_manager import AnnotationManager
from LabGym.annotator.core.data_models import (
    BEHAVIOR_MODE_INTERACTIVE_ADVANCED,
    BEHAVIOR_MODE_INTERACTIVE_BASIC,
    AnnotationSession,
)
from LabGym.annotator.core.example_generator import ExampleGenerator
from LabGym.training.example_sort import (
    sort_examples_from_annotations,
    sort_examples_from_csv_subject_aware,
)
from LabGym.training.soft_labels import (
    write_soft_labels_sidecar,
)


def load_session(path: Union[str, Path]) -> AnnotationSession:
    return AnnotationManager.load_from_json(path).session


def export_label_tables(
    session: AnnotationSession,
    output_dir: Union[str, Path],
    video_path: Optional[str] = None,
) -> List[Path]:
    """Write frame label CSVs for all subjects (and group if mode 1)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gen = ExampleGenerator(session, video_path=video_path or session.video_path)
    written = gen.export_frame_labels_all_subjects(output_dir, combined=True)

    # Mode 1: also write group labels as frame_labels.csv (classic LabGym)
    if int(session.behavior_mode) == BEHAVIOR_MODE_INTERACTIVE_BASIC:
        # Temporary: treat group ethogram via generator with mode already set
        from LabGym.training.soft_labels import dense_frame_labels_from_session
        import pandas as pd

        names, arr = dense_frame_labels_from_session(session, use_group=True)
        data = {"frame": list(range(arr.shape[0]))}
        for i, n in enumerate(names):
            data[n] = arr[:, i].astype(int).tolist()
        path = output_dir / "frame_labels_group.csv"
        pd.DataFrame(data).to_csv(path, index=False)
        written.append(path)

    # Mode 2: export partner-aware bout table
    if int(session.behavior_mode) == BEHAVIOR_MODE_INTERACTIVE_ADVANCED:
        written.append(_export_partner_bout_table(session, output_dir))

    gen.close()
    return written


def _export_partner_bout_table(
    session: AnnotationSession, output_dir: Path
) -> Path:
    import pandas as pd

    rows = []
    for subj in session.subjects:
        bmap = session.bouts_for_subject(subj.subject_id)
        for beh, blist in bmap.items():
            for bout in blist:
                rows.append(
                    {
                        "subject_id": subj.subject_id,
                        "display_name": subj.display_name,
                        "behavior": beh,
                        "start_frame": bout.start_frame,
                        "end_frame": bout.end_frame,
                        "partner_ids": ",".join(str(p) for p in bout.partner_ids),
                    }
                )
    path = output_dir / "interaction_role_bouts.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def export_soft_labels_for_examples(
    session: AnnotationSession,
    examples_dir: Union[str, Path],
    *,
    window_len: int = 15,
    edge_smooth: int = 2,
) -> Path:
    return write_soft_labels_sidecar(
        examples_dir,
        session,
        window_len=window_len,
        edge_smooth=edge_smooth,
    )


def sort_unsorted_examples(
    annotations_path: Union[str, Path],
    examples_dir: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    also_soft_labels: bool = True,
    window_len: int = 15,
    copy: bool = False,
) -> Dict[str, int]:
    """Sort LabGym unsorted examples using annotation session; optional soft sidecar.

    Legacy path (dense generate_data then sort). Prefer
    ``generate_training_pairs_from_ethogram`` for ethogram-first workflows.
    """
    counts = sort_examples_from_annotations(
        annotations_path, examples_dir, out_dir, copy=copy
    )
    if also_soft_labels:
        session = load_session(annotations_path)
        # Soft labels relative to sorted folders are less useful; write next to out_dir root
        # Prefer writing for the original unsorted set if still present
        src = Path(examples_dir)
        if any(src.glob("*.avi")):
            write_soft_labels_sidecar(src, session, window_len=window_len)
        else:
            write_soft_labels_sidecar(out_dir, session, window_len=window_len)
    return counts


def generate_training_pairs_from_ethogram(
    annotations_path: Union[str, Path],
    tracklets_dir: Union[str, Path],
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    length: int = 15,
    sampling: str = "dense_in_bout",
    stride: int = 0,
    min_bout_frames: int = 1,
    social_distance: float = 0.0,
    write_soft_labels: bool = True,
    analysis_start_frame: Optional[int] = None,
) -> Dict:
    """Ethogram-first: bout windows → sorted LabGym .avi/.jpg pairs + soft_labels."""
    from LabGym.training.ethogram_examples import (
        GenerationConfig,
        generate_examples_from_ethogram,
    )

    cfg = GenerationConfig(
        video_path=str(video_path),
        annotations_path=str(annotations_path),
        tracklets_dir=str(tracklets_dir),
        output_dir=str(output_dir),
        length=int(length),
        sampling=sampling,
        stride=int(stride),
        min_bout_frames=int(min_bout_frames),
        social_distance=float(social_distance),
        write_soft_labels=write_soft_labels,
        analysis_start_frame=analysis_start_frame,
    )
    return generate_examples_from_ethogram(cfg)
