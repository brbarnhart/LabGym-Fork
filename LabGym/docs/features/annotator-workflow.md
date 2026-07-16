# Multi-animal annotation & soft-label training

This document describes the merged **LabGym Behavior Annotator** (PySide6) and
how frame-by-frame labels feed categorizer training.

## Workflow

1. **Preprocess** videos (legacy wx module).
2. **Detect & track** with a Detector so identity tracklets exist.
3. **ID review** (legacy) to correct identity switches.
4. **Annotate** with `LabGym-annotate` or **Tools → Behavior Annotator** /
   `LabGym-workflow`.
5. **Extract / sort** training examples (annotation session or subject-aware CSV).
6. **Train** categorizer with hard / soft label modes.
7. **Analyze** videos with the trained categorizer.

## Launch

```bash
# Full legacy GUI
LabGym

# Annotator only
LabGym-annotate
# or
python -m LabGym.annotator

# Lightweight PySide workflow shell (launches tools as needed)
LabGym-workflow
```

## Multi-subject annotation

- Load tracklets from an `id_review` folder (**Tracks → Load Tracklets…** or
  auto-detect next to the video).
- Cycle subjects with **`[` / `]`**.
- Behavior modes:
  - **Non-interactive (0):** per-subject ethogram
  - **Interactive basic (1):** group ethogram → `interaction_bouts.group`
  - **Interactive advanced (2):** per-subject roles + optional **partner** IDs

Schema v2 JSON is written on save (v1 files auto-migrate to subject `0`).

## Exports

| Export | Purpose |
|--------|---------|
| `frame_labels_subject{N}.csv` | Per-subject one-hot frames |
| `frame_labels_all_subjects.csv` | Combined multi-subject table |
| `soft_labels.csv` | Window soft targets for training |
| `interaction_role_bouts.csv` | Mode-2 partner-aware bout table |
| Clips `{video}_sub{id}_{behavior}_…` | Curated examples |

## Sorting examples

In **Training → Sort Behavior Examples (from .csv)**:

- Classic CSV sort (unchanged)
- **Subject-aware CSV** (parses animal id + frame from filenames)
- **Sort from annotation session** (`.annotations.json`)

## Soft-label training

In **Train Categorizers → Specify label mode**:

| Mode | Loss |
|------|------|
| `hard_only` | Folder hard labels only (legacy) |
| `hard_soft_aux` **(default)** | `L_hard + λ L_soft` |
| `soft_primary` | Soft primary + small hard term |

Place `soft_labels.csv` in the **prepared examples** folder (or pick a path).
If soft labels are missing, training falls back to `hard_only`.

Generate soft labels from the annotator:

**Tools → Export soft_labels.csv for examples folder…**

Recommended: `λ ≈ 0.3–0.5`, exclusive ethograms for categorizer export.

## Related modules

- `LabGym.annotator` — GUI + session schema
- `LabGym.training` — soft labels, losses, example sort
- `LabGym.gui_pyside` — workflow shell
