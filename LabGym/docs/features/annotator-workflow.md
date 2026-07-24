# Ethogram-first training workflow

This is the **recommended** path for multi-animal behavior categorizer training in
this LabGym workspace build. Ethograms are the durable ground truth; training
clip length and sampling can change later **without re-annotating**.

## Pipeline

```text
Raw video
  → Detect & track (LabGym detector)
  → Fix ID swaps (ID review → save remapped tracklets)
  → Annotate ethograms (LabGym Behavior Annotator)
  → Save video.annotations.json  ← source of truth
  → Generate LabGym pairs FROM ethogram + fixed tracklets
  → Train categorizer (hard ± soft labels)
  → Analyze new videos
```

**Not** the classic path: generate many unlabeled windows → manually sort clips.
Sorting dense `generate_data*` output is still available as a **legacy** option.

## Launch

```bash
LabGym                 # full legacy GUI (detect, ID review, train, analyze)
LabGym-annotate        # standalone multi-subject ethogram annotator
LabGym-workflow        # PySide6 ethogram-first shell (recommended)
python -m LabGym.gui_pyside
```

### LabGym-workflow (PySide6 workbench shell)

FreeCAD-style **workbenches** (top bar) with **tabs** per subtask. See repo-root
`specifications.md` and `implementation-plan.md`.

| Workbench | Role |
|-----------|------|
| **Preprocess** | **Preprocess videos** + **Draw markers** |
| **Detector** | Detect + track, Review IDs, **Train/Test detector** |
| **Categorizer** | Generate training data, **Train/Test categorizer**; Process videos still Phase 7 |
| **Results** | Coming soon placeholder |

**Projects** (`*.labproj.json`): root folder + explicit video list + defaults.
File → New/Open/Save; Project → Edit Project.

**Annotate / Generate:** pick a project video, load tracklets from `id_review` (or
per-video `detection_dir`), save `*.annotations.json`, then generate sorted pairs.
Temporary **legacy wx** remains on unported tabs only.

```bash
# CLI ethogram → training pairs
python -m LabGym.training.ethogram_examples \
  --annotations path/to/video.annotations.json \
  --tracklets path/to/id_review \
  --video path/to/video.avi \
  --out path/to/examples \
  --length 15 \
  --sampling dense_in_bout
```

## Stage details

### 1–2. Detect & track; fix IDs

Use LabGym analysis / detector so that `id_review/{kind}_tracklets.npz` exists
**after** ID remaps are applied. These tracklets are the frozen identity layer
for annotation and example generation (no re-detection required).

### 3. Annotate ethogram

```bash
python -m LabGym.annotator
```

- Open the video; tracklets auto-load when found beside the video.
- Mode **0 / 1 / 2** = non-interactive / interactive basic / interactive advanced.
- Annotate with hotkeys; save **`video.annotations.json`**.
- Ethogram does **not** bake in training window length.

### 4. Generate examples from ethogram (Stage C)

In the annotator: **Tools → Generate LabGym training pairs from ethogram…**

| Parameter | Meaning |
|-----------|---------|
| Window length | LabGym `time_step` (animation length) |
| Sampling | `dense_in_bout`, `bout_end`, `bout_center`, `coverage` |
| Stride | For dense sampling (0 = length/3) |
| Tracklets folder | Post–ID-review directory |

**Outputs** (already sorted by behavior):

```text
examples/
  approach/
    clip_mouse_0_123_len15.avi
    clip_mouse_0_123_len15.jpg
  fight/
    ...
  soft_labels.csv
  generation_config.json
```

Re-run with a new `--length` anytime; ethogram stays the same.

### 5. Train categorizer

LabGym **Train Categorizers** → select the sorted folders from Stage C.

- Optional **hard_soft_aux** with `soft_labels.csv` next to prepared examples.
- Then analyze with the trained model as usual.

## Modes (behavior)

| Code | Ethogram | Example geometry |
|------|----------|------------------|
| 0 | Per subject | Per-ID blob + pattern |
| 1 | Group `interaction_bouts` | All animals in joint crop (`_itbs`) |
| 2 | Per subject + partners | Main + costars (`_itadv`) |

## Legacy path

1. LabGym **Generate Behavior Examples** (dense sample).  
2. **Sort from annotation session** or subject-aware CSV.  

Prefer ethogram-first generation so only labeled windows become examples.

## Modules

| Module | Role |
|--------|------|
| `LabGym.annotator` | Ethogram GUI |
| `LabGym.training.ethogram_examples` | Bout → LabGym pairs |
| `LabGym.training.soft_labels` | Soft targets |
| `LabGym.id_review` | Tracklets + ID fixes |
