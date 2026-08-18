# Ethogram-first training workflow

This is the **recommended** path for multi-animal behavior categorizer training.
Ethograms are the durable ground truth; training clip length and sampling can
change later **without re-annotating**.

## Pipeline

```text
Raw video
  → Detect & track (LabGym detector)
  → Fix ID swaps (ID review → save remapped tracklets)
  → Annotate ethograms (LabGym Behavior Annotator)
  → Save video.annotations.json  ← source of truth
  → Generate LabGym pairs FROM ethogram + fixed tracklets
  → Train categorizer (hard ± soft labels)
  → Process / analyze new videos
```

**Primary path** is ethogram-first. Dense “generate many unsorted clips then
sort” remains available as a **power-user tool** (Tools → Dense generate + sort
examples…), not as the default workbench workflow.

## Launch

```bash
LabGym                 # default: PySide6 ethogram-first workbench shell
LabGym-workflow        # same shell (compat alias)
LabGym-annotate        # standalone multi-subject ethogram annotator
python -m LabGym.gui_pyside
```

### LabGym (PySide6 workbench shell)

FreeCAD-style **workbenches** (top bar) with **tabs** per subtask. See repo-root
`specifications.md` and `implementation-plan.md`.

| Workbench | Role |
|-----------|------|
| **Preprocess** | Preprocess videos + Draw markers |
| **Detector** | Generate training data (frames + annotate placeholder), Train/Test, Detect + track, Review IDs |
| **Categorizer** | Ethogram annotate / generate pairs, Train/Test, Process videos |
| **Results** | Mine results, behavior plots, distance calculations |

**Projects** (`*.labproj.json`): root folder + explicit video list + defaults.
File → New/Open/Save; Project → Edit Project.

**Annotate / Generate:** pick a project video, load tracklets from `id_review` (or
per-video `detection_dir`), save `*.annotations.json`, then generate sorted pairs.

**Dense classic pipeline:** Tools → **Dense generate + sort examples…** (pop-out
window for unsorted generation + manual/CSV/annotations sort).

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

Use Detector → Detect + track (writes **raw tracklets**), then Detector →
Review IDs → **Save** to publish **remapped tracklets** (**accepted identities**).
An empty switch list is how you accept detector IDs. Annotate, generate
examples, and Process videos all require that save. Process videos categorizes
those remapped outlines and does **not** run the detector again.

### 3. Annotate ethogram

```bash
python -m LabGym.annotator
```

Or open **Categorizer → Generate training data → Annotate ethogram** in the
workbench.

- Open the video; tracklets auto-load when found beside the video.
- Mode **0 / 1 / 2** = non-interactive / interactive basic / interactive advanced.
- Annotate with hotkeys; save **`video.annotations.json`**.
- Ethogram does **not** bake in training window length.

### 4. Generate examples from ethogram

In the workbench: **Categorizer → Generate training data → Generate examples**,
or in the annotator: **Tools → Generate LabGym training pairs from ethogram…**

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

**Categorizer → Train categorizer** → select the sorted folders from Stage 4.

- Optional **hard_soft_aux** with `soft_labels.csv` next to prepared examples.
- Then **Process videos** with the trained model.

## Modes (behavior)

| Code | Ethogram | Example geometry |
|------|----------|------------------|
| 0 | Per subject | Per-ID blob + pattern |
| 1 | Group `interaction_bouts` | All animals in joint crop (`_itbs`) |
| 2 | Per subject + partners | Main + costars (`_itadv`) |

## Classic dense path (Tools menu)

1. Tools → Dense generate + sort → **Generate unsorted**.  
2. Sort with **manual keys**, **CSV**, or **annotations JSON**.  

Prefer ethogram-first generation so only labeled windows become examples.

## Modules

| Module | Role |
|--------|------|
| `LabGym.annotator` | Ethogram GUI |
| `LabGym.training.ethogram_examples` | Bout → LabGym pairs |
| `LabGym.training.soft_labels` | Soft targets |
| `LabGym.id_review` | Tracklets + ID fixes |
| `LabGym.gui_pyside` | Workbench shell |
| `LabGym.gui_pyside.tools_windows` | Dense generate + sort pop-out |
