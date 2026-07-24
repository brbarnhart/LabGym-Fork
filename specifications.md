# Product UI Specification

**Status:** agreed draft (pre-implementation)  
**Goal:** Replace the legacy wxPython LabGym GUI with a **PySide6** application laid out for the multi-animal / ethogram-first workflow.  
**Inspiration:** FreeCAD-style **workbenches** (major task groups) with **tabs** for subtasks.

This document is the shared product/UI contract. Implementation details (file layout, class names, PR order) belong in a separate plan.

---

## Decisions locked in

| Topic | Decision |
|-------|----------|
| Workbench chrome | **Top toolbar icons** for workbenches; subtask **tabs** under them |
| Project model | **Both**: experiment root folder **and** editable explicit video list |
| Generate training data | **Two subtabs**: Annotate ethogram \| Generate examples |
| Dense generate-then-sort | **Drop from UI** (ethogram-first only) |
| Results workbench (first usable release) | **Placeholder only** |

**Recommended defaults** (can revisit later):

- Subject **roles**: free text; optional project-level vocabulary later  
- Batch detect/process concurrency: **one video at a time** for MVP  
- Branding: keep **LabGym** unless renamed later  
- Legacy wx: temporary bridge only; end state pure PySide  

---

## 1. Product intent

A single PySide6 desktop app so a researcher can:

1. Prepare videos  
2. Detect and track animals; correct IDs; assign experimental names/roles  
3. Manually build ethograms and generate categorizer training data **from those ethograms**  
4. Train/test a categorizer and process new videos  
5. (Later) export results for stats / figures  

**Primary training philosophy (ethogram-first):**

```text
Detect & track → Fix IDs / assign roles → Annotate ethograms (JSON = source of truth)
  → Generate LabGym-style training pairs FROM ethograms
  → Train categorizer → Process / analyze new videos
```

**Not in the new UI:** dense “generate many unsorted clips then hand-sort” as a product path. Ethogram-first only.

**Near-term non-goals:**

- Rewriting Detectron2 / training kernels  
- Full in-app statistics package (Results is future / placeholder)  
- Detector image-example authoring tool (future tab)

---

## 2. Layout philosophy

### 2.1 Shell regions

| UI region | Role |
|-----------|------|
| **Workbench switcher** | **Top toolbar icons** — Preprocessing, Detector, Categorizer, Results |
| **Tab strip** | Subtasks for the **active** workbench only |
| **Main content** | Active tab (forms, video player, tables, progress) |
| **Project context** | Shared experiment settings; always visible (status) and via File/Project menu |
| **Status / log** | Progress and messages; long jobs do not freeze the UI |

### 2.2 Sketch

```text
┌─────────────────────────────────────────────────────────────────────┐
│ File  Project  Help              Project: MyExp.labproj.json        │
├─────────────────────────────────────────────────────────────────────┤
│ [Preprocess] [Detector] [Categorizer] [Results]   ← workbench icons │
├─────────────────────────────────────────────────────────────────────┤
│ [ Tab A ] [ Tab B ] [ Tab C ]                     ← subtasks        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                     Active tab content                              │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ video list · paths · detector · categorizer · job status            │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Navigation rules

- Switching workbench **keeps** the same project open.  
- Tabs do **not** auto-start long jobs.  
- Missing upstream artifacts → clear empty state + jump to the right tab.  
- **End state:** all primary workflows in-process **PySide6** (no wx). wx only as a temporary migration bridge if needed.

---

## 3. Workbenches and tabs

### 3.1 Preprocessing (MVP)

| Tab | Purpose |
|-----|---------|
| **Preprocess videos** | Trim/crop/resize/enhance (current LabGym preprocess capability) |
| **Draw markers** | Existing marker-drawing workflow |

**Outputs:** processed videos (+ any marker artifacts LabGym already produces).

### 3.2 Detector

| Tab | Priority | Purpose |
|-----|----------|---------|
| **Create & annotate image examples** | Future | In-app detector training images (not required now) |
| **Train detector** | MVP | Train detector (existing backend, new UI) |
| **Test detector** | MVP | Test detector |
| **Detect + track subjects** | MVP high | **Batch** multi-video detect/track; save durable outputs for later review |
| **Review IDs & assign names/roles** | MVP high | Fix ID swaps; re-save corrected tracklets; assign experimental names/roles |

#### Detect + track (batch)

- Multi-select videos or use project video list.  
- Shared detector/parameters.  
- Per-video progress; write reloadable **identity package**.  
- MVP concurrency: **one video at a time** (parallelism later if needed).

#### Review IDs & names/roles

- Open detection output for a video.  
- Contact-aware ID correction; **persist remapped tracklets**.  
- Assign **display names** and **roles** (free text; optional project role vocabulary later).  
- Browse all project videos that have detection output.

**Identity package (per video, conceptual):**

```text
corrected tracklets
subject table: { id, display_name, role, color, … }
analysis_start_frame / detector metadata
```

### 3.3 Categorizer

| Tab | Priority | Purpose |
|-----|----------|---------|
| **Generate training data** | MVP high | Ethogram annotation + example generation (see subtabs) |
| **Train categorizer** | MVP | Train on ethogram-generated sorted folders; optional soft labels |
| **Test categorizer** | MVP | Test trained categorizer |
| **Process videos** | MVP high | Batch: video + identity package → categorizer/analysis outputs |

#### Generate training data — two subtabs

1. **Annotate ethogram**  
   - Video + identity package (tracklets, names).  
   - Behaviors (name, hotkey, color, reorder).  
   - Multi-subject; modes 0 / 1 / 2; partners; bout editor (including bulk partners).  
   - Save `*.annotations.json` (schema v2) as ground truth.  

2. **Generate examples**  
   - Window length, sampling, output folder, mode-aware geometry.  
   - Write **already sorted** LabGym pairs from ethogram + fixed tracklets.  
   - Optional soft labels.  
   - Re-run new length without re-annotating.  

Context testing of a categorizer on this experiment may live under **Test categorizer** or a light panel later; not blocking MVP.

### 3.4 Results / data export

| Area | Priority |
|------|----------|
| Ethogram figures, R tables, extra analyses | Future |
| Workbench shell entry | **Placeholder** for first usable release (“Coming soon”) |

---

## 4. Project concept

A **Project** is a saved experiment context shared across workbenches.

### 4.1 Contents

- **Root experiment folder** (optional but usual)  
- **Explicit list of videos** included in the project (editable)  
- Default paths: processed videos, detection outputs, annotations, examples, models  
- Defaults: behavior mode, exclusive mode, example window length, last detector/categorizer  
- Optional notes; optional role vocabulary  
- On-disk file e.g. `MyExperiment.labproj.json`; recent projects in QSettings  

### 4.2 Project vs video session

| Concept | Scope |
|---------|--------|
| **Project** | Whole experiment (many videos, models, defaults) |
| **Video session** | One video open in Annotate / Review IDs |

### 4.3 How workbenches use the project

| Tab | Uses project for |
|-----|------------------|
| Preprocess | Default in/out folders; video list |
| Detect + track | Video list, detector path, output root |
| Review IDs | Videos with detection output |
| Annotate ethogram / Generate examples | Current video, tracklets, annotations, mode, length |
| Train / Test / Process | Models, example folders, video batch |
| Results (future) | Analysis output root |

Auto-resolve sidecars when conventions match (`video.annotations.json`, `id_review/` next to video).

---

## 5. Cross-cutting requirements

- **Toolkit:** PySide6 end state; no wx for daily workflow once migration complete.  
- **Jobs:** train/detect/generate/process off the UI thread; progress (+ cancel where feasible).  
- **Durability:** ethogram JSON + corrected tracklets are first-class; regenerating clips does not require re-annotation.  
- **Modes:** 0 non-interactive · 1 interactive basic · 2 interactive advanced (partners/costars).  
- **Identity:** numeric track ID ≠ experimental display name/role.  
- **Usability:** empty states, keyboard annotation shortcuts, undo for annotations where present.

---

## 6. Migration order (after this spec is agreed)

1. Shell: top workbench bar + tabs + project open/save  
2. Categorizer → Generate training data (Annotate + Generate subtabs; largely existing annotator)  
3. Detector → Review IDs & names/roles  
4. Detector → Detect + track batch  
5. Preprocessing tabs  
6. Train/test detector & categorizer; Process videos  
7. Results (real); remove any remaining wx main window  

---

## 7. Acceptance (usable without wx)

User can preprocess → batch detect/track → review IDs/names → annotate ethogram → generate pairs → train categorizer → process videos, all in the PySide app. Results may still be a placeholder.

---

## 8. Mapping to current code (orientation only)

| Spec area | Building blocks |
|-----------|-----------------|
| Preprocess / markers | `gui_preprocessor.py` (wx → port) |
| Train/test detector | `gui_detector.py`, `detector.py` |
| Detect + track | `gui_analyzer.py`, `analyzebehavior*`, tracklet export |
| Review IDs | `gui_id_review.py`, `id_review/*` |
| Annotate + generate pairs | `annotator/*`, `training/ethogram_examples.py` |
| Train/test categorizer | `gui_categorizer.py`, `categorizer.py` |
| Early PySide shell | `gui_pyside/*` — reshape into workbench shell |

---

## 9. Original outline (preserved)

> FreeCAD-like workbenches; each first-level bullet is a workbench, each subbullet a tab.  
>
> - **Preprocessing:** preprocess videos; draw markers  
> - **Detector:** (future) image examples; train; test; detect+track bulk; review IDs + names/roles  
> - **Categorizer:** generate training data (manual ethogram + examples); train; test; process videos  
> - **Results:** future ethogram figures / R tables / analyses  
>
> Project idea for shared saved settings is desirable and is specified in §4.
