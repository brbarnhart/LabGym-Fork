# Product UI Specification

**Status:** implemented (PySide workbench is the default LabGym UI)  
**Goal:** LabGym as a **PySide6** application laid out for the multi-animal / ethogram-first workflow.  
**Inspiration:** FreeCAD-style **workbenches** (major task groups) with **tabs** for subtasks.

This document is the shared product/UI contract. Implementation details (file layout, class names, PR order) belong in [`implementation-plan.md`](./implementation-plan.md).

---

## Decisions locked in

| Topic | Decision |
|-------|----------|
| Workbench chrome | **Top toolbar icons** for workbenches; subtask **tabs** under them |
| Project model | **Both**: experiment root folder **and** editable explicit video list |
| Categorizer generate training data | **Two subtabs**: Annotate ethogram \| Generate examples (ethogram-first) |
| Dense generate-then-sort | **Tools menu pop-out** (power user); not a workbench tab |
| Detector generate training data | **Two subtabs**: Generate images \| Annotate images (placeholder → EZannot for now) |
| Results workbench | **Mine results / Behavior plot / Calculate distances** shipped |
| Toolkit | **PySide6 only** (no wx / no `--legacy-wx`) |

**Recommended defaults** (can revisit later):

- Subject **roles**: free text; optional project-level vocabulary later  
- Batch detect/process concurrency: **one video at a time** for MVP  
- Branding: keep **LabGym** unless renamed later  

---

## 1. Product intent

A single PySide6 desktop app so a researcher can:

1. Prepare videos  
2. Detect and track animals; correct IDs; assign experimental names/roles  
3. Manually build ethograms and generate categorizer training data **from those ethograms**  
4. Train/test a categorizer and process new videos  
5. Mine results, plot behaviors, and compute distances from analysis outputs  

**Primary training philosophy (ethogram-first):**

```text
Detect & track → Fix IDs / assign roles → Annotate ethograms (JSON = source of truth)
  → Generate LabGym-style training pairs FROM ethograms
  → Train categorizer → Process / analyze new videos
```

**Secondary (Tools):** dense generate unsorted clips + sort (manual / CSV / annotations).

**Near-term non-goals:**

- Rewriting Detectron2 / training kernels  
- Full in-app statistics package beyond mine/plot/distances  
- Full in-app detector mask annotation (EZannot recommended until built-in lands)

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
| **Tools menu** | Secondary windows (dense generate + sort) |

### 2.2 Sketch

```text
┌─────────────────────────────────────────────────────────────────────┐
│ File  Project  Tools  Help           Project: MyExp.labproj.json    │
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
- **Toolkit:** primary workflows are in-process **PySide6** only.

---

## 3. Workbenches and tabs

### 3.1 Preprocessing

| Tab | Purpose |
|-----|---------|
| **Preprocess videos** | Trim/crop/resize/enhance |
| **Draw markers** | Marker-drawing workflow |

### 3.2 Detector

| Tab | Purpose |
|-----|---------|
| **Generate training data** | Subtabs: **Generate images** (frame extract); **Annotate images** (future; point to EZannot) |
| **Train detector** | Train Mask R-CNN from COCO annotations |
| **Test detector** | Evaluate detector |
| **Detect + track subjects** | Batch multi-video detect/track; durable identity package |
| **Review IDs & assign names/roles** | Fix ID swaps; remapped tracklets; display names/roles |

### 3.3 Categorizer

| Tab | Purpose |
|-----|---------|
| **Generate training data** | Subtabs: Annotate ethogram \| Generate examples (sorted pairs) |
| **Train categorizer** | Train on sorted folders; optional soft labels / export-onfly aug |
| **Test categorizer** | Test trained categorizer |
| **Process videos** | Batch: video + identity package → analysis outputs |

### 3.4 Results / data export

| Tab | Purpose |
|-----|---------|
| **Mine results** | Statistical comparison of LabGym summary folders |
| **Behavior plot** | Raster plot from `all_events.xlsx` |
| **Calculate distances** | Shortest / travel distances from analysis folders |

Further ethogram figures / R-ready exports remain future enhancements.

### 3.5 Tools (not a workbench)

| Entry | Purpose |
|-------|---------|
| **Dense generate + sort examples…** | Classic unsorted generation + manual/CSV/annotations sort |

---

## 4. Project concept

A **Project** is a saved experiment context shared across workbenches (`*.labproj.json`).

Contents: root folder, explicit video list, default paths/models/modes, optional notes/role vocabulary, recent files in QSettings.

---

## 5. Cross-cutting requirements

- **Toolkit:** PySide6 only for the product GUI.  
- **Jobs:** train/detect/generate/process off the UI thread; progress (+ cancel where feasible).  
- **Durability:** ethogram JSON + corrected tracklets are first-class.  
- **Modes:** 0 non-interactive · 1 interactive basic · 2 interactive advanced.  
- **Identity:** numeric track ID ≠ experimental display name/role.

---

## 6. Acceptance

User can preprocess → batch detect/track → review IDs/names → annotate ethogram → generate pairs → train categorizer → process videos → mine/plot/distances, all in the PySide app, without wxPython.

---

## 7. Mapping to code

| Spec area | Building blocks |
|-----------|-----------------|
| Shell + project | `gui_pyside/main_window.py`, `gui_pyside/project/*` |
| Preprocess / markers | `workbenches/preprocessing/*`, `tools` |
| Detector tabs | `workbenches/detector/*`, `detector.py`, `detection/batch_detect.py` |
| Review IDs | `workbenches/detector/review_ids_tab.py`, `id_review/*` |
| Annotate + generate pairs | `annotator/*`, `training/ethogram_examples.py` |
| Train/test categorizer | `workbenches/categorizer/*`, `categorizer.py` |
| Process videos | `analysis/process_videos.py` |
| Results | `workbenches/results/*`, `minedata.py`, `tools.plot_events` |
| Dense generate + sort | `gui_pyside/tools_windows/*` |
