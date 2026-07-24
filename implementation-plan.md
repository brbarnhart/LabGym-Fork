# Implementation Plan — PySide Workbench GUI

**Against:** [`specifications.md`](./specifications.md)  
**Status:** ready to execute  
**Principle:** ship usable ethogram-first workflow early; port remaining LabGym surfaces behind the same shell; pure PySide end state.

---

## 0. Current baseline

| Area | State today |
|------|-------------|
| Ethogram annotator | PySide6, multi-subject, modes 0/1/2, bout editor, partners bulk edit |
| Ethogram → training pairs | `training/ethogram_examples.py` + generate dialog/tab |
| ID review / tracklets | Logic in `id_review/*`; UI still mostly wx (`gui_id_review.py`) |
| Detect / preprocess / train / analyze | wx panels (`gui_*.py`) + backend modules |
| Early shell | `gui_pyside/*` — flat tabs + project QSettings, not FreeCAD-style workbenches |
| Entry points | `LabGym` (wx), `LabGym-annotate`, `LabGym-workflow` |

**Reuse heavily:** annotator, ethogram example gen, tracklets bridge, ID review apply/remap.  
**Reshape:** `gui_pyside` into the workbench shell.  
**Port later:** preprocess, detector train/test, batch detect+track, categorizer train/test/process.

---

## 1. Target architecture

```text
LabGym/gui_pyside/                    # primary application package
  app.py                              # QApplication bootstrap
  main_window.py                      # shell: menus, workbench bar, stack, status
  project/
    model.py                          # Project dataclass + JSON schema
    controller.py                     # open/save/dirty, signals
    video_list.py                     # root folder + explicit video entries
  shell/
    workbench_bar.py                  # top icon toolbar (exclusive workbench)
    workbench_host.py                 # swaps tab widget per workbench
  workbenches/
    preprocessing/
      __init__.py                     # registers tabs
      preprocess_tab.py
      draw_markers_tab.py
    detector/
      train_tab.py
      test_tab.py
      detect_track_tab.py             # batch
      review_ids_tab.py
      image_examples_tab.py           # stub "future"
    categorizer/
      generate_training_tab.py        # host with 2 subtabs
      annotate_ethogram_tab.py
      generate_examples_tab.py
      train_tab.py
      test_tab.py
      process_videos_tab.py
    results/
      placeholder_tab.py
  jobs/
    worker.py                         # QThread / QObject patterns
    sequential_queue.py               # one-video-at-a-time batch runner
  legacy/                             # temporary only
    wx_bridge.py                      # optional subprocess launch while porting
```

**Patterns:**

- **Workbench** = `QWidget` that owns a `QTabWidget` of its tabs.  
- **ProjectController** = single shared object injected into every tab.  
- **Jobs** = never block the GUI thread; emit progress/cancel where feasible.  
- **Empty states** = each tab checks project/upstream artifacts and offers “Go to …”  
- **Annotator** = refactor `annotator.ui.main_window.MainWindow` so core workspace can embed as `QWidget` (or keep QMainWindow-as-widget embed used today). Prefer extracting `AnnotatorWorkspace(QWidget)` if menus fight the shell.

**Entry points (end of migration):**

| Command | Behavior |
|---------|----------|
| `LabGym` | Prefer PySide workbench shell |
| `LabGym-workflow` | Alias to same shell (compat) |
| `LabGym-annotate` | Optional deep-link: open shell → Categorizer → Annotate ethogram |
| `LabGym --legacy-wx` | Temporary escape hatch to old GUI until removed |

---

## 2. Phased PR plan

Phases are ordered for **daily-usable ethogram-first path first**, then fill the rest of the spec. Each phase should leave `main` runnable.

### Phase 0 — Spec freeze & scaffolding (docs only if already done)

- [x] `specifications.md` agreed  
- [ ] Point README / `docs/features` at workbench + ethogram-first docs  
- [ ] Optional: `LabGym --legacy-wx` flag stub so default can flip later without surprise  

**Exit:** team uses this plan + spec as contract.

---

### Phase 1 — Workbench shell + Project (foundation) ✅

**Goal:** App looks and navigates like the spec; project is real on disk.

| Work | Detail | Status |
|------|--------|--------|
| Shell chrome | Top exclusive workbench toolbar: Preprocess, Detector, Categorizer, Results | Done |
| Tab host | Switching workbench swaps the whole tab strip + content | Done |
| Project model | JSON file `*.labproj.json`: root folder, explicit video list, default paths, modes, last models, notes | Done |
| Project UI | File → New/Open/Save/Save As/Recent; Project → Edit; status bar | Done |
| Results | Placeholder workbench (“Coming soon”) | Done |
| Other tabs | Placeholder panels + temporary legacy button | Done |
| Migrate | New `gui_pyside/project`, `shell`, `workbenches`; old flat tabs kept for Phase 2 | Done |

**Project schema (v1 sketch):**

```json
{
  "schema_version": 1,
  "name": "MyExp",
  "root_dir": "D:/experiments/MyExp",
  "videos": [
    {"path": "videos/a.avi", "enabled": true}
  ],
  "paths": {
    "detection_output_root": "detection",
    "annotations_root": "",
    "examples_root": "examples",
    "models_root": "models"
  },
  "defaults": {
    "behavior_mode": 0,
    "exclusive_mode": true,
    "window_length": 15,
    "sampling": "dense_in_bout",
    "detector_name": "",
    "categorizer_name": ""
  },
  "notes": ""
}
```

**Acceptance:**

- Open app → switch four workbenches; tabs change correctly.  
- Create/save/reload project with root + video list.  
- Restart app → recent project restores.

**Tests:** unit tests for project load/save/round-trip; smoke test shell constructs without video.

---

### Phase 2 — Categorizer: Generate training data (Annotate + Generate) ✅

**Goal:** Spec §3.3 Generate training data fully usable inside the shell (highest leverage; mostly exists).

| Work | Detail | Status |
|------|--------|--------|
| Subtabs | Under Categorizer: **Annotate ethogram** \| **Generate examples** | Done |
| Annotate | Embed annotator; load video/tracklets/ann/mode from project | Done |
| Video picker | Project video combo; path resolution helpers | Done |
| Generate | Params + background `ethogram_examples`; project defaults | Done |
| Empty states | Edit project / Go to Annotate prompts | Done |
| Dense generate-then-sort | Not offered | Done |

**Acceptance:**

- From a saved project, load video+tracklets, annotate, Ctrl+S annotations, generate sorted pairs — without wx.  
- Re-generate with new length without re-annotating.

**Tests:** existing schema/partner tests remain green; project→path resolution unit tests.

---

### Phase 3 — Detector: Review IDs & assign names/roles ✅

**Goal:** Identity layer is correct and named before ethograms (spec §3.2 high priority).

| Work | Detail | Status |
|------|--------|--------|
| PySide Review IDs tab | Risk timeline, mark swap, undo, video scrub | Done |
| Persist | `finalize_switch_annotations` + remapped tracklets from baseline | Done |
| Names/roles | `subjects.json` via SubjectsTable | Done |
| Project integration | Video combo + open package folder | Done |
| Annotator bridge | `load_tracklets_for_annotator` merges subjects.json | Done |

**Identity package layout (propose, document in spec addendum if needed):**

```text
<video_stem>_or_id_review/
  *_tracklets.npz (+ meta)
  subjects.json   # [{id, display_name, role, color}, ...]
```

**Acceptance:**

- Fix an ID swap, save, reopen → remap held.  
- Assign names/roles → Annotate ethogram shows them.  
- No wx required for this path.

**Tests:** subjects.json round-trip; remap apply unit tests (extend `id_review` tests).

---

### Phase 4 — Detector: Detect + track (batch) ✅

**Goal:** Batch produce identity packages for the project video list.

| Work | Detail | Status |
|------|--------|--------|
| UI | Detect + track tab: detector, params, multi-select videos | Done |
| Runner | Sequential `SequentialJobQueue` | Done |
| Output | `detection/<stem>/id_review` + `detection_dir` on video entries | Done |
| Backend | `LabGym.detection.batch_detect.detect_and_track_video` | Done |

**Risk:** `analyzebehavior*` is large and wx-coupled in places. Strategy:

1. Extract/identify a **headless** function: `(video, detector, out_dir, params) → identity package`.  
2. If extraction is hard, Phase 4a: subprocess CLI wrapper; Phase 4b: true in-process API.

**Acceptance:**

- Queue ≥2 videos, get per-video outputs, Review IDs can open them.

**Tests:** adapter mock or small fixture video if available; otherwise integration checklist.

---

### Phase 5 — Preprocessing workbench ✅

| Tab | Work | Status |
|-----|------|--------|
| Preprocess videos | PySide form + `tools.preprocess_video` batch | Done |
| Draw markers | Canvas + burn markers onto videos | Done |

---

### Phase 6 — Train / test detector & categorizer ✅

| Tab | Work | Status |
|-----|------|--------|
| Train detector | `Detector.train` off UI thread | Done |
| Test detector | `Detector.test` off UI thread | Done |
| Train categorizer | prepare + `Categorizers.train_*` + soft labels | Done |
| Test categorizer | `Categorizers.test_categorizer` | Done |

**Acceptance:** train + test both model types from shell; paths default from project.

**Risk:** long training jobs — require robust logging UI and non-blocking workers; cancel best-effort.

---

### Phase 7 — Process videos (categorizer batch)

| Work | Detail |
|------|--------|
| Inputs | Project videos + identity package (+ categorizer) |
| Backend | Headless path from `analyzebehavior*` categorizer-on-tracks pipeline |
| Outputs | Behavior time series / LabGym analysis products under project output root |
| Queue | Sequential per video |

**Acceptance:** end-to-end without wx: detect → review → (optional annotate/train) → process.

---

### Phase 8 — Default entry + retire wx

| Work | Detail |
|------|--------|
| `LabGym` main | Launch PySide shell by default |
| Legacy | `--legacy-wx` only; document deprecation |
| Remove | `legacy_launch` from normal tabs; delete dead wx menu paths when unused |
| Results | Still placeholder unless starting Results epic |

**Acceptance:** daily workflow matches spec §7 without opening wx.

---

### Phase 9+ — Future (out of MVP)

- Results: ethogram figures, R-ready tables  
- Detector: create & annotate image examples  
- Parallel batch jobs  
- Project role vocabulary UI  

---

## 3. Dependency graph

```text
Phase 1 Shell + Project
    │
    ├─► Phase 2 Categorizer Generate (Annotate | Generate)
    │
    ├─► Phase 3 Review IDs + names/roles  ──► improves Phase 2 load quality
    │
    ├─► Phase 4 Detect+track batch ──► feeds Phase 3
    │
    ├─► Phase 5 Preprocess (parallelizable with 3–4)
    │
    ├─► Phase 6 Train/test models
    │         │
    │         └─► Phase 7 Process videos
    │
    └─► Phase 8 Make shell default / retire wx
```

**Critical path to “ethogram-first daily use”:**  
`1 → 2` (immediate) + `3` (identity quality) + `4` (batch detect) → then `6–7`.

**Minimum lovable product (MLP):** Phases **1 + 2 + 3**, with detect still via temporary legacy bridge only if Phase 4 slips.

---

## 4. Cross-cutting workstreams

### 4.1 Jobs framework (introduce in Phase 1–2, reuse always)

- `JobWorker(QObject)` + `QThread`  
- Signals: `progress(int,int,str)`, `finished(object)`, `error(str)`  
- Sequential batch runner for multi-video tabs  

### 4.2 Empty-state helper

- Standard widget: message + primary button “Open workbench X / tab Y”  
- Shell API: `main_window.goto(workbench_id, tab_id)`

### 4.3 Identity package API

- Single module e.g. `LabGym.identity.package` used by detect, review, annotator, process  
- Avoid three different folder conventions  

### 4.4 Testing strategy

| Layer | What |
|-------|------|
| Unit | Project JSON, path resolution, identity subjects, ethogram schema (existing) |
| Widget smoke | Construct shell + each workbench without display if possible (`QT_QPA_PLATFORM=offscreen`) |
| Manual checklist | Per-phase acceptance in this doc |

### 4.5 Temporary legacy bridge policy

- Allowed only on unported tabs, labeled **“Open legacy LabGym (temporary)”**.  
- Removed when that tab’s Phase lands.  
- Never the path for Annotate ethogram / Generate examples.

---

## 5. Suggested sprint slicing

Assuming focused work (~1 person familiar with the repo):

| Sprint | Deliver |
|--------|---------|
| S1 | Phase 1 shell + project file |
| S2 | Phase 2 annotate + generate subtabs wired to project |
| S3 | Phase 3 Review IDs + names/roles + subjects.json |
| S4 | Phase 4 batch detect adapter (or CLI bridge → in-process) |
| S5 | Phase 5 preprocess (+ markers if time) |
| S6 | Phase 6 train/test UIs |
| S7 | Phase 7 process videos + Phase 8 default entry |

Adjust if headless extract for detect/process takes longer than expected (split S4/S7).

---

## 6. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Analysis code tightly bound to wx | Adapter + optional subprocess; don’t block shell on full rewrite |
| Embedding annotator menus inside shell | Extract `AnnotatorWorkspace` widget; shell owns global menus |
| Project path sprawl | One schema version field; migration function on load |
| Scope creep (Results, image examples) | Explicitly Future; placeholder only |
| CUDA / long train UX | Reuse existing train entrypoints; log pane + disable double-submit |

---

## 7. Definition of done (spec §7)

Without wx, user can:

1. Create a project (root + videos)  
2. Preprocess (after Phase 5)  
3. Batch detect+track  
4. Review IDs and assign names/roles  
5. Annotate ethogram and generate training pairs  
6. Train categorizer and process videos  

Results may remain a placeholder.

---

## 8. Immediate next action

**Execute Phase 1:** implement workbench shell + `*.labproj.json` project model, reshape `gui_pyside`, leave non-ported tabs as placeholders (optional legacy button).

When Phase 1 merges, start Phase 2 immediately so the ethogram path lives in the new chrome.
