# LabGym codebase audit

**Status:** Phase 0 inventory complete · Phase 1 findings drafted (living document)  
**Branch intent:** `docs/codebase-audit` (findings only; no product rewrites)  
**Against:** `main` at audit start (`f2174f6` Detect+track cleanup and later)  
**Companion:** root `implementation-plan.md` / `specifications.md` (migration complete)

---

## 0. Scope and exclusions

| In scope | Out of deep rewrite |
|----------|---------------------|
| `LabGym/gui_pyside/**` | Vendored `LabGym/detectron2/**` |
| Domain packages: `detection`, `analysis`, `training`, `id_review`, `identity`, `annotator` | Bundled weights under `LabGym/detectors/` |
| Core engines: `analyzebehavior*.py`, `categorizer.py`, `tools.py`, `detector.py`, `minedata.py` | Local data tree `testing_ground/` (gitignored; do not delete without ask) |
| Infra: config, logging, registration, probes, packaging, tests, docs | Pure Detectron2 style nits |

**Principles:** audit ≠ rewrite; short-lived themed fix branches after prioritization; preserve analysis behavior without characterization tests.

---

## 1. Mechanical inventory (Phase 0)

### 1.1 Layer sizes (approx. non-blank lines, excl. detectron2)

| Layer | ~LOC | Notes |
|-------|------|--------|
| `gui_pyside/` | ~8.0k | 46 Python modules; new workbench shell |
| Core root engines | ~6.6k | Four mega-files dominate |
| `annotator/` | ~6.3k | Standalone + embeddable ethogram UI |
| `id_review/` + `identity/` | ~2.2k | Modular; good unit tests |
| `training/` + `analysis/` + `detection/` | ~2.2k | Adapters for UI |
| `tests/unit` | ~2.8k | Strong infra/id_review; weak GUI/core loops |
| `detectron2/` | ~214 modules | Vendor — excluded |

### 1.2 Files ≥ 500 lines (LabGym, excl. detectron2)

| Lines | Path |
|------:|------|
| 2076 | `LabGym/analyzebehavior_dt.py` |
| 1907 | `LabGym/categorizer.py` |
| 1135 | `LabGym/analyzebehavior.py` |
| 1134 | `LabGym/tools.py` |
| 1037 | `LabGym/annotator/ui/main_window.py` |
| 861 | `LabGym/annotator/ui/bout_list.py` |
| 824 | `LabGym/gui_pyside/workbenches/categorizer/train_categorizer_tab.py` |
| 795 | `LabGym/training/ethogram_examples.py` |
| 765 | `LabGym/gui_pyside/workbenches/detector/review_ids_tab.py` |
| 702 | `LabGym/gui_pyside/workbenches/detector/detect_track_tab.py` |
| 684 | `LabGym/annotator/core/annotation_manager.py` |
| 553 | `LabGym/annotator/core/example_generator.py` |
| 504 | `LabGym/id_review/dataset.py` |

Also notable (400–500): `dense_generate_sort.py` (~490), `process_videos_tab.py` (~494).

### 1.3 Entry points (`pyproject.toml`)

| Script | Target |
|--------|--------|
| `LabGym` | `LabGym.__main__:main` → PySide workbench |
| `LabGym-workflow` | `LabGym.gui_pyside.main_window:main` |
| `LabGym-annotate` | `LabGym.annotator.__main__:main` |

No `--legacy-wx` / wxPython dependency (migration complete).

### 1.4 Dead / legacy references

| Pattern | Result |
|---------|--------|
| `import wx` / `mywx` / `legacy-wx` in product code | **None** (clean) |
| Source `LabGym/gui_*.py` | **Removed** |
| Comments mentioning old GUIs | `process_videos.py` (“gui_analyzer style”), `dense_backend.py` (“mirrors gui_categorizer”), `batch_detect.py` (wx ID review note), survey docstring history |
| Stale bytecode | `LabGym/__pycache__/gui_*.pyc` still present locally (gitignored; harmless but confuses greps) |

### 1.5 Test matrix (unit)

| Area | Coverage signal |
|------|-----------------|
| config / argparse / logging / probes / pkghash / registration / survey | Present |
| `id_review` / `identity` | Strong |
| `batch_detect` / `process_videos` / ethogram examples / soft labels / augment | Present (mostly mocked) |
| Workbench | Construct smoke only (`test_workbench_phases56`, `test_workbench_project`) |
| `gui_pyside/jobs`, progress, refresh-during-batch | **Missing** |
| Core acquire loops / full train | **Missing** (expected; heavy) |
| `tests/integration/` | **Empty** |

### 1.6 Repo hygiene

| Item | State |
|------|--------|
| `__pycache__/` | gitignored |
| `testing_ground/` | gitignored; can be huge locally |
| Root planning | `implementation-plan.md`, `specifications.md` mark UI MVP complete |
| Docs | `docs/module-structure.md` aligned with workbench; this audit is additive |

### 1.7 Workbench surface map

| Workbench | Tabs / surfaces |
|-----------|-----------------|
| Preprocess | preprocess, draw markers |
| Detector | generate images, **annotate images (placeholder)**, train, test, detect+track, review IDs |
| Categorizer | annotate ethogram, generate examples, train, test, process videos |
| Results | mine, plot, distances |
| Tools | dense generate + sort pop-out |

---

## 2. Finding schema

| Field | Meaning |
|-------|---------|
| ID | `AUD-<AREA>-NNN` |
| Severity | P0 blocker · P1 high · P2 medium · P3 low · P4 note |
| Type | bug · architecture · consistency · dead · test · docs · hygiene · product-gap |
| Effort | S · M · L |
| Wave | Suggested implementation wave (3a–3f) |

---

## 3. Findings (Phase 1 draft)

### 3.1 GUI / jobs / project state

#### AUD-GUI-001 — Process videos table status wipe (same class as Detect+track bug)

| | |
|--|--|
| **Severity** | P1 |
| **Type** | bug / consistency |
| **Location** | `process_videos_tab.py` `refresh_videos`, `project.changed` |
| **Evidence** | `refresh_videos` always rebuilds rows to `"pending"`; `_start` calls `mark_dirty()` which emits `changed`. Mid-batch completion may also dirty the project. No `_batch_active` / sticky status. |
| **Proposed fix** | Port Detect+track pattern: path job ids, sticky status, skip rebuild while batch active. |
| **Effort** | S–M · **Wave** 3b |

#### AUD-GUI-002 — Process videos uses row-index job ids (`p{r}`)

| | |
|--|--|
| **Severity** | P2 |
| **Type** | architecture / consistency |
| **Location** | `process_videos_tab.py` ~`jid = f"p{r}"` |
| **Evidence** | Detect+track now uses path as `job_id`; process videos still row-based → fragile under any table rebuild. |
| **Proposed fix** | `job_id=path` + same status helper. |
| **Effort** | S · **Wave** 3b |

#### AUD-GUI-003 — No structured frame progress on Process videos

| | |
|--|--|
| **Severity** | P2 |
| **Type** | consistency / UX |
| **Location** | `process_videos_tab.py`, `analysis/process_videos.py`, `analyzebehavior*` |
| **Evidence** | Detect+track has `JobProgress.frame` + dialog; process path is message-only. Long videos give weak feedback. |
| **Proposed fix** | Reuse `JobProgress` / optional frame callback through process adapter; optional progress dialog. |
| **Effort** | M · **Wave** 3b |

#### AUD-GUI-004 — `project.changed` fan-out rebuilds many tabs

| | |
|--|--|
| **Severity** | P2 |
| **Type** | architecture |
| **Location** | `controller.mark_dirty` → `changed`; listeners in detect_track, process, preprocess, markers, review_ids, generate_*, annotate ethogram, main title |
| **Evidence** | Dirty flag always emits `changed` even when already dirty. Any tab that fully rebuilds UI can thrash or reset widgets. |
| **Proposed fix** | Split signals (`dirty_changed` vs `project_data_changed`); or document “batch-active skip” as standard pattern; consider not emitting on every `mark_dirty` when already dirty if listeners only need dirty chrome. |
| **Effort** | M · **Wave** 3b |

#### AUD-GUI-005 — Dual meaning of job “done”

| | |
|--|--|
| **Severity** | P2 |
| **Type** | architecture |
| **Location** | `sequential_queue.py` + DetectTrackResult / ProcessVideoResult |
| **Evidence** | Queue sets `status="done"` when runner returns; result may have `ok=False`. Summary UIs re-derive success. |
| **Proposed fix** | Normalize in queue (inspect result protocol) or require raise on failure; document contract. |
| **Effort** | S–M · **Wave** 3b |

#### AUD-GUI-006 — Large workbench tabs approaching / past healthy size

| | |
|--|--|
| **Severity** | P2 |
| **Type** | architecture |
| **Location** | train_categorizer (~824), review_ids (~765), detect_track (~702), process_videos (~494), dense_generate_sort (~490) |
| **Evidence** | Form + worker + dialog + project IO in one file. |
| **Proposed fix** | Extract progress dialogs, video tables, model pickers into small modules (behavior-preserving). |
| **Effort** | M each · **Wave** 3c |

#### AUD-GUI-007 — Duplicated detector / model browse-scan UX

| | |
|--|--|
| **Severity** | P3 |
| **Type** | consistency |
| **Location** | detect_track, process_videos, train_*, generate_*, test_* |
| **Evidence** | Similar browse/scan/`resource_filename` roots repeated. |
| **Proposed fix** | Shared `ModelPicker` / scan helper under `gui_pyside`. |
| **Effort** | M · **Wave** 3c |

#### AUD-GUI-008 — Progress dialog patterns not shared

| | |
|--|--|
| **Severity** | P3 |
| **Type** | consistency |
| **Location** | `DetectTrackProgressDialog`, `TrainProgressDialog`, ad-hoc workers elsewhere |
| **Evidence** | Two dialog styles; preprocess/process use log lines only. |
| **Proposed fix** | Optional shared base or documented standard: pop-out for long jobs. |
| **Effort** | M · **Wave** 3c |

#### AUD-GUI-009 — Broad `except Exception` in UI paths

| | |
|--|--|
| **Severity** | P3 |
| **Type** | bug risk |
| **Location** | Many `gui_pyside` tabs (~35 hits); some bare `except Exception: pass` (scan roots, optional loads) |
| **Evidence** | Soft-fails hide misconfiguration. |
| **Proposed fix** | Prefer typed catches; log or surface user-visible errors for path/metadata failures. |
| **Effort** | M · **Wave** 3a/3b |

### 3.2 Product gaps (intentional but track)

#### AUD-PROD-001 — Annotate images is external-tool placeholder

| | |
|--|--|
| **Severity** | P2 (product) / P4 (code quality) |
| **Type** | product-gap |
| **Location** | `annotate_images_tab.py` |
| **Evidence** | Explicit EZannot / COCO workflow; no in-app labeling. Spec already locks this. |
| **Proposed fix** | Keep placeholder; track as roadmap item, not audit cleanup. |
| **Effort** | L (product) · **Wave** later |

### 3.3 Domain adapters

#### AUD-DOM-001 — Headless adapters are the right layer; keep expanding them

| | |
|--|--|
| **Severity** | P4 |
| **Type** | architecture (positive) |
| **Location** | `detection/batch_detect.py`, `analysis/process_videos.py`, `training/*` |
| **Evidence** | UI should call adapters, not engines directly. |
| **Proposed fix** | New features go through adapters; add tests when behavior changes. |
| **Effort** | — · **Wave** 3d/3e as needed |

#### AUD-DOM-002 — Stale wx / old-GUI comments in adapters

| | |
|--|--|
| **Severity** | P3 |
| **Type** | docs / hygiene |
| **Location** | `batch_detect.py`, `process_videos.py`, `dense_backend.py` |
| **Proposed fix** | Rephrase comments to PySide / workbench wording. |
| **Effort** | S · **Wave** 3a |

### 3.4 Core engines (map only — no rewrite in audit)

#### AUD-CORE-001 — Mega-modules concentrate risk

| | |
|--|--|
| **Severity** | P2 |
| **Type** | architecture |
| **Location** | `analyzebehavior_dt.py`, `categorizer.py`, `analyzebehavior.py`, `tools.py` |
| **Evidence** | >1k LOC each; mixed I/O, CV, TF/PyTorch, mode branches. |
| **Proposed fix** | Only after call-graph from UI + characterization tests; extract pure helpers first. |
| **Effort** | L · **Wave** 3f (optional) |

#### AUD-CORE-002 — UI call surface is smaller than public class surface

| | |
|--|--|
| **Severity** | P3 |
| **Type** | architecture |
| **Evidence** | Workbench primarily uses prepare/acquire/craft/export paths via adapters; many methods are legacy interactive modes. |
| **Proposed fix** | Document “supported entry methods” for adapters; do not delete unused methods without usage grep + tests. |
| **Effort** | M (docs/map) · **Wave** Phase 1 complete / 3f |

#### AUD-CORE-003 — Broad exceptions in detector / categorizer

| | |
|--|--|
| **Severity** | P3 |
| **Type** | bug risk |
| **Location** | `detector.py`, `categorizer.py` |
| **Proposed fix** | Narrow catches when touching those paths; do not mass-edit. |
| **Effort** | S opportunistic · **Wave** 3a/3d |

### 3.5 Annotator

#### AUD-ANN-001 — Dual shell (standalone + workbench host)

| | |
|--|--|
| **Severity** | P3 |
| **Type** | architecture |
| **Location** | `annotator/__main__.py`, categorizer `annotate_ethogram_tab` |
| **Evidence** | Large UI (`main_window` >1k, `bout_list` ~861). |
| **Proposed fix** | Boundary audit: schema/export contracts first; deep UI polish later. |
| **Effort** | M–L · **Wave** later |

#### AUD-ANN-002 — Annotator has some unit tests; not GUI interaction tests

| | |
|--|--|
| **Severity** | P3 |
| **Type** | test |
| **Location** | `test_annotator_schema`, `test_annotator_tracklets` |
| **Proposed fix** | Keep schema/tracklet tests green; add export regression fixtures if bugs appear. |
| **Effort** | S–M · **Wave** 3e |

### 3.6 Tests / CI / packaging

#### AUD-TEST-001 — GUI covered only by construct smoke

| | |
|--|--|
| **Severity** | P2 |
| **Type** | test |
| **Location** | `test_workbench_phases56.py` |
| **Evidence** | Instantiates tabs; no queue/progress/status behavior. |
| **Proposed fix** | Unit-test `JobProgress` + a pure “status map” helper; optional Qt offscreen tests for refresh skip. |
| **Effort** | M · **Wave** 3e |

#### AUD-TEST-002 — Empty integration suite

| | |
|--|--|
| **Severity** | P3 |
| **Type** | test |
| **Location** | `tests/integration/` |
| **Proposed fix** | Optional smoke script or one marked integration test; not blocking hygiene. |
| **Effort** | M · **Wave** 3e |

#### AUD-TEST-003 — No tests for jobs package

| | |
|--|--|
| **Severity** | P2 |
| **Type** | test |
| **Location** | `gui_pyside/jobs/sequential_queue.py` |
| **Proposed fix** | Threadless unit tests for JobProgress + worker status transitions with fake runner. |
| **Effort** | S · **Wave** 3e |

### 3.7 Hygiene / docs

#### AUD-HYG-001 — Stale `gui_*.pyc` under package `__pycache__`

| | |
|--|--|
| **Severity** | P4 |
| **Type** | hygiene |
| **Proposed fix** | Local clean (`Remove-Item -Recurse LabGym\__pycache__\gui_*.pyc`); already gitignored. |
| **Effort** | S · **Wave** 3a |

#### AUD-HYG-002 — Root one-off notes vs Sphinx docs

| | |
|--|--|
| **Severity** | P4 |
| **Type** | docs |
| **Location** | `Parallelization_PRs.md`, `offline_detector_augment.md`, root guides |
| **Proposed fix** | Keep or move under `docs/contributing/` with index link; avoid silent drift. |
| **Effort** | S · **Wave** 3a |

#### AUD-HYG-003 — This audit document

| | |
|--|--|
| **Severity** | — |
| **Type** | docs |
| **Location** | `docs/codebase-audit.md` |
| **Proposed fix** | Update after each implementation wave; link from contributing index when merged. |
| **Effort** | S |

---

## 4. Recommended priority (for Phase 2 workshop)

### Now (high leverage, bounded)

1. ~~**AUD-GUI-001 + AUD-GUI-002**~~ — **Done** (wave 3b): Process videos sticky status + path job ids.  
2. ~~**AUD-TEST-003**~~ — **Done** (wave 3b): `tests/unit/test_sequential_queue.py`.  
3. ~~**AUD-DOM-002 + AUD-HYG**~~ — **Done** / ongoing: old-GUI comment cleanup; local `gui_*.pyc` remains gitignored.  

### Next

4. ~~**AUD-GUI-004**~~ — **Partial (wave 3b follow-up):** batch-active / preserve selection on Preprocess, Draw markers, Review IDs; controller docs. Full signal split still optional later.  
5. ~~**AUD-GUI-003**~~ — **Done** (wave 3b): process_video frame progress + status note.  
6. ~~**AUD-GUI-005**~~ — **Done** (wave 3b): soft-fail `ok=False` on queue.  
7. ~~**AUD-GUI-006**~~ — **Wave 3c:** train progress/workers, detect-track progress, review_ids package/render/markers modules extracted.  
7b. ~~**AUD-GUI-007 / 008**~~ — **Wave 3c polish:** `model_paths` scan + `widgets/path_browse` + `JobProgressDialogBase` shared by train/detect-track dialogs.  
8. **AUD-TEST-001** — Expand workbench behavioral tests.  

### Later

9. **AUD-CORE-001 / 002** — Engine decomposition only with characterization.  
10. **AUD-ANN-001** — Annotator UI deep clean.  
11. **AUD-PROD-001** — In-app detector annotation (product).  

### Never / vendor

- Detectron2 style refactors  
- Deleting `testing_ground/` without explicit request  
- Mass formatting of legacy tab-indented core files without purpose  

---

## 5. Suggested implementation waves (post-prioritization)

| Wave | Theme | Finding IDs |
|------|-------|-------------|
| **3a** Hygiene | AUD-DOM-002, AUD-HYG-001, AUD-HYG-002, opportunistic AUD-CORE-003 |
| **3b** Cross-UI jobs/status | AUD-GUI-001–005, AUD-GUI-009 (targeted) |
| **3c** Tab decomposition | AUD-GUI-006–008 |
| **3d** Domain API | AUD-DOM-001 as policy; adapter cleanups as needed |
| **3e** Tests | AUD-TEST-001–003, AUD-ANN-002 |
| **3f** Core (optional) | AUD-CORE-001–002 |

Each wave: branch off `main` → small PR → smoke relevant workbench path → merge.

---

## 6. Smoke matrix (manual; file P0/P1 if fail)

1. Launch workbench; new/open project  
2. Preprocess one clip  
3. Detect+track short video (status stickiness + progress dialog)  
4. Review IDs open package  
5. Annotate ethogram → generate tiny examples  
6. Process videos batch (watch for status wipe — **AUD-GUI-001**)  
7. Results mine/plot/distances if fixtures exist  
8. Tools dense generate+sort open/close  
9. `LabGym-annotate` standalone  

---

## 7. Phase status

| Phase | Status |
|-------|--------|
| 0 Inventory | **Done** (this doc §1) |
| 1 Findings draft | **Done** (this doc §3; may grow) |
| 2 Prioritization with owner | Wave 3b + follow-up accepted |
| 3 Implementation waves | 3a/3b done; **3c started** (train + detect-track progress extract) |

---

## 8. Changelog

| Date | Note |
|------|------|
| 2026-07-27 | Initial Phase 0–1 audit on `docs/codebase-audit` after Detect+track progress cleanup on `main`. |
| 2026-07-27 | Wave 3b: process videos sticky status, queue soft-fail, frame progress, job unit tests. |
| 2026-07-27 | Wave 3a/3b follow-up: Preprocess status+batch-active; Draw markers / Review IDs selection preserve; survey/controller hygiene; audit status update. |
| 2026-07-27 | Wave 3c: extract `train_progress_dialog` / `train_workers`; extract `detect_track_progress`. |
| 2026-07-27 | Wave 3c: Review IDs split — `review_ids_package`, `review_ids_render`, `review_ids_markers`. |
| 2026-07-27 | Wave 3c polish: shared `model_paths`, path browse widgets, `JobProgressDialogBase`. |
