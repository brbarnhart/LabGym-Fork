# AGENTS.md — LabGym

Decision guide for AI coding agents. Prefer this file over rediscovering the tree each session. Human contributor setup: `docs/contributing/`. Full module map: `docs/module-structure.md`.

---

## Product

LabGym quantifies **user-defined** animal/object behaviors from video/images: detect → track → (optional ID fix) → classify → quantify kinematics.

**Primary training path (ethogram-first):**

```text
Detect+track → Review IDs → Annotate ethogram → Generate pairs → Train categorizer → Process videos
```

Dense “generate unsorted clips then sort” is a **Tools** power feature, not the default workbench path. See `docs/features/annotator-workflow.md`.

**Behavior modes** (analysis / categorizer):

| Code | Mode |
|-----:|------|
| 0 | Non-interactive |
| 1 | Interactive basic |
| 2 | Interactive advanced |
| 3 | Static image |

Version: `LabGym/__init__.py` (`__version__`). License: **GPL-3.0**. Default branch in this workspace: **`main`** (some upstream docs still say `master`).

---

## Stack (do not invent alternatives)

| Layer | Choice |
|-------|--------|
| Language | Python **3.9–3.10 only** (`requires-python = ">=3.9,<3.11"`) |
| GUI | **PySide6** FreeCAD-style workbenches — **no wx** |
| Detector | PyTorch **2.7.1** + vendored **Detectron2** (`LabGym/detectron2/`) |
| Categorizer | TensorFlow / Keras (Windows: TF `<2.11`) |
| CV / data | OpenCV, NumPy `<=1.26.4`, pandas, scikit-image/learn, matplotlib/seaborn |
| Packaging | `pyproject.toml` (pdm-backend), **`uv.lock`**, optional `nox` |
| Tests | pytest (`pytest.ini` markers: `slow`, `integration`, `gui`) |

**CUDA:** On Windows/Linux, torch comes from the `pytorch-cu118` index via `[tool.uv.sources]`. Do not let installs silently switch to CPU-only torch (breaks detector batch-size / GPU UI assumptions).

Do **not** introduce another GUI toolkit, job framework, progress system, or ML stack.

---

## Entry points & run

| Command | What it launches |
|---------|------------------|
| `LabGym` | Default workbench shell (`LabGym.__main__:main`) |
| `LabGym-workflow` | Same shell (compat alias) |
| `LabGym-annotate` | Standalone ethogram annotator |
| `python -m LabGym.gui_pyside` | Same workbench |
| `python -m LabGym.annotator` | Annotator |
| `python -m LabGym.training.ethogram_examples` | CLI: ethogram → training pairs |

```powershell
# Preferred install (repo root)
uv sync
# Alternate
py -3.10 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -e .

# Run
.\.venv\Scripts\python.exe -m LabGym
# or after activate / console scripts on PATH: LabGym
```

User config lives under **`~/.labgym/`** (`config.toml`, `logging.yaml`, `registration.yaml`). Override config file with `LABGYM_CONFIGFILE` or `--configfile`.

---

## Architecture (where code goes)

```text
LabGym.__main__
  → gui_pyside.WorkbenchMainWindow
       workbenches: preprocessing | detector | categorizer | results
       project (*.labproj.json) | jobs | widgets | tools_windows
  → engines: detector.py | categorizer.py | analyzebehavior*.py | tools.py | minedata.py
  → adapters: detection/ | training/ | analysis/ | id_review/ | identity/ | annotator/
```

| Path | Responsibility |
|------|----------------|
| `LabGym/gui_pyside/` | Shell, project, jobs, shared widgets, workbench tabs |
| `LabGym/gui_pyside/project/` | `*.labproj.json` model, controller, path resolution |
| `LabGym/gui_pyside/jobs/` | `SequentialJobQueue`, `JobItem`, `JobProgress` |
| `LabGym/gui_pyside/widgets/` | `path_browse`, `JobProgressDialogBase` |
| `LabGym/annotator/` | Ethogram UI (embed + standalone) |
| `LabGym/detection/` | Batch detect/track, detector continue-train, train progress hooks |
| `LabGym/training/` | Ethogram → examples, soft labels, train progress helpers |
| `LabGym/analysis/` | Process-videos adapter |
| `LabGym/id_review/`, `identity/` | Tracklet packages, remaps, hard-case helpers |
| Root engines (`detector.py`, …) | Scientific / training / analysis implementations |
| `LabGym/detectron2/` | **Vendored** Detectron2 — treat as third-party |

**Workbench tabs (product surface):**

- **Preprocess** — preprocess videos; draw markers  
- **Detector** — generate images; annotate images (**placeholder**); train/test; detect+track; review IDs  
- **Categorizer** — annotate ethogram; generate examples; train/test; process videos  
- **Results** — mine; plot; distances  
- **Tools** — dense generate + sort (pop-out)

---

## Data & path contracts

| Artifact | Rule |
|----------|------|
| Project file | `*.labproj.json` — root folder, video list, defaults, path roots (`PROJECT_SCHEMA_VERSION = 1`) |
| Annotations | Default **sidecar**: `Path(video).with_suffix(".annotations.json")`; overridable per video / project |
| Tracklets | Prefer post–ID-review `id_review/` under detection output (`*_tracklets.npz`, meta JSON) |
| Examples | Project `examples_root` (default `examples/`); ethogram generation writes sorted behavior folders + optional `soft_labels.csv` |
| Models | Project `models_root`; package defaults under `LabGym/detectors/`, `LabGym/models/` |
| Path API | Use `gui_pyside/project/paths.py` (`annotations_path_for`, tracklet discovery, `ResolvedVideoContext`) — do not ad-hoc reimplement |

---

## Coding rules

### GUI thin, engines thick

- Tabs/workers call engines and adapters; put science/ML logic in engines/adapters, not in widgets.
- Prefer extending existing packages over new top-level modules.

### Where to put new work

| If you are adding… | Put it here |
|--------------------|-------------|
| Workbench tab UI | Matching `gui_pyside/workbenches/<area>/*_tab.py` |
| Shared folder/file picker | Reuse `widgets/path_browse.py` — do not copy `QFileDialog` glue |
| Long-running progress dialog | Subclass `widgets/JobProgressDialogBase` |
| Multi-video / multi-item batch | `jobs/sequential_queue.py` with **path** (or stable id) as `job_id` |
| Project path resolution | `project/paths.py` (+ model fields in `project/model.py` if schema grows) |
| Detector/categorizer root listing | `gui_pyside/model_paths.py` |
| Batch detect/track logic | `detection/batch_detect.py` (GUI stays in detect_track tab) |
| Detector continue-train / loss hooks | `detection/continue_train.py`, `detection/train_progress.py` |
| Ethogram → training pairs | `training/ethogram_examples.py` (+ tab already hosts UI) |
| Process-videos batch adapter | `analysis/process_videos.py` |
| ID remaps / tracklet packages | `id_review/`, `identity/` |
| Ethogram annotation UX | `annotator/` (standalone + embed from categorizer tab) |
| Dense unsorted generate/sort | `tools_windows/` — not a new workbench |
| Core detect/classify/quantify math | Root engines (`detector.py`, `categorizer.py`, `analyzebehavior*.py`, …) |
| Unit test | `tests/unit/test_<area>.py` (mirror feature; mark `gui` / `slow` when needed) |

### Reuse these UI building blocks

| Need | Use |
|------|-----|
| Folder/file pickers | `LabGym.gui_pyside.widgets.path_browse` |
| Long-job progress window | Subclass `JobProgressDialogBase` |
| Batch sequential work | `SequentialJobQueue` + path-based `job_id` (not row index) |
| Progress text + frames | `JobProgress` / `as_frame_callback` in `jobs/sequential_queue.py` |
| Detector/categorizer model roots | `gui_pyside.model_paths` |
| Train epoch/iter UI hooks | Existing `train_progress_cb` / `detection.train_progress` patterns |

### Style

- New/changed public functions: type hints + Google-style docstrings.
- Modules `snake_case`; tabs `*_tab.py`; tests `tests/unit/test_*.py`.
- Avoid new blanket `except Exception: pass` on user-visible paths.
- Do not grow already-large tabs past ~1k LOC without extraction.

---

## Known gotchas (read before editing UI / jobs)

These are recurring footguns. Full inventory: `docs/codebase-audit.md` (IDs there are optional).

1. **`project.changed` fan-out** — `mark_dirty()` emits `changed` even when already dirty. Many tabs rebuild widgets on that signal. Mid-job dirty can wipe tables or reset controls.
2. **Batch-active refresh pattern** — While a batch runs, **skip full rebuilds** and keep **sticky** per-row status. Detect+track is the reference; **Process videos** is weaker (row-index job ids like `p{r}`, rebuild can reset to `"pending"`). When touching process videos, move toward **path job ids + sticky status**.
3. **Dual meaning of job “done”** — Queue may set `status="done"` when the runner returns, while the result object has `ok=False`. Always inspect structured results; use `soft_error_from_result` / status summary helpers.
4. **Prefer path-based `job_id`** — Row indices break if the table rebuilds under a running queue.
5. **Frame progress is uneven** — Detect+track supports `JobProgress.frame`; process-videos is often message-only. Add structured frame callbacks through adapters when you touch long video jobs.
6. **Bare `pytest` on PATH** — May not be the repo `.venv` (conda pollution). Always `.\.venv\Scripts\python.exe -m pytest …`.
7. **CPU-only torch install** — Breaks GPU batch-size UX on Windows/Linux; keep `uv` CUDA 11.8 sources.
8. **Stale `__pycache__`** — Local pyc for deleted modules can confuse greps; sources of truth are `.py` files.
9. **Logging dual stack** — Boot path uses `mylogging` in `__main__`; `central_logging` also exists. Follow the pattern already used in the module you edit; do not invent a third logger setup.
10. **Upstream doc drift** — Some docs say `master` / old wx flows; product code is PySide + `main`.

---

## ML-specific decisions

| Topic | Decision |
|-------|----------|
| Detectron2 | Vendor tree only — no style cleanup; touch only for proven LabGym integration bugs |
| Detector continue-train | Supported via `detection/continue_train.py`: **same classes**, warm-start from `model_final.pth`; **not** adding new animal categories |
| Detector train UI loss | Live loss via progress callback / EventStorage; dialog is **monitor-only** (no reliable mid-train cancel) |
| Categorizer continue-train | **Not shipped** — do not assume weight warm-start UI/API exists |
| Soft labels | Ethogram generation can emit `soft_labels.csv`; train supports `hard_soft_aux` / `label_mode` in categorizer |
| Annotate images (Detector) | Placeholder product gap — not a full in-app COCO annotator |

---

## Tests & quality

**Always use the repo venv.** Bare `pytest` on PATH may hit a different Python (e.g. conda) and fail collection.

```powershell
# Fast focused (example — verified on this repo)
.\.venv\Scripts\python.exe -m pytest tests/unit/test_path_browse.py tests/unit/test_progress_dialog_base.py tests/unit/test_detector_continue_train.py tests/unit/test_detector_train_progress.py -q

# Broader local unit suite
.\.venv\Scripts\python.exe -m pytest tests/unit -q

# Markers
.\.venv\Scripts\python.exe -m pytest -m "not slow" -q
.\.venv\Scripts\python.exe -m pytest -m integration -q
.\.venv\Scripts\python.exe -m pytest -m gui -q

# CI matrix (3.9 + 3.10)
nox -s tests
```

| Present | Weak / missing |
|---------|----------------|
| config, registration, id_review, path_browse, jobs helpers, continue-train, many adapters | Full acquire loops, full GPU train, deep GUI automation |
| Integration smokes under `tests/integration/` | Not a substitute for unit coverage of engines |

Lint: `[tool.ruff.lint]` in `pyproject.toml` (E4/E7/E9/F/I001); optional `tests/linting/pylint*`. **No mypy/typecheck gate in CI.** Docs: Sphinx under `docs/` (`nox -s docs`).

---

## Do not touch (without explicit ask)

1. **`LabGym/detectron2/**`** — vendored third-party.
2. **`testing_ground/`** — local data/experiments (gitignored); never delete or “clean”.
3. **Core mega-files** without characterization tests first: `analyzebehavior_dt.py`, `categorizer.py`, `analyzebehavior.py`, `tools.py`.
4. **wx / `gui_*.py` / `--legacy-wx`** — removed; do not revive.
5. User data under **`~/.labgym/`**, large local videos, or model weights not part of the requested change.
6. Force-push, `git reset --hard`, or rewriting published history without confirmation.

---

## Git hygiene (this workspace)

- Prefer **short feature branches**; merge to `main` when asked.
- **Confirm before** `git push`, force-push, or amending published commits.
- Do not commit secrets, huge binaries, or `testing_ground` dumps.
- Upstream human docs: `CONTRIBUTING.md`, `docs/contributing/developing.md`.

---

## Read next (depth on demand)

| Doc | When |
|-----|------|
| `docs/module-structure.md` | Full package map |
| `docs/features/annotator-workflow.md` | Ethogram-first pipeline details |
| `docs/codebase-audit.md` | Known debt, size map, fix waves |
| `specifications.md` / `implementation-plan.md` | Workbench MVP (complete) baseline |
| `pyproject.toml` | Deps, scripts, uv torch sources |
| `pytest.ini` / `noxfile.py` | Test config and CI sessions |

---

## Agent skills

### Issue tracker

GitHub Issues on `brbarnhart/LabGym-Fork` (via `gh`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (`CONTEXT.md` + `docs/adr/` at repo root). See `docs/agents/domain.md`.
