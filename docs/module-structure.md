# LabGym Module Structure and Responsibilities

This document describes the repository layout and the responsibilities of each major package, module, and supporting subsystem in LabGym (version 3.x). It is intended for developers, maintainers, and contributors who need a map of the codebase—not as an end-user guide.

LabGym is a desktop application (**PySide6 workbench GUI**, CLI entry points) for detecting animals/objects in videos or images, recognizing user-defined behaviors, and quantifying those behaviors with kinematic and statistical outputs.

---

## 1. High-level architecture

LabGym is organized around FreeCAD-style **workbenches** (Preprocess, Detector, Categorizer, Results), backed by core analysis engines, shared computer-vision utilities, a vendored Detectron2 stack, and infrastructure for configuration, logging, and packaging.

```text
                        ┌─────────────────────────────────┐
                        │  Entry: LabGym.__main__:main    │
                        │  (CLI script "LabGym")          │
                        └───────────────┬─────────────────┘
                                        │
              config / logging / probes / registration / selftest
                                        │
                        ┌───────────────▼─────────────────┐
                        │  gui_pyside WorkbenchMainWindow │
                        │  workbench bar + project shell  │
                        └───┬──────────┬──────────┬───────┘
                            │          │          │
              Preprocessing │ Detector │ Categorizer │ Results
                            │          │          │
              workbenches/* tabs  tools_windows (dense gen+sort)
                            │          │          │
                            ▼          ▼          ▼
                         tools.py  detector.py  analyzebehavior*.py
                                   categorizer.py  minedata.py
                                   id_review / annotator / training
                            │
                    OpenCV / TensorFlow-Keras / PyTorch
                    LabGym.detectron2 (instance segmentation)
```

**User workflow (workbenches):**

| Workbench | Purpose |
|-----------|---------|
| **Preprocess** | Prepare videos (trim, crop, contrast, FPS, markers). |
| **Detector** | Frame extraction, train/test detectors, batch detect+track, review IDs. |
| **Categorizer** | Ethogram-first training data, train/test categorizer, process videos. |
| **Results** | Mine summary stats, behavior plots, distance calculations. |

**Detection strategies:**

1. **Background subtraction** — Fast; needs stable lighting and a static background (`AnalyzeAnimal` + `tools.extract_background`).
2. **Trained Detector (Detectron2)** — More robust under variable conditions and multi-individual interaction (`AnalyzeAnimalDetector` + `Detector`).

**Behavior modes (Categorizer / analysis):**

| Mode | Code value | Responsibility |
|------|------------|----------------|
| Non-interactive | `0` | Solitary behaviors of individuals. |
| Interactive basic | `1` | Whole interacting group as one entity (faster). |
| Interactive advanced | `2` | Per-individual roles within multi-subject interactions. |
| Static image | `3` | Behaviors in still images (non-interactive). |

---

## 2. Repository top-level layout

| Path | Responsibility |
|------|----------------|
| `LabGym/` | Installable Python package: application code, assets, vendored Detectron2. |
| `tests/` | Unit tests, lint config, integration placeholders, developer notes. |
| `docs/` | Sphinx/MyST documentation (installation, contributing, features, walkthroughs). |
| `Examples/` | Demo GIFs and images used in README and marketing materials. |
| `pyproject.toml` | Package metadata, dependencies, console script entry point, build config (PDM). |
| `noxfile.py` | Nox automation sessions (dev tooling). |
| `pytest.ini` | Pytest configuration. |
| `README.md` | Product overview, high-level usage, citation. |
| `LabGym_Zoo.md` | Catalog of shared trained models and training examples. |
| `LabGym_extended_user_guide.pdf` / `LabGym_practical_guide.pdf` | End-user guides. |
| `LICENSE.txt` / `NOTICE.txt` / `COPYRIGHT.txt` | Licensing and attribution. |
| `CONTRIBUTING.md` / `CODE_OF_CONDUCT.md` | Contribution process and community norms. |
| `.github/` | CI workflows (e.g. `ci.yml`, `python-publish.yml`) and helper scripts. |

---

## 3. Package entry and identity

### `LabGym/__init__.py`

- Defines package version (`__version__`).
- Minimal surface: version is also the PDM dynamic version source.

### `LabGym/__main__.py`

- Console entry point (`LabGym = LabGym.__main__:main` in `pyproject.toml`).
- Bootstraps deferred logging and configures logging.
- Optional **selftest** path: if configured, runs tests and exits.
- Checks PyPI for newer versions and prints upgrade advice.
- Launches the **PySide6 workbench shell** (`gui_pyside`).

Aliases: `LabGym-workflow` (same shell), `LabGym-annotate` (standalone ethogram annotator).

---

## 4. User interface layer (`gui_pyside`)

The default GUI is a FreeCAD-style shell: top **workbench bar**, per-workbench **tabs**, shared **project** (`*.labproj.json`).

### Shell

| Module / class | Responsibility |
|----------------|----------------|
| `gui_pyside/main_window.py` → `WorkbenchMainWindow` | Menus, workbench bar, stack of workbenches, project status. |
| `gui_pyside/shell/workbench_bar.py` | Exclusive workbench switcher. |
| `gui_pyside/shell/workbench_host.py` | Hosts active workbench widget. |
| `gui_pyside/project/*` | Project model, controller, path resolution, editor dialog. |
| `gui_pyside/jobs/*` | Sequential job queue for long-running batch work. |

### Workbenches

| Package | Tabs (high level) |
|---------|-------------------|
| `workbenches/preprocessing/` | Preprocess videos; Draw markers |
| `workbenches/detector/` | Generate training data (images + annotate placeholder); Train/Test; Detect + track; Review IDs |
| `workbenches/categorizer/` | Generate training data (ethogram + examples); Train/Test; Process videos |
| `workbenches/results/` | Mine results; Behavior plot; Calculate distances |

### Secondary tools

| Module | Responsibility |
|--------|----------------|
| `gui_pyside/tools_windows/` | Tools → Dense generate + sort (classic unsorted examples + sort) |

### Other UI

| Module | Responsibility |
|--------|----------------|
| `annotator/` | Multi-subject ethogram annotation (embeddable + standalone). |
| `assets/icons/` | Icon resources (`.ico`, `.icns`, `.png`). |

---

## 5. Core analysis and ML engines

These modules implement the scientific pipeline invoked by the workbench (and usable programmatically).

### `detector.py` — Object/animal Detector (Detectron2)

Class `Detector`:

| Method | Responsibility |
|--------|----------------|
| `train` | Register COCO instance-segmentation data; train Mask R-CNN (R50-FPN) via Detectron2. |
| `test` | Run COCO evaluation on a test set. |
| `load` | Load a trained detector and animal-kind mapping for inference. |
| `inference` | Run detection on input images/frames. |

Uses CUDA when available; otherwise CPU.

### `categorizer.py` — Behavior Categorizer (TensorFlow/Keras)

| Component | Responsibility |
|-----------|----------------|
| `DatasetFromPath_AA` | Keras `Sequence` for animation (+ pattern) training batches. |
| `DatasetFromPath` | Keras `Sequence` for pattern-image-only training. |
| `Categorizers` | Full categorizer lifecycle: prepare labels, augment data, build networks, train, test. |

**`Categorizers` capabilities:**

- **Data prep:** `rename_label`, `build_data` (augmentations, resize, behavior modes).
- **Architectures:** simple VGG/TVGG, ResNet-style blocks (`simple_resnet` / `simple_tresnet`), combined dual-stream network (`combined_network`).
- **Training modes:**
  - Pattern Recognizer only (static spatial pattern of motion/outline).
  - Animation Analyzer only (spatiotemporal clip).
  - Combined network (both streams).
  - “On-the-fly” variants that load batches from disk during training.
- **Evaluation:** `test_categorizer` against ground-truth sorted examples.

### `analyzebehavior.py` — Analysis via background subtraction

Class `AnalyzeAnimal`:

| Stage | Methods | Responsibility |
|-------|---------|----------------|
| Setup | `prepare_analysis` | Video metadata, backgrounds, timing windows, categorizer dims. |
| Track | `track_animal` | Associate contours/centers across frames. |
| Features | `acquire_information`, `acquire_information_interact_basic`, `craft_data` | Build animations and pattern images per track. |
| Classify | `categorize_behaviors` | Run trained categorizer; apply uncertainty / min-length filters. |
| Output | `annotate_video`, `analyze_parameters`, `export_results` | Annotated video, kinematics (count, duration, latency, speed, etc.), spreadsheets. |
| Training data | `generate_data`, `generate_data_interact_basic` | Emit unlabeled behavior examples for later sorting/training. |

### `analyzebehavior_dt.py` — Analysis via trained Detector

Class `AnalyzeAnimalDetector` — parallel pipeline for Detectron2 detection, with extra support for interactive and multi-kind scenarios:

| Additional capability | Responsibility |
|----------------------|----------------|
| `detect_track_individuals` / `detect_track_interact` | Detect + track individuals or interacting groups. |
| `track_animal_interact` | Interactive-mode tracking with co-occurring contours. |
| `generate_data_interact_advance` | Examples for interactive-advanced (main character + costars). |
| `correct_identity` | Re-label identities based on specific behavior cues. |
| `analyze_images_individuals` | Static-image analysis path. |

Also provides the same high-level stages as `AnalyzeAnimal`: prepare → acquire → craft → categorize → annotate → parameterize → export.

### `minedata.py` — Statistical data mining

Class `data_mining`:

- Loads analysis result tables (optionally with a control group).
- Tests normality; chooses appropriate two-group or multi-group tests (SciPy / scikit-posthocs).
- Writes significant findings to `data_mining_results.xlsx`.

### `tools.py` — Shared computer-vision and I/O utilities

Cross-cutting helpers used by preprocessing, training-example generation, and analysis:

| Function group | Examples | Responsibility |
|----------------|----------|----------------|
| Background | `extract_background`, `estimate_constants` | Static background estimation; animal size/intensity constants. |
| Contours / blobs | `contour_frame`, `crop_frame`, `extract_blob_*`, `get_inner` | Segment animals from frames; extract ROIs and inner textures. |
| Pattern images | `generate_patternimage`, `generate_patternimage_all`, `generate_patternimage_interact` | Build motion/outline pattern images for the Pattern Recognizer. |
| Preprocess | `preprocess_video`, `extract_frames` | Video trim/crop/contrast/FPS; frame dumps for annotation. |
| Visualization | `plot_events` | Temporal event rasters. |
| Post-analysis | `parse_all_events_file`, `calculate_distances`, `sort_examples_from_csv` | Parse event logs; distances; bulk example sorting. |

---

## 6. Configuration, logging, and operations

| Module | Responsibility |
|--------|----------------|
| `config.py` | Merge defaults + config file + environment + CLI args; cache full config; expand paths (`detectors`, `models`, logging, feature flags, etc.). |
| `myargparse.py` | Custom CLI argument parsing into a nested dict (feeds config). |
| `mylogging.py` | Deferred log capture, then configure handlers from YAML; non-fatal if logging fails. |
| `logging.yaml` | Default logging configuration resource. |
| `central_logging.py` | Optional remote/HTTP logging (e.g. registration telemetry); queue listener cleanup. |
| `registration.py` | Optional first-run user registration form; store and optionally forward registration info. |
| `probes.py` | Pre-start sanity checks (certs, network, userdata layout, feature flags). |
| `userdata_survey.py` | Detect legacy models/detectors stored inside the package tree; guide users to external data dirs. |
| `pkghash/` | Version string augmented with content hash for reproducible support reporting. |
| `selftest/` | Run pytest-based selftests when LabGym is started with the selftest configuration. |
| `mypkg_resources.py` | Compatibility shim replacing deprecated `pkg_resources` patterns. |

---

## 7. Subpackages and data directories

### `LabGym/detectron2/`

Vendored [Detectron2](https://github.com/facebookresearch/detectron2)-style library used for instance segmentation Detectors. Major subpackages:

| Subpackage | Responsibility |
|------------|----------------|
| `config/` | Config system (`CfgNode`), defaults, lazy/instantiate helpers. |
| `data/` | Dataset catalog, COCO/LVIS/etc. loaders, mappers, samplers, transforms. |
| `modeling/` | Backbones, meta-architectures (R-CNN, RetinaNet, …), ROI heads, RPN. |
| `engine/` | Training loop, default trainer, hooks, launch helpers. |
| `evaluation/` | COCO and other evaluators. |
| `checkpoint/` | Model checkpoint loading. |
| `structures/` | Boxes, masks, instances, keypoints. |
| `layers/` | NMS, ROI align, batch norm wrappers, etc. |
| `solver/` | Optimizers and LR schedules. |
| `model_zoo/` | Predefined YAML/Python model configs and weight URLs. |
| `utils/` | Logging, visualization, distributed helpers. |
| `export/` | Caffe2/TorchScript export utilities. |
| `tracking/` | Multi-object tracking helpers (IoU/Hungarian variants). |
| `projects/` | Optional research projects (DeepLab, PointRend, etc.). |

LabGym’s `detector.py` is the primary application-level consumer; most of this tree is infrastructure.

### `LabGym/detectors/` and `LabGym/models/`

- Placeholder package directories for **Detectors** and **Categorizers**.
- Runtime paths are normally configured to **user-writable external folders** (see `config` / `userdata_survey`), not the package install tree.

### `LabGym/assets/`

Static GUI assets (application icons).

---

## 8. Tests and quality tooling

```text
tests/
├── unit/           # Module-focused unit tests (config, logging, argparse, registration, …)
├── integration/    # Reserved for integration tests
├── linting/        # Pylint configuration and helper scripts
├── notes/          # Developer notes (import time, pytest, OpenCV, etc.)
└── conftest.py     # Shared pytest fixtures
```

| Item | Responsibility |
|------|----------------|
| `tests/unit/test_*.py` | Automated coverage of infrastructure and small pure modules. |
| `tests/linting/` | Style/lint conventions for contributors. |
| `.github/workflows/ci.yml` | Continuous integration. |
| `.github/workflows/python-publish.yml` | Package publish pipeline. |

Heavy CV/ML modules (`analyzebehavior*`, `categorizer`, `detector`) are primarily exercised through the GUI workflow and selftests rather than extensive unit tests in-tree.

---

## 9. Documentation project (`docs/`)

Sphinx site (MyST Markdown) published to Read the Docs:

| Area | Responsibility |
|------|----------------|
| `installation/` | Platform install guides (Windows, macOS, Linux). |
| `features/` | Feature docs for preprocessing, training, analysis (stubs may still say “Coming soon”). |
| `walkthroughs/` | Task-oriented guides (background subtraction, detector, static images). |
| `contributing/` | Dev setup, docs authoring, internal notes. |
| `what-can-labgym-do.md` | Capability overview. |
| `changelog.md` / `issues.md` | Release history and issue reporting. |

---

## 10. Dependency roles (runtime)

| Stack | Used for |
|-------|----------|
| **PySide6** | Desktop workbench GUI (and Qt registration/survey dialogs). |
| **OpenCV** (`opencv-python`, contrib) | Video I/O, contours, preprocessing, annotation drawing. |
| **TensorFlow / Keras** | Categorizer neural networks (Animation Analyzer, Pattern Recognizer). |
| **PyTorch + Detectron2 (vendored)** | Detector training and inference. |
| **NumPy / SciPy / scikit-image / scikit-learn** | Arrays, stats, image ops. |
| **pandas / openpyxl / xlsxwriter** | Spreadsheet export and data mining I/O. |
| **matplotlib / seaborn** | Plots and event rasters. |
| **PyYAML / tomli / packaging / requests** | Config, version checks, HTTP. |

Python requirement (from packaging metadata): **3.9–3.10**.

---

## 11. Typical call paths

### A. Preprocess a video

```text
gui_pyside Preprocess tab
  → tools.preprocess_video
```

### B. Train a Detector

```text
Detector → Generate images → tools.extract_frames
  → (external annotation: EZannot / Roboflow COCO JSON)
Detector → Train detector → detector.Detector.train
```

### C. Train a Categorizer (ethogram-first)

```text
Categorizer → Annotate ethogram → annotator / annotations JSON
Categorizer → Generate examples → training.ethogram_examples
Categorizer → Train → categorizer.Categorizers.train_*
```

Dense (Tools) alternative:

```text
Tools → Dense generate → analyzebehavior*.generate_data*
Tools → Sort (manual/CSV/annotations)
→ Train categorizer on sorted folders
```

### D. Analyze behaviors (background subtraction)

```text
analysis backends (AnalyzeAnimal) still support background subtraction;
workbench Process videos currently emphasizes detector + categorizer path.
  → AnalyzeAnimal.prepare_analysis
  → acquire_information* / craft_data
  → categorize_behaviors (Keras model)
  → annotate_video / analyze_parameters / export_results
```

### E. Analyze behaviors (Detector)

```text
gui_pyside Process videos / analysis.process_videos
  → AnalyzeAnimalDetector.prepare_analysis
  → Detector.load + detect_track_*
  → craft_data → categorize_behaviors → export
```

### F. Mine results

```text
gui_pyside Results → Mine results
  → minedata.data_mining.statistical_analysis
```

---

## 12. Design notes for contributors

1. **GUI vs. engine separation** — Prefer keeping algorithm changes in `tools`, `analyzebehavior*`, `categorizer`, and `detector`. Workbench tabs should orchestrate parameters and paths, not reimplement CV logic.
2. **Two analysis pipelines** — Background subtraction and Detector paths are deliberately parallel. Shared behavior (export schemas, pattern-image conventions) should stay consistent when one side is changed.
3. **User data outside the package** — Models and detectors belong in configured external directories; `userdata_survey` exists to prevent packaging user data into installs.
4. **Detectron2 is vendored** — Prefer extending `detector.py` and configs used by LabGym rather than forking deep Detectron2 internals unless necessary for upstream sync.
5. **Logging is best-effort** — Infrastructure modules treat logging configuration failures as non-fatal so analysis can continue.
6. **Behavior modes are a cross-cutting contract** — Mode integers (`0–3`) appear in categorizer training, example generation, and analysis UIs; changing meaning requires coordinated updates.

---

## 13. Quick file index (application code)

| File / package | One-line role |
|----------------|---------------|
| `__main__.py` | Process entry, upgrade check, workbench launch. |
| `__init__.py` | Package version. |
| `gui_pyside/` | PySide6 workbench shell, project, workbenches, tools windows. |
| `annotator/` | Ethogram annotation UI + export helpers. |
| `tools.py` | Shared CV, preprocess, pattern images, plots. |
| `detector.py` | Detectron2 Detector train/load/infer. |
| `categorizer.py` | Keras Categorizer train/test architectures. |
| `analyzebehavior.py` | Background-subtraction analysis pipeline. |
| `analyzebehavior_dt.py` | Detector-based analysis pipeline. |
| `analysis/process_videos.py` | Batch process adapter used by Process videos tab. |
| `minedata.py` | Statistical mining of result tables. |
| `id_review/` | Tracklets, contact risk, remaps. |
| `training/` | Ethogram example gen, soft labels, aug export, sort helpers. |
| `config.py` | Hierarchical configuration. |
| `myargparse.py` | CLI parsing. |
| `mylogging.py` | Logging bootstrap. |
| `central_logging.py` | Optional remote logging. |
| `registration.py` | Optional user registration (Qt dialog). |
| `probes.py` | Startup health checks. |
| `userdata_survey.py` | Userdata placement warnings (Qt dialogs). |
| `mypkg_resources.py` | Resource loading compatibility. |
| `pkghash/` | Version + hash reporting. |
| `selftest/` | On-demand pytest selftest runner. |
| `detectron2/` | Vendored detection framework. |

---

*Generated from the LabGym repository structure and module responsibilities. For end-user procedures, see the README and the Extended / Practical PDF guides.*
