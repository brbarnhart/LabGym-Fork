# Mouse Behavior Annotator for LabGym

**Working title:** etho_annotator / behavior_video_annotator

> **Note:** Active development has moved into the LabGym package as
> `LabGym.annotator` (PySide6, multi-subject schema v2, tracklet overlays).
> Prefer:
>
> ```bash
> cd ../LabGym
> uv run python -m LabGym.annotator
> # or: LabGym-annotate
> ```
>
> See `../LabGym/docs/features/annotator-workflow.md`. This folder remains as a
> reference / historical standalone PyQt6 app.

A desktop GUI tool for precise manual annotation of user-defined behaviors in video recordings of mice (or other animals). Built to complement [LabGym](https://github.com/umyelab/LabGym).

## Goals
- High-temporal-precision manual annotation with keyboard-driven workflow.
- Automatic computation of standard metrics (latency, bout counts/durations, frequency).
- Structured JSON save/load.
- Export curated short video clips ("behavioral examples" / animations).
- Export LabGym-compatible artifacts (`frame_labels.csv` for auto-sorting + ready-to-use clips).

See `project_plan.md` for the full detailed specification and development phases.

## Prerequisites
- [uv](https://docs.astral.sh/uv/) (Python package & environment manager)
- Python 3.11+ (uv will install a suitable version if needed)

Install uv if you don't have it:

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

1. Create the virtual environment and install dependencies (from the project root):

   ```powershell
   uv sync
   ```

   This creates `.venv/`, installs runtime deps from `pyproject.toml`, and pins versions in `uv.lock`.

   Include optional groups as needed:

   ```powershell
   # Dev tools (pytest)
   uv sync --group dev

   # Dark theme / advanced timeline extras
   uv sync --group polish

   # Everything
   uv sync --all-groups
   ```

2. Run the app:

   ```powershell
   uv run python main.py
   ```

   Or activate the environment and run normally:

   ```powershell
   # Windows
   .\.venv\Scripts\Activate.ps1
   python main.py
   ```

   ```bash
   # macOS / Linux
   source .venv/bin/activate
   python main.py
   ```

### Note on decord (Windows)
`decord` is excluded on Windows in `pyproject.toml` because wheels are often unavailable. The app falls back to OpenCV. On Linux/macOS, `decord` is installed automatically for better random frame access.

### Common uv commands

| Command | Purpose |
|---------|---------|
| `uv sync` | Create/update `.venv` from lockfile |
| `uv lock` | Refresh `uv.lock` after editing deps |
| `uv add <pkg>` | Add a runtime dependency |
| `uv add --group dev <pkg>` | Add a dev dependency |
| `uv remove <pkg>` | Remove a dependency |
| `uv run <cmd>` | Run a command in the project environment |
| `uv run pytest` | Run tests (after `uv sync --group dev`) |

## Current Status
This project is under active development following the approved plan (see session plan.md).

**Implemented (strong Phase 1 MVP + LabGym export support):**
- Project scaffolding, requirements, structure
- VideoHandler (opencv primary on Windows + decord path)
- Full playback: play/pause (Space), seek slider/spinbox/arrows, variable speed
- Behavior palette (add/rename/delete/color/hotkey)
- Annotation toggle via hotkeys (e.g. '1','2') or double-click in palette
- Live colored behavior overlay on video + active state
- Basic multi-row timeline (QPainter) with click-to-seek
- Save / load JSON annotations (exact schema from project_plan)
- Basic metrics calculation + dialog + .xlsx export
- **Generate Examples for LabGym** (Tools menu):
  - Export short curated MP4 clips (centered / random / all non-overlapping)
  - Always writes `frame_labels.csv` (frame-wise one-hot labels) for LabGym auto-sorting
- Keyboard-first design started

Open a video → annotate with hotkeys → save JSON → Tools → Generate Examples or Metrics. Ready for real use.

## Key Features (Target)
- Frame-accurate video seeking and playback (decord / cv2).
- Toggle annotation via hotkeys (e.g. press "1" to start/stop a "grooming" bout at current frame).
- Live colored overlay of active behaviors.
- Interactive timeline + bout list.
- Per-behavior metrics + Excel export.
- Curated clip export + `frame_labels.csv`.

## LabGym Integration
This tool generates human-curated ground truth that can be used two ways with LabGym:
1. Directly use the exported short MP4 clips as high-quality training examples.
2. Export `frame_labels.csv` alongside a video so that LabGym can automatically sort the (unsorted) behavior examples that LabGym itself generates from that video.

See the export dialog and README section on frame-wise labels for details.

## Development
Dependencies and Python environment are managed with **uv** (`pyproject.toml` + `uv.lock`).

Follow the ordered steps in the implementation plan.

## License
To be decided (likely MIT or similar for research tools).
