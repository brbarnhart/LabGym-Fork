**PROJECT NAME SUGGESTION**  
**Mouse Behavior Annotator for LabGym** (or `etho_annotator`, `labgym_prep`, `behavior_labeler`)

**GOAL**  
Build a desktop GUI tool that lets a researcher:
1. Manually annotate user-defined behaviors in mice videos with high temporal precision.
2. Automatically compute key metrics (latency to first occurrence, bout durations, etc.).
3. Save structured annotation data.
4. Generate curated short video clips (“behavioral examples” / animations) of a user-specified length in frames for each behavior.
5. Export data in formats useful for LabGym (especially frame-wise or bout-wise labels for its auto-sorting feature, plus ready-to-use animation clips).

The tool is **complementary** to LabGym: it provides precise human-curated ground truth that can be used either to generate high-quality training examples directly or to auto-sort the examples that LabGym itself generates.

---

### 1. Recommended Technology Stack

- **Language**: Python 3.11+
- **GUI**: **PyQt6** (best balance of power + video/timeline control). Use `QThread` + `QTimer` for smooth playback.
- **Video backend**: `opencv-python` (cv2) for core reading/seeking + `decord` (recommended for accurate random frame access on long videos) or fall back to cv2.
- **Data**: `dataclasses` + `pandas` (metrics & exports), `json` for annotations.
- **Video clip export**: `opencv-python` (VideoWriter) or `imageio[ffmpeg]`.
- **UI polish (optional but recommended)**: `qdarkstyle` or custom QSS for dark theme; `pyqtgraph` for advanced timeline (Phase 2).
- **Packaging (later)**: PyInstaller.

**Why PyQt6 over Tkinter?**  
You need precise frame seeking, interactive timeline with colored segments, keyboard-driven annotation, and non-blocking video playback. PyQt6 handles this cleanly.

---

### 2. High-Level Architecture

```
etho_annotator/
├── main.py
├── core/
│   ├── data_models.py          # Dataclasses: Behavior, Bout, AnnotationSession
│   ├── annotation_manager.py   # Load/save JSON, bout logic, validation
│   ├── video_handler.py        # Video loading, frame seeking, playback thread
│   ├── metrics_calculator.py
│   └── example_generator.py    # Cut short clips from bouts
├── ui/
│   ├── main_window.py
│   ├── video_display.py        # QLabel + overlay for current behaviors
│   ├── playback_controls.py
│   ├── behavior_palette.py     # List of behaviors + add/edit/delete + colors/hotkeys
│   ├── timeline_widget.py      # Custom QWidget (or pyqtgraph) showing colored bouts
│   ├── bout_list.py            # Editable table of bouts per behavior
│   ├── metrics_panel.py
│   └── export_dialog.py
├── utils/
│   └── helpers.py
├── pyproject.toml
├── uv.lock
└── README.md
```

---

### 3. Core Data Model (core/data_models.py)

Use Python `dataclasses`:

```python
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class Behavior:
    name: str
    color: str = "#FF5555"      # hex for timeline + overlay
    hotkey: str | None = None   # e.g. "1", "g", "r"

@dataclass
class Bout:
    start_frame: int
    end_frame: int

@dataclass
class AnnotationSession:
    video_path: str
    fps: float
    total_frames: int
    behaviors: List[Behavior] = field(default_factory=list)
    bouts: Dict[str, List[Bout]] = field(default_factory=dict)  # behavior_name -> list of Bout
    # metrics can be computed on demand or cached
```

**JSON on-disk format** (one file per video, e.g. `video_annotations.json`):

```json
{
  "video_path": "/path/to/video.mp4",
  "fps": 30.0,
  "total_frames": 12345,
  "behaviors": [
    {"name": "grooming", "color": "#00AAFF", "hotkey": "1"},
    {"name": "rearing", "color": "#FFAA00", "hotkey": "2"}
  ],
  "bouts": {
    "grooming": [
      {"start_frame": 120, "end_frame": 380},
      {"start_frame": 1450, "end_frame": 1620}
    ]
  }
}
```

---

### 4. Key Features & Implementation Order (Recommended Phases)

**Phase 1 – Minimum Viable Player + Annotation (do this first)**

- Load video → read `fps` and `total_frames` (use `decord` or `cv2`).
- Video display widget (`QLabel` updated from `QImage`).
- Playback controls: Play/Pause (Space), frame step forward/back, speed control (0.25×–4×), seek slider + exact frame spinbox.
- Sidebar: Behavior palette (add/rename/delete behaviors, pick color, assign hotkey).
- Annotation UX (most important):
  - Select behavior (click or press hotkey).
  - “Mark Start” / “Mark End” buttons **or better**: Toggle mode — press hotkey again to close the current bout for that behavior.
  - Visual overlay on video: current active behaviors (text + colored bar).
  - Bout list table (per behavior) — show start/end/duration, ability to delete or jump to bout.
- Basic timeline bar at bottom (colored segments). Clicking seeks.

**Phase 2 – Timeline + Polish**

- Build a proper interactive timeline widget (`QWidget` with `QPainter` or `pyqtgraph`).
  - One horizontal track per behavior.
  - Colored rectangles for bouts.
  - Zoom + pan support.
  - Click/drag to seek or adjust bout edges (advanced but very powerful).

**Phase 3 – Metrics & Saving**

Implement `metrics_calculator.py`:

For each behavior compute:
- Number of bouts
- Total duration (frames + seconds)
- Mean / median / min / max bout duration
- **Latency to first occurrence** (frames or seconds from video start; handle “never observed”)
- Bout frequency (bouts per minute of video)
- Optional: total time any behavior was active, etc.

Display in a nice table. Export to `.xlsx` (multiple sheets: Summary, All Bouts, Frame-wise labels).

**Phase 4 – Example Generation for LabGym**

In `example_generator.py` + UI dialog:

User settings:
- Output folder
- Desired example length in **frames** (e.g. 30, 45, 60)
- Sampling mode:
  - One representative clip per bout (centered on the bout if possible)
  - N random clips per behavior (within annotated bouts)
  - All non-overlapping clips of that length inside bouts

For each selected bout/clip:
- Extract exact frame range from original video using OpenCV / imageio.
- Save as MP4 (H.264) with clear filename, e.g. `grooming_bout001_f0120-0150.mp4`
- Create subfolders per behavior name.

**Bonus (highly recommended)**: Also export a `frame_labels.csv` that LabGym can use for automatic sorting of its own generated examples:

```csv
frame, grooming, rearing, sniffing, ...
120, 1, 0, 0, ...
121, 1, 0, 0, ...
...
```

(You can make it one-hot or use a single `label` column. Check LabGym’s sorting code or test with a small example.)

---

### 5. User Workflow (what the final app should feel like)

1. Open app → (optional) load a behavior template (common mouse behaviors).
2. Load video.
3. Review/adjust behavior list + colors/hotkeys.
4. Play video and annotate using hotkeys (very fast once muscle memory is built).
5. Scrub timeline or bout list to review/edit.
6. Save annotations (auto-save option nice).
7. Open “Metrics” panel → review numbers → export Excel report.
8. Open “Generate Examples for LabGym” → set clip length → generate folder of curated animation clips.
9. (Optional) Export `frame_labels.csv` → use inside LabGym to auto-sort examples it generates from the same video.

---

### 6. Important Technical Notes for the LLM

- **Frame-accurate seeking**: OpenCV’s `set(CAP_PROP_POS_FRAMES)` is not always reliable on compressed video. Strongly recommend starting with the `decord` library (very fast and accurate random access) and falling back to cv2.
- **Threading**: Video playback and especially clip export **must** run in `QThread` so the GUI never freezes.
- **Bout validation**: For the same behavior, new bouts should not overlap previous ones. Auto-close previous bout or warn user.
- **Keyboard-first design**: Power users will use this for hours. Make every common action have a hotkey.
- **LabGym compatibility**:
  - The animation clips you generate are “raw” (with background). This is still useful.
  - The real power for many users will be the `frame_labels.csv` for LabGym’s auto-sorting feature.
- **Future extensibility** (mention in README):
  - Multi-animal support (subject ID per bout)
  - Integration with pose estimation (SLEAP/DLC) for automatic suggestion of bouts
  - Background subtraction option when generating clips

---

### 7. Suggested Development Order for the Local AI

1. Create project structure + `pyproject.toml` (managed with uv)
2. Implement `VideoHandler` class (load + get frame by number)
3. Build basic PyQt6 main window + video display + playback controls (Phase 1)
4. Add behavior palette + simple annotation (Mark Start/End + bout list)
5. Add timeline bar + seeking
6. Add JSON save/load + metrics calculation
7. Add example clip generator
8. Add `frame_labels.csv` export + documentation for LabGym workflow
9. Polish (dark theme, keyboard shortcuts everywhere, better timeline, error handling, recent files)

---

### 8. Bonus / Nice-to-Have Features (Phase 5+)

- Load a folder of videos and batch-annotate (show progress across videos)
- Import existing annotations from BORIS or other tools (CSV/JSON)
- Template system for common mouse behavior sets (grooming, rearing, sniffing, digging, climbing, immobility, etc.)
- Option to generate simple pattern-image-like summaries (motion history images) alongside the animation clips
- Command-line interface (headless) for scripted use

---
