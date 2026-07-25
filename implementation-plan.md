# Implementation Plan — PySide Workbench GUI

**Against:** [`specifications.md`](./specifications.md)  
**Status:** **complete for MVP + wx retirement** (Phases 1–8 and Results / dense Tools / detector frames)  
**Principle:** ethogram-first primary path; classic dense pipeline under Tools; pure PySide product UI.

---

## 0. Current baseline (post-migration)

| Area | State |
|------|--------|
| Shell + project | PySide workbench + `*.labproj.json` |
| Preprocess / markers | Workbench tabs |
| Detector | Generate training data (images + annotate placeholder), train/test, detect+track, review IDs |
| Categorizer | Ethogram annotate/generate, train/test, process videos |
| Results | Mine / plot / distances |
| Dense generate + sort | Tools menu pop-out |
| Entry | `LabGym` → workbench only (no wx / no `--legacy-wx`) |
| Dependency | PySide6; **wxPython removed** |

---

## 1. Target architecture (shipped)

```text
LabGym/gui_pyside/
  main_window.py
  project/          # model, controller, paths, editor
  shell/            # workbench bar + host
  workbenches/
    preprocessing/
    detector/       # generate images, annotate placeholder, train/test, detect, review
    categorizer/    # ethogram generate host, train/test, process
    results/        # mine, plot, distances
  tools_windows/    # dense generate + sort
  jobs/
```

**Entry points:**

| Command | Behavior |
|---------|----------|
| `LabGym` | PySide workbench shell |
| `LabGym-workflow` | Same shell (compat) |
| `LabGym-annotate` | Standalone ethogram annotator |

---

## 2. Phased plan — status

| Phase | Goal | Status |
|-------|------|--------|
| 0 Spec freeze | `specifications.md` | ✅ |
| 1 Shell + project | Workbench bar, `*.labproj.json` | ✅ |
| 2 Categorizer generate | Annotate ethogram + Generate examples | ✅ |
| 3 Review IDs + roles | Contact-aware remaps, subjects | ✅ |
| 4 Detect + track batch | Sequential queue, identity package | ✅ |
| 5 Preprocess + markers | Ported tabs | ✅ |
| 6 Train/test models | Detector + categorizer UIs | ✅ |
| 7 Process videos | Batch analysis adapter | ✅ |
| 8 Default entry | PySide default | ✅ |
| A Detector generate training data | Generate images + Annotate placeholder | ✅ |
| B Results | Mine / plot / distances | ✅ |
| C Dense Tools | Pop-out generate + sort | ✅ |
| D Qt registration/survey | No wx dialogs at startup | ✅ |
| E Remove wx | Delete `gui_*.py`, `mywx`, deps, flags | ✅ |

### Phase 8 (historical detail)

Originally introduced `--legacy-wx` as a temporary escape hatch. **Phase E removed** that flag, the Tools legacy launcher, and the wxPython dependency. Default entry is PySide only.

### Future (post-MVP)

- In-app detector annotation (EZannot-like), replacing the Annotate images placeholder  
- Richer Results (ethogram figures, R-ready tables)  
- Parallel multi-video batch jobs  
- Project role vocabulary UI  
- Analysis parameter presets UI (backend module exists)  

---

## 3. Definition of done

Without wx, user can:

1. Create a project (root + videos)  
2. Preprocess  
3. Batch detect+track  
4. Review IDs and assign names/roles  
5. Annotate ethogram and generate training pairs  
6. Train categorizer and process videos  
7. Mine results / plot / distances  
8. Optionally use Tools → Dense generate + sort  

---

## 4. Immediate next actions (optional polish)

- Push / PR `UI_migration` → `main`  
- Manual smoke on clean install without wxPython  
- Optional: wire analysis presets into Process videos  
