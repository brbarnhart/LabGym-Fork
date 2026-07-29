# Non-destructive dataset manifest and effective training set

Status: accepted

Curation (exclude, recategorize, category merge/exclude) must not destroy ethogram-generated examples or make multi-model history unrecoverable. We record exclusions, label overrides, soft overrides, split assignment, and taxonomy operations in a **dataset manifest** next to each **example store** root. Training and evaluation consume the **effective training set** (store + manifest) by default, with an optional ignore-manifest escape hatch. Physical folder moves/renames are not the primary curation mechanism.

## Considered options

- **Manifest + effective view (chosen)** — undoable, safe when several categorizers share one store, preserves original hard labels and soft label store.
- **Move/rename files on disk** — matches classic LabGym simplicity; weak undo; breaks paths in old evaluation runs.
- **Export a new curated folder on every apply** — clear snapshots; disk-heavy path sprawl.

## Consequences

- Train, prepare, and Evaluate must resolve the effective view (not only glob the raw store).
- Soft training uses original soft label store plus soft overrides and **soft projection** under taxonomy ops; do not silently rewrite `soft_labels.csv`.
- Review actions apply to the manifest immediately (with undo), not via a separate staging commit.
