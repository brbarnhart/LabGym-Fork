# Soft labels: store intact, overrides and projection for effective view

Status: accepted

The ethogram-derived **soft label store** (e.g. `soft_labels.csv`) is a recoverable source of truth, parallel to the example store for media and original hard labels. Effective soft vectors for training come from:

1. Original soft vectors (or a per-example **soft override** in the dataset manifest), then  
2. **Soft projection** into the active taxonomy after category merge/exclude (e.g. sum mass of merged classes onto the target; drop excluded mass and renormalize).

Recategorizing the hard label alone does not invent a new soft distribution. Regenerating soft labels from ethograms remains an advanced rebuild action, not the daily path after every taxonomy edit.

## Considered options

- **Overrides + deterministic projection (chosen)** — non-destructive; merge preserves ethogram ambiguity mass; undo re-reads originals.
- **Rewrite soft_labels.csv on every override/merge** — easy to train against; destroys undo and ethogram-derived detail unless the file is versioned heavily.
- **Disable soft modes whenever any manifest override exists** — simple; throws away soft training for curated sets.

## Consequences

- Effective-view code must project soft vectors whenever active class lists differ from the soft store’s classnames.
- If projection yields an empty or unusable vector, fall back toward hard-only for that example (or the run) with a clear warning—consistent with today’s “no soft file → hard_only” spirit.
- First Review UI can focus on hard keep/exclude/recategorize; soft-override editing can follow later.
