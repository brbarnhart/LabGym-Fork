GitHub: https://github.com/brbarnhart/LabGym-Fork/issues/7

## Parent

Part of #1

## What to build

Annotate ethogram, generate examples, and Process videos share one gate: **accepted identities** are required for videos with per-animal tracks. Annotate **does not load the video** when the gate fails. Interactive basic is exempt (no identity package, no Review IDs step).

## Acceptance criteria

- [ ] One predicate answers “may this video be annotated, used for examples, or processed?”
- [ ] Without accepted identities, Annotate refuses and does not open the video (per-animal modes).
- [ ] Generate examples and Process videos still refuse without accepted identities (per-animal modes).
- [ ] Interactive basic is allowed through the gate without accepted identities.
- [ ] Tests cover the predicate and the annotate refuse path (no full Qt playback).

## Blocked by

- #2 Publish remapped tracklets from raw only
- #6 Detect always writes raw (modes 0 and 2)
