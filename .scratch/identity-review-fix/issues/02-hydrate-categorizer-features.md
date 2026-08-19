GitHub: https://github.com/brbarnhart/LabGym-Fork/issues/3

## Parent

Part of #1

## What to build

Process videos hydrates from **remapped tracklets** and rebuilds **categorizer** features the same *kind* as the live path: a real rolling animation clip (last `length` blobs, zeros when that ID is absent), and body-part inners when the categorizer was trained with body parts. The remapped package is the kind set: an **empty kind** is valid; a **missing kind** is not invented; a remapped kind is never dropped so the run can “succeed.”

## Acceptance criteria

- [ ] Animations at an analysis frame are a temporal window of blobs, not the same still stacked `length` times (a moving outline produces differing slices).
- [ ] When the categorizer includes body parts, inners (and other-inners in interactive advanced) are recomputed from remapped outlines + video, as in the live path.
- [ ] When the categorizer does not include body parts, inners stay omitted.
- [ ] An empty kind (in the remapped package, no valid tracks) hydrates without error.
- [ ] A missing kind (asked for, not in the package) errors. A remapped kind that cannot be loaded into the analyzer errors. No silent skip of remapped data.
- [ ] Tests sit on the hydrate seam. Process videos still takes no detector and does not re-track.

## Blocked by

None — can start immediately.
