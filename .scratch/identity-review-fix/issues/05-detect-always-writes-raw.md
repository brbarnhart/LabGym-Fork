GitHub: https://github.com/brbarnhart/LabGym-Fork/issues/6

## Parent

Part of #1

## What to build

Detect + track **always** writes **raw tracklets** (the identity package) for behavior modes that produce per-animal tracks (non-interactive and interactive advanced). Export is not optional — the checkbox is removed or ignored. Interactive basic writes no identity package (no per-animal IDs to review).

## Acceptance criteria

- [ ] A detect run in mode 0 or 2 writes raw even if a caller would have passed “do not export.”
- [ ] The Detect + track UI no longer offers skipping the identity package for those modes.
- [ ] A detect run in interactive basic (mode 1) does not write an identity package.
- [ ] Re-running detect for modes 0/2 is still a new tracking world (replace raw, unpublish remapped, clear switches) after the existing confirm.
- [ ] Tests sit on the detect-export / identity-package write, not full GPU acquire.

## Blocked by

- #2 Publish remapped tracklets from raw only
