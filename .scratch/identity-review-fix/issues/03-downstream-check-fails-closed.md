GitHub: https://github.com/brbarnhart/LabGym-Fork/issues/4

## Parent

Part of #1

## What to build

Before Review IDs save, if an ethogram or **example store** already exists, the user is warned that those files may go stale and are not rewritten. If that lookup **fails**, the user is told the check failed and is still asked before save (default No). An error is never treated as “nothing downstream.” The existing re-detect confirm (accepted identities or switch markers, default No) stays as it is.

## Acceptance criteria

- [ ] When the ethogram/examples lookup raises, the caller receives a failed-check signal, not an empty all-clear.
- [ ] Save still asks for confirmation in that case (default No); the real error is logged or shown.
- [ ] No blanket exception handler that returns silence for this check.
- [ ] Re-detect overwrite confirm is unchanged: ask when accepted identities or switch markers exist; default No.
- [ ] Tests cover the check’s result object/signal, not dialog wording.

## Blocked by

None — can start immediately.
