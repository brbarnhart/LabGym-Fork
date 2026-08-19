GitHub: https://github.com/brbarnhart/LabGym-Fork/issues/8

## Parent

Part of #1

## What to build

Make the branch pass a second Standards review: public hydrate functions have type hints and Google-style docstrings; leftover detector path/batch controls are gone from Process videos; frame progress reports during hydrate rebuild (not only at the end); leftover “accepted is not None” checks and duplicated publish/load/unpublish loops are removed; no new blanket exception swallows on user-visible paths.

## Acceptance criteria

- [ ] Public hydrate functions are typed and documented (Google-style).
- [ ] Process videos has no leftover detector path / detector batch UI or unused wiring.
- [ ] Hydrate/process reports frame progress during rebuild, not only a 100% tick at the end.
- [ ] No leftover `accepted is not None` style checks; no duplicate publish path after ticket 1.
- [ ] No new blanket `except Exception` on user-visible hydrate or gate paths.
- [ ] Existing unit/smoke tests that referenced removed detector fields are updated and pass.

## Blocked by

- #3 Hydrate categorizer features correctly from remapped tracklets
- #7 One downstream gate, including Annotate
