GitHub: https://github.com/brbarnhart/LabGym-Fork/issues/5

## Parent

Part of #1

## What to build

With no **raw tracklets**, no switch-list edits: mark swap, delete, remove-at-frame, undo, or any reorder. Names and roles may still change. Opening Review IDs on an **uncorrected** pack that has public tracklets but no raw **asks** before moving those files into raw (this may mean the data is in a bad state). Declining leaves the pack unchanged. **Accepted identities** (including old “corrected” packs) are never offered this migrate.

## Acceptance criteria

- [ ] Switch-edit policy is false without raw and true with raw (covers add, delete, undo — not only mark swap).
- [ ] Names and roles remain editable without raw.
- [ ] Migrating an uncorrected public pack into raw is an explicit operation the UI can confirm, not an implicit side effect of load.
- [ ] Declining migrate leaves files and status unchanged; switch edits stay locked.
- [ ] Accepted / legacy corrected packs are not offered migrate (remapped is never copied in as raw).
- [ ] Tests sit on the identity-package seam; the tab only shows the confirm.

## Blocked by

- #2 Publish remapped tracklets from raw only
