GitHub: https://github.com/brbarnhart/LabGym-Fork/issues/2

## Parent

Part of #1

## What to build

Every Review IDs save publishes **remapped tracklets** from **raw tracklets** plus the current **switch-marker** list — including empty-switch accept (public remapped equals raw). There is one publish path. Rebuild is refused when raw is missing. Public remapped files are never used as the remap baseline, so a second save cannot double-apply switches. Any older analyzer resave uses this same path, not a “only if some remaps applied” fork.

## Acceptance criteria

- [ ] Saving Review IDs with an empty switch-marker list writes public remapped tracklets equal to raw and records **accepted identities**.
- [ ] Saving twice with the same switch markers does not double-apply (IDs are not swapped twice).
- [ ] Publish without raw refuses to rebuild remapped geometry; it does not load public files as the baseline.
- [ ] The analyzer/resave path calls the same publish behavior (no `n_applied > 0` exception that leaves remapped unpublished).
- [ ] Tests sit on the identity-package seam (disk + status), not the Qt tab.

## Blocked by

None — can start immediately.
