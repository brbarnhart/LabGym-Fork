# Ship Hungarian association; defer live occlusion freeze and split detection

Soak-testing on real videos showed that **Hungarian matching to predicted centers of mass** (with **parked slots**) reduced identity switches, while live **split detection** and **occlusion freeze** increased the number of manual **switch markers** needed per video. This release therefore ships only the live matcher from #12 (plus the identity-continuity prefactor and the Hard cases popup). Live freeze, split-detection hygiene, and **proposed switch markers** stay off `main` until a design beats Hungarian-only on labeled Review IDs saves.

ADR-0007 remains the longer-term identity-continuity design. It is not implemented in full in v1.0.1.

## Considered options

- **Merge the full feature branch (freeze + split on, optional Detect + track toggles)** — rejected; default-on freeze/split made Review IDs worse. Toggles are for lab comparison, not the product default.
- **Ship freeze/split with the checkboxes default off** — rejected for this push; `main` would still carry an unproven association path. Keep that code on `feat/identity-occlusion-followup`.
- **Revert parked slots as well** — rejected; parked slots shipped with the Hungarian matcher (#12) and are part of the improvement.

## Consequences

- New Detect + track on `main` always uses Hungarian assignment to constant-velocity predicted COMs and parks unmatched **identity slots**. No freeze, no split-detection steal rule, no association checkboxes.
- **Raw tracklets** / **accepted identities** / Review IDs save rules are unchanged (ADR-0006).
- Issue #13 (live split detection and occlusion freeze) and #14 (proposed switch markers) are not in this release. #14 stays blocked on an occlusion design that actually lowers switch labor.
- Follow-up work lives on `feat/identity-occlusion-followup` (does not merge until it beats Hungarian-only on switch-marker counts).
- Identity association only affects **new** Detect + track runs. Videos detected with freeze/split on need a re-detect to pick up Hungarian-only raw tracklets.
