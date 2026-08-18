# Raw tracklets, accepted identities, and categorize-without-redetect

Identity corrections must stay editable after the first Review IDs save, and later analysis must use that reviewed identity layer rather than running the detector again. We keep **raw tracklets** immutable, treat the **switch-marker** list as the only editable source of truth, and publish **remapped tracklets** only when Review IDs is saved (**accepted identities**). Annotate ethogram, generate examples, and Process videos all require that save — including refusing to open the video in the annotator until then. Process videos hydrates from remapped tracklets and rebuilds categorizer features from those outlines; it does not take a detector or re-track.

## Considered options

- **Two mapping files (original vs preferred updated)** — rejected; two switch lists plus remapped geometry, unclear ID space.
- **Overwrite the only tracklet file on first save and freeze switches** — current code; blocks fixing mistakes; no raw to rebuild from.
- **Process videos re-tracks, then replays switches** — rejected; new tracks are a new identity world, and it skips the Review IDs nudge.
- **Silent re-detect when remapped tracklets are missing** — rejected; IDs and experimental names would never be forced through review.
- **Phased ship (editable Review IDs first, old Process videos until later)** — rejected; one effort so the workbench never ships a tab that violates the rule.

## Consequences

- Detect + track always writes raw for modes that produce per-animal tracks (non-interactive and interactive advanced). That export is not optional. Interactive basic has no per-animal tracks: no identity package, and Annotate / Generate / Process are exempt from the accepted-identities gate.
- Opening Review IDs on an uncorrected pack with no raw asks before moving public tracklets into raw. Silent migrate is rejected: that situation may mean the data is in a bad state, and the user should see it.
- Re-running Detect + track replaces raw, unpublishes remapped tracklets, and clears switches (new tracking world). If any selected video has accepted identities or switch markers, Detect + track asks first (default No).
- Review IDs previews from raw + the current switch list; save rebuilds remapped. No real raw snapshot (legacy baked packs) → no switch-list edits at all (add, delete, undo). Names and roles may still change. Non-tail marker edits warn only.
- Changing remapped tracklets after an ethogram or examples exist warns and does not rewrite those files. If that check itself fails, say so and still ask before save — do not treat an error as “nothing downstream.”
- Public `*_tracklets.npz` is remapped; raw is a sibling artifact that kind-discovery must not treat as another animal kind.
- One publish path: rebuild only from raw (refuse if there is no raw). Every Review IDs save writes public remapped, including empty-switch accept (public equals raw). Never remap starting from the public files.
- Old unsaved detect packs are not accepted identities until Review IDs is saved.
- Process videos uses the remapped package’s kind set. An **empty kind** (reviewed; nobody present) is valid. A **missing kind** (not in the package) is not invented from the detector class list. Silently dropping a remapped kind is an error.
