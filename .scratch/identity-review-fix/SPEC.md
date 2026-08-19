## Problem Statement

I can save identity corrections in Detector → Review IDs, and I want later work (annotate ethogram, generate examples, Process videos) to use those **accepted identities**. The first implementation of editable review is on the branch, but a code review showed it can still produce the wrong science or the wrong nudge:

- Process videos can rebuild **categorizer** inputs that are not the same *kind* of features the categorizer saw in training (frozen animations; pattern images without body-part inners).
- It can silently drop a remapped animal kind, or invent a kind the package never accepted.
- Two different save paths disagree about when **remapped tracklets** are published. One of them can apply **switch markers** on top of already-remapped public files (double-apply).
- Annotate ethogram still opens the video after a warning, so I can draw an ethogram without accepted identities. Generate examples and Process videos already refuse.
- Legacy packs with no **raw tracklets** still allow delete/undo of switch markers even though remapped geometry cannot be rebuilt.
- Detect + track can skip writing the identity package (checkbox, or interactive basic).
- An old unsaved detect pack is migrated into raw silently — that may mean the data is in a bad state and I want to be told.
- The warning that an existing ethogram or **example store** may go stale can vanish if the check throws.

I need these gaps closed so the workbench matches ADR 0006 and the glossary: raw stays immutable, switch markers are the source of truth only when we can rebuild, public remapped is what analysis consumes, and I am told when something destructive or uncertain is about to happen.

## Solution

Close the review gaps behind the three existing seams (no new packages):

1. **Identity package** — one publish path; raw-only rebuild; every Review IDs save publishes remapped (including empty-switch accept); no switch-list edits without raw; confirm before snapshotting an uncorrected pack into raw; Detect always writes raw for non-interactive and interactive advanced; interactive basic has no package.
2. **Hydrate from remapped tracklets** — Process videos rebuilds a real rolling animation clip and body-part inners when the categorizer needs them; the remapped package is the kind set (**empty kind** valid, **missing kind** not invented, never drop remapped data silently).
3. **Downstream gate** — Annotate, generate examples, and Process videos share one answer: accepted identities required, except interactive basic. Annotate does not load the video on refuse.

Destructive or uncertain steps still ask (re-detect; migrate to raw; stale ethogram/examples; failed downstream check). Standards cleanup from the same review (typed hydrate APIs, leftover detector fields, progress, no blanket swallows) rides along so the branch is shippable.

## User Stories

1. As an experimenter, I want Process videos to score animals using the identities I accepted in Review IDs, so that I do not re-run the detector and scramble IDs.
2. As an experimenter, I want those scores to use the same kind of animation clip the categorizer was trained on (a short rolling window, not a frozen still), so that motion-using models are not fed paused animals.
3. As an experimenter, I want pattern images at Process videos time to include body-part inners when my categorizer was trained that way, so that analysis matches training.
4. As an experimenter with a categorizer trained without body parts, I want pattern images built without inners, so that I do not invent structure the model never saw.
5. As an experimenter whose video has two detector kinds but only kind A present, I want Process videos to run (empty kind B is valid), so that assays that sometimes lack a kind still analyze.
6. As an experimenter, I want Process videos to fail if it would throw away a remapped kind that is in the package, so that I never get “success” with a reviewed animal omitted.
7. As an experimenter, I want Process videos to fail if I asked to analyze a missing kind that the remapped package does not contain, so that the detector class list cannot invent animals this video never accepted.
8. As an experimenter, I want every Review IDs save to publish remapped tracklets, even if I made no swaps, so that “I looked and the detector IDs are fine” is enough to unlock later steps.
9. As an experimenter, I want that publish to always start from raw tracklets, so that saving twice cannot double-apply switch markers.
10. As an experimenter with a legacy pack that has no raw, I want Save to refuse to rebuild remapped, so that baked geometry is not silently rewritten from the wrong baseline.
11. As an experimenter with no raw, I want Mark swap, delete, remove-at-frame, and undo all blocked, so that the switch list cannot diverge from remapped geometry I cannot rebuild.
12. As an experimenter with no raw, I still want to edit display names and roles and save subjects, so that experimental labels are not locked to the legacy geometry problem.
13. As an experimenter, I want Annotate ethogram to refuse to open a video until I have accepted identities (except interactive basic), so that I cannot draw an ethogram that does not know who is who.
14. As an experimenter, I want generate examples to keep refusing without accepted identities, so that training clips are not cut from unreviewed tracks.
15. As an experimenter, I want Process videos to keep refusing without accepted identities, so that I cannot score a video I never reviewed.
16. As an experimenter using interactive basic, I want Annotate / generate / Process to run without an identity package, so that a group-blob mode is not blocked by a Review IDs step that has no individuals to name.
17. As an experimenter running Detect + track in non-interactive or interactive advanced, I want an identity package with raw tracklets written every time, so that I cannot skip Review IDs by unchecking an export box.
18. As an experimenter running Detect + track in interactive basic, I do not want a fake identity package, so that Review IDs is not ceremony over a single group blob.
19. As an experimenter who re-runs Detect + track on a video I already reviewed, I want to be asked first (default No) if accepted identities or switch markers exist, so that I do not destroy a finished review by hitting Run again.
20. As an experimenter who understands the confirm, I want re-detect to replace raw, unpublish remapped, and clear switch markers, so that the new tracking world is clean and later steps block until I save Review IDs again.
21. As an experimenter opening Review IDs on an old unsaved detect pack with no raw folder, I want to be asked before those public files are moved into raw, so that I can notice the pack may be in a bad state.
22. As an experimenter who declines that migrate, I want the files left as they are and switch edits still locked until raw exists, so that nothing is rewritten behind my back.
23. As an experimenter who accepts that migrate, I want the files moved into raw and a later Save to publish remapped, so that I can then edit switch markers safely.
24. As an experimenter who already accepted identities (including old “corrected” packs), I do not want that migrate offered, so that remapped geometry is never copied in as raw.
25. As an experimenter who already has an ethogram or example store for the video, I want a warning before Save rebuilds remapped, so that I know those files may now describe the wrong animals and will not be rewritten.
26. As an experimenter, if that downstream check itself fails, I want to be told the check failed and still asked before save (default No), so that an error is not treated as “nothing downstream.”
27. As an experimenter previewing in Review IDs, I want the picture to come from raw plus the current switch-marker list, so that I can add and remove swaps after the first save and see the truth.
28. As an experimenter, I want non-tail switch-marker edits to keep warning that later markers may now mean something else, so that I do not silently scramble a sequential remap.
29. As a developer, I want one publish function used by Review IDs save and any analyzer resave, so that the two paths cannot disagree about empty-switch accept.
30. As a developer, I want hydrate APIs typed and documented, leftover detector fields removed from Process videos, and frame progress during rebuild, so that the branch meets repo standards and the first review’s cleanup list.
31. As a future reader, I want empty kind vs missing kind and the mode-1 exemption to match the glossary and ADR 0006, so that we do not re-litigate silent fallbacks.

## Implementation Decisions

- Three seams only: identity package; hydrate from remapped tracklets; downstream gate. Qt tabs call these; they do not own the rules.
- Identity package: rebuild remapped only from raw. If raw is absent, refuse to rebuild. Never load public remapped files as the remap baseline.
- Every successful Review IDs save that is allowed to rebuild writes public remapped for every kind in the raw snapshot, including empty-switch accept (public equals raw). Status records accepted identities. The analyzer resave path must call this same publish, not a “only if some remaps applied” fork.
- Switch-list mutations (add, delete, remove-at-frame, undo, any reorder) require raw. Names and roles do not.
- Snapshot of an uncorrected public pack into raw is offered, not silent. Declining leaves the pack unchanged. Accepted / legacy “corrected” packs are never offered this migrate.
- Detect + track always writes raw for behavior modes that produce per-animal tracks (0 and 2). The export checkbox is removed or ignored. Mode 1 does not write an identity package.
- Downstream gate: one predicate. Modes 0 and 2 (and any other per-animal mode) require accepted identities. Mode 1 is exempt. Annotate, generate examples, and Process videos all use it. Annotate does not load the video when the gate fails.
- Hydrate: Process videos still takes no detector and does not re-track. It prepares analysis with a falsy detector path, fills geometry from remapped stores, then rebuilds categorizer inputs from those outlines plus the video.
- Animations: at each analysis frame, a real clip of the last `length` blobs (zeros when that ID is absent), matching the live acquire path — not the same blob stacked `length` times.
- Pattern images: when the categorizer’s `include_bodyparts` / inner code says body parts are in, recompute inners (and other-inners in interactive advanced) the same way the live path does (`get_inner` from outline + frame). Otherwise inners are omitted.
- Kind set: Process videos uses the remapped package’s kinds. Empty kind (in the package, no valid tracks) is valid. Missing kind (caller asked for a kind the package does not contain) is an error. A remapped kind that cannot be loaded into the analyzer is an error. Never skip a remapped kind to “succeed.”
- Re-detect confirm stays: if any selected video has accepted identities or switch markers, ask; default No. Then new tracking world (replace raw, unpublish remapped, clear switches).
- Downstream stale-file check: on lookup failure, report that the check failed and still confirm save (default No). No blanket exception that returns “no downstream files.”
- Standards cleanup in the same effort: public hydrate functions get type hints and Google-style docstrings; no new blanket `except Exception` on user-visible paths; Process videos frame progress during rebuild (not only at the end); remove leftover detector path / batch controls and unused accepted-is-not-None leftovers; dedupe load/unpublish loops if they remain after the single publish path.
- Do not rewrite the acquire loops in the core analysis engine. Hydrate reuses existing blob / inner / pattern helpers.
- Do not treat raw as a discovered animal kind. Public files remain remapped; raw stays a sibling snapshot.
- Glossary and ADR 0006 are already updated for these decisions; implementation must not contradict them.

## Testing Decisions

Good tests assert observable behavior at the three seams: what is on disk after save/export, what hydrate puts into analyzer animations/pattern images, whether a video is allowed through the gate, and which errors are raised. They do not assert private helper names, widget trees, or exact dialog wording beyond the decision (ask vs silent; refuse vs load).

- **Identity package:** empty-switch save publishes remapped equal to raw and marks accepted; second save from the same markers does not double-apply; publish without raw refuses; switch-edit policy is false without raw and true with raw; migrate-to-raw is a distinct operation the UI can confirm, not an implicit load side effect; detect export for modes 0/2 writes raw even if a caller would have passed “do not export”; mode 1 does not write a package. Prior art: existing identity package, raw store, review-pack load, and detect-export unit tests.
- **Hydrate:** a moving outline across frames produces an animation whose slices differ (not `length` copies of one blob); with body parts on, inners are passed into pattern generation; with body parts off, they are not; empty kind hydrates without error; missing kind and dropped remapped kind raise. Prior art: existing hydrate geometry test (extend it; do not only test centers).
- **Downstream gate:** annotate-style load refuses without accepted identities for a per-animal video and does not open the video; generate and process keep refusing; mode 1 is allowed without accepted identities. Prior art: process-videos unit tests (already patch hydrate), generate-examples refusal, workbench smoke (patch message boxes; assert batch does not start). Prefer testing the predicate and the annotate apply-to return, not full Qt playback.
- **Downstream check:** when the ethogram/examples lookup raises, the caller receives a failed-check signal, not an empty “all clear.”
- Use the repo venv pytest. Mark `gui` only if a test must construct widgets. Do not require GPU or a real detector for these slices.

## Out of Scope

- Rewriting acquire loops in the core detector analysis engine.
- Vendored Detectron2, `testing_ground`, user `~/.labgym/` data, or large local videos/weights.
- Changing how switch markers compose sequentially (non-tail still warns only).
- Auto-rewriting existing ethograms or example stores after a remap.
- Making interactive basic grow per-animal IDs or a real Review IDs workflow.
- Categorizer continue-train, annotate-images detector placeholder, or Tools dense-generate/sort.
- New GUI toolkit, job system, or ML stack.
- Committing, pushing, or merging the feature branch (human decision after tickets land).

## Further Notes

- Branch: `feat/editable-identity-review` (uncommitted WIP on top of the first implementation). Tickets assume that work stays on this branch; they fix it, they do not re-implement ADR 0006 from scratch.
- Pile B from the code review (types, leftover fields, progress, duplicated loops) is in Implementation Decisions so the branch can pass a second Standards + Spec review.
- After tickets: each `/implement` in a fresh session, `/tdd` inside, `/code-review` at the end of each ticket. Do not implement this spec as one blob in the grilling session.
- Domain terms: raw tracklets, switch marker, remapped tracklets, accepted identities, empty kind, missing kind. Avoid “original/updated mappings,” “corrected tracklets” as the name of remapped, and “knowingly accept raw” as the name of accepted identities.
