## Problem Statement

When two or more animals of the same kind share a video, LabGym assigns each detection to a fixed identity slot by greedy nearest last center of mass. That assignment switches identities — and the switch often sticks — in three situations that matter for later review and analysis:

1. Wrestling / piling (occlusion). The animals occupy the same place. The detector typically outlines only the *visible* part of a hidden animal, so the two outlines usually sit almost touching and often do **not** intersect. Nearest last center of mass follows the bodies, not the identities. Several switches can happen *inside* one bout. Identities during the bout still matter, because those frames are themselves behaviors.
2. Chasing. The animals are close but not piled. The pursuer is often nearer the leader’s last center of mass than the leader is, so greedy assignment swaps them.
3. Split detection. For a frame or two the detector draws two overlapping outlines on one animal and misses the other. Greedy assignment gives both nearby blobs to the two nearest slots, and the steal can last after both animals are detected again.

Review IDs already lets a human fix a switch with a switch marker on immutable raw tracklets, then save accepted identities. That path stays. What is missing is better identity association on the way into raw tracklets, and machine-suggested proposed switch markers so the human does not have to find every remaining swap by hand. Contact-risk warnings on the timeline are already ignored: they treat a whole chase as one long bout, they miss split detection, and they do not offer a mapping to accept.

Animals per kind is a known maximum (not every slot is occupied every frame). Default UX assumes unmarked animals. Appearance similarity is a later option, not this work.

## Solution

Identity continuity is improved in two layers without changing how raw tracklets, switch markers, remapped tracklets, or accepted identities work.

**Live Detect + track (always on).** Identity association uses Hungarian matching to constant-velocity predicted centers of mass. Unmatched identity slots park instead of teleporting to a dummy point. Split detection (raw outline overlap plus an unmatched slot) does not steal. During an occlusion bout, association freezes (last center of mass and velocity do not update off the pile). An occlusion bout is a composite test — fattened-outline intersection, small raw-contour gap, collapsed detection area next to another detection, or an unmatched slot with a leftover detection nearby — and does **not** require raw outlines to intersect. Mere closeness is not freeze; that is chasing, handled by predicted centers of mass. At freeze-lift, rematch uses the last pre-freeze identities.

**Offline, including videos already detected.** Opening Review IDs computes, from current raw tracklets only, occlusion bouts, proximity risks (close centers of mass, not occluding), and proposed switch markers. A proposed switch marker is emitted only when rematch is decisive (typically at a freeze-lift, or a leftover chase/split error). Weaker bouts stay timeline risk. Proposals sit beside the switch-marker list: distinct timeline ticks plus a compact list. Accept promotes a proposal to a switch marker (an N-way permutation of the involved slots). Dismiss persists on the identity package and is not applied on save. Save still publishes accepted identities from the human-owned switch list (which may be empty). Opening Review IDs does not un-accept an already accepted package. Re-detect still replaces raw and clears switches and dismissals.

**Review IDs layout.** Hard cases → detector training images moves to a popup so the proposals list has room. Extract behavior is unchanged.

**Later, not this spec.** A marked-animal appearance cost on the same rematch cost matrix; a true per-kind animals-per-kind control in the Detect + track form.

Success is fewer remaining true switches on a small set of videos the user accepts in Review IDs (including intra-bout and chase), without a flood of false proposals.

## User Stories

1. As an analyst, I want identity slots to stay with the same animals through a chase, so I do not spend Review IDs time undoing nearest-center-of-mass swaps.
2. As an analyst, I want identity slots to stay with the same animals through a wrestle, including frames *during* the bout, so behaviors that happen in the pile are scored on the right individual.
3. As an analyst, I want a one- or two-frame split detection (two overlapping outlines on one animal, the other missed) not to steal or permanently swap identity slots, so a detector hiccup is a gap, not a lasting remap.
4. As an analyst, I want an animal that leaves the frame or is missed to park its identity slot, so that slot does not snap onto another animal across the arena.
5. As an analyst, I want animals per kind to remain a hard maximum, so extra detections cannot create new individuals.
6. As an analyst with two mice of one kind, I want the first shipped path to be correct for that scene, so my current videos improve immediately.
7. As an analyst with more than two animals or more than one kind, I want rematch and accept to apply an N-way permutation of the involved slots, so a three-animal pile is not a second architecture.
8. As an analyst, I want Detect + track to use the new identity association automatically, so I do not have to remember a checkbox to avoid dummy-center teleports.
9. As an analyst, I want raw tracklets to remain the detector’s immutable geometry, so I can still undo identity decisions after the first save.
10. As an analyst, I want Review IDs to remain required before annotate, generate examples, or Process videos, so accepted identities stay a human gate.
11. As an analyst opening Review IDs on a newly detected video, I want to see occlusion bouts and proximity risks on the timeline, so I can jump to piles versus close chases.
12. As an analyst, I want a chase to show as proximity risk, not as one long occlusion bout, so the timeline is usable again.
13. As an analyst, I want a wrestle to show as an occlusion bout even when the two raw outlines do not intersect, so piles are not invisible just because the detector outlined only the visible parts.
14. As an analyst, I want a fully hidden animal (one leftover detection, a slot unmatched) to count as an occlusion bout, so a one-blob pile is not treated as a chase.
15. As an analyst, I want proposed switch markers only when rematch is decisive, so I do not ignore the list the way I already ignore noisy contact warnings.
16. As an analyst, I want a decisive rematch at each freeze-lift inside a long wrestle to become its own proposed switch marker, so several intra-bout switches can be accepted in order.
17. As an analyst, I want each proposed switch marker to carry a concrete mapping and a frame, so I am not guessing which IDs to swap or where.
18. As an analyst, I want proposed switch markers as distinct ticks on the timeline and as a compact list, so I can jump to them without hunting.
19. As an analyst, I want Accept on a proposal to add it to the switch-marker list and update the preview, so I can see the result before save.
20. As an analyst, I want Dismiss on a proposal to hide it next time I open that package, so I am not re-rejecting the same suggestion.
21. As an analyst, I want dismissals not to be switch markers and not to apply on save, so dismissing is not an identity decision.
22. As an analyst who re-runs Detect + track, I want switches and dismissals to clear with the new raw, so suggestions and corrections are not from a previous tracking world.
23. As an analyst with videos I already detected, I want opening Review IDs to compute proposals from current raw without rewriting raw, so I get the offline half without re-detecting.
24. As an analyst who already saved accepted identities, I want opening Review IDs to leave that acceptance in place, so new proposals are extras I may accept and save again — not a silent un-accept.
25. As an analyst, I want proposals that match a switch marker already on the list not to appear again, so I do not accept the same swap twice.
26. As an analyst, I want Save with zero accepted proposals to still mean “I accept raw,” so empty switch lists remain valid accepted identities.
27. As an analyst, I want the manual Swap IDs control to stay a two-identity swap, so everyday two-mouse corrections stay simple.
28. As an analyst, I want Hard cases → detector training images moved out of the crowded right column into a popup, so the proposals list has room and extract still works the same.
29. As an analyst, I want Process videos to keep hydrating remapped tracklets and not re-track, so a better association only appears in analysis after I save Review IDs (and after I re-detect, if I want better raw).
30. As an analyst growing a small labeled set, I want success judged by fewer remaining true switches on videos I accept in Review IDs — including intra-bout and chase — so we do not ship a noisier timeline.
31. As a future user of marked animals, I want the rematch cost matrix left as a hook, so a paint-dot appearance cost can be added later without a new identity model.
32. As a future user of unmarked animals, I want the default product to assume animals are not marked, so appearance is never required to get the association improvements.
33. As a developer, I want identity association tested through one identity-continuity interface with synthetic detections, so wrestle, chase, and split-detection behavior does not require GPU or Qt.
34. As a developer, I want proposed switch markers and dismissals tested through the existing identity-package persistence, so reopen and re-detect behave like switch markers already do.
35. As an analyst with mixed kinds, I want identity association to stay independent per animal kind, so a mouse never takes an object’s identity slot.

## Implementation Decisions

- **Two-layer identity continuity, one module.** A single identity-continuity module owns live identity association and offline proposal. Detect + track and Review IDs are adapters. The module’s interface is the spec’s test surface.
- **Live operation.** Input: this frame’s detections (centers, outlines or test stand-in shapes, areas) plus animals per kind and prior slot state. Output: assignment of detections to identity slots, updated slot state (active, parked, frozen), and occlusion / split flags. Hungarian matching to constant-velocity predicted centers of mass. No learned appearance.
- **Offline operation.** Input: raw tracklet store (centers, validity, heights, contours). Output: occlusion bouts, proximity risks, and proposed switch markers. Same occlusion tests as live freeze. Proposals only when rematch is decisive, typically at freeze-lift against last pre-freeze identities. Cost matrix is internal (geometry now; appearance later is the same matrix, not a second public seam until a second adapter exists).
- **Occlusion bout (composite), not raw outline intersection.** Fire if any of: fattened outlines intersect; minimum gap between raw contours is small; one detection’s area has collapsed versus that kind’s typical area while another is nearby; a slot is unmatched and leftover detection(s) sit where that animal should be. Typical wrestle footage has almost-touching visible parts, so dilation and gap do most of the work; area-collapse and unmatched-slot cover the one-blob / hidden animal. Thresholds are module constants, tuned against synthetic fixtures and the user’s small labeled set — not user-facing controls in this spec.
- **Proximity risk.** Close centers of mass that do not satisfy occlusion tests (chase). Timeline hint. Propose only if a rematch is still decisive.
- **Split detection.** Raw contour overlap plus an unmatched slot of that kind. Live pass refuses the steal. May also emit a short-gap proposal if a lasting swap remains in raw. Not an occlusion bout.
- **Parked slot.** Unmatched slot stays reserved; it does not bind to a detection that is a better match to an active slot and does not teleport to a dummy far-away center. A leftover detection after present animals are matched may claim it only if it is not a split detection.
- **Freeze.** During an occlusion bout, do not update last center of mass or velocity off the pile. Freeze-lift is the first frame the occlusion tests no longer hold. Rematch from last pre-freeze identities, not the pile center.
- **Contact event** is the umbrella name for a timeline risk bout (occlusion bout or proximity risk). Existing COM-distance-only “everything close is one long contact” behavior is replaced by this split.
- **Proposed switch markers** are not switch markers. They persist on the identity package beside the switch list. Accept copies the mapping into the switch-marker list (user-accepted, including N-way permutation). Dismiss persists as a rejection keyed so the same suggestion does not return on reopen. Dismissals are not applied on save. Re-detect / new raw clears switches and dismissals.
- **Switch marker apply path is unchanged:** suffix permutation from a frame forward, rebuilt remapped tracklets only on Review IDs save, raw never rewritten. Multiple accepted freeze-lift markers in one wrestle are sequential suffix remaps.
- **Always on** for new Detect + track. No enable checkbox. Offline proposals also run when Review IDs opens on existing raw.
- **Already accepted packages:** computing proposals does not clear accepted identities. Accepting new proposals and saving publishes a new remapped layer; existing downstream-artifact warnings still apply.
- **Manual Swap IDs** remains a two-identity swap. Accept of a proposal may apply an N-way mapping.
- **Hard cases → detector training images** moves to a popup opened from a button. Sampling, output directory, and extract behavior stay the same.
- **Animals per kind** remains a declared maximum per kind. The Detect + track form still copies one spinbox onto every kind; a true per-kind control is a sibling, not this spec. Association already consumes a per-kind count.
- **Interactive basic** still has no per-animal identity slots and is out of this work.
- **ADR-0006 stands.** ADR-0007 records this two-layer split, occlusion-as-composite, proposals-are-not-switch-markers, and related rejections.

## Testing Decisions

- Test **external behavior** of the identity-continuity module: given detections (or a raw tracklet store), assert identity-slot assignments, occlusion vs proximity classification, proposed mappings and frames, and freeze / park / split outcomes. Do not assert internal cost numbers, Hungarian call shapes, or dilation kernel size except insofar as they change those outcomes.
- **Seam 1 — identity continuity (synthetic detections).** Short in-memory sequences, same style as existing contact-event tests (hand-built centers, now with simple outlines/areas). Fixtures at minimum:
  - two animals chase (close, not occluding) → slots follow predicted motion, not last center of mass
  - two animals pile with almost-touching non-intersecting outlines → occlusion bout, freeze, correct rematch at freeze-lift
  - several freeze-lifts in one wrestle → multiple decisive proposals, not one suffix at the final exit only
  - split detection for one or two frames → no lasting steal
  - one animal missing / off-frame → parked slot does not snap across the arena
  - one leftover detection during a pile (hidden animal) → occlusion, not proximity
  - three involved slots → rematch is a permutation, not only a 2-swap
  - proximity without occlusion → risk, no proposal unless rematch is decisive
- **Seam 2 — identity package persistence.** Follow existing switch-marker and package tests: write/read proposals and dismissals; reopen hides dismissed items; accept promotes to a switch marker that apply already understands; proposals matching an existing switch marker are suppressed; re-detect / new raw clears proposals and dismissals; computing proposals does not rewrite raw or un-accept a saved package.
- **Adapters.** Detect + track calls the live operation instead of greedy last-center-of-mass. Review IDs shows two risk kinds, proposal ticks, compact list, Accept/Dismiss, and the hard-cases popup. Prefer package-level and existing Review IDs package-load tests over widget screenshot tests. Hard-cases extract keeps its current unit tests; the popup is housing.
- **Not required for this spec:** GPU detector inference, full-video integration, formal MOT/IDF1 on a public benchmark. The user’s growing set of Review IDs saves is the product acceptance set, not a unit-test fixture.

## Out of Scope

- Learned appearance / re-ID models, marked-animal toggle, paint-dot or tail-color features. The rematch cost matrix is only a later hook.
- Changing or skipping accepted identities; rewriting raw tracklets; auto-inserting proposals into the switch-marker list.
- Per-kind animals-per-kind UI (sibling). Variable animal count / creating new identity slots mid-video.
- Freeze on mere center-of-mass closeness; treating a whole chase as one occlusion bout.
- Replacing Review IDs, Process videos re-tracking, or interactive basic identity.
- Vendored Detectron2 tracker code; a second ML stack; idtracker.ai / SORT as an external replacement.
- User-facing threshold sliders for occlusion, dilation, or “decisive” rematch.
- Growing already-large detector-analysis modules with the new algorithm (the algorithm lives in the identity-continuity module; the engine calls it).

## Further Notes

- Branch: `feat/appearance-aware-id-assignment`. Glossary and ADR-0007 are already updated; implement against those names (occlusion bout, not “overlap bout”).
- First scene: two animals, one kind, top-down. Engine must already be N-way.
- Typical occlusion outlines almost touch; far-side head/rump fragments were not observed. Keep unmatched-slot and area-collapse anyway for the one-blob hidden animal.
- A red sharpie tail often lies outside the outline. If appearance is added later, crops need a margin beyond the contour; paint-dots on the back are the mark that can land inside the outline.
- Only two videos currently have accepted identities. The user will accept more while this is built. Do not invent a second identity ground-truth format.
- `possible_swap` geometry on today’s contact events is a hint, not a proposed switch marker, unless rematch is decisive under the new rules.
