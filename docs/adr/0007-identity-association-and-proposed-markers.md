# Live association hygiene and proposed switch markers

Identity continuity is improved in two layers without changing ADR-0006. Detect + track keeps **raw tracklets** as the immutable detector geometry. The live pass uses Hungarian matching to **constant-velocity predicted COMs**, refuses bad rebinds (**split-detection** hygiene, conservative freeze during an **occlusion bout**, **parked slots**), and does **not** freeze on mere closeness (chasing). An occlusion bout is a composite test: fattened-outline intersection, small contour gap, collapsed detection area, or an unmatched slot with a leftover detection nearby. It does **not** require raw contours to intersect — LabGym detectors typically outline only the visible part of a piled animal, and those visible parts usually sit almost touching. Identities during a wrestle or chase still matter. Keep-versus-swap that the live pass cannot settle is decided offline and offered as **proposed switch markers** only when the rematch is decisive; weaker bouts stay timeline risk. A proposal is not a **switch marker** until the user accepts it; save still publishes **accepted identities** from the human-owned switch list (which may be empty). Review IDs shows proposals as distinct timeline ticks plus a compact list (Accept / Dismiss). Appearance is not in this slice; rematch is a pre/post cost matrix so a marked-animal cost can be added later. Default UX assumes unmarked animals.

## Considered options

- **Rematch during Detect + track and bake keep/swap into raw** — rejected; the post-contact window does not exist yet, and a wrong rematch would be frozen into raw.
- **Auto-insert proposals into the switch-marker list** — rejected; Save would publish machine errors as accepted identities if the user does not inspect each one.
- **Rewrite raw after offline rematch** — rejected; contradicts ADR-0006.
- **Learned appearance / re-ID as the first matcher** — deferred; animals are assumed unmarked; a paint-dot toggle is later and optional.
- **One global animals-per-kind spinbox as the association model** — rejected; association already consumes a per-kind count. A true per-kind UI is a sibling change, not this work.
- **Freeze whenever animals are close (contact-distance)** — rejected; that freezes chases and stops IDs updating while they run.
- **Post-separation IDs only** — rejected; intra-bout identities are used for some behaviors.
- **Propose a swap on every contact/split** — rejected as the only tier; users already ignore noisy contact warnings. Decisive rematches become proposals; weaker bouts stay risk hints.
- **Keep a Detect + track checkbox (default off) until soak-tested** — rejected; dummy-COM teleports and greedy last-COM are not worth preserving. Proposals remain opt-in per marker via Accept.
- **Proposals only after a fresh Detect + track** — rejected; the offline half can run on existing raw without modifying it.
- **Occlusion = raw contour / mask IoU** — rejected; wrestling outlines usually do not intersect.
- **Occlusion = COM distance alone** — rejected; that is chase, and fragment COMs are a poor proxy.
- **Dilated contours only** — not sufficient as the sole test; hidden-animal / one-blob frames still need the unmatched-slot and area-collapse clauses.

## Consequences

- Review IDs remains required. Proposals sit beside the switch-marker list as timeline ticks plus a compact list; accept promotes them, dismiss drops them.
- Chasing is in this work via predicted COM, not via freeze.
- Wrestling rematch uses **occlusion bouts** (composite test; typical footage has almost-touching visible parts, not far-side head/rump fragments). A decisive rematch at each **freeze-lift** can become a **proposed switch marker**. **Proximity risk** is a weak timeline hint. **Split detection** remains a separate short-gap rule and *does* use raw contour overlap (that is the double-outline case).
- Unmatched slots park; they do not teleport to a dummy COM after two seconds.
- New Detect + track always uses the new association (no checkbox). Offline proposals also compute from existing **raw tracklets** when Review IDs opens; they never rewrite raw.
- Hard cases → detector training images moves to a popup in this work so the proposals list has room. Behavior of extract is unchanged.
- Per-kind count UI is out of this branch. Appearance is a later cost on the same matrix; no marked-animal checkbox until that cost exists.
- Ground truth for this work is a small set of videos the user will accept in Review IDs (only two exist today).
- Dismissed proposals persist in the identity package and are not switch markers. Re-detect / new raw clears dismissals.
- Accept supports an N-way permutation of involved slots. The manual Swap IDs control stays 2-ID.
