# LabGym — domain language

Ubiquitous language for LabGym product and analysis workflows. Implementation detail does not belong here.

## Training data & categorizer evaluation

**Example**:
A short labeled clip (or static image window) used to train or evaluate a categorizer. Examples live in an on-disk example store organized by behavior category.
_Avoid_: sample (when meaning a training clip), pair (unless referring to animal–animal interactive pairs specifically)

**Example store**:
The durable on-disk collection of examples as generated or prepared (folders, filenames, optional soft labels). It is the recoverable source of truth for raw media and original hard labels.
_Avoid_: dataset folder (ambiguous with project roots), training data (broader)

**Dataset manifest**:
A durable, non-destructive record of exclusions, label overrides, soft overrides, split assignment (train / validation / sealed test), and taxonomy operations for one example store root. Lives with that store; projects point at the store rather than owning a second override reality.
_Avoid_: annotation file (reserved for ethogram/video annotations), project-only overrides (as the sole home)

**Effective training set**:
The view of an example store after applying the dataset manifest (exclusions removed; overrides as active labels; soft projection when soft training is enabled). Training and evaluation that use a store consume this view by default; an explicit ignore-manifest mode can use the raw store.
_Avoid_: cleaned dataset, filtered folder

**Label override**:
A manifest entry that changes an example’s active behavior category without erasing the original category recorded when the example was created.
_Avoid_: relabel (unless speaking casually), move to folder

**Exclusion**:
A manifest entry that removes an example from the effective training set while leaving it in the example store so it can be restored.
_Avoid_: delete, remove from disk

**Recategorize**:
The user action of setting or changing a label override on an example.
_Avoid_: reclassify (prefer for model predictions), move

**Keep** (review decision):
A review outcome that leaves the example in the effective training set with its current active label and clears any “needs review” flag for that example in the current review context. Review decisions apply to the dataset manifest immediately (with undo), not via a separate staging commit.
_Avoid_: accept, dismiss (ambiguous)

**Categorizer**:
A trained behavior classifier (network weights + model parameters) that maps examples or video windows to behavior categories.
_Avoid_: model (alone, when detector vs categorizer is ambiguous), network

**Behavior category**:
A named class in a categorizer’s taxonomy (e.g. a behavior ethogram label used as a hard class).
_Avoid_: class (ok in metrics prose), label (prefer when meaning the assignment on one example)

**Manage dataset**:
The categorizer workbench area for taxonomy operations, example review (keep / exclude / recategorize), split management (including sealed test), and evaluation metrics over example stores and categorizers. Host with three areas: Categories, Review examples, Evaluate (split tooling may live under Categories or Evaluate as long as it edits the store’s dataset manifest).
_Avoid_: using “Test categorizer” for curation or multi-model comparison

**Test categorizer** (workbench tab):
A thin, quick path to score one categorizer on one ground-truth example folder. Shares the same evaluation engine as Evaluate; does not own comparison, review queues, or taxonomy ops.
_Avoid_: treating Test as a separate metrics implementation

**Train partition**:
Examples in the effective set used to update categorizer weights.
_Avoid_: training set (ok casually when contrast is clear)

**Validation partition** (train hold-out):
A stratified portion of the effective set used during training only for monitoring and training-time decisions (e.g. validation loss/accuracy, early stopping, learning-rate schedules)—**not** for weight updates. Also the default source of quick hold-out metrics after train. Not a substitute for a sealed test.
_Avoid_: test set, hold-out (alone—ambiguous with sealed test)

**Sealed test partition**:
A user-declared subset of an example store (recorded in the dataset manifest) that is **excluded from all training-time use**: no weight updates, no validation/early-stopping/LR signals, no train-time metric that could influence the run. Used later via Evaluate / Test / model comparison. Dataset Management can create or edit this split explicitly (fraction or hand-picked membership), independent of the train/validation split.
_Avoid_: hold-out (alone), validation set, dedicated test store (prefer when meaning an entirely separate folder)

**Dedicated test store**:
An entire example store (or its effective view) kept outside the training store—an alternative to a sealed test partition when test examples live in a separate folder. Same isolation rule: never used during training of the models under comparison.
_Avoid_: external examples (vague), ground truth folder (UI-only phrasing)

**Evaluation run**:
One scoring of a categorizer against a declared ground-truth set (train hold-out or dedicated test store). Always produces: confusion matrix (counts and row-normalized), classification report, macro and per-class F1 (worst-first), top confused pairs, per-example predictions with confidence, and a model-settings plus ground-truth-set snapshot. Train runs may also attach a high-loss table for the train-for-weights split. Durable artifacts live with the categorizer (per-model eval runs), not in the dataset manifest.
_Avoid_: test (alone), validation

**Split assignment**:
The durable assignment in the dataset manifest of each active example to **train**, **validation**, or **sealed test** (or unassigned until a split action). Stable by default: exclusions drop members without reshuffling; new examples stay unassigned until regenerate or assign-new. Sealed test membership must not leak into train or validation loaders.
_Avoid_: random split (when meaning an unpersisted one-off), hold-out split (ambiguous)

**High-loss example**:
An example from the train-for-weights split whose end-of-training loss ranks among the highest for that run. Flagged for human review; high loss alone does not imply a wrong label.
_Avoid_: hard negative (detection jargon), outlier

**Misclassified example**:
An example from an evaluation run whose predicted category differs from its active ground-truth category.
_Avoid_: error (alone), failure case

**Review queue**:
The set of examples surfaced in Review examples, typically the union of high-loss examples and misclassified examples, each tagged with its source (which train run or evaluation run).
_Avoid_: todo list, flagged set (ok casually)

**Taxonomy operation**:
A reversible manifest change to the set of active behavior categories or how they map—especially merge and category-level exclusion—without rewriting the example store’s original labels.
_Avoid_: remapping (ok for implementation), class edit

**Category merge**:
A taxonomy operation that maps two or more behavior categories to one active category for the effective training set, while retaining enough history to restore the finer original categories later.
_Avoid_: collapse classes, rename (rename is identity-preserving name change only)

**Category exclusion**:
A taxonomy operation that removes a behavior category from the active taxonomy so its examples do not participate in the effective training set (typically via bulk exclusion or an inactive-class rule), without deleting the example store.
_Avoid_: drop class, delete category

**Model comparison**:
A side-by-side evaluation of two or more already-trained categorizers on the same declared ground-truth set, highlighting metrics and training settings—not an automated hyperparameter sweep.
_Avoid_: sweep, experiment grid, NAS

**Taxonomy drift**:
A mismatch between a categorizer’s trained class list and the example store’s current active taxonomy (after merges, exclusions, or renames). Evaluation still uses the model’s label space; curation always updates the current dataset manifest.
_Avoid_: class mismatch (ok casually), broken model

**Soft label**:
A per-example distribution over behavior categories (typically frame-occupancy fractions over the example window from an ethogram), used when training is not hard-only. Distinct from the hard active label.
_Avoid_: probability (model output), confidence

**Soft label store**:
The durable ethogram-derived soft table for an example store (today: `soft_labels.csv`). Source of truth for original soft vectors, analogous to the example store for media and original hard labels.
_Avoid_: soft_labels.csv as the only name for the concept

**Soft override**:
A manifest entry that replaces an example’s soft vector for the effective view without rewriting the soft label store.
_Avoid_: edited soft_labels row (implementation)

**Soft projection**:
Deriving an effective soft vector in the active taxonomy from an original or overridden soft vector (e.g. summing mass of merged categories, dropping excluded categories and renormalizing). Deterministic and reversible by re-reading originals after undoing taxonomy operations.
_Avoid_: remapped soft labels (ambiguous with permanent rewrite)

## Identity review

**Raw tracklets**:
The detector’s per-video track geometry before any human identity corrections. They are not overwritten when switch markers change.
_Avoid_: original mappings, uncorrected tracks (ok casually)

**Switch marker**:
A user-recorded identity swap at one analysis frame. The current list of switch markers is the source of truth for identity corrections.
_Avoid_: mapping (alone), updated mappings, original mappings

**Remapped tracklets**:
Track geometry derived by applying the current switch-marker list to raw tracklets. Created or regenerated only when Review IDs is saved; this is what later annotation and analysis consume.
_Avoid_: updated mappings, corrected tracklets (ok in UI copy)

**Accepted identities**:
The user-confirmed identity layer for a video that has per-animal tracks, created by saving in Review IDs (the switch-marker list may be empty). Required before annotate, generate examples, or Process videos for those videos. Interactive basic has no per-animal tracks and does not use this gate.
_Avoid_: knowingly accept raw, corrected package (ok casually)

**Empty kind**:
An animal kind that is in that video’s accepted remapped tracklets but has no valid tracks. The kind was reviewed; no animal of that kind was present.
_Avoid_: missing kind, absent kind (ambiguous)

**Missing kind**:
An animal kind someone asked to analyze that is not in that video’s accepted remapped tracklets at all.
_Avoid_: empty kind, absent kind (ambiguous)
