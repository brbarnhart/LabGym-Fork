# Manage dataset workbench and already-trained model comparison

Status: accepted

Categorizer workbench gains a **Manage dataset** host with three areas: **Categories**, **Review examples**, and **Evaluate**. That is the home for taxonomy ops, sealed-test/split tooling (as needed), high-loss and misclassification review, metrics browsing, and multi-model comparison.

**Test categorizer** remains a thin one-model, one-folder shortcut and must call the same evaluation engine—not a second metrics implementation.

**Model comparison** means scoring **already-trained** categorizers on a **shared** declared ground-truth set (prefer sealed test or dedicated test store), side-by-side with settings from `model_parameters.txt` (time_step, network levels, label_mode, lambda_soft, etc.). There is **no** in-app hyperparameter sweep or auto-train grid in this feature set.

## Considered options

- **Manage dataset host + compare under Evaluate (chosen)** — matches Generate training data’s nested pattern; keeps Test lightweight.
- **Fold everything into Test categorizer** — overloads a simple path; weak home for taxonomy and review.
- **Full sweep UI** — high cost; out of scope for “how did my trained settings compare?”

## Consequences

- Delivery order: evaluation engine → train hooks (val metrics + high-loss + minimal manifest/splits) → Evaluate UI → Review + train effective view → Categories + soft projection → compare polish.
- Implementation proceeds on a dedicated feature branch with incremental commits per phase.
