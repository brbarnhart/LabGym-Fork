# Evaluation artifacts live with the categorizer

Status: accepted

Metrics and review queues need durable, reloadable **evaluation runs**, not only console logs. Each run is stored under the trained categorizer directory:

```text
<model_dir>/eval/<run_id>/
```

including confusion matrices (counts + row-normalized), classification report, macro and worst-first per-class F1, top confused pairs, per-example predictions with confidence, model-settings and ground-truth-set snapshots, and (for training runs) end-of-train **high-loss** ranks on the train partition only. Mid-epoch hard-example mining is out of scope for v1.

Dataset curation state stays in the **example store’s dataset manifest**; model scores stay with the **model**. Review examples builds queues from selected runs and applies keep/exclude/recategorize to the current manifest.

## Considered options

- **Per-categorizer `eval/<run_id>/` (chosen)** — natural home next to `model_parameters.txt`; easy multi-model browse.
- **All runs under the example store** — couples many models’ scores to one data folder; messy ownership.
- **Log-only / in-memory** — cannot rebuild review queues or compare later.

## Consequences

- Test categorizer and Manage dataset → Evaluate share one evaluation engine that writes this shape.
- Taxonomy drift: scoring uses the **model’s class list**; curation still updates the **current** manifest; UI shows a drift banner rather than silently remapping network outputs into a new taxonomy.
- ROC/PR curves and calibration are deferred; prediction tables are required so misclassified review works.
