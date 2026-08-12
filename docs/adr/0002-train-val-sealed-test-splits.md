# Train, validation, and sealed test partitions

Status: accepted

A single example store needs three roles so users can iterate without contaminating final claims:

| Partition | Weight updates | Training-time decisions (val loss, early stopping, LR, etc.) | Later metrics |
|-----------|----------------|---------------------------------------------------------------|---------------|
| **Train** | Yes | — | High-loss mining |
| **Validation** | No | Yes | Quick hold-out metrics |
| **Sealed test** | No | **No** | Honest test / model comparison |

Split membership lives in the dataset manifest, is **stable by default** (exclusions drop members without reshuffling; new examples stay unassigned until assign/regenerate), and can be regenerated explicitly. Dataset Management owns creating and editing the **sealed test** partition (fraction or selection). A **dedicated test store** (separate folder) remains valid as an alternative to an in-store sealed test, with the same isolation rule.

LabGym’s historical 80/20 split is treated as train vs **validation**, not as a sealed test—validation may influence the training run and must not be sold as fully held-out test performance.

## Considered options

- **Three-way split in the manifest (chosen)** — sealed test never enters train or validation loaders.
- **Validation only (no sealed test)** — simpler; encourages over-interpreting val metrics.
- **External test folder only** — still useful, but harder to carve a pure test subset from one generated store without a second copy.

## Consequences

- Train loaders must filter out sealed test membership completely.
- Evaluation runs record which ground-truth set they used (validation split generation, sealed test membership/generation, or dedicated store path + content identity).
- Fair model comparison prefers the same sealed test (or same dedicated store), not mismatched validation-only runs.
