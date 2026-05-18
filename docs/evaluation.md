# evaluation.md

Evaluation rigor, metrics, and ablation suite.

## 1. Test Set Isolation

- The **test split** (15% of dataset, stratified) is **never** used for:
  - training,
  - validation / early stopping,
  - checkpoint selection,
  - decision threshold tuning,
  - any ablation that touches model weights.
- Threshold tuning is done on the **val split**; once frozen, it is reported on the test split exactly once per locked baseline.
- Re-running test evaluation after tuning on the test split invalidates the result.

## 2. Metrics — Definitions

| Metric | Definition | Why it matters here |
|---|---|---|
| **Confusion matrix** | TP, FP, TN, FN counts on the test split | Base for all derived metrics |
| **Precision** | `TP / (TP + FP)` for **Violence** | High precision = few false alarms |
| **Recall** | `TP / (TP + FN)` for **Violence** | High recall = few missed Violence events |
| **F1** | `2 · P · R / (P + R)` for **Violence** | Single-number balance metric |
| **AUC-ROC** | Area under the ROC curve over thresholds in `[0, 1]` | Threshold-free quality of the sigmoid score |

All per-class metrics are reported with **Violence as the positive class** (`y = 1`).

## 3. Confusion Matrix Interpretation

```text
                 Predicted
                 V       NV
Actual  V    [ TP    | FN  ]
        NV   [ FP    | TN  ]
```

- **FP (Predicted Violence, Actually NonViolence)**: false alarm. Costly in surveillance UX.
- **FN (Predicted NonViolence, Actually Violence)**: missed event. Costly in safety scenarios.
- The relative cost drives the threshold choice (see §7).

## 4. Precision / Recall / F1 / AUC-ROC Usage

- Report **all four** alongside the confusion matrix.
- Use **AUC-ROC** to compare *models* (threshold-free).
- Use **precision / recall / F1** at a chosen threshold to compare *deployments*.
- Always state the **threshold** under which precision / recall / F1 were computed.

## 5. Threshold Analysis

- Sweep threshold over a grid (e.g. `0.30, 0.35, ..., 0.90`) on the **val split**.
- Plot precision–recall vs threshold.
- Choose the threshold per use case (see §7).
- The **initial** deployment threshold is **0.7** (locked baseline). It is a starting point, not an empirical optimum.

## 6. Ablation Experiments

All ablations follow the same protocol:

1. Start from the locked baseline.
2. Change exactly one factor.
3. Re-train (if the change touches the dataset or model) using the same training schedule.
4. Re-evaluate on **val** (for tuning) and on **test** (for the final number).
5. Compare against the baseline using the metrics in §2.

### 6.1 Threshold Study

| Factor | Baseline | Sweep |
|---|---|---|
| Decision threshold | **0.7** | `{0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90}` |
| Re-train required? | No | No |

Goal: characterize precision/recall trade-off across operating points and pick a deployment threshold per use case.

### 6.2 Interaction Feature Study

| Factor | Baseline | Variant |
|---|---|---|
| Normalized bbox center distance | **included** in the 69-dim vector | **removed** (feature vector becomes shorter; record the new dimensionality) |
| Re-train required? | — | **Yes** |

Goal: quantify the contribution of the two-person interaction signal.

### 6.3 Normalization Study

| Factor | Baseline | Variants |
|---|---|---|
| Skeletal normalization | **hip centering + shoulder–hip scaling** | (a) no normalization; (b) hip centering only; (c) shoulder–hip scaling only |
| Re-train required? | — | **Yes** for each variant |

Goal: confirm that both normalization steps contribute, and quantify each.

### 6.4 Motion Filter Threshold Study

| Factor | Baseline | Sweep |
|---|---|---|
| Motion filter θ (Violence only) | **0.05** | `{0.00 (disabled), 0.025, 0.05, 0.075, 0.10}` |
| Re-train required? | — | **Yes** for each θ |

Goal: characterize how aggressively low-motion Violence windows should be removed; trade off label-noise reduction vs. data loss.

## 7. How to Choose the Final Threshold per Use Case

- **High-recall surveillance** (must not miss events): pick the lowest threshold whose precision is still acceptable. Bias toward `< 0.7`.
- **Low-false-alarm dashboards** (operator fatigue): pick the highest threshold whose recall is still acceptable. Bias toward `> 0.7`.
- **Balanced default**: maximize F1 on the val split.
- Always **freeze the threshold on val** before reporting on **test**.

## 8. How to Report Results

For each experiment, report:

- Run ID and config (seed, hyperparameters, motion-filter θ, etc.).
- Confusion matrix on **test**.
- Precision, recall, F1 at the **frozen** threshold (state the threshold value).
- AUC-ROC (threshold-free).
- A short delta vs. the locked baseline.

Recommended single-table summary:

| Run | Threshold | Precision | Recall | F1 | AUC-ROC | Notes |
|---|---|---|---|---|---|---|
| baseline | 0.70 | TBD | TBD | TBD | TBD | locked |
| threshold-sweep@0.5 | 0.50 | TBD | TBD | TBD | — | same weights |
| no-interaction | 0.70 | TBD | TBD | TBD | TBD | feature ablation |
| no-normalization | 0.70 | TBD | TBD | TBD | TBD | normalization ablation |
| motion-θ=0.0 | 0.70 | TBD | TBD | TBD | TBD | motion filter ablation |

## 9. Reporting Hygiene

- Do not move from `val` to `test` until the threshold is frozen.
- Do not change two factors in one run.
- Do not silently rerun the baseline with a new seed and call it "the baseline"; treat it as a new entry.

## 10. Assumptions / Open Questions

- Window-level vs. video-level metric aggregation: **TBD** (default: window-level).
- Whether to also report calibration (e.g. ECE / Brier): **TBD.**
- Confidence intervals via bootstrap: **TBD.**
- Whether multiple seeds are required per ablation row: **TBD.**
