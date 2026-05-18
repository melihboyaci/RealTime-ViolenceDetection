# model.md

GRU sequence classifier specification.

## 1. Input

- **Shape**: `(batch, 30, 69)`
  - `30` = sequence window length (sliding window over 10 FPS frames)
  - `69` = per-frame feature vector (2 persons × 17 COCO keypoints + normalized bbox center distance, on hip-centered + shoulder–hip-scaled skeletons)
- **Type**: `float32`
- **Source**: produced by the shared offline/online feature builder.

## 2. Layer-by-Layer Architecture

| Layer | Type | Notes |
|---|---|---|
| 1 | Input | `(batch, 30, 69)` |
| 2 | **GRU** | hidden size: **TBD**, num_layers: **TBD**, dropout: **TBD** |
| 3 | Pooling / last-step | **TBD** (commonly: take the last hidden state or mean over time) |
| 4 | Linear | output dim **1** |
| 5 | **Sigmoid** | probability of **Violence** |

> Concrete hidden size, number of layers, dropout, and the head's reduction strategy are **Not specified in the source PDF**. Once chosen, they must be recorded in `decision_log.md` and frozen as a baseline before any ablation.

## 3. Why GRU over LSTM

| Criterion | GRU | LSTM |
|---|---|---|
| Parameter count | Lower | Higher |
| Training speed | Faster | Slower |
| Performance on **short** sequences (~30 steps) | Comparable | Comparable |
| Memory footprint | Lower | Higher |
| Suitability for real-time inference | Better | Slightly worse |

The project chooses **GRU** because the sequence is short (30 frames at 10 FPS, ~3 seconds) and the model must run in real time downstream of YOLOv8n-Pose, where every saved millisecond matters.

## 4. Output Interpretation

- The model outputs a single scalar `p ∈ [0, 1]` per sequence window.
- `p` is the probability that the window depicts **Violence**.
- A decision is made by thresholding: `p ≥ 0.7` (initial) → **Violence**, otherwise **NonViolence**.
- The threshold is configurable per use case; see `evaluation.md`.

## 5. Loss Function

- **`BCELoss`** (binary cross-entropy on the sigmoid output).
- If a `BCEWithLogitsLoss` formulation is used for numerical stability, the final activation must remain logically sigmoid; the `0.7` threshold is on the probability, not the logit.

## 6. Optimizer

- **Adam**.
- Learning rate, betas, weight decay: **Not specified in the source PDF** (TBD; record in `decision_log.md`).

## 7. Scheduler

- **Not specified in the source PDF.** Default assumption: no LR scheduler unless explicitly chosen and documented.

## 8. Parameter Count Summary

- **Not specified in the source PDF.** Must be reported once the GRU hidden size and depth are fixed.

## 9. Model Limitations

- **Pose-only input**: cannot recover information lost when YOLOv8n-Pose misses small or occluded persons.
- **Fixed window length (30)**: events shorter than ~3 seconds may be diluted; events longer than the window must rely on consecutive predictions.
- **Two-person cap**: scenes with more than 2 active participants are truncated to top 2.
- **No temporal context beyond 30 frames**: the model has no memory across windows during training; online inference reuses the same window length via the FIFO buffer.
- **Sensitive to x-axis sort**: the multi-person ordering can flip when persons overlap, momentarily disrupting features.

## 10. Suggested Ablation Points (Model)

The following ablations are model-side and complement those in `evaluation.md`:

| Ablation | What changes | Why |
|---|---|---|
| Hidden size sweep | GRU hidden size | Capacity vs. real-time budget |
| Depth sweep | `num_layers` | Marginal gain vs. cost |
| GRU vs. LSTM | sequence backbone | Validate the locked GRU choice |
| Pooling | last-step vs. mean vs. attention | Aggregation across the 30-frame window |
| Dropout | values 0.0 / 0.2 / 0.5 | Regularization sensitivity |

> Any ablation that touches the locked decisions in `decision_log.md` must update `decision_log.md` if it overturns them.

## 11. Assumptions / Open Questions

- GRU hidden size, layers, dropout: **Not specified in the source PDF.**
- Aggregation strategy (last hidden state vs. mean vs. attention): **TBD.**
- Whether `BCELoss` or `BCEWithLogitsLoss` is used internally: **TBD** (output semantics must remain sigmoid-thresholded at 0.7).
- Initialization scheme for GRU/Linear: **TBD.**
