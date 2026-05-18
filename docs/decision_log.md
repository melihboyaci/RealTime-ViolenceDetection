# decision_log.md

Authoritative record of architectural and methodological decisions. Any future change must update this file **before** propagating elsewhere.

## 1. Decisions Table

| # | Decision Area | Selected Option | Rejected Option(s) | Reason | Source Reference |
|---|---|---|---|---|---|
| D1 | Target FPS | **10 FPS** | Native FPS; 5 FPS; 25/30 FPS | Consistent ~3-second window across diverse source FPS; balances motion detail and compute | Source PDF — preprocessing section |
| D2 | Frame ops | **Conditional CLAHE on dark frames + Gaussian blur 3×3 + resize 640×640 + BGR→RGB** | Always-on CLAHE; no blur; smaller resize; keep BGR | Conditional CLAHE avoids overexposing already-bright frames; 3×3 blur reduces noise without smearing keypoints; 640×640 matches the YOLOv8 input domain; RGB matches PyTorch convention | Source PDF — preprocessing section |
| D3 | Pose model | **YOLOv8n-Pose** | YOLOv8s/m/l-Pose; HRNet; OpenPose; MediaPipe | Smallest YOLOv8-Pose variant; suitable for real-time; uses standard COCO 17 keypoints | Source PDF — pose estimation section |
| D4 | Multi-person strategy | **Top 2 persons (B+ strategy), sorted by X axis, with normalized bbox center distance as interaction feature** | Single-person; all-persons; sort by detection confidence; raw inter-person distance | Two-person scope captures the dominant interaction case; X-sort gives stable indexing; normalized distance is scale-invariant | Source PDF — multi-person section |
| D5 | Identity ordering | **X-axis sort** | IoU tracking; ByteTrack; appearance ReID | Simplest stable ordering; no extra latency. **Known limitation**: unstable under person overlap (see `limitations.md`) | Source PDF — multi-person section |
| D6 | Interaction feature | **Normalized bbox center distance** | Raw pixel distance; keypoint-pair distances; relative angles | Scale-invariant single scalar; cheap to compute | Source PDF — multi-person section |
| D7 | Sequence length | **30 frames** (sliding window) | 15; 60; 90 | At 10 FPS this is ~3 seconds, matching short violent altercations; keeps GRU input small | Source PDF — sequence creation section |
| D8 | Sequence model | **GRU** | LSTM; Transformer; TCN | Fewer parameters than LSTM, comparable on short sequences, faster real-time inference | Source PDF — model section |
| D9 | Split | **Stratified 70 / 15 / 15** train / val / test | 80/10/10; 60/20/20; non-stratified | Preserves class proportions; reserves enough data for both val (early stopping / threshold tuning) and isolated test | Source PDF — dataset section |
| D10 | Initial decision threshold | **0.7** | 0.5; 0.6; 0.8 | Bias toward fewer false positives at deployment start; tunable per use case | Source PDF — inference section |
| D11 | Motion filter (Violence only) | **θ = 0.05** | 0.0 (disabled); 0.025; 0.075; 0.10 | Drops likely-mislabeled low-motion Violence windows while preserving most positives | Source PDF — sequence filtering section |
| D12 | Keypoint confidence cutoff | **0.5** | 0.3; 0.4; 0.6 | Standard cutoff for unreliable keypoints; balances recall and noise | Source PDF — pose estimation section |
| D13 | Per-frame feature dim | **69** | Other dimensionalities | Encodes 2 persons × 17 COCO keypoints + the interaction distance under the chosen normalization | Source PDF — feature vector section |
| D14 | Skeletal normalization | **Hip centering + shoulder–hip scaling** | No normalization; bbox normalization; image-plane normalization | Position- and scale-invariant; standard for pose-based action recognition | Source PDF — feature vector section |
| D15 | Loss function | **BCELoss** | CE on 2-class softmax; focal loss | Matches a binary task with sigmoid head; simple and stable | Source PDF — model section |
| D16 | Optimizer | **Adam** | SGD; AdamW; RMSprop | Robust default for sequence models; minimal tuning to converge | Source PDF — model section |
| D17 | Training schedule | **max 100 epochs, batch 32, early stopping on val_loss, best by lowest val_loss** | Fixed N epochs; best by val accuracy/F1 | Lowest val_loss is the most stable selection criterion under BCELoss | Source PDF — training section |
| D18 | Inference buffer | **FIFO 30 frames** | Variable-length; smaller window; sliding sub-windows | Matches the training-time sequence length exactly; prevents distribution mismatch | Source PDF — inference section |
| D19 | Phase separation | **Strict offline / online split with shared per-frame builder** | Single combined pipeline | Keeps heavy work cached; guarantees no train/serve skew via shared builder | Source PDF — architecture section |
| D20 | Evaluation metrics | **Confusion matrix, precision, recall, F1, AUC-ROC** | Accuracy alone; AUC-PR alone | Covers both threshold-bound and threshold-free quality | Source PDF — evaluation section |

## 2. Locked Decisions

The following are **locked**. Changing any of them requires:

1. A new entry / amendment in this table.
2. Re-running the relevant ablation in `evaluation.md`.
3. Propagation across all docs that mention the value.

- D1 — Target FPS = **10**
- D2 — Frame ops order: **conditional CLAHE → Gaussian 3×3 → 640×640 → BGR→RGB**
- D3 — Pose model = **YOLOv8n-Pose**
- D4 — Multi-person = **top 2, X-sorted, normalized bbox center distance**
- D5 — **X-axis sort** for identity ordering
- D6 — Interaction feature = **normalized bbox center distance**
- D7 — Sequence length = **30 frames**
- D8 — Model = **GRU**
- D9 — Split = **stratified 70 / 15 / 15**
- D10 — Initial decision threshold = **0.7**
- D11 — Motion filter θ = **0.05** (Violence only)
- D12 — Keypoint confidence cutoff = **0.5**
- D13 — Per-frame feature dim = **69**
- D14 — Normalization = **hip centering + shoulder–hip scaling**
- D15 — Loss = **BCELoss**
- D16 — Optimizer = **Adam**
- D17 — Training schedule (max 100 epochs, batch 32, early stop on val_loss, best by lowest val_loss)
- D18 — Inference buffer = **FIFO 30 frames**
- D19 — Strict offline / online phase separation with shared per-frame builder
- D20 — Evaluation metric set

## 3. What May Still Change

The following are explicitly **not** locked and are expected to evolve as ablations and operational data come in:

- **Deployment decision threshold** — initial **0.7** is a starting point. May be retuned per use case using val-split metrics. Test split must remain isolated.
- **GRU hyperparameters** — hidden size, num_layers, dropout: **Not specified in the source PDF.**
- **Adam hyperparameters** — learning rate, betas, weight decay: **Not specified in the source PDF.**
- **Scheduler** — none assumed; **Not specified in the source PDF.**
- **Early-stopping patience / min-delta**: **Not specified in the source PDF.**
- **Stride** of the sliding sequence window: **TBD.**
- **Replacement rule** for sub-threshold keypoints (`< 0.5`): **TBD.**
- **Placeholder rule** for missing person 2 / zero-person frames: **TBD.**
- **Random seed** for the stratified split: **TBD.**
- **Class-balancing strategy** beyond the motion filter: **TBD.**
- **Motion statistic** used inside the motion filter: **TBD** (must be computed on normalized keypoints).
- **On-disk artifact layout**: **TBD** (the structure in `data_pipeline.md` is a recommendation).
- **Inference UI / overlay details**: **TBD.**
- **Temporal smoothing of consecutive online decisions**: **TBD.**

## 4. Assumptions / Open Questions

- Whether the project will adopt an identity tracker upstream of the multi-person selector (potentially overturning D5): **TBD.**
- Whether `BCELoss` is implemented as `BCELoss(sigmoid(...))` or `BCEWithLogitsLoss`: **TBD** (output semantics must remain sigmoid-thresholded at 0.7).
- Whether multi-source training data is in scope for v2: **TBD.**
