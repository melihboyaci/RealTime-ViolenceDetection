# state.md

This file captures the **current project state** so any AI agent or human contributor can pick the work up without re-reading the source PDF.

## 1. What Is Already Decided (Locked)

| Area | Decision |
|---|---|
| Task | Binary classification: **Violence / NonViolence** |
| Phases | Strict separation of **offline preprocessing** and **online inference** |
| Dataset | Real Life Violence Situations Dataset (Kaggle), 2000 videos |
| Split | Stratified **70 / 15 / 15** train / val / test |
| Target FPS | **10 FPS** |
| Frame ops | conditional CLAHE on dark frames → Gaussian blur **3×3** → resize **640×640** → BGR→RGB |
| Pose model | **YOLOv8n-Pose**, COCO **17 keypoints** |
| Keypoint confidence threshold | **0.5** (below = unreliable) |
| Multi-person policy | top **2** people, **sort by X axis**, **normalized bbox center distance** as interaction feature |
| Feature vector | **69 dims / frame** |
| Normalization | hip centering + shoulder–hip scaling |
| Sequence window | **30 frames**, sliding |
| Motion filter | applied to **Violence** windows only, **θ = 0.05** |
| Model | **GRU** sequence classifier, **sigmoid** output |
| Loss / optimizer | **BCELoss**, **Adam** |
| Training | max **100 epochs**, batch **32**, **early stopping on val_loss**, **best by lowest val_loss** |
| Inference buffer | **FIFO 30 frames** |
| Initial decision threshold | **0.7** |
| Evaluation | confusion matrix, precision, recall, F1, AUC-ROC, plus ablations |
| Stack | OpenCV, YOLOv8n-Pose, PyTorch, GRU |

## 2. What Is Not Yet Decided

| Area | Status |
|---|---|
| GRU layer count / hidden size / dropout | **Not specified in the source PDF** |
| Optimizer learning rate / scheduler | **Not specified in the source PDF** |
| Random seed policy for splits | **TBD** |
| Hardware target for online inference | **Not specified in the source PDF** |
| Final tuned decision threshold (post-evaluation) | **TBD** (initial = 0.7) |
| Logging / experiment tracker (e.g. TensorBoard, W&B) | **TBD** |
| Deployment surface (CLI / web / RTSP) | **TBD** |
| Class-balancing strategy beyond motion filter | **TBD** |

## 3. Current Pipeline Stage Assumptions

Unless evidence on disk says otherwise, future agents should assume:

- No code has been implemented yet.
- No `.npy` feature artifacts exist.
- No trained checkpoint exists.
- No evaluation report has been generated.
- The `docs/` folder is the only authoritative artifact.

## 4. Implementation Status Template

Each module/file should be tracked using this template (mirrored in `project-plan.md`):

```text
- [ ] <module name>           # status: pending | in-progress | done
      owner: <agent or human>
      depends-on: <module(s)>
      acceptance: <link to specs.md requirement IDs>
      notes: <any TBD or blocker>
```

## 5. Risks and Blockers

| Risk | Phase | Mitigation |
|---|---|---|
| X-axis sorting instability under person overlap | offline + online | Documented as accepted limitation; ablation may revisit |
| Video-level label noise leaking into Violence sequences | offline | **Motion filter θ = 0.05** drops low-motion Violence windows |
| Zero-person frames (no skeleton) | offline + online | Define explicit handling (TBD); see `inference.md` |
| Domain shift from a single YouTube-sourced dataset | model generalization | Acknowledged limitation; future multi-source data |
| Missing hyperparameters in source PDF | training | Treat as TBD; do not invent |

## 6. Last Known Stable Decisions

These are the points an agent **must not silently change**:

1. **10 FPS** standardization.
2. Frame ops order: **CLAHE (conditional) → Gaussian 3×3 → resize 640×640 → BGR→RGB**.
3. **YOLOv8n-Pose** with **17** keypoints, confidence cutoff **0.5**.
4. Multi-person **top-2 by X-axis sort**, **normalized bbox center distance**.
5. **69-dim** per-frame feature vector.
6. **Hip centering + shoulder–hip scaling**.
7. **30-frame** sliding sequence windows.
8. **Motion filter θ = 0.05** on Violence windows.
9. **GRU** model with sigmoid + **BCELoss** + **Adam**.
10. **70 / 15 / 15** stratified split.
11. Inference **FIFO 30 frames**, initial **threshold 0.7**.

Any change to these requires a new entry in `decision_log.md` and re-running the relevant ablations.

## 7. Open Questions

- Should keypoint imputation (e.g. last-known) replace zero-confidence keypoints, or should they be zero-filled? **TBD.**
- Behavior when fewer than 2 people are detected: zero-pad person 2 or skip frame? **TBD.**
- Online behavior while the FIFO buffer has fewer than 30 frames: emit "warming up" or no decision? **TBD.**
- Does the project ship a CLI, an SDK, or both? **TBD.**
- Reproducibility expectation: bit-exact or seed-only? **TBD.**

## 8. Assumptions / Open Questions

This file should be updated whenever a new decision is locked or a TBD is resolved. Treat it as the **single status surface** for the project.
