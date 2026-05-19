# Real-Time Violence Detection via Skeletal Pose Analysis

A real-time violence detection system that operates entirely on **skeletal pose features** — no raw pixels are fed to the classifier. The system extracts 17-keypoint poses with YOLOv8n-Pose, builds 69-dimensional per-frame feature vectors, and classifies 30-frame sequences as **Violence / NonViolence** using a two-layer GRU.

```
[Raw Video]
    │
    ▼  Offline Preprocessing (Kaggle GPU)
    ├─ 10 FPS sampling
    ├─ Conditional CLAHE → Gaussian 3×3 → Resize 640×640 → BGR→RGB
    ├─ YOLOv8n-Pose → 17 COCO keypoints per person
    ├─ Top-2 person selection (by bbox area, X-sorted)
    ├─ Hip centering + shoulder–hip scaling
    ├─ 69-dim feature vector / frame
    ├─ 30-frame sliding windows (stride=15)
    └─ Motion filter θ=0.05 on Violence windows only
           │
           ▼  (N, 30, 69) .npy sequences
    ┌──────────────────────────────┐
    │  GRU Classifier (Training)   │
    │  2-layer GRU 128→64          │
    │  BCELoss + Adam, max 100 ep  │
    │  Early stopping on val_loss  │
    └──────────────────────────────┘
           │
           ▼  models/best_model.pt
[Live Camera / Video]
    │
    ▼  Online Inference (real-time)
    ├─ Same frame preprocessing
    ├─ FIFO buffer (30 frames)
    ├─ GRU → sigmoid score
    ├─ Triple-zone decision (t=0.45):
    │    score < 0.35  →  NonViolence
    │    0.35–0.45     →  Suspicious
    │    score ≥ 0.45  →  Violence
    ├─ Temporal smoothing (3-window majority)
    └─ Entry suppression (30f on new person)
```

## 1. Results

### Final Model (Blended RLVS + RWF-2000, threshold = 0.45)

| Metric      | Value      |
| ----------- | ---------- |
| Precision   | 0.7749     |
| Recall      | 0.8700     |
| **F1**      | **0.8197** |
| **AUC-ROC** | **0.9272** |
| Accuracy    | 85.7%      |

Test set: 740 sequences (RLVS-only, isolated). TP=241, FP=70, FN=36, TN=393.

### Ablation Summary

| Experiment                                | F1        | AUC-ROC   | ΔF1      |
| ----------------------------------------- | --------- | --------- | -------- |
| Baseline (RLVS-only, t=0.70)              | 0.667     | 0.921     | —        |
| Threshold sweep → t=0.40                  | 0.827     | 0.921     | +0.160   |
| **Blended model (RLVS+RWF-2000, t=0.45)** | **0.820** | **0.927** | deployed |
| P7.2 No interaction feature (68-dim)      | 0.825     | 0.931     | +0.005   |
| P7.3 No normalization                     | 0.827     | 0.938     | +0.007   |
| P7.3 Hip centering only                   | 0.823     | 0.937     | +0.003   |
| P7.3 Scale only (no centering)            | 0.819     | 0.925     | -0.001   |
| P7.4 Motion-filter θ=0.00 (disabled)      | 0.831     | 0.923     | +0.011   |
| P7.4 Motion-filter θ=0.025                | 0.819     | 0.919     | -0.000   |
| P7.4 Motion-filter θ=0.075                | 0.798     | 0.923     | -0.022   |
| P7.4 Motion-filter θ=0.10                 | 0.830     | 0.931     | +0.010   |

> **P7.3:** All normalization variants within ±1% F1. Centering more important than scaling. Full normalization kept.
>
> **P7.4:** Motion filter has marginal impact — θ≥0.025 yields near-identical data (±8 windows). Variance in F1 across θ=0.025–0.10 is due to training stochasticity, not data quality. Baseline θ=0.05 provides a reasonable operating point.

## 2. Dataset

| Split | Sequences | Violence | NonViolence | Source               |
| ----- | --------- | -------- | ----------- | -------------------- |
| Train | 6 423     | 3 227    | 3 196       | RLVS + RWF-2000      |
| Val   | 1 435     | 683      | 752         | RLVS + RWF-2000      |
| Test  | 740       | 277      | 463         | RLVS only (isolated) |

Raw video preprocessing is done via `notebooks/kaggle_preprocessing.ipynb` on Kaggle (GPU T4 x2, ~1 hour for 4000 videos).

## 3. How to Run

### Requirements

```bash
# GPU (CUDA 12.x recommended):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### Real-Time Inference (webcam)

```bash
python -m src.inference
```

### Inference on a Video File

```bash
python -m src.inference --source path/to/video.mp4
```

### Evaluate on Test Split

```bash
python -m src.evaluate
```

### Train from Scratch

```bash
# 1. Run notebooks/kaggle_preprocessing.ipynb on Kaggle to generate .npy sequences
# 2. Place sequences in data/sequences/{train,val,test}/
python -m src.train
```

## 4. Repository Structure

```
configs/        — config.py (all hyperparameters, thresholds, paths)
docs/           — project documentation (architecture, pipeline, evaluation, limitations…)
models/         — best_model.pt checkpoint + training log
notebooks/      — kaggle_preprocessing.ipynb (offline preprocessing)
                — kaggle_ablation_norm_*.ipynb (P7.3 ablation variants)
scripts/        — ablation studies, threshold sweep, utility scripts
src/            — preprocessing.py, model.py, dataset.py, train.py, evaluate.py, inference.py
```

## 5. Key Design Decisions

| Decision               | Choice                               | Rationale                                       |
| ---------------------- | ------------------------------------ | ----------------------------------------------- |
| Feature representation | 69-dim skeletal pose                 | Privacy-preserving, lighting-invariant          |
| Normalization          | Hip centering + shoulder–hip scaling | Camera-distance invariant pose                  |
| Sequence length        | 30 frames @ 10 FPS (~3 sec)          | Covers typical violence onset duration          |
| Motion filter          | θ=0.05 on Violence only              | Removes mislabeled calm segments                |
| Model                  | 2-layer GRU                          | Temporal sequence; lightweight for real-time    |
| Dataset blend          | RLVS + RWF-2000                      | Reduces domain overfitting (+21% training data) |
| Decision zones         | 3-zone (NV / Suspicious / Violence)  | Reduces false alarm fatigue                     |

Full decision log: [`docs/decision_log.md`](docs/decision_log.md)

## 6. Known Limitations

- **Two-person cap:** only the top 2 detected persons are encoded; crowd scenes lose information.
- **Pose-only:** weapons, blood, and context invisible to the model.
- **X-axis sorting instability:** persons can flip identity during crossing/overlap (see `docs/limitations.md`).
- **Single dataset domain:** trained on YouTube clips; performance may drop on top-down/fisheye surveillance cameras.
- **Video-level labels:** label noise mitigated by motion filter but not eliminated.

Full limitations: [`docs/limitations.md`](docs/limitations.md)

## 7. Documentation Index

| File                    | Purpose                                        |
| ----------------------- | ---------------------------------------------- |
| `docs/architecture.md`  | Component boundaries + Mermaid flow diagrams   |
| `docs/data_pipeline.md` | Full data lifecycle, normalization, sequencing |
| `docs/evaluation.md`    | Metrics, threshold analysis, ablation suite    |
| `docs/limitations.md`   | Accepted trade-offs and known issues           |
| `docs/decision_log.md`  | Authoritative locked decisions (D1–D25)        |
| `docs/state.md`         | Current project state snapshot                 |
