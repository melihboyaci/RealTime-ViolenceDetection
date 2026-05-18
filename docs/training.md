# training.md

Practical training playbook for the GRU sequence classifier.

## 1. Training Loop Overview

```text
for epoch in range(MAX_EPOCHS):
    for batch in train_loader:           # batch shape: (32, 30, 69)
        x, y = batch
        p = model(x)                     # sigmoid output, shape (32,)
        loss = BCELoss(p, y)             # binary labels: Violence=1, NonViolence=0
        loss.backward()
        optimizer.step()                 # Adam
        optimizer.zero_grad()

    val_loss = evaluate(model, val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)           # best-by-lowest-val_loss

    if early_stopping_triggered(val_loss):
        break
```

## 2. Batch Size

- **32** sequences per batch.
- Each sequence is `(30, 69)`, so a batch tensor is `(32, 30, 69)`.

## 3. Epoch Strategy

- **Max epochs**: **100**.
- **Early stopping** on **`val_loss`**.
- Patience and min-delta for early stopping: **Not specified in the source PDF** (TBD; record in `decision_log.md`).

## 4. Early Stopping

- Monitor: **`val_loss`** (not val accuracy, not F1).
- Direction: minimize.
- Trigger: when `val_loss` does not improve for `patience` consecutive epochs (patience **TBD**).
- On trigger: stop training, **load** the best-by-lowest-val_loss checkpoint, do not save the latest weights.

## 5. Best Checkpoint Logic

- Save a checkpoint **only when `val_loss` improves**.
- The saved checkpoint is the **single source of truth** for evaluation and online inference.
- Checkpoint file content (recommended; layout TBD):
  - model state dict
  - hyperparameters used in this run
  - best `val_loss` and the epoch it was reached
  - data split hash / seed
  - feature builder version (so offline/online stay aligned)

## 6. Validation Protocol

- Run validation **once per epoch**, after the train pass.
- Use the **val split** (15% of dataset, stratified).
- Compute at least: `val_loss`. Other metrics (precision, recall, F1, AUC-ROC) are recommended but only `val_loss` drives early stopping and checkpoint selection.
- The **test split is never used** during training or validation.

## 7. Overfitting Signals

Watch for any of the following during training:

- `train_loss` continues decreasing while `val_loss` plateaus or rises for several epochs.
- Train F1/AUC keeps climbing while val F1/AUC stalls.
- Class-conditional precision / recall on the val split diverge sharply from train.
- Sudden drop in `val_loss` followed by a noisy rebound (suggests overfitting on a noisy minibatch).

If observed, prefer (in order): more aggressive early stopping → dropout tuning → motion-filter θ review → data augmentation strategies (TBD).

## 8. Tuning Levers

| Lever | Effect | Risk if changed silently |
|---|---|---|
| Learning rate | Convergence speed and stability | Diverging training; must be re-ablated |
| GRU hidden size / depth | Capacity | Real-time budget impact |
| Dropout | Generalization | Underfitting if too high |
| Batch size | Gradient noise / memory | Changing it invalidates direct checkpoint comparisons |
| Motion-filter θ | Class balance + label noise | Changes the dataset; full re-train + re-eval needed |
| Decision threshold (post-train) | Precision/recall trade-off | Does **not** require retraining; tune on val, freeze before test |

## 9. Recommended Experiment Naming Convention

```text
exp/<YYYYMMDD>-<short-description>/<seed>/
  config.yaml
  checkpoints/
    best.pt
  metrics/
    train.csv
    val.csv
  logs/
    stdout.log
```

Examples:

- `exp/20260518-baseline/0/`
- `exp/20260518-motion-theta-0.03/0/`
- `exp/20260518-no-interaction-feature/0/`

## 10. Reproducibility Notes

- Persist:
  - random seed,
  - split hash,
  - feature builder version,
  - all hyperparameters (LR, hidden size, depth, dropout, batch, max epochs, early-stopping patience, motion-filter θ).
- Pin library versions for OpenCV, Ultralytics (YOLOv8n-Pose), PyTorch.
- Determinism level expected: **TBD** (bit-exact vs. seed-only).

## 11. What Not to Touch Without Re-Running Ablations

These are baseline-defining choices. If any of them changes, the corresponding ablation **must be re-run** and `decision_log.md` updated:

- Target FPS = **10**
- Frame ops order: conditional CLAHE → Gaussian 3×3 → 640×640 → BGR→RGB
- Pose model = **YOLOv8n-Pose**, 17 keypoints, conf cutoff **0.5**
- Multi-person policy: top 2, **X-axis sort**, normalized bbox center distance
- Per-frame feature dim = **69**
- Normalization: hip centering + shoulder–hip scaling
- Sequence window = **30 frames**, sliding
- Motion filter θ = **0.05** (Violence only)
- Loss = **BCELoss**, optimizer = **Adam**
- Batch = **32**, max epochs = **100**, early stopping on **val_loss**, best by **lowest val_loss**
- Initial decision threshold = **0.7**
- Split = stratified **70 / 15 / 15**

## 12. Assumptions / Open Questions

- Early-stopping patience and min-delta: **TBD.**
- Learning rate, weight decay, betas: **Not specified in the source PDF.**
- Class-balancing inside the train loader: **TBD.**
- Whether to log metrics to TensorBoard / W&B / CSV-only: **TBD.**
