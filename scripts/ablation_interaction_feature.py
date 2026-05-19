"""
P7.2 — Interaction-Feature Ablation Study

Removes the normalized bbox center distance (last feature, dim index 68)
from the 69-dim feature vector, resulting in a 68-dim input.

Goal: Quantify the contribution of the two-person interaction signal.
Protocol: same training schedule as baseline, evaluated on isolated test split.
"""

import os
import sys
import csv
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE, GRADIENT_CLIP_MAX_NORM,
    SCHEDULER_PATIENCE, SCHEDULER_FACTOR, SCHEDULER_MIN_LR,
    GRU_HIDDEN_SIZE_1, GRU_HIDDEN_SIZE_2, DROPOUT_RATE, DENSE_UNITS,
    SEQUENCES_DIR, MODELS_DIR, DECISION_THRESHOLD,
)
from src.model import ViolenceGRU

ABLATION_INPUT_DIM = 68  # 69 - 1 (interaction distance removed)
ABLATION_MODEL_PATH = os.path.join(MODELS_DIR, "ablation_no_interaction.pt")
EVAL_THRESHOLD = DECISION_THRESHOLD


class AblationDataset(Dataset):
    """Loads sequences and drops the last feature (interaction distance)."""

    def __init__(self, split: str, data_dir: str = SEQUENCES_DIR):
        x_path = os.path.join(data_dir, split, f"X_{split}.npy")
        y_path = os.path.join(data_dir, split, f"y_{split}.npy")
        X = np.load(x_path).astype(np.float32)
        self.X = X[:, :, :ABLATION_INPUT_DIM]  # drop last feature
        self.y = np.load(y_path).astype(np.float32)
        print(f"[{split.upper()}] shape={self.X.shape}  "
              f"V={int((self.y==1).sum())}  NV={int((self.y==0).sum())}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.FloatTensor([self.y[idx]]),
        )


def make_loader(split, shuffle=False):
    ds = AblationDataset(split)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


def train_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Input dim: {ABLATION_INPUT_DIM} (interaction feature REMOVED)\n")

    train_loader = make_loader("train", shuffle=True)
    val_loader = make_loader("val")
    test_loader = make_loader("test")

    model = ViolenceGRU(input_size=ABLATION_INPUT_DIM).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR, min_lr=SCHEDULER_MIN_LR,
    )

    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"{'='*60}")
    print("Training (no-interaction ablation)")
    print(f"{'='*60}")

    for epoch in range(MAX_EPOCHS):
        t0 = time.time()
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_MAX_NORM)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                preds = model(Xb)
                val_losses.append(criterion(preds, yb).item())
                correct += ((preds >= 0.5) == yb).sum().item()
                total += yb.size(0)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = correct / total
        scheduler.step(val_loss)
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ABLATION_MODEL_PATH)
            improved = " ★ saved"
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Acc: {val_acc:.3f} | {time.time()-t0:.1f}s{improved}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # ── Evaluate on Test ──────────────────────────────────────────
    model.load_state_dict(torch.load(ABLATION_MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            all_probs.extend(model(Xb.to(device)).cpu().numpy().flatten())
            all_labels.extend(yb.numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs >= EVAL_THRESHOLD).astype(int)

    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    acc = (preds == all_labels).mean()
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    print(f"\n{'='*60}")
    print(f"P7.2 No-Interaction Ablation — Test Results (t={EVAL_THRESHOLD})")
    print(f"{'='*60}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"\nBaseline (blended, with-interaction, t=0.45):")
    print(f"  F1=0.8197  AUC=0.9272  Prec=0.7749  Recall=0.8700")
    delta_f1 = f1 - 0.8197
    delta_auc = auc - 0.9272
    print(f"\nDelta vs baseline:  ΔF1={delta_f1:+.4f}  ΔAUC={delta_auc:+.4f}")

    # Save result
    result_path = os.path.join(MODELS_DIR, "ablation_no_interaction_result.txt")
    with open(result_path, 'w') as f:
        f.write(f"P7.2 Interaction-Feature Ablation\n")
        f.write(f"Input dim: {ABLATION_INPUT_DIM} (interaction distance REMOVED)\n")
        f.write(f"Threshold: {EVAL_THRESHOLD}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1:        {f1:.4f}\n")
        f.write(f"AUC-ROC:   {auc:.4f}\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}\n")
        f.write(f"DeltaF1 vs baseline: {delta_f1:+.4f}\n")
    print(f"Result saved: {result_path}")

    return {"prec": prec, "rec": rec, "f1": f1, "auc": auc, "acc": acc}


if __name__ == "__main__":
    train_ablation()
