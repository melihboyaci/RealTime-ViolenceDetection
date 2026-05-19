"""
Run training + evaluation for a single ablation variant.

Usage:
    python scripts/run_ablation_eval.py <ablation_name> <data_dir>

Example:
    python scripts/run_ablation_eval.py norm_none data/sequences_ablation_norm_none

The script:
1. Loads sequences from <data_dir>/{train,val,test}/X_{split}.npy
2. Trains ViolenceGRU with the same schedule as baseline
3. Evaluates on test split
4. Prints results + delta vs blended baseline
5. Saves checkpoint + results to models/ablation_<name>.pt
"""

import os
import sys
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
    SEQUENCE_LENGTH, MODELS_DIR, DECISION_THRESHOLD,
)
from src.model import ViolenceGRU

# Blended baseline results for comparison
BASELINE = {"f1": 0.8197, "auc": 0.9272, "prec": 0.7749, "rec": 0.8700}


class AblationDataset(Dataset):
    def __init__(self, split, data_dir):
        x_path = os.path.join(data_dir, split, f"X_{split}.npy")
        y_path = os.path.join(data_dir, split, f"y_{split}.npy")
        self.X = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)
        self.feature_dim = self.X.shape[2]
        print(f"[{split.upper()}] shape={self.X.shape}  "
              f"V={int((self.y==1).sum())}  NV={int((self.y==0).sum())}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])


def run_ablation(name, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Ablation: {name}")
    print(f"Data dir: {data_dir}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Data
    train_ds = AblationDataset("train", data_dir)
    val_ds = AblationDataset("val", data_dir)
    test_ds = AblationDataset("test", data_dir)
    feature_dim = train_ds.feature_dim

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nFeature dim: {feature_dim}")

    # Model
    model = ViolenceGRU(input_size=feature_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR, min_lr=SCHEDULER_MIN_LR,
    )

    model_path = os.path.join(MODELS_DIR, f"ablation_{name}.pt")
    best_val_loss = float('inf')
    patience_counter = 0

    # Train
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
        scheduler.step(val_loss)

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            improved = " ★"
        else:
            patience_counter += 1

        print(f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Acc: {correct/total:.3f} | {time.time()-t0:.1f}s{improved}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Evaluate
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            all_probs.extend(model(Xb.to(device)).cpu().numpy().flatten())
            all_labels.extend(yb.numpy().flatten())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    t = DECISION_THRESHOLD
    preds = (all_probs >= t).astype(int)

    prec = precision_score(all_labels, preds, zero_division=0)
    rec = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    acc = (preds == all_labels).mean()
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    print(f"\n{'='*60}")
    print(f"Ablation [{name}] — Test Results (t={t})")
    print(f"{'='*60}")
    print(f"  Precision : {prec:.4f}  (baseline: {BASELINE['prec']:.4f}  Δ={prec-BASELINE['prec']:+.4f})")
    print(f"  Recall    : {rec:.4f}  (baseline: {BASELINE['rec']:.4f}  Δ={rec-BASELINE['rec']:+.4f})")
    print(f"  F1        : {f1:.4f}  (baseline: {BASELINE['f1']:.4f}  Δ={f1-BASELINE['f1']:+.4f})")
    print(f"  AUC-ROC   : {auc:.4f}  (baseline: {BASELINE['auc']:.4f}  Δ={auc-BASELINE['auc']:+.4f})")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    # Save
    result_path = os.path.join(MODELS_DIR, f"ablation_{name}_result.txt")
    with open(result_path, 'w') as f:
        f.write(f"Ablation: {name}\n")
        f.write(f"Data: {data_dir}\n")
        f.write(f"Feature dim: {feature_dim}\n")
        f.write(f"Threshold: {t}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1:        {f1:.4f}\n")
        f.write(f"AUC-ROC:   {auc:.4f}\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}\n")
        f.write(f"ΔF1:  {f1-BASELINE['f1']:+.4f}\n")
        f.write(f"ΔAUC: {auc-BASELINE['auc']:+.4f}\n")
    print(f"\nModel saved: {model_path}")
    print(f"Result saved: {result_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/run_ablation_eval.py <ablation_name> <data_dir>")
        print("Example: python scripts/run_ablation_eval.py norm_none data/sequences_ablation_norm_none")
        sys.exit(1)
    run_ablation(sys.argv[1], sys.argv[2])
