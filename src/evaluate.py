"""
Evaluation script for the ViolenceGRU model.

Produces: Confusion Matrix, Precision, Recall, F1, AUC-ROC on the test split.
Violence is the positive class (y=1).

Locked decisions:
    D10/D18: Initial threshold 0.7
    D9:      Test split is strictly isolated
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    BEST_MODEL_PATH,
    DECISION_THRESHOLD,
    SEQUENCES_DIR,
    BATCH_SIZE,
    MODELS_DIR,
)
from src.model import ViolenceGRU
from src.dataset import create_dataloaders


def evaluate(threshold: float = DECISION_THRESHOLD):
    """Run evaluation on the test split."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load Model ────────────────────────────────────────────
    model = ViolenceGRU().to(device)
    model.load_state_dict(
        torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"Model loaded: {BEST_MODEL_PATH}")

    # ── Load Test Data ────────────────────────────────────────
    _, _, test_loader = create_dataloaders(
        data_dir=SEQUENCES_DIR,
        batch_size=BATCH_SIZE,
    )

    # ── Inference ─────────────────────────────────────────────
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            probs = model(X_batch).cpu().numpy()      # (batch, 1)
            labels = y_batch.cpu().numpy()              # (batch, 1)
            all_probs.extend(probs.flatten().tolist())
            all_labels.extend(labels.flatten().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ── Metrics ───────────────────────────────────────────────
    predictions = (all_probs >= threshold).astype(int)

    cm = confusion_matrix(all_labels, predictions)
    precision = precision_score(all_labels, predictions, zero_division=0)
    recall = recall_score(all_labels, predictions, zero_division=0)
    f1 = f1_score(all_labels, predictions, zero_division=0)
    auc_roc = roc_auc_score(all_labels, all_probs)

    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS (threshold={threshold})")
    print(f"{'='*60}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 V       NV")
    print(f"  Actual  V    [{tp:5d}  | {fn:5d}]")
    print(f"          NV   [{fp:5d}  | {tn:5d}]")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print(f"\nTotal samples: {len(all_labels)}")
    print(f"Violence: {int((all_labels == 1).sum())}")
    print(f"NonViolence: {int((all_labels == 0).sum())}")

    print(f"\n{classification_report(all_labels, predictions, target_names=['NonViolence', 'Violence'])}")

    # ── Save Results ──────────────────────────────────────────
    results_path = os.path.join(MODELS_DIR, "evaluation_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"AUC-ROC: {auc_roc:.4f}\n")
        f.write(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n")
    print(f"\nResults saved: {results_path}")

    # ── Plot Training Curves ──────────────────────────────────
    log_path = os.path.join(MODELS_DIR, "training_log.csv")
    if os.path.exists(log_path):
        import pandas as pd
        log_df = pd.read_csv(log_path)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(log_df['epoch'], log_df['train_loss'], label='Train Loss')
        ax1.plot(log_df['epoch'], log_df['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(log_df['epoch'], log_df['val_acc'], label='Val Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(MODELS_DIR, "training_curves.png")
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Training curves saved: {fig_path}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_roc": auc_roc,
        "threshold": threshold,
    }


if __name__ == "__main__":
    evaluate()
