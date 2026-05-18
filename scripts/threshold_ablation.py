"""Threshold ablation study on the test split."""
import os, sys, torch, numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.config import BEST_MODEL_PATH, SEQUENCES_DIR, BATCH_SIZE
from src.model import ViolenceGRU
from src.dataset import create_dataloaders

device = torch.device("cpu")
model = ViolenceGRU().to(device)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device, weights_only=True))
model.eval()

_, _, test_loader = create_dataloaders(data_dir=SEQUENCES_DIR, batch_size=BATCH_SIZE)

all_probs, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        probs = model(X_batch).numpy().flatten()
        labels = y_batch.numpy().flatten()
        all_probs.extend(probs)
        all_labels.extend(labels)

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
auc = roc_auc_score(all_labels, all_probs)

print(f"AUC-ROC: {auc:.4f}\n")
header = f"{'Thresh':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Acc':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'TN':>5s}"
print(header)
print("-" * len(header))

best_f1, best_t = 0, 0
for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    preds = (all_probs >= t).astype(int)
    p = precision_score(all_labels, preds, zero_division=0)
    r = recall_score(all_labels, preds, zero_division=0)
    f1 = f1_score(all_labels, preds, zero_division=0)
    acc = (preds == all_labels).mean()
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    marker = "  <-- best" if f1 > best_f1 else ""
    if f1 > best_f1:
        best_f1, best_t = f1, t
    print(f"{t:8.2f} {p:8.4f} {r:8.4f} {f1:8.4f} {acc:8.4f} {tp:5d} {fp:5d} {fn:5d} {tn:5d}{marker}")

print(f"\nBest F1: {best_f1:.4f} at threshold={best_t}")
