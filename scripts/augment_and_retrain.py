"""
Data augmentation + retrain script.

Augmentation strategies (applied only to TRAIN):
1. Gaussian jitter: small noise to skeleton features
2. Temporal shift: shift sequence by 1-3 frames (with zero padding)
3. Original data kept as-is

Val/Test remain untouched.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.config import SEQUENCES_DIR, FEATURE_DIM, SEQUENCE_LENGTH

# ── Load original data ───────────────────────────────────────
X_train = np.load(os.path.join(SEQUENCES_DIR, "train", "X_train.npy"))
y_train = np.load(os.path.join(SEQUENCES_DIR, "train", "y_train.npy"))

print(f"Original train: {X_train.shape}, V={int((y_train==1).sum())}, NV={int((y_train==0).sum())}")

np.random.seed(42)

# ── Augmentation 1: Gaussian Jitter ──────────────────────────
# Add small noise to skeleton features (0-67), leave feature 68 (distance) alone
def gaussian_jitter(X, noise_std=0.02):
    """Add Gaussian noise to skeleton features only."""
    X_aug = X.copy()
    noise = np.random.normal(0, noise_std, X_aug[:, :, :68].shape).astype(np.float32)
    X_aug[:, :, :68] += noise
    return X_aug

# ── Augmentation 2: Temporal Shift ───────────────────────────
def temporal_shift(X, max_shift=3):
    """Shift sequences forward by 1-3 frames, pad with zeros."""
    X_aug_list = []
    for seq in X:
        shift = np.random.randint(1, max_shift + 1)
        shifted = np.zeros_like(seq)
        shifted[shift:] = seq[:-shift]
        X_aug_list.append(shifted)
    return np.array(X_aug_list, dtype=np.float32)

# ── Augmentation 3: Reverse temporal order ───────────────────
def temporal_reverse(X):
    """Reverse the frame order in each sequence."""
    return X[:, ::-1, :].copy()

# ── Apply augmentations ──────────────────────────────────────
X_jitter = gaussian_jitter(X_train, noise_std=0.02)
y_jitter = y_train.copy()

X_shift = temporal_shift(X_train, max_shift=3)
y_shift = y_train.copy()

X_reverse = temporal_reverse(X_train)
y_reverse = y_train.copy()

# ── Combine ──────────────────────────────────────────────────
X_augmented = np.concatenate([X_train, X_jitter, X_shift, X_reverse], axis=0)
y_augmented = np.concatenate([y_train, y_jitter, y_shift, y_reverse], axis=0)

# Shuffle
perm = np.random.permutation(len(X_augmented))
X_augmented = X_augmented[perm]
y_augmented = y_augmented[perm]

print(f"Augmented train: {X_augmented.shape}, V={int((y_augmented==1).sum())}, NV={int((y_augmented==0).sum())}")
print(f"Augmentation ratio: {len(X_augmented) / len(X_train):.1f}x")

# ── Save augmented data ──────────────────────────────────────
aug_dir = os.path.join(SEQUENCES_DIR, "train")
np.save(os.path.join(aug_dir, "X_train_orig.npy"), X_train)
np.save(os.path.join(aug_dir, "y_train_orig.npy"), y_train)
np.save(os.path.join(aug_dir, "X_train.npy"), X_augmented)
np.save(os.path.join(aug_dir, "y_train.npy"), y_augmented)

print(f"\nOriginal data backed up as X_train_orig.npy / y_train_orig.npy")
print(f"Augmented data saved as X_train.npy / y_train.npy")
print(f"\nReady for retraining!")
