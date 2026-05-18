"""
PyTorch Dataset for loading preprocessed sequence .npy files.

Expected data layout (from Kaggle output):
    data/sequences/train/X_train.npy  — shape (N, 30, 69)
    data/sequences/train/y_train.npy  — shape (N,)
    data/sequences/val/X_val.npy
    data/sequences/val/y_val.npy
    data/sequences/test/X_test.npy
    data/sequences/test/y_test.npy
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    SEQUENCES_DIR,
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    FEATURE_DIM,
)


class ViolenceDataset(Dataset):
    """
    Loads (30, 69) sequence windows and binary labels from .npy files.
    """

    def __init__(self, split: str, data_dir: str = SEQUENCES_DIR):
        """
        Args:
            split: 'train', 'val', or 'test'
            data_dir: base directory containing sequences/{split}/
        """
        x_path = os.path.join(data_dir, split, f"X_{split}.npy")
        y_path = os.path.join(data_dir, split, f"y_{split}.npy")

        if not os.path.exists(x_path):
            raise FileNotFoundError(
                f"Sequence file not found: {x_path}\n"
                f"Kaggle Notebook çıktılarını data/sequences/ altına kopyaladınız mı?"
            )

        self.X = np.load(x_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)

        assert self.X.shape[1] == SEQUENCE_LENGTH, (
            f"Expected sequence length {SEQUENCE_LENGTH}, got {self.X.shape[1]}"
        )
        assert self.X.shape[2] == FEATURE_DIM, (
            f"Expected feature dim {FEATURE_DIM}, got {self.X.shape[2]}"
        )

        print(f"[{split.upper()}] Loaded {len(self.X)} sequences, "
              f"shape={self.X.shape}, "
              f"Violence={int((self.y == 1).sum())}, "
              f"NonViolence={int((self.y == 0).sum())}")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.FloatTensor(self.X[idx])      # (30, 69)
        y = torch.FloatTensor([self.y[idx]])     # (1,)
        return x, y


def create_dataloaders(
    data_dir: str = SEQUENCES_DIR,
    batch_size: int = BATCH_SIZE,
):
    """
    Creates train, val, test DataLoaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = ViolenceDataset("train", data_dir)
    val_dataset = ViolenceDataset("val", data_dir)
    test_dataset = ViolenceDataset("test", data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
