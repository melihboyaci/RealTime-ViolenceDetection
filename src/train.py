"""
Training script for the ViolenceGRU model.

Locked decisions:
    D15: BCELoss
    D16: Adam optimizer
    D17: max 100 epochs, batch 32, early stopping on val_loss,
         best checkpoint by lowest val_loss
"""

import os
import sys
import csv
import time
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_MAX_NORM,
    SCHEDULER_PATIENCE,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    BEST_MODEL_PATH,
    MODELS_DIR,
    SEQUENCES_DIR,
)
from src.model import ViolenceGRU
from src.dataset import create_dataloaders


def train():
    """Main training function."""

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, _ = create_dataloaders(
        data_dir=SEQUENCES_DIR,
        batch_size=BATCH_SIZE,
    )

    # ── Model ─────────────────────────────────────────────────
    model = ViolenceGRU().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Loss, Optimizer, Scheduler ────────────────────────────
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
        min_lr=SCHEDULER_MIN_LR,
    )

    # ── Training State ────────────────────────────────────────
    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Training log CSV
    log_path = os.path.join(MODELS_DIR, "training_log.csv")
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr", "time_sec"])

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Max epochs: {MAX_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"  Gradient clip: {GRADIENT_CLIP_MAX_NORM}")
    print(f"{'='*60}\n")

    # ── Training Loop ─────────────────────────────────────────
    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()

        # ── Train Phase ───────────────────────────────────────
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)       # (batch, 1)
            loss = criterion(predictions, y_batch)
            loss.backward()

            # Gradient clipping (exploding gradient koruması)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM
            )

            optimizer.step()
            train_losses.append(loss.item())

        # ── Validation Phase ──────────────────────────────────
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_losses.append(loss.item())

                # Accuracy (threshold 0.5 for val monitoring)
                predicted_labels = (predictions >= 0.5).float()
                val_correct += (predicted_labels == y_batch).sum().item()
                val_total += y_batch.size(0)

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        # ── Scheduler ─────────────────────────────────────────
        scheduler.step(val_loss)

        # ── Checkpoint ────────────────────────────────────────
        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            improved = " ★ saved"
        else:
            patience_counter += 1

        # ── Log ───────────────────────────────────────────────
        print(
            f"Epoch {epoch+1:3d}/{MAX_EPOCHS} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Acc: {val_acc:.3f} | "
            f"LR: {current_lr:.2e} | "
            f"{epoch_time:.1f}s{improved}"
        )

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}",
                f"{val_acc:.4f}", f"{current_lr:.2e}", f"{epoch_time:.1f}"
            ])

        # ── Early Stopping ────────────────────────────────────
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(patience={EARLY_STOPPING_PATIENCE})")
            break

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best val_loss: {best_val_loss:.6f}")
    print(f"Model saved: {BEST_MODEL_PATH}")
    print(f"Log saved: {log_path}")
    print(f"{'='*60}")

    return best_val_loss


if __name__ == "__main__":
    train()
