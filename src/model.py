"""
GRU-based binary sequence classifier for Violence / NonViolence detection.

Architecture (from model.md and decision_log.md D8):
    Input (batch, 30, 69)
    → GRU Layer 1: 128 units, return_sequences=True
    → Dropout(0.3)
    → GRU Layer 2: 64 units, return_sequences=False
    → Dropout(0.3)
    → Dense: Linear(64→32) + ReLU
    → Output: Linear(32→1) + Sigmoid → P(Violence) ∈ [0, 1]
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from configs.config import (
    GRU_INPUT_SIZE,
    GRU_HIDDEN_SIZE_1,
    GRU_HIDDEN_SIZE_2,
    DROPOUT_RATE,
    DENSE_UNITS,
)


class ViolenceGRU(nn.Module):
    """
    Two-layer GRU sequence classifier with sigmoid output.

    Locked decisions:
        - D8:  GRU (not LSTM)
        - D13: input_size = 69
        - D15: BCELoss (sigmoid output)
    """

    def __init__(
        self,
        input_size: int = GRU_INPUT_SIZE,
        hidden_size_1: int = GRU_HIDDEN_SIZE_1,
        hidden_size_2: int = GRU_HIDDEN_SIZE_2,
        dropout_rate: float = DROPOUT_RATE,
        dense_units: int = DENSE_UNITS,
    ):
        super().__init__()

        self.gru_layer_1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout_1 = nn.Dropout(p=dropout_rate)

        self.gru_layer_2 = nn.GRU(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout_2 = nn.Dropout(p=dropout_rate)

        self.dense = nn.Linear(hidden_size_2, dense_units)
        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, sequence_length, feature_dim) — e.g. (B, 30, 69)

        Returns:
            (batch_size, 1) — sigmoid probability of Violence
        """
        # GRU Layer 1: (B, 30, 69) → (B, 30, 128)
        out, _ = self.gru_layer_1(x)
        out = self.dropout_1(out)

        # GRU Layer 2: (B, 30, 128) → last step only → (B, 64)
        out, _ = self.gru_layer_2(out)
        out = out[:, -1, :]  # return_sequences=False equivalent
        out = self.dropout_2(out)

        # Dense: (B, 64) → (B, 32)
        out = self.dense(out)
        out = self.relu(out)

        # Output: (B, 32) → (B, 1)
        out = self.output_layer(out)
        out = self.sigmoid(out)

        return out


if __name__ == "__main__":
    model = ViolenceGRU()
    print(f"Model:\n{model}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    dummy_input = torch.randn(32, 30, 69)
    output = model(dummy_input)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
