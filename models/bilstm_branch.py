# models/bilstm_branch.py
# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM branch: forecasts Dissolved Oxygen 12 hours ahead from a
# 24-hour window of water quality sensor readings.
#
# Architecture:
#   Input  : (batch, window_size=24, n_features=5)
#   BiLSTM : hidden=128, layers=2, bidirectional → output (batch, 24, 256)
#   Dropout: 0.3
#   Take last timestep → (batch, 256)
#   Linear : 256 → 1   → predicted DO (standardised scalar)
#
# Why Bidirectional?
#   Reading the 24-hour window BOTH forward (morning→night) and backward
#   (night→morning) lets the model capture:
#     Forward  → rising turbidity trend, daytime DO spike patterns
#     Backward → the nocturnal DO crash trajectory working backwards
#   Together they give a richer representation of the bloom cycle.
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    N_FEATURES, WINDOW_SIZE,
    BILSTM_HIDDEN, BILSTM_LAYERS, BILSTM_DROPOUT,
)


class BiLSTMForecaster(nn.Module):
    """
    Bidirectional LSTM for DO forecasting.

    Parameters
    ----------
    input_size  : int — number of input features (default: N_FEATURES = 5)
    hidden_size : int — hidden units per direction (default: 128)
    num_layers  : int — number of stacked LSTM layers (default: 2)
    dropout     : float — dropout between LSTM layers (default: 0.3)
    """

    def __init__(
        self,
        input_size:  int   = N_FEATURES,
        hidden_size: int   = BILSTM_HIDDEN,
        num_layers:  int   = BILSTM_LAYERS,
        dropout:     float = BILSTM_DROPOUT,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # ── Bidirectional LSTM ────────────────────────────────────────────
        # output size per timestep = hidden_size * 2 (forward + backward)
        self.bilstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        # ── Regularisation ────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=dropout)

        # ── Output layer ──────────────────────────────────────────────────
        # hidden_size * 2 because bidirectional
        self.fc = nn.Linear(hidden_size * 2, 1)

        self._init_weights()

    def _init_weights(self):
        """Xavier init for LSTM weights; zeros for biases."""
        for name, param in self.bilstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1.0 (helps long-term memory)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (batch, window_size, n_features)
            Standardised water quality readings over the past 24 hours.

        Returns
        -------
        torch.Tensor  shape (batch, 1)
            Predicted standardised DO value at t + forecast_horizon.
        """
        # x: (batch, T=24, F=5)
        lstm_out, _ = self.bilstm(x)     # (batch, T, hidden*2)
        last        = lstm_out[:, -1, :] # take last timestep: (batch, hidden*2)
        last        = self.dropout(last)
        return self.fc(last)             # (batch, 1)

    def predict_sequence(self, x: torch.Tensor, steps: int = 12) -> torch.Tensor:
        """
        Autoregressively predict DO for the next `steps` hours.
        Used for visualisation/evaluation only — not during training.

        Parameters
        ----------
        x     : (1, window_size, n_features) — single sample
        steps : number of future timesteps to predict

        Returns
        -------
        torch.Tensor shape (steps,) — predicted DO values
        """
        self.eval()
        predictions = []
        window = x.clone()  # (1, T, F)
        do_idx = 0          # DO is the first feature in TABULAR_FEATURES

        with torch.no_grad():
            for _ in range(steps):
                pred = self(window)        # (1, 1)
                predictions.append(pred.squeeze().item())

                # Slide window forward: drop oldest, append new step
                new_step = window[:, -1:, :].clone()
                new_step[:, 0, do_idx] = pred.squeeze()
                window = torch.cat([window[:, 1:, :], new_step], dim=1)

        return torch.tensor(predictions)


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = BiLSTMForecaster()
    model.eval()

    # Simulate a batch of 8 ponds, each with 24-hour WQ readings, 5 features
    dummy_batch = torch.randn(8, WINDOW_SIZE, N_FEATURES)

    with torch.no_grad():
        pred_do = model(dummy_batch)

    print("\n── BiLSTM Smoke Test ───────────────────────────────")
    print(f"  Input shape        : {dummy_batch.shape}")
    print(f"  Predicted DO shape : {pred_do.shape}")
    print(f"  Sample predictions : {pred_do[:4].squeeze().tolist()}")

    # Autoregressive forecast
    single = dummy_batch[:1]
    forecast = model.predict_sequence(single, steps=12)
    print(f"\n  12-hour forecast   : {[f'{v:.3f}' for v in forecast.tolist()]}")
    print("──────────────────────────────────────────────────── ✓")

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters   : {total:,}")
