# models/risk_classifier.py
# ─────────────────────────────────────────────────────────────────────────────
# Tabular risk classifier: combines the current Water Quality features (5-d)
# and the forecasted DO (1-d) to predict Low / Moderate / High mortality risk.
#
# Inputs:
#   - Current WQ sensors: DO, pH, Temperature, Turbidity, Ammonia (5-d)
#   - BiLSTM Forecasted DO (1-d)
#   - Concatenated Input: 6-d
#
# Architecture:
#   Linear(6 → 64) → BatchNorm1d → ReLU → Dropout(0.4) → Linear(64 → 3)
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLASSIFIER_HIDDEN, CLASSIFIER_DROPOUT, NUM_RISK_CLASSES


class RiskClassifier(nn.Module):
    """
    MLP Classifier for predicting fish mortality risk.
    Inputs are tabular only.
    """

    def __init__(
        self,
        input_dim: int       = 6,  # 5 WQ features + 1 predicted DO
        hidden_dim: int      = CLASSIFIER_HIDDEN,
        dropout: float       = CLASSIFIER_DROPOUT,
        num_classes: int     = NUM_RISK_CLASSES,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        current_wq: torch.Tensor,
        predicted_do: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        current_wq : torch.Tensor  shape (batch, 5)
            The WQ features at the latest timestep.
        predicted_do : torch.Tensor  shape (batch, 1)
            The forecasted DO output from the BiLSTM forecaster.

        Returns
        -------
        torch.Tensor  shape (batch, 3)
            Logits for [Low, Moderate, High] risk classes.
        """
        if predicted_do.dim() == 1:
            predicted_do = predicted_do.unsqueeze(1)

        # Concatenate current features and predicted DO -> (batch, 6)
        x = torch.cat([current_wq, predicted_do], dim=1)
        return self.net(x)

    def predict(
        self,
        current_wq: torch.Tensor,
        predicted_do: torch.Tensor,
    ) -> tuple:
        """
        Convenience method: returns (class_index, probability, label_string).
        """
        label_names = ["Low", "Moderate", "High"]
        with torch.no_grad():
            logits = self.forward(current_wq, predicted_do)
            probs  = torch.softmax(logits, dim=1)
            ids    = probs.argmax(dim=1)
        labels = [label_names[i.item()] for i in ids]
        return ids, probs, labels


class FullTabularPipeline(nn.Module):
    """
    End-to-end inference pipeline wrapping the BiLSTM and the Risk Classifier.
    Takes the 24h WQ sequence (batch, 24, 5) as input.
    """

    def __init__(self, bilstm_model, classifier_model):
        super().__init__()
        self.bilstm     = bilstm_model
        self.classifier = classifier_model

    def forward(self, wq_sequence: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        wq_sequence : torch.Tensor  shape (batch, 24, 5)
            Sequence of WQ readings.

        Returns
        -------
        torch.Tensor  shape (batch, 3)
            Mortality risk logits.
        """
        # Forecast DO 12h ahead -> (batch, 1)
        pred_do = self.bilstm(wq_sequence)

        # Latest timestep features -> (batch, 5)
        current_wq = wq_sequence[:, -1, :]

        # Predict Risk -> (batch, 3)
        return self.classifier(current_wq, pred_do)


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    from models.bilstm_branch import BiLSTMForecaster
    from config import WINDOW_SIZE, N_FEATURES

    bilstm = BiLSTMForecaster()
    classifier = RiskClassifier()

    bilstm.eval()
    classifier.eval()

    # Simulate batch of 4 ponds
    B = 4
    wq_sequence = torch.randn(B, WINDOW_SIZE, N_FEATURES)

    with torch.no_grad():
        pred_do = bilstm(wq_sequence)  # (4, 1)
        current_wq = wq_sequence[:, -1, :]  # (4, 5)
        logits = classifier(current_wq, pred_do)  # (4, 3)
        ids, probs, labels = classifier.predict(current_wq, pred_do)

    print("\n── RiskClassifier Smoke Test ───────────────────────")
    print(f"  wq_sequence shape : {wq_sequence.shape}")
    print(f"  current_wq shape  : {current_wq.shape}")
    print(f"  predicted_do shape: {pred_do.shape}")
    print(f"  logits shape      : {logits.shape}")
    print("\n  Per-sample results:")
    for i in range(B):
        print(f"    Pond {i+1}: {labels[i]:>8} Risk (conf: {probs[i].max().item():.1%})")

    # End-to-end pipeline test
    pipeline = FullTabularPipeline(bilstm, classifier)
    with torch.no_grad():
        out = pipeline(wq_sequence)
    print(f"\n  FullTabularPipeline output shape: {out.shape}")
    print("──────────────────────────────────────────────────── ✓")
