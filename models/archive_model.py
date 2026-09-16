import os
import pickle
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_NAMES = ["DO", "temperature", "pH", "turbidity", "ammonia", "fish_weight"]
RISK_LABELS = ["Low", "Moderate", "High"]
WINDOW_SIZE = 24


class ArchiveRiskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=6,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(self, sequence):
        output, _ = self.lstm(sequence)
        return self.head(output[:, -1, :])


class _ScalerUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_scaler(path):
    with open(path, "rb") as scaler_file:
        return _ScalerUnpickler(scaler_file).load()


def load_scaler(path):
    return _load_scaler(path)


@lru_cache(maxsize=2)
def load_archive_model(interval):
    if interval not in {"5min", "30min"}:
        raise ValueError("interval must be '5min' or '30min'")

    model_dir = os.path.join(BASE_DIR, f"archive_{interval}", "trained_model")
    checkpoint_path = os.path.join(model_dir, "fish_mortality_lstm_model.pth")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    model = ArchiveRiskModel()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, _load_scaler(scaler_path)


def predict_archive(interval, values):
    model, scaler = load_archive_model(interval)
    raw_values = np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
    scaled_values = scaler.transform(raw_values.reshape(1, -1))[0]
    sequence = np.tile(scaled_values, (WINDOW_SIZE, 1))
    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]

    predicted_id = int(probabilities.argmax().item())
    return {
        "interval": interval,
        "risk_label": RISK_LABELS[predicted_id],
        "confidence": float(probabilities[predicted_id].item()),
        "all_probs": {
            label: float(probabilities[index].item())
            for index, label in enumerate(RISK_LABELS)
        },
    }