# models/cnn_branch.py
# ─────────────────────────────────────────────────────────────────────────────
# Vision branch: classifies pond/water-surface photos into algal bloom
# severity (Low / Moderate / High risk) using a MobileNetV2 backbone
# pretrained on ImageNet, fine-tuned on the labeled `algal_blooms/` photo set.
#
# Architecture:
#   MobileNetV2 features (frozen) → GAP → Linear(1280 → 128) → ReLU → Dropout
#     → Linear(128 → 3)
#
# The 128-d embedding is exposed separately so this branch can later feed
# a multimodal fusion head alongside the BiLSTM branch (see models/risk_classifier.py).
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_RISK_CLASSES

EMBED_DIM = 128


class CNNBranch(nn.Module):
    """
    MobileNetV2-based image classifier for algal bloom severity.
    Backbone is frozen; only the embedding + classification heads train.
    """

    def __init__(
        self,
        embed_dim: int   = EMBED_DIM,
        dropout: float   = 0.3,
        num_classes: int = NUM_RISK_CLASSES,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features  # (B, 1280, 7, 7) for 224x224 input
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.embed_head = nn.Sequential(
            nn.Linear(1280, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for layer in [*self.embed_head, self.classifier]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, 224, 224) → embedding (B, embed_dim)"""
        x = self.features(images)
        x = self.pool(x).flatten(1)
        return self.embed_head(x)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, 224, 224) → logits (B, num_classes)"""
        return self.classifier(self.embed(images))

    def predict(self, images: torch.Tensor) -> tuple:
        """Convenience method: returns (class_index, probability, label_string)."""
        label_names = ["Low", "Moderate", "High"]
        with torch.no_grad():
            logits = self.forward(images)
            probs = torch.softmax(logits, dim=1)
            ids = probs.argmax(dim=1)
        labels = [label_names[i.item()] for i in ids]
        return ids, probs, labels


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = CNNBranch()
    model.eval()

    B = 4
    images = torch.randn(B, 3, 224, 224)

    with torch.no_grad():
        logits = model(images)
        embedding = model.embed(images)
        ids, probs, labels = model.predict(images)

    print("\n── CNNBranch Smoke Test ────────────────────────────")
    print(f"  input shape     : {images.shape}")
    print(f"  embedding shape : {embedding.shape}")
    print(f"  logits shape    : {logits.shape}")
    print("\n  Per-sample results:")
    for i in range(B):
        print(f"    Sample {i+1}: {labels[i]:>8} Risk (conf: {probs[i].max().item():.1%})")
    print("──────────────────────────────────────────────────── ✓")
