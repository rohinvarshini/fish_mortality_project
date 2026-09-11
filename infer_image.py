# infer_image.py
# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI: classify algal bloom severity risk from a single photo,
# using the trained CNN vision branch (checkpoints/cnn_best.pt).
#
# Usage:
#   python infer_image.py path/to/photo.jpg
# ─────────────────────────────────────────────────────────────────────────────

import sys
import os
import torch
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DEVICE, CHECKPOINTS
from models.cnn_branch import CNNBranch

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image_path: str):
    checkpoint_path = os.path.join(CHECKPOINTS, "cnn_best.pt")
    if not os.path.exists(checkpoint_path):
        print("[!] No trained CNN checkpoint found at checkpoints/cnn_best.pt")
        print("    Run: python training/train_cnn.py")
        sys.exit(1)

    model = CNNBranch(pretrained=False, freeze_backbone=True).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = probs.argmax().item()

    label_names = ["Low", "Moderate", "High"]
    print(f"\n  Photo: {image_path}")
    print(f"  Risk : {label_names[pred_id]}  (confidence {probs[pred_id]:.1%})")
    print(f"  Probabilities: Low={probs[0]:.1%}  Moderate={probs[1]:.1%}  High={probs[2]:.1%}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infer_image.py path/to/photo.jpg")
        sys.exit(1)
    predict(sys.argv[1])
