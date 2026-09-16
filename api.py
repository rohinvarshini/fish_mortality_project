from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import io
import shap
from PIL import Image
from torchvision import transforms

# Suppress SHAP progress bar noise
import warnings
warnings.filterwarnings("ignore")

# Append project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from models.archive_model import load_scaler

app = FastAPI(title="Fish Mortality Prediction API")

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WaterQualityInput(BaseModel):
    DO: float
    pH: float
    temperature: float
    turbidity: float
    ammonia: float
    fish_weight: float = 0.0  # accepted for UI compatibility; not used by the real pipeline


class ArchivePredictionInput(WaterQualityInput):
    interval: str = "5min"

# ── Load Scaler (fit on the 5 real TABULAR_FEATURES) ───────────────────────────
scaler_path = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")
scaler = load_scaler(scaler_path)
scale_mean = scaler.mean_
scale_std = scaler.scale_

# ── Load Real Trained Pipeline: BiLSTM forecaster + Risk Classifier ───────────
from models.bilstm_branch import BiLSTMForecaster
from models.risk_classifier import RiskClassifier, FullTabularPipeline
from training.utils import load_checkpoint
from config import WINDOW_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bilstm = BiLSTMForecaster().to(device)
load_checkpoint(bilstm, "bilstm_best", device=device)
bilstm.eval()

classifier = RiskClassifier().to(device)
load_checkpoint(classifier, "classifier_best", device=device)
classifier.eval()

pipeline = FullTabularPipeline(bilstm, classifier).to(device)
pipeline.eval()

# ── Feature Names ─────────────────────────────────────────────────────────────
FEATURE_NAMES = ["DO (mg/L)", "pH", "Temperature (C)", "Turbidity (NTU)", "Ammonia (mg/L)"]

# ── Load CNN Vision Branch (algal bloom severity from photo) ──────────────────
from models.cnn_branch import CNNBranch

cnn_model = CNNBranch(pretrained=False, freeze_backbone=True).to(device)
cnn_checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "cnn_best.pt")
cnn_ready = False
if os.path.exists(cnn_checkpoint_path):
    cnn_state = torch.load(cnn_checkpoint_path, map_location=device, weights_only=False)
    cnn_model.load_state_dict(cnn_state["model"])
    cnn_ready = True
cnn_model.eval()

from explainability.risk_guidance import get_guidance

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── SHAP: Prediction function (maps 5 raw features → 3 class probs) ───────────
def predict_fn_shap(inputs: np.ndarray) -> np.ndarray:
    """
    inputs: (n, 5) raw (unscaled) sensor values
    returns: (n, 3) softmax probabilities [Low, Moderate, High]
    """
    results = []
    for row in inputs:
        scaled = (row - scale_mean) / scale_std
        seq = np.tile(scaled, (WINDOW_SIZE, 1))  # (24, 5) — steady-state assumption
        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = pipeline(tensor)
            probs = torch.softmax(logits, dim=1)
        results.append(probs.cpu().numpy()[0])
    return np.array(results)

# ── SHAP: Background dataset (typical healthy pond readings) ──────────────────
# These represent a range of normal aquaculture conditions used as SHAP baseline
background_data = np.array([
    [6.0, 7.5, 28.0, 30.0,  0.30],
    [5.5, 7.2, 27.0, 25.0,  0.20],
    [7.0, 8.0, 29.0, 40.0,  0.40],
    [4.5, 7.8, 30.0, 60.0,  0.60],
    [5.0, 7.0, 28.5, 35.0,  0.30],
    [6.5, 7.5, 27.5, 20.0,  0.25],
    [3.5, 8.2, 31.0, 80.0,  0.80],
    [7.5, 7.3, 26.0, 15.0,  0.15],
    [4.0, 8.5, 32.0, 100.0, 1.00],
    [5.8, 7.6, 29.5, 45.0,  0.35],
], dtype=np.float64)

print("[SHAP] Initializing KernelExplainer with background data...")
shap_explainer = shap.KernelExplainer(predict_fn_shap, background_data)
print("[SHAP] KernelExplainer ready.")

RISK_LABELS = ["Low", "Moderate", "High"]
RISK_RANK = {"Low": 0, "Moderate": 1, "High": 2}


def _run_sensor_prediction(DO, pH, temperature, turbidity, ammonia, with_shap=True):
    """Runs the real BiLSTM+classifier pipeline on 5 raw sensor values."""
    raw_values = np.array([DO, pH, temperature, turbidity, ammonia], dtype=np.float64)
    scaled_values = (raw_values.astype(np.float32) - scale_mean) / scale_std
    # Tile to sequence shape (1, 24, 5) — steady-state assumption (no real
    # 24h history available from a single form submission)
    seq = np.tile(scaled_values, (WINDOW_SIZE, 1))
    input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = pipeline(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_id = probs.argmax(dim=1).item()

    risk_label = RISK_LABELS[pred_id]
    confidence = probs[0, pred_id].item()
    all_probs = probs[0].tolist()

    result = {
        "risk_label": risk_label,
        "confidence": float(confidence),
        "all_probs": {
            "Low": float(all_probs[0]),
            "Moderate": float(all_probs[1]),
            "High": float(all_probs[2]),
        },
        "guidance": get_guidance(risk_label, source="sensor"),
    }

    if with_shap:
        current_input = raw_values.reshape(1, -1)  # (1, 5)
        # nsamples=50 gives a fast approximation (~3-8 seconds)
        shap_vals = shap_explainer.shap_values(current_input, nsamples=50)
        # Handle different SHAP output formats (ndarray of shape (1, 5, 3) vs list of arrays)
        if isinstance(shap_vals, np.ndarray):
            if len(shap_vals.shape) == 3:
                shap_for_pred = shap_vals[0, :, pred_id].tolist()
            else:
                shap_for_pred = shap_vals[0].tolist()
        elif isinstance(shap_vals, list):
            shap_for_pred = shap_vals[pred_id][0].tolist()
        else:
            shap_for_pred = np.array(shap_vals)[0].tolist()

        result["shap"] = {
            "values": shap_for_pred,
            "feature_names": FEATURE_NAMES,
            "predicted_class": risk_label,
        }

    return result


async def _run_image_prediction(file: UploadFile):
    """Runs the CNN vision branch on an uploaded photo."""
    if not cnn_ready:
        raise HTTPException(
            status_code=503,
            detail="CNN model not trained yet. Run: python training/train_cnn.py",
        )
    raw_bytes = await file.read()
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = cnn_model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_id = probs.argmax(dim=1).item()

    risk_label = RISK_LABELS[pred_id]
    confidence = probs[0, pred_id].item()
    all_probs = probs[0].tolist()

    return {
        "risk_label": risk_label,
        "confidence": float(confidence),
        "all_probs": {
            "Low": float(all_probs[0]),
            "Moderate": float(all_probs[1]),
            "High": float(all_probs[2]),
        },
        "guidance": get_guidance(risk_label, source="image"),
    }


# ── Predict Endpoint (sensor branch) ───────────────────────────────────────────
@app.post("/api/predict")
async def predict_risk(data: WaterQualityInput):
    try:
        import time
        t0 = time.time()
        result = _run_sensor_prediction(data.DO, data.pH, data.temperature, data.turbidity, data.ammonia)
        result["latency_ms"] = round((time.time() - t0) * 1000)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/archive-predict")
async def predict_archive_risk(data: ArchivePredictionInput):
    if data.interval not in {"5min", "30min"}:
        raise HTTPException(status_code=400, detail="interval must be '5min' or '30min'")

    try:
        import time
        from models.archive_model import predict_archive

        started_at = time.time()
        values = data.model_dump(exclude={"interval"})
        result = predict_archive(data.interval, values)
        result["latency_ms"] = round((time.time() - started_at) * 1000)
        return result
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=f"Archive model is missing: {error}")
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

# ── Predict-From-Image Endpoint (vision branch) ────────────────────────────────
@app.post("/api/predict_image")
async def predict_risk_from_image(file: UploadFile = File(...)):
    try:
        import time
        t0 = time.time()
        result = await _run_image_prediction(file)
        result["latency_ms"] = round((time.time() - t0) * 1000)
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── Fusion Endpoint: combine both branches into one cautious verdict ──────────
@app.post("/api/predict_fusion")
async def predict_risk_fusion(
    file: UploadFile = File(...),
    DO: float = Form(...),
    pH: float = Form(...),
    temperature: float = Form(...),
    turbidity: float = Form(...),
    ammonia: float = Form(...),
):
    """
    Runs the sensor branch and the vision branch independently, then combines
    them with a precautionary rule: the fused verdict is whichever branch
    reports the HIGHER risk. This is deliberate, not a trained network — there
    is no real dataset pairing the same pond's sensor readings and photo at
    the same moment, so training a fusion network would only be learning to
    reproduce fake, synthetically-matched labels. A safety system should never
    let a calm reading silently outvote a visible warning sign (or vice
    versa), so "trust the more cautious branch" is the honest, defensible
    combination rule here.
    """
    try:
        import time
        t0 = time.time()

        sensor_result = _run_sensor_prediction(DO, pH, temperature, turbidity, ammonia, with_shap=False)
        image_result = await _run_image_prediction(file)

        sensor_rank = RISK_RANK[sensor_result["risk_label"]]
        image_rank = RISK_RANK[image_result["risk_label"]]

        if image_rank > sensor_rank:
            fused_label = image_result["risk_label"]
            fused_source = "image"
        elif sensor_rank > image_rank:
            fused_label = sensor_result["risk_label"]
            fused_source = "sensor"
        else:
            fused_label = sensor_result["risk_label"]
            fused_source = "sensor"  # tie -> either is equivalent; default to sensor's guidance text

        latency_ms = round((time.time() - t0) * 1000)

        return {
            "risk_label": fused_label,
            "driven_by": fused_source,  # which branch's (higher) risk determined the fused verdict
            "sensor_result": sensor_result,
            "image_result": image_result,
            "guidance": get_guidance(fused_label, source=fused_source),
            "latency_ms": latency_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
