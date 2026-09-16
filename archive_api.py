import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from models.archive_model import predict_archive


class ArchivePredictionInput(BaseModel):
    interval: str = "5min"
    DO: float
    pH: float
    temperature: float
    turbidity: float
    ammonia: float
    fish_weight: float = 0.0


app = FastAPI(title="Archived Fish Mortality Models API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "models": ["5min", "30min"]}


@app.post("/api/archive-predict")
def predict_archive_risk(data: ArchivePredictionInput):
    if data.interval not in {"5min", "30min"}:
        raise HTTPException(status_code=400, detail="interval must be '5min' or '30min'")

    started_at = time.time()
    values = data.model_dump(exclude={"interval"})
    result = predict_archive(data.interval, values)
    result["latency_ms"] = round((time.time() - started_at) * 1000)
    return result