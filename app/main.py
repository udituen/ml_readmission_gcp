import joblib
import time
import json
import numpy as np

from fastapi import FastAPI
from app.schemas import PredictionRequest
from app.logging_config import get_logger

app = FastAPI()
logger = get_logger()

artifact = joblib.load("app/model/model.joblib")
model = artifact["model"]
MODEL_VERSION = artifact["version"]

@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(request: PredictionRequest):

    start_time = time.time()

    data = np.array([[
        request.age,
        request.num_procedures,
        request.num_medications,
        request.gender,
        request.admission_type
    ]])

    prob = model.predict_proba(data)[0][1]

    latency = time.time() - start_time

    logger.info(json.dumps({
        "event": "inference",
        "model_version": MODEL_VERSION,
        "latency_ms": round(latency * 1000, 2),
        "prediction_probability": float(prob)
    }))

    return {
        "readmission_probability": float(prob),
        "model_version": MODEL_VERSION,
        "latency_ms": round(latency * 1000, 2)
    }