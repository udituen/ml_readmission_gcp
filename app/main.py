from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import os

from app.schema import PatientInput, PredictionResponse

# ── Model loading ──────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/xgb_model_v2.pkl")

ml_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup
    try:
        ml_model["pipeline"] = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise
    yield
    # Cleanup on shutdown
    ml_model.clear()


# ── App setup ──────────────────────────────────────────────────
app = FastAPI(
    title="Diabetes Readmission Prediction API",
    description="Production MLOps pipeline — XGBoost model served via FastAPI on GCP Cloud Run",
    version="1.0.0",
    lifespan=lifespan
)

# Column name mapping — handles hyphenated column names
HYPHEN_COLS = {
    "glyburide_metformin": "glyburide-metformin",
    "glipizide_metformin": "glipizide-metformin",
    "glimepiride_pioglitazone": "glimepiride-pioglitazone",
    "metformin_rosiglitazone": "metformin-rosiglitazone",
    "metformin_pioglitazone": "metformin-pioglitazone",
}

# ── Endpoints ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Diabetes Readmission Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health():
    model_loaded = "pipeline" in ml_model
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    if "pipeline" not in ml_model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to dict using aliases so hyphenated cols are correct
        input_dict = patient.model_dump(by_alias=True)

        # Build DataFrame
        input_df = pd.DataFrame([input_dict])

        # Get prediction and probability
        pipeline = ml_model["pipeline"]
        prediction = int(pipeline.predict(input_df)[0])
        probability = float(pipeline.predict_proba(input_df)[0][1])

        return PredictionResponse(
            prediction=prediction,
            prediction_label="Readmitted" if prediction == 1 else "Not Readmitted",
            probability_readmitted=round(probability, 4),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model-info")
def model_info():
    return {
        "model_type": "XGBClassifier",
        "features": {
            "categorical": ["age", "gender"],
            "numerical": [
                "num_lab_procedures", "num_procedures", "num_medications",
                "number_outpatient", "number_emergency",
                "number_inpatient", "number_diagnoses"
            ],
            "medications": 23
        },
        "target": "Readmitted (1) / Not Readmitted (0)",
        "version": "1.0.0"
    }
