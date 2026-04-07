# Diabetes Readmission MLOps Pipeline

A production-grade MLOps pipeline that trains, serves, and automatically deploys a diabetes readmission prediction model on Google Cloud Platform.

**Live API:** https://diabetes-api-565467329909.us-central1.run.app/docs

---

## Architecture

<img width="391" height="601" alt="architecture" src="https://github.com/user-attachments/assets/fc8672ff-0765-42b0-ad14-ba5eee018874" />

---

## ML Pipeline

- **Model:** XGBoost Classifier (`n_estimators=100`, `max_depth=5`, `learning_rate=0.1`)
- **Preprocessing:** Scikit-learn `ColumnTransformer` with three parallel pipelines:
  - Categorical features (`age`, `gender`): SimpleImputer + OneHotEncoder
  - Numerical features (7 columns): SimpleImputer + StandardScaler
  - Medication features (23 columns): OrdinalEncoder
- **Target:** Binary — Readmitted (1) / Not Readmitted (0)
- **Dataset:** UCI Diabetes 130-US Hospitals dataset

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service info |
| `/health` | GET | Health check + model status |
| `/predict` | POST | Run inference on patient data |
| `/model-info` | GET | Feature names and model metadata |
| `/docs` | GET | Interactive Swagger UI |

### Sample Request

```bash
curl -X POST https://diabetes-api-565467329909.us-central1.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": "[80-90)",
    "gender": "Female",
    "num_lab_procedures": 72,
    "num_procedures": 6,
    "num_medications": 21,
    "number_outpatient": 0,
    "number_emergency": 3,
    "number_inpatient": 5,
    "number_diagnoses": 9,
    "metformin": "Steady",
    "repaglinide": "No",
    "nateglinide": "No",
    "chlorpropamide": "No",
    "glimepiride": "No",
    "acetohexamide": "No",
    "glipizide": "Steady",
    "glyburide": "No",
    "tolbutamide": "No",
    "pioglitazone": "No",
    "rosiglitazone": "No",
    "acarbose": "No",
    "miglitol": "No",
    "troglitazone": "No",
    "tolazamide": "No",
    "examide": "No",
    "citoglipton": "No",
    "insulin": "Up",
    "glyburide-metformin": "No",
    "glipizide-metformin": "No",
    "glimepiride-pioglitazone": "No",
    "metformin-rosiglitazone": "No",
    "metformin-pioglitazone": "No"
  }'
```

### Sample Response

```json
{
  "prediction": 1,
  "prediction_label": "Readmitted",
  "probability_readmitted": 0.7643,
  "model_version": "1.0.0"
}
```

---

## Performance Metrics

| Metric | Value |
|---|---|
| API latency (warm) | ~110ms |
| Cold start latency | <320ms |
| CI/CD pipeline runtime | ~2m 30s |
| Deployment method | Zero-downtime Cloud Run |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | XGBoost + Scikit-learn Pipeline |
| API | FastAPI + Pydantic |
| Containerisation | Docker (python:3.12-slim) |
| Container Registry | GCP Artifact Registry |
| Serving | GCP Cloud Run (serverless) |
| CI/CD | GitHub Actions |
| Authentication | GCP Workload Identity Federation (OIDC) |

---

## Project Structure

```
diabetes-mlops/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, endpoints, model loading
│   └── schema.py        # Pydantic input/output schemas
├── models/
│   └── xgb_model_v2.pkl # Trained sklearn pipeline artifact
├── data/
│   └── diabetic_data.csv
├── train.py             # Training pipeline
├── Dockerfile
├── requirements.txt
└── .github/
    └── workflows/
        └── deploy.yml   # GitHub Actions CI/CD
```

---


## Running Locally

```bash
# Clone repo
git clone https://github.com/udituen/ml_readmission_gcp
cd ml_readmission_gcp

# Install dependencies
pip install -r requirements.txt

# Train model (generates models/xgb_model_v2.pkl)
python train.py

# Run API
uvicorn app.main:app --reload --port 8080

# Visit http://localhost:8080/docs
```

### Run with Docker

```bash
docker build -t diabetes-api .
docker run -p 8080:8080 diabetes-api
```

---

## Author

**Uduak Ituen** — AI Engineer | MLOps | LLM & RAG Systems

[LinkedIn](https://linkedin.com/in/uduak-ituen) · [GitHub](https://github.com/udituen) · [DocsQA Project](https://huggingface.co/spaces/udituen/DocsQA)
