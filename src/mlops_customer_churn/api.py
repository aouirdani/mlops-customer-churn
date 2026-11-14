
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field



import os

MODEL_PATH = Path(
    os.environ.get(
        "CHURN_MODEL_PATH",
        str(
            Path(__file__).resolve().parents[2]
            / "models/churn_model.joblib"
        ),
    )
)


model = None


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    yield
    model = None


app = FastAPI(
    title="Customer Churn Prediction API",
    description="API de prédiction du churn avec scikit-learn",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(customer: Customer):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle indisponible",
        )

    data = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(data)[0, 1])
    prediction = int(model.predict(data)[0])

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_level": (
            "high" if probability >= 0.7
            else "medium" if probability >= 0.4
            else "low"
        ),
    }
