
import pytest
from fastapi.testclient import TestClient

from mlops_customer_churn.api import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def customer():
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5,
        "TotalCharges": 191.0,
    }


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict(client, customer):
    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    data = response.json()

    assert data["churn_prediction"] in [0, 1]
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_level"] in ["low", "medium", "high"]


def test_predict_missing_field(client, customer):
    del customer["gender"]

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_invalid_tenure(client, customer):
    customer["tenure"] = -5

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_extra_field(client, customer):
    customer["unknown_column"] = "test"

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
