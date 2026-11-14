
import pytest
from pydantic import ValidationError

from mlops_customer_churn.api import Customer


@pytest.fixture
def valid_customer():
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


def test_customer_valid(valid_customer):
    customer = Customer(**valid_customer)
    assert customer.tenure == 2


def test_negative_tenure(valid_customer):
    valid_customer["tenure"] = -1

    with pytest.raises(ValidationError):
        Customer(**valid_customer)


def test_invalid_senior_citizen(valid_customer):
    valid_customer["SeniorCitizen"] = 3

    with pytest.raises(ValidationError):
        Customer(**valid_customer)


def test_missing_required_field(valid_customer):
    del valid_customer["gender"]

    with pytest.raises(ValidationError):
        Customer(**valid_customer)
