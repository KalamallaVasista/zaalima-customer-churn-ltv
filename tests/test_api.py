import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Add the project root directory to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.main import app


# Create FastAPI test client
client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")

    assert response.status_code == 200


def test_health_endpoint():
    """Test the API health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)


def test_single_ltv_prediction():
    """Test single-customer LTV prediction."""

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 65.50,
        "TotalCharges": 1572.00
    }

    response = client.post("/predict/ltv", json=customer)

    assert response.status_code == 200

    data = response.json()

    assert "predicted_ltv" in data
    assert "ltv_segment" in data

    assert data["predicted_ltv"] >= 0

    assert data["ltv_segment"] in [
        "Low Value",
        "Medium Value",
        "High Value"
    ]