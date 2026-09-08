from pathlib import Path
from typing import List

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LTV_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ltv_model.joblib"
)

CHURN_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_model.joblib"
)


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="Customer Churn & LTV API",
    description=(
        "Zaalima Development - Customer Churn Prediction "
        "and Lifetime Value Engine"
    ),
    version="2.0.0",
)


# ---------------------------------------------------------
# Load Models
# ---------------------------------------------------------

if not LTV_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"LTV model not found at: {LTV_MODEL_PATH}"
    )

if not CHURN_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Churn model not found at: {CHURN_MODEL_PATH}"
    )

ltv_model = joblib.load(
    LTV_MODEL_PATH
)

churn_model = joblib.load(
    CHURN_MODEL_PATH
)


# ---------------------------------------------------------
# Customer Input Schema
# ---------------------------------------------------------

class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int

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

    MonthlyCharges: float
    TotalCharges: float


# ---------------------------------------------------------
# Shared Feature Engineering
# ---------------------------------------------------------

def calculate_engineered_features(
    customer: CustomerInput,
):

    if customer.tenure > 0:
        avg_monthly_spend = (
            customer.TotalCharges
            / customer.tenure
        )
    else:
        avg_monthly_spend = (
            customer.MonthlyCharges
        )

    if customer.tenure <= 12:
        tenure_group = "0-12"

    elif customer.tenure <= 24:
        tenure_group = "13-24"

    elif customer.tenure <= 48:
        tenure_group = "25-48"

    else:
        tenure_group = "49-72"

    service_values = [
        customer.PhoneService,
        customer.MultipleLines,
        customer.OnlineSecurity,
        customer.OnlineBackup,
        customer.DeviceProtection,
        customer.TechSupport,
        customer.StreamingTV,
        customer.StreamingMovies,
    ]

    num_services = sum(
        value == "Yes"
        for value in service_values
    )

    is_month_to_month = int(
        customer.Contract
        == "Month-to-month"
    )

    has_internet = int(
        customer.InternetService
        != "No"
    )

    auto_payment_methods = [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    auto_payment = int(
        customer.PaymentMethod
        in auto_payment_methods
    )

    has_security_support = int(
        customer.OnlineSecurity == "Yes"
        or customer.TechSupport == "Yes"
    )

    return {
        "AvgMonthlySpend": avg_monthly_spend,
        "TenureGroup": tenure_group,
        "NumServices": num_services,
        "IsMonthToMonth": is_month_to_month,
        "HasInternet": has_internet,
        "AutoPayment": auto_payment,
        "HasSecuritySupport": (
            has_security_support
        ),
    }


# ---------------------------------------------------------
# LTV Feature Preparation
# ---------------------------------------------------------

def prepare_ltv_features(
    customer: CustomerInput,
):

    engineered = (
        calculate_engineered_features(
            customer
        )
    )

    model_input = {
        "gender": customer.gender,
        "SeniorCitizen": (
            customer.SeniorCitizen
        ),
        "Partner": customer.Partner,
        "Dependents": customer.Dependents,
        "tenure": customer.tenure,

        "PhoneService": (
            customer.PhoneService
        ),
        "MultipleLines": (
            customer.MultipleLines
        ),

        "InternetService": (
            customer.InternetService
        ),
        "OnlineSecurity": (
            customer.OnlineSecurity
        ),
        "OnlineBackup": (
            customer.OnlineBackup
        ),
        "DeviceProtection": (
            customer.DeviceProtection
        ),
        "TechSupport": (
            customer.TechSupport
        ),

        "StreamingTV": (
            customer.StreamingTV
        ),
        "StreamingMovies": (
            customer.StreamingMovies
        ),

        "Contract": customer.Contract,
        "PaperlessBilling": (
            customer.PaperlessBilling
        ),
        "PaymentMethod": (
            customer.PaymentMethod
        ),
        "MonthlyCharges": (
            customer.MonthlyCharges
        ),

        **engineered,
    }

    return model_input


# ---------------------------------------------------------
# Churn Feature Preparation
# ---------------------------------------------------------

def prepare_churn_features(
    customer: CustomerInput,
):

    engineered = (
        calculate_engineered_features(
            customer
        )
    )

    model_input = {
        "gender": customer.gender,
        "SeniorCitizen": (
            customer.SeniorCitizen
        ),
        "Partner": customer.Partner,
        "Dependents": customer.Dependents,
        "tenure": customer.tenure,

        "PhoneService": (
            customer.PhoneService
        ),
        "MultipleLines": (
            customer.MultipleLines
        ),

        "InternetService": (
            customer.InternetService
        ),
        "OnlineSecurity": (
            customer.OnlineSecurity
        ),
        "OnlineBackup": (
            customer.OnlineBackup
        ),
        "DeviceProtection": (
            customer.DeviceProtection
        ),
        "TechSupport": (
            customer.TechSupport
        ),

        "StreamingTV": (
            customer.StreamingTV
        ),
        "StreamingMovies": (
            customer.StreamingMovies
        ),

        "Contract": customer.Contract,
        "PaperlessBilling": (
            customer.PaperlessBilling
        ),
        "PaymentMethod": (
            customer.PaymentMethod
        ),

        "MonthlyCharges": (
            customer.MonthlyCharges
        ),
        "TotalCharges": (
            customer.TotalCharges
        ),

        **engineered,
    }

    return model_input


# ---------------------------------------------------------
# LTV Segment Function
# ---------------------------------------------------------

def get_ltv_segment(
    predicted_ltv: float,
):

    if predicted_ltv < 1500:
        return "Low Value"

    elif predicted_ltv < 4500:
        return "Medium Value"

    return "High Value"


# ---------------------------------------------------------
# Churn Risk Segment Function
# ---------------------------------------------------------

def get_risk_segment(
    churn_probability: float,
):

    if churn_probability >= 70:
        return "High Risk"

    elif churn_probability >= 30:
        return "Medium Risk"

    return "Low Risk"


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": (
            "Customer Churn & LTV API "
            "is running"
        )
    }


# ---------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "churn_model_loaded": True,
        "ltv_model_loaded": True,
    }


# ---------------------------------------------------------
# Single Churn Prediction
# ---------------------------------------------------------

@app.post("/predict/churn")
def predict_churn(
    customer: CustomerInput,
):

    try:

        model_input = (
            prepare_churn_features(
                customer
            )
        )

        customer_df = pd.DataFrame(
            [model_input]
        )

        probability = (
            churn_model
            .predict_proba(
                customer_df
            )[0][1]
        )

        probability_percent = round(
            float(probability) * 100,
            2,
        )

        predicted_class = int(
            probability >= 0.5
        )

        return {
            "churn_probability": (
                probability_percent
            ),
            "predicted_churn": (
                "Yes"
                if predicted_class == 1
                else "No"
            ),
            "risk_segment": (
                get_risk_segment(
                    probability_percent
                )
            ),
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ---------------------------------------------------------
# Batch Churn Prediction
# ---------------------------------------------------------

@app.post("/predict/churn/batch")
def predict_churn_batch(
    customers: List[CustomerInput],
):

    try:

        if len(customers) == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Customer list "
                    "cannot be empty."
                ),
            )

        prepared_customers = [
            prepare_churn_features(
                customer
            )
            for customer in customers
        ]

        batch_df = pd.DataFrame(
            prepared_customers
        )

        probabilities = (
            churn_model
            .predict_proba(
                batch_df
            )[:, 1]
        )

        results = []

        for index, probability in enumerate(
            probabilities
        ):

            probability_percent = round(
                float(probability) * 100,
                2,
            )

            predicted_class = int(
                probability >= 0.5
            )

            results.append(
                {
                    "customer_number": (
                        index + 1
                    ),
                    "churn_probability": (
                        probability_percent
                    ),
                    "predicted_churn": (
                        "Yes"
                        if predicted_class == 1
                        else "No"
                    ),
                    "risk_segment": (
                        get_risk_segment(
                            probability_percent
                        )
                    ),
                }
            )

        return {
            "total_customers": (
                len(results)
            ),
            "predictions": results,
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ---------------------------------------------------------
# Single LTV Prediction
# ---------------------------------------------------------

@app.post("/predict/ltv")
def predict_ltv(
    customer: CustomerInput,
):

    try:

        model_input = (
            prepare_ltv_features(
                customer
            )
        )

        customer_df = pd.DataFrame(
            [model_input]
        )

        prediction = (
            ltv_model.predict(
                customer_df
            )[0]
        )

        predicted_ltv = round(
            float(prediction),
            2,
        )

        segment = get_ltv_segment(
            predicted_ltv
        )

        return {
            "predicted_ltv": (
                predicted_ltv
            ),
            "ltv_segment": segment,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ---------------------------------------------------------
# Batch LTV Prediction
# ---------------------------------------------------------

@app.post("/predict/ltv/batch")
def predict_ltv_batch(
    customers: List[CustomerInput],
):

    try:

        if len(customers) == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Customer list "
                    "cannot be empty."
                ),
            )

        prepared_customers = [
            prepare_ltv_features(
                customer
            )
            for customer in customers
        ]

        batch_df = pd.DataFrame(
            prepared_customers
        )

        predictions = (
            ltv_model.predict(
                batch_df
            )
        )

        results = []

        for index, prediction in enumerate(
            predictions
        ):

            predicted_ltv = round(
                float(prediction),
                2,
            )

            segment = get_ltv_segment(
                predicted_ltv
            )

            results.append(
                {
                    "customer_number": (
                        index + 1
                    ),
                    "predicted_ltv": (
                        predicted_ltv
                    ),
                    "ltv_segment": (
                        segment
                    ),
                }
            )

        return {
            "total_customers": (
                len(results)
            ),
            "predictions": results,
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )