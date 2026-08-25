from pathlib import Path
from typing import List

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

MODEL_PATH = Path("models/ltv_model.joblib")


# ---------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------

app = FastAPI(
    title="Customer Churn & LTV API",
    description=(
        "Zaalima Development - Customer Churn Prediction "
        "and Lifetime Value Engine"
    ),
    version="1.0.0"
)


# ---------------------------------------------------------
# Load LTV Model
# ---------------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"LTV model not found at: {MODEL_PATH}"
    )

ltv_model = joblib.load(MODEL_PATH)


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
# Feature Engineering Function
# ---------------------------------------------------------

def prepare_customer_features(customer: CustomerInput):

    customer_data = customer.model_dump()

    # Average Monthly Spend
    if customer.tenure > 0:
        avg_monthly_spend = (
            customer.TotalCharges / customer.tenure
        )
    else:
        avg_monthly_spend = customer.MonthlyCharges

    # Tenure Group
    if customer.tenure <= 12:
        tenure_group = "0-12"

    elif customer.tenure <= 24:
        tenure_group = "13-24"

    elif customer.tenure <= 48:
        tenure_group = "25-48"

    else:
        tenure_group = "49-72"

    # Number of Active Services
    service_values = [
        customer.PhoneService,
        customer.MultipleLines,
        customer.OnlineSecurity,
        customer.OnlineBackup,
        customer.DeviceProtection,
        customer.TechSupport,
        customer.StreamingTV,
        customer.StreamingMovies
    ]

    num_services = sum(
        value == "Yes"
        for value in service_values
    )

    # Month-to-Month Contract Flag
    is_month_to_month = int(
        customer.Contract == "Month-to-month"
    )

    # Internet Flag
    has_internet = int(
        customer.InternetService != "No"
    )

    # Automatic Payment Flag
    auto_payment_methods = [
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

    auto_payment = int(
        customer.PaymentMethod
        in auto_payment_methods
    )

    # Security / Support Flag
    has_security_support = int(
        customer.OnlineSecurity == "Yes"
        or customer.TechSupport == "Yes"
    )

    # Build Model Input
    model_input = {
        "gender": customer_data["gender"],
        "SeniorCitizen": customer_data["SeniorCitizen"],
        "Partner": customer_data["Partner"],
        "Dependents": customer_data["Dependents"],
        "tenure": customer_data["tenure"],
        "PhoneService": customer_data["PhoneService"],
        "MultipleLines": customer_data["MultipleLines"],
        "InternetService": customer_data["InternetService"],
        "OnlineSecurity": customer_data["OnlineSecurity"],
        "OnlineBackup": customer_data["OnlineBackup"],
        "DeviceProtection": customer_data["DeviceProtection"],
        "TechSupport": customer_data["TechSupport"],
        "StreamingTV": customer_data["StreamingTV"],
        "StreamingMovies": customer_data["StreamingMovies"],
        "Contract": customer_data["Contract"],
        "PaperlessBilling": customer_data[
            "PaperlessBilling"
        ],
        "PaymentMethod": customer_data["PaymentMethod"],
        "MonthlyCharges": customer_data["MonthlyCharges"],

        # Engineered Features
        "AvgMonthlySpend": avg_monthly_spend,
        "TenureGroup": tenure_group,
        "NumServices": num_services,
        "IsMonthToMonth": is_month_to_month,
        "HasInternet": has_internet,
        "AutoPayment": auto_payment,
        "HasSecuritySupport": has_security_support
    }

    return model_input


# ---------------------------------------------------------
# LTV Segment Function
# ---------------------------------------------------------

def get_ltv_segment(predicted_ltv: float):

    if predicted_ltv < 1500:
        return "Low Value"

    elif predicted_ltv < 4500:
        return "Medium Value"

    else:
        return "High Value"


# ---------------------------------------------------------
# Root Endpoint
# ---------------------------------------------------------

@app.get("/")
def root():

    return {
        "message": (
            "Customer Churn & LTV API is running"
        )
    }


# ---------------------------------------------------------
# Health Endpoint
# ---------------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "ltv_model_loaded": True
    }


# ---------------------------------------------------------
# Single Customer LTV Prediction
# ---------------------------------------------------------

@app.post("/predict/ltv")
def predict_ltv(customer: CustomerInput):

    try:

        model_input = prepare_customer_features(
            customer
        )

        customer_df = pd.DataFrame(
            [model_input]
        )

        prediction = ltv_model.predict(
            customer_df
        )[0]

        predicted_ltv = round(
            float(prediction),
            2
        )

        segment = get_ltv_segment(
            predicted_ltv
        )

        return {
            "predicted_ltv": predicted_ltv,
            "ltv_segment": segment
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ---------------------------------------------------------
# Batch Customer LTV Prediction
# ---------------------------------------------------------

@app.post("/predict/ltv/batch")
def predict_ltv_batch(
    customers: List[CustomerInput]
):

    try:

        if len(customers) == 0:
            raise HTTPException(
                status_code=400,
                detail="Customer list cannot be empty."
            )

        prepared_customers = []

        for customer in customers:

            prepared_customer = (
                prepare_customer_features(
                    customer
                )
            )

            prepared_customers.append(
                prepared_customer
            )

        batch_df = pd.DataFrame(
            prepared_customers
        )

        predictions = ltv_model.predict(
            batch_df
        )

        results = []

        for index, prediction in enumerate(
            predictions
        ):

            predicted_ltv = round(
                float(prediction),
                2
            )

            segment = get_ltv_segment(
                predicted_ltv
            )

            results.append(
                {
                    "customer_number": index + 1,
                    "predicted_ltv": predicted_ltv,
                    "ltv_segment": segment
                }
            )

        return {
            "total_customers": len(results),
            "predictions": results
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )