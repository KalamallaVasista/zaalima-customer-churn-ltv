from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_customer_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "churn_model.joblib"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_risk_segments_full.csv"
)


def assign_risk(probability):
    if probability >= 70:
        return "High Risk"
    elif probability >= 30:
        return "Medium Risk"
    else:
        return "Low Risk"


def main():
    print("Loading churn model and customer data...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature data not found: {DATA_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Churn model not found: {MODEL_FILE}"
        )

    df = pd.read_csv(DATA_FILE)
    model = joblib.load(MODEL_FILE)

    if "customerID" not in df.columns:
        raise ValueError("customerID column not found.")

    customer_ids = df["customerID"].copy()

    X = df.drop(
        columns=[
            "customerID",
            "Churn",
        ]
    )

    probabilities = (
        model.predict_proba(X)[:, 1] * 100
    )

    output = pd.DataFrame(
        {
            "customerID": customer_ids,
            "ChurnProbability": probabilities.round(2),
        }
    )

    output["RiskSegment"] = (
        output["ChurnProbability"]
        .apply(assign_risk)
    )

    # Add useful business fields for analysis/dashboard.
    for column in [
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]:
        if column in df.columns:
            output[column] = df[column]

    output = output.sort_values(
        by="ChurnProbability",
        ascending=False,
    ).reset_index(drop=True)

    output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nFull Churn Risk Prediction")
    print("--------------------------")
    print(f"Total Customers: {len(output)}")

    print("\nRisk Segment Distribution:")
    print(
        output["RiskSegment"]
        .value_counts()
        .to_string()
    )

    print(
        "\nAverage Churn Probability: "
        f"{output['ChurnProbability'].mean():.2f}%"
    )

    print("\nTop 10 Highest-Risk Customers:")
    print(
        output[
            [
                "customerID",
                "ChurnProbability",
                "RiskSegment",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nFull churn-risk data saved to:")
    print(OUTPUT_FILE)

    print(
        "\nFull customer churn inference "
        "completed successfully."
    )


if __name__ == "__main__":
    main()