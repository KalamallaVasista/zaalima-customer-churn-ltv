import pandas as pd
from pathlib import Path


INPUT_PATH = Path("data/processed/telco_customer_churn_clean.csv")
OUTPUT_PATH = Path("data/processed/telco_customer_features.csv")


def create_features():
    df = pd.read_csv(INPUT_PATH)

    print("=" * 60)
    print("TELCO CUSTOMER CHURN - FEATURE ENGINEERING")
    print("=" * 60)

    print(f"\nOriginal Shape: {df.shape}")

    # 1. Average monthly spend
    df["AvgMonthlySpend"] = df["MonthlyCharges"]

    mask = df["tenure"] > 0

    df.loc[mask, "AvgMonthlySpend"] = (
        df.loc[mask, "TotalCharges"]
        / df.loc[mask, "tenure"]
    )

    # 2. Tenure group
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12",
            "13-24",
            "25-48",
            "49-72"
        ]
    )

    # 3. Number of active services
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    df["NumServices"] = 0

    for column in service_columns:
        df["NumServices"] += (
            df[column] == "Yes"
        ).astype(int)

    # 4. Month-to-month contract flag
    df["IsMonthToMonth"] = (
        df["Contract"] == "Month-to-month"
    ).astype(int)

    # 5. Internet availability flag
    df["HasInternet"] = (
        df["InternetService"] != "No"
    ).astype(int)

    # 6. Automatic payment flag
    df["AutoPayment"] = df["PaymentMethod"].isin(
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    ).astype(int)

    # 7. Security/support flag
    df["HasSecuritySupport"] = (
        (df["OnlineSecurity"] == "Yes")
        | (df["TechSupport"] == "Yes")
    ).astype(int)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nNew Shape: {df.shape}")

    print("\nCreated Features:")
    print(
        [
            "AvgMonthlySpend",
            "TenureGroup",
            "NumServices",
            "IsMonthToMonth",
            "HasInternet",
            "AutoPayment",
            "HasSecuritySupport"
        ]
    )

    print(
        f"\nFeature dataset saved to: {OUTPUT_PATH}"
    )

    print("\nFeature engineering completed successfully.")


if __name__ == "__main__":
    create_features()