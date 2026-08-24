import pandas as pd
from pathlib import Path


FEATURE_DATA_PATH = Path(
    "data/processed/telco_customer_features.csv"
)

FEATURE_COLUMNS = [
    "AvgMonthlySpend",
    "TenureGroup",
    "NumServices",
    "IsMonthToMonth",
    "HasInternet",
    "AutoPayment",
    "HasSecuritySupport"
]


def validate_features():
    df = pd.read_csv(FEATURE_DATA_PATH)

    print("=" * 60)
    print("TELCO CUSTOMER CHURN - FEATURE VALIDATION")
    print("=" * 60)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nMissing Values in Engineered Features:")
    print(df[FEATURE_COLUMNS].isnull().sum())

    print("\nTenure Group Distribution:")
    print(df["TenureGroup"].value_counts().sort_index())

    print("\nNumber of Services Range:")
    print("Minimum:", df["NumServices"].min())
    print("Maximum:", df["NumServices"].max())

    binary_features = [
        "IsMonthToMonth",
        "HasInternet",
        "AutoPayment",
        "HasSecuritySupport"
    ]

    print("\nBinary Feature Values:")

    for column in binary_features:
        print(
            f"{column}:",
            sorted(df[column].unique().tolist())
        )

    print("\nAverage Monthly Spend:")
    print("Minimum:", round(df["AvgMonthlySpend"].min(), 2))
    print("Maximum:", round(df["AvgMonthlySpend"].max(), 2))

    print("\nFeature validation completed successfully.")


if __name__ == "__main__":
    validate_features()