import pandas as pd
from pathlib import Path


LTV_DATA_PATH = Path(
    "data/processed/ltv_active_customers.csv"
)


def validate_ltv_data():
    df = pd.read_csv(LTV_DATA_PATH)

    print("=" * 60)
    print("TELCO CUSTOMER CHURN - LTV DATA VALIDATION")
    print("=" * 60)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nChurn Values:")
    print(df["Churn"].value_counts())

    print("\nMissing Values:")
    print(
        df[
            [
                "CurrentRevenue",
                "Projected12MonthRevenue",
                "ProjectedLTV"
            ]
        ].isnull().sum()
    )

    print("\nProjected LTV Range:")
    print("Minimum:", round(df["ProjectedLTV"].min(), 2))
    print("Maximum:", round(df["ProjectedLTV"].max(), 2))

    print("\nInvalid Negative LTV Values:")
    print((df["ProjectedLTV"] < 0).sum())

    print("\nLTV data validation completed successfully.")


if __name__ == "__main__":
    validate_ltv_data()