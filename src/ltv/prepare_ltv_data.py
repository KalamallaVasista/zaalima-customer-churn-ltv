import pandas as pd
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/telco_customer_features.csv"
)

OUTPUT_PATH = Path(
    "data/processed/ltv_active_customers.csv"
)


def prepare_ltv_data():
    df = pd.read_csv(INPUT_PATH)

    print("=" * 60)
    print("TELCO CUSTOMER CHURN - LTV DATA PREPARATION")
    print("=" * 60)

    print(f"\nOriginal Customers: {len(df)}")

    # LTV modeling is performed for currently active customers
    active_df = df[df["Churn"] == "No"].copy()

    print(f"Active Customers: {len(active_df)}")

    # Current observed revenue
    active_df["CurrentRevenue"] = active_df["TotalCharges"]

    # 12-month future revenue projection
    active_df["Projected12MonthRevenue"] = (
        active_df["MonthlyCharges"] * 12
    )

    # LTV proxy target
    active_df["ProjectedLTV"] = (
        active_df["CurrentRevenue"]
        + active_df["Projected12MonthRevenue"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    active_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nLTV Features Created:")
    print(
        [
            "CurrentRevenue",
            "Projected12MonthRevenue",
            "ProjectedLTV"
        ]
    )

    print("\nProjected LTV Summary:")
    print(active_df["ProjectedLTV"].describe())

    print(
        f"\nLTV dataset saved to: {OUTPUT_PATH}"
    )

    print("\nLTV data preparation completed successfully.")


if __name__ == "__main__":
    prepare_ltv_data()