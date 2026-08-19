import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/telco_customer_churn.csv")


def investigate_total_charges():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("TOTAL CHARGES INVESTIGATION")
    print("=" * 60)

    # Convert TotalCharges to numeric temporarily.
    # Invalid values will become NaN.
    numeric_total_charges = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    invalid_rows = df[numeric_total_charges.isna()]

    print("\nInvalid / Blank TotalCharges Count:")
    print(len(invalid_rows))

    print("\nAffected Customer Records:")
    print(
        invalid_rows[
            [
                "customerID",
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "Churn"
            ]
        ]
    )


if __name__ == "__main__":
    investigate_total_charges()