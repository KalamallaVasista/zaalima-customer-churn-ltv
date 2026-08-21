import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/telco_customer_churn.csv")
PROCESSED_DATA_PATH = Path("data/processed/telco_customer_churn_clean.csv")


def clean_data():
    # Load original dataset
    df = pd.read_csv(RAW_DATA_PATH)

    print("=" * 60)
    print("TELCO CUSTOMER CHURN - DATA CLEANING")
    print("=" * 60)

    print(f"\nOriginal Shape: {df.shape}")

    # Convert TotalCharges from text to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    print(
        "\nMissing TotalCharges after conversion:",
        df["TotalCharges"].isna().sum()
    )

    # The 11 blank TotalCharges records have tenure = 0,
    # so they are new customers with no accumulated charges yet.
    zero_tenure_mask = (
        df["TotalCharges"].isna()
        & (df["tenure"] == 0)
    )

    df.loc[zero_tenure_mask, "TotalCharges"] = 0.0

    print(
        "Missing TotalCharges after handling:",
        df["TotalCharges"].isna().sum()
    )

    # Remove exact duplicate rows if any exist
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates()

    print("Duplicates Found:", duplicates_before)
    print("Duplicates Remaining:", df.duplicated().sum())

    # Save cleaned dataset
    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(f"\nCleaned Shape: {df.shape}")
    print(f"Cleaned dataset saved to: {PROCESSED_DATA_PATH}")

    print("\nData cleaning completed successfully.")


if __name__ == "__main__":
    clean_data()