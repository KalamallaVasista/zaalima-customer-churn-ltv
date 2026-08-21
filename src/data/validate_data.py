import pandas as pd
from pathlib import Path


CLEAN_DATA_PATH = Path(
    "data/processed/telco_customer_churn_clean.csv"
)


def validate_clean_data():
    df = pd.read_csv(CLEAN_DATA_PATH)

    print("=" * 60)
    print("TELCO CUSTOMER CHURN - DATA VALIDATION")
    print("=" * 60)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nTotalCharges Data Type:")
    print(df["TotalCharges"].dtype)

    print("\nTotalCharges Missing Values:")
    print(df["TotalCharges"].isnull().sum())

    print("\nChurn Distribution:")
    print(df["Churn"].value_counts())

    print("\nValidation completed successfully.")


if __name__ == "__main__":
    validate_clean_data()