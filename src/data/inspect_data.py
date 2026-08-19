import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/raw/telco_customer_churn.csv")


def inspect_dataset():
    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("TELCO CUSTOMER CHURN - DATA INSPECTION")
    print("=" * 60)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nChurn Distribution:")
    print(df["Churn"].value_counts())

    print("\nInspection completed successfully.")


if __name__ == "__main__":
    inspect_dataset()