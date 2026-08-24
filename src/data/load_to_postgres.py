import pandas as pd
from pathlib import Path

from db_connection import engine


CLEAN_DATA_PATH = Path(
    "data/processed/telco_customer_churn_clean.csv"
)

TABLE_NAME = "telco_customers"


def load_data_to_postgres():
    print("=" * 60)
    print("TELCO CUSTOMER CHURN - POSTGRESQL INGESTION")
    print("=" * 60)

    # Read cleaned dataset
    df = pd.read_csv(CLEAN_DATA_PATH)

    print(f"\nRows to load: {len(df)}")
    print(f"Columns to load: {len(df.columns)}")

    # Load dataframe into PostgreSQL
    df.to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",
        index=False
    )

    print(
        f"\nData successfully loaded into PostgreSQL table: "
        f"{TABLE_NAME}"
    )


if __name__ == "__main__":
    load_data_to_postgres()