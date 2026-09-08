from pathlib import Path

import pandas as pd
from sqlalchemy import inspect

from db_connection import engine


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retention_dashboard_data.csv"
)

TABLE_NAME = "retention_dashboard"


def main():
    print("Loading retention dashboard dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dashboard dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(f"Rows to load: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"- {column}")

    print(
        f"\nLoading data into PostgreSQL table: {TABLE_NAME}"
    )

    df.to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=500,
    )

    inspector = inspect(engine)

    if TABLE_NAME not in inspector.get_table_names():
        raise RuntimeError(
            f"Table {TABLE_NAME} was not created."
        )

    with engine.connect() as connection:
        result = connection.exec_driver_sql(
            f'SELECT COUNT(*) FROM "{TABLE_NAME}"'
        )

        database_count = result.scalar()

    print("\nPostgreSQL verification")
    print("-----------------------")
    print(f"CSV rows: {len(df)}")
    print(f"Database rows: {database_count}")

    if database_count != len(df):
        raise ValueError(
            "Database row count does not match CSV row count."
        )

    print(
        "\nDashboard data loaded successfully."
    )


if __name__ == "__main__":
    main()