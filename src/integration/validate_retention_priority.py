from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETENTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_retention_priority.csv"
)


def main():
    print("Loading retention priority data...")

    if not RETENTION_FILE.exists():
        raise FileNotFoundError(
            f"Retention priority file not found: {RETENTION_FILE}"
        )

    df = pd.read_csv(RETENTION_FILE)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    required_columns = {
        "customerID",
        "ChurnProbability",
        "RiskSegment",
        "ProjectedLTV",
        "LTVSegment",
        "RetentionPriority",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    missing_customer_ids = df["customerID"].isna().sum()

    if missing_customer_ids > 0:
        raise ValueError(
            f"Missing customerID values found: {missing_customer_ids}"
        )

    duplicate_customers = df["customerID"].duplicated().sum()

    if duplicate_customers > 0:
        raise ValueError(
            f"Duplicate customer records found: {duplicate_customers}"
        )

    invalid_probability = (
        (df["ChurnProbability"] < 0)
        | (df["ChurnProbability"] > 100)
    ).sum()

    if invalid_probability > 0:
        raise ValueError(
            f"Invalid churn probabilities found: {invalid_probability}"
        )

    negative_ltv = (df["ProjectedLTV"] < 0).sum()

    if negative_ltv > 0:
        raise ValueError(
            f"Negative LTV values found: {negative_ltv}"
        )

    valid_priorities = {
        "Low Priority",
        "Medium Priority",
        "High Priority",
    }

    invalid_priorities = set(
        df["RetentionPriority"].dropna().unique()
    ) - valid_priorities

    if invalid_priorities:
        raise ValueError(
            f"Invalid retention priorities found: "
            f"{sorted(invalid_priorities)}"
        )

    missing_values = df[
        [
            "customerID",
            "ChurnProbability",
            "RiskSegment",
            "ProjectedLTV",
            "LTVSegment",
            "RetentionPriority",
        ]
    ].isna().sum().sum()

    if missing_values > 0:
        raise ValueError(
            f"Missing values found in required fields: {missing_values}"
        )

    print("\nValidation Results")
    print("------------------")
    print("Required columns: PASS")
    print("Missing customer IDs: 0")
    print("Duplicate customers: 0")
    print("Churn probability range: PASS")
    print("Projected LTV values: PASS")
    print("Retention priority values: PASS")
    print("Missing required values: 0")

    print("\nRetention Priority Distribution:")
    print(
        df["RetentionPriority"]
        .value_counts()
        .to_string()
    )

    print("\nRetention priority validation completed successfully.")


if __name__ == "__main__":
    main()