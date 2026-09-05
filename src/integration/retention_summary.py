from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RETENTION_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_retention_priority.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retention_priority_summary.csv"
)


def main():
    print("Loading retention priority data...")

    if not RETENTION_FILE.exists():
        raise FileNotFoundError(
            f"Retention priority file not found: {RETENTION_FILE}"
        )

    df = pd.read_csv(RETENTION_FILE)

    required_columns = {
        "RetentionPriority",
        "ChurnProbability",
        "ProjectedLTV",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    summary = (
        df.groupby("RetentionPriority")
        .agg(
            CustomerCount=("RetentionPriority", "size"),
            AvgChurnProbability=("ChurnProbability", "mean"),
            AvgProjectedLTV=("ProjectedLTV", "mean"),
        )
        .reset_index()
    )

    summary["AvgChurnProbability"] = (
        summary["AvgChurnProbability"].round(2)
    )

    summary["AvgProjectedLTV"] = (
        summary["AvgProjectedLTV"].round(2)
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print("\nRetention Priority Analytics Summary")
    print("------------------------------------")
    print(summary.to_string(index=False))

    print("\nOverall Metrics")
    print("---------------")
    print(f"Total Customers: {len(df)}")
    print(
        f"Average Churn Probability: "
        f"{df['ChurnProbability'].mean():.2f}%"
    )
    print(
        f"Average Projected LTV: "
        f"{df['ProjectedLTV'].mean():.2f}"
    )

    print(f"\nSummary saved to:")
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()