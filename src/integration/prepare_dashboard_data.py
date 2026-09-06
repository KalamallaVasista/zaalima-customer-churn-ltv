from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_retention_priority.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "retention_dashboard_data.csv"
)


def main():
    print("Preparing retention dashboard data...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Retention priority file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "customerID",
        "ChurnProbability",
        "RiskSegment",
        "ProjectedLTV",
        "LTVSegment",
        "RetentionPriority",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    dashboard_df = df[required_columns].copy()

    priority_order = {
        "High Priority": 1,
        "Medium Priority": 2,
        "Low Priority": 3,
    }

    dashboard_df["PriorityOrder"] = (
        dashboard_df["RetentionPriority"]
        .map(priority_order)
    )

    if dashboard_df["PriorityOrder"].isna().any():
        raise ValueError(
            "Unknown retention priority values found."
        )

    dashboard_df = dashboard_df.sort_values(
        by=[
            "PriorityOrder",
            "ChurnProbability",
            "ProjectedLTV",
        ],
        ascending=[True, False, False],
    )

    dashboard_df = dashboard_df.reset_index(drop=True)

    dashboard_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nDashboard Dataset Summary")
    print("-------------------------")
    print(f"Total Customers: {len(dashboard_df)}")

    print("\nRetention Priority Distribution:")
    print(
        dashboard_df["RetentionPriority"]
        .value_counts()
        .to_string()
    )

    print("\nRisk Segment Distribution:")
    print(
        dashboard_df["RiskSegment"]
        .value_counts()
        .to_string()
    )

    print("\nLTV Segment Distribution:")
    print(
        dashboard_df["LTVSegment"]
        .value_counts()
        .to_string()
    )

    print("\nTop 5 Retention Priority Customers:")
    print(
        dashboard_df[
            [
                "customerID",
                "ChurnProbability",
                "ProjectedLTV",
                "RetentionPriority",
            ]
        ]
        .head()
        .to_string(index=False)
    )

    print("\nDashboard data saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()