from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Full churn predictions for all 7,043 customers
CHURN_RISK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_risk_segments_full.csv"
)

# LTV information for active customers only
LTV_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ltv_customer_segments.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_retention_priority.csv"
)


def assign_retention_priority(row):
    risk = row["RiskSegment"]
    ltv = row["LTVSegment"]

    if risk == "High Risk":
        if ltv in ["High Value", "Medium Value"]:
            return "High Priority"
        return "Medium Priority"

    if risk == "Medium Risk":
        if ltv == "High Value":
            return "High Priority"
        elif ltv == "Medium Value":
            return "Medium Priority"
        return "Low Priority"

    if risk == "Low Risk":
        if ltv == "High Value":
            return "Medium Priority"
        return "Low Priority"

    return "Unknown"


def main():
    print("Loading churn risk and LTV data...")

    if not CHURN_RISK_FILE.exists():
        raise FileNotFoundError(
            f"Churn risk file not found: {CHURN_RISK_FILE}"
        )

    if not LTV_FILE.exists():
        raise FileNotFoundError(
            f"LTV file not found: {LTV_FILE}"
        )

    churn_df = pd.read_csv(CHURN_RISK_FILE)
    ltv_df = pd.read_csv(LTV_FILE)

    print(f"Churn-risk records: {len(churn_df)}")
    print(f"LTV active customers: {len(ltv_df)}")

    required_churn_columns = {
        "customerID",
        "ChurnProbability",
        "RiskSegment",
    }

    required_ltv_columns = {
        "customerID",
        "ProjectedLTV",
        "LTVSegment",
    }

    missing_churn_columns = (
        required_churn_columns
        - set(churn_df.columns)
    )

    missing_ltv_columns = (
        required_ltv_columns
        - set(ltv_df.columns)
    )

    if missing_churn_columns:
        raise ValueError(
            "Missing churn columns: "
            f"{sorted(missing_churn_columns)}"
        )

    if missing_ltv_columns:
        raise ValueError(
            "Missing LTV columns: "
            f"{sorted(missing_ltv_columns)}"
        )

    if churn_df["customerID"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs found in churn-risk data."
        )

    if ltv_df["customerID"].duplicated().any():
        raise ValueError(
            "Duplicate customer IDs found in LTV data."
        )

    churn_subset = churn_df[
        [
            "customerID",
            "ChurnProbability",
            "RiskSegment",
        ]
    ].copy()

    ltv_subset = ltv_df[
        [
            "customerID",
            "ProjectedLTV",
            "LTVSegment",
        ]
    ].copy()

    retention_df = pd.merge(
        churn_subset,
        ltv_subset,
        on="customerID",
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Matched active customers: "
        f"{len(retention_df)}"
    )

    retention_df["RetentionPriority"] = (
        retention_df.apply(
            assign_retention_priority,
            axis=1,
        )
    )

    if (
        retention_df["RetentionPriority"]
        == "Unknown"
    ).any():
        unknown_count = (
            retention_df["RetentionPriority"]
            == "Unknown"
        ).sum()

        raise ValueError(
            "Unknown retention priority found for "
            f"{unknown_count} customers."
        )

    priority_order = {
        "High Priority": 1,
        "Medium Priority": 2,
        "Low Priority": 3,
    }

    retention_df["_PriorityOrder"] = (
        retention_df["RetentionPriority"]
        .map(priority_order)
    )

    retention_df = retention_df.sort_values(
        by=[
            "_PriorityOrder",
            "ChurnProbability",
            "ProjectedLTV",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    retention_df = retention_df.drop(
        columns=["_PriorityOrder"]
    )

    retention_df = retention_df.reset_index(
        drop=True
    )

    retention_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nRetention Priority Distribution:")
    print(
        retention_df["RetentionPriority"]
        .value_counts()
        .to_string()
    )

    print("\nRisk Segment Distribution:")
    print(
        retention_df["RiskSegment"]
        .value_counts()
        .to_string()
    )

    print("\nLTV Segment Distribution:")
    print(
        retention_df["LTVSegment"]
        .value_counts()
        .to_string()
    )

    print("\nTop 10 Retention Priority Customers:")
    print(
        retention_df[
            [
                "customerID",
                "ChurnProbability",
                "RiskSegment",
                "ProjectedLTV",
                "LTVSegment",
                "RetentionPriority",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nRetention priority data saved to:")
    print(OUTPUT_FILE)

    print(
        "\nChurn and LTV integration "
        "completed successfully."
    )


if __name__ == "__main__":
    main()