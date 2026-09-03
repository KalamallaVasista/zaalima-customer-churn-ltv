from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHURN_FILE = PROJECT_ROOT / "data" / "processed" / "customer_risk_segments.csv"
LTV_FILE = PROJECT_ROOT / "data" / "processed" / "ltv_customer_segments.csv"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_retention_priority.csv"
)


def assign_retention_priority(risk_segment, ltv_segment):
    """
    Combine churn risk and customer value to determine
    retention priority.
    """

    if risk_segment == "High Risk":
        if ltv_segment in ["High Value", "Medium Value"]:
            return "High Priority"
        return "Medium Priority"

    if risk_segment == "Medium Risk":
        if ltv_segment == "High Value":
            return "High Priority"

        if ltv_segment == "Medium Value":
            return "Medium Priority"

        return "Low Priority"

    if risk_segment == "Low Risk":
        if ltv_segment == "High Value":
            return "Medium Priority"

        return "Low Priority"

    return "Unknown"


def main():
    print("Loading churn-risk data...")
    churn_df = pd.read_csv(CHURN_FILE)

    print("Loading LTV data...")
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

    missing_churn = required_churn_columns - set(churn_df.columns)
    missing_ltv = required_ltv_columns - set(ltv_df.columns)

    if missing_churn:
        raise ValueError(
            f"Missing churn columns: {sorted(missing_churn)}"
        )

    if missing_ltv:
        raise ValueError(
            f"Missing LTV columns: {sorted(missing_ltv)}"
        )

    churn_selected = churn_df[
        [
            "customerID",
            "ChurnProbability",
            "RiskSegment",
        ]
    ].copy()

    ltv_selected = ltv_df[
        [
            "customerID",
            "ProjectedLTV",
            "LTVSegment",
        ]
    ].copy()

    retention_df = pd.merge(
        churn_selected,
        ltv_selected,
        on="customerID",
        how="inner",
        validate="one_to_one",
    )

    retention_df["RetentionPriority"] = retention_df.apply(
        lambda row: assign_retention_priority(
            row["RiskSegment"],
            row["LTVSegment"],
        ),
        axis=1,
    )

    retention_df = retention_df.sort_values(
        by=["RetentionPriority", "ChurnProbability", "ProjectedLTV"],
        ascending=[True, False, False],
    )

    retention_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nRetention priority integration completed.")
    print(f"Matched active customers: {len(retention_df)}")

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

    print(f"\nSaved output to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()