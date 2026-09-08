from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Full churn predictions for all customers
CHURN_RISK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_risk_segments_full.csv"
)

# LTV data contains active customers only
LTV_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ltv_customer_segments.csv"
)


def main():
    print("Checking churn and LTV integration coverage...")

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
        required_churn_columns - set(churn_df.columns)
    )

    missing_ltv_columns = (
        required_ltv_columns - set(ltv_df.columns)
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

    churn_ids = set(
        churn_df["customerID"]
        .dropna()
        .astype(str)
    )

    ltv_ids = set(
        ltv_df["customerID"]
        .dropna()
        .astype(str)
    )

    matched_ids = churn_ids & ltv_ids

    active_without_risk = (
        ltv_ids - churn_ids
    )

    risk_without_active_ltv = (
        churn_ids - ltv_ids
    )

    total_churn_risk = len(churn_ids)
    total_active_ltv = len(ltv_ids)
    total_matched = len(matched_ids)

    if total_active_ltv == 0:
        integration_coverage = 0.0
    else:
        integration_coverage = (
            total_matched
            / total_active_ltv
            * 100
        )

    print("\nIntegration Coverage Report")
    print("---------------------------")

    print(
        f"Total churn-risk customers: "
        f"{total_churn_risk}"
    )

    print(
        f"Total active LTV customers: "
        f"{total_active_ltv}"
    )

    print(
        f"Matched active customers: "
        f"{total_matched}"
    )

    print(
        f"Active customers without churn risk: "
        f"{len(active_without_risk)}"
    )

    print(
        f"Churn-risk customers without active LTV: "
        f"{len(risk_without_active_ltv)}"
    )

    print(
        f"Integration coverage: "
        f"{integration_coverage:.2f}%"
    )

    print("\nCoverage Interpretation")
    print("-----------------------")

    if len(active_without_risk) == 0:
        print(
            "All active LTV customers have "
            "churn-risk predictions."
        )
    else:
        print(
            "Some active customers do not have "
            "churn-risk predictions."
        )

    if integration_coverage == 100:
        print(
            "Churn and LTV integration coverage "
            "is complete."
        )
    else:
        print(
            "Churn and LTV integration coverage "
            "is incomplete."
        )

    print(
        "\nIntegration coverage check "
        "completed successfully."
    )


if __name__ == "__main__":
    main()