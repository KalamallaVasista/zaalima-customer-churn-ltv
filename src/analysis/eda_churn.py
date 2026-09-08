from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_customer_churn_clean.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "eda"
)


def save_bar_chart(
    series,
    title,
    xlabel,
    ylabel,
    filename,
):
    plt.figure(figsize=(8, 5))

    series.plot(kind="bar")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_file = OUTPUT_DIR / filename

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def churn_rate_by(df, column):
    result = (
        df.groupby(column, observed=False)["ChurnFlag"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    return result


def main():
    print("Loading cleaned Telco dataset...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Clean dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df["ChurnFlag"] = (
        df["Churn"]
        .map(
            {
                "No": 0,
                "Yes": 1,
            }
        )
    )

    print(f"Dataset Shape: {df.shape}")

    print("\nChurn Distribution")
    print("------------------")

    churn_counts = df["Churn"].value_counts()
    churn_percent = (
        df["Churn"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    churn_summary = pd.DataFrame(
        {
            "Count": churn_counts,
            "Percentage": churn_percent,
        }
    )

    print(churn_summary)

    churn_summary.to_csv(
        OUTPUT_DIR / "churn_distribution.csv"
    )

    save_bar_chart(
        churn_counts,
        "Customer Churn Distribution",
        "Churn",
        "Number of Customers",
        "churn_distribution.png",
    )

    analysis_columns = {
        "Contract": "churn_by_contract.png",
        "InternetService": "churn_by_internet_service.png",
        "PaymentMethod": "churn_by_payment_method.png",
        "TechSupport": "churn_by_tech_support.png",
        "OnlineSecurity": "churn_by_online_security.png",
    }

    for column, filename in analysis_columns.items():
        rates = churn_rate_by(
            df,
            column,
        )

        print(
            f"\nChurn Rate by {column}"
        )
        print("-" * 40)
        print(rates.round(2))

        rates.to_csv(
            OUTPUT_DIR
            / f"{column.lower()}_churn_rate.csv",
            header=["ChurnRate"],
        )

        save_bar_chart(
            rates,
            f"Churn Rate by {column}",
            column,
            "Churn Rate (%)",
            filename,
        )

    tenure_summary = (
        df.groupby(
            "Churn",
            observed=False,
        )["tenure"]
        .mean()
        .round(2)
    )

    monthly_charge_summary = (
        df.groupby(
            "Churn",
            observed=False,
        )["MonthlyCharges"]
        .mean()
        .round(2)
    )

    print("\nAverage Tenure by Churn")
    print("-----------------------")
    print(tenure_summary)

    print("\nAverage Monthly Charges by Churn")
    print("--------------------------------")
    print(monthly_charge_summary)

    save_bar_chart(
        tenure_summary,
        "Average Tenure by Churn Status",
        "Churn",
        "Average Tenure (Months)",
        "average_tenure_by_churn.png",
    )

    save_bar_chart(
        monthly_charge_summary,
        "Average Monthly Charges by Churn Status",
        "Churn",
        "Average Monthly Charges",
        "average_monthly_charges_by_churn.png",
    )

    print("\nEDA figures saved to:")
    print(OUTPUT_DIR)

    print(
        "\nEDA completed successfully."
    )


if __name__ == "__main__":
    main()