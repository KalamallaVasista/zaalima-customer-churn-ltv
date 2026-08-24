import pandas as pd
from pathlib import Path


INPUT_PATH = Path(
    "data/processed/ltv_active_customers.csv"
)

OUTPUT_PATH = Path(
    "data/processed/ltv_customer_segments.csv"
)


def segment_customers():
    print("=" * 60)
    print("CUSTOMER LIFETIME VALUE - SEGMENTATION")
    print("=" * 60)

    df = pd.read_csv(INPUT_PATH)

    print(f"\nActive Customers: {len(df)}")

    # Create three value-based groups using quantiles
    df["LTVSegment"] = pd.qcut(
        df["ProjectedLTV"],
        q=3,
        labels=[
            "Low Value",
            "Medium Value",
            "High Value"
        ]
    )

    print("\nLTV Segment Distribution:")
    print(df["LTVSegment"].value_counts())

    print("\nAverage LTV by Segment:")
    print(
        df.groupby(
            "LTVSegment",
            observed=True
        )["ProjectedLTV"]
        .mean()
        .round(2)
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSegmented dataset saved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        "\nLTV customer segmentation completed successfully."
    )


if __name__ == "__main__":
    segment_customers()