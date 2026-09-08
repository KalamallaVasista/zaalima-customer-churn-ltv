from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_customer_features.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "churn_model.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

IMPORTANCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "churn_shap_feature_importance.csv"
)

BAR_PLOT_FILE = (
    OUTPUT_DIR
    / "churn_shap_bar.png"
)

SUMMARY_PLOT_FILE = (
    OUTPUT_DIR
    / "churn_shap_summary.png"
)


def main():
    print("Loading churn model and feature data...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature data not found: {DATA_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Churn model not found: {MODEL_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    X = df.drop(
        columns=[
            "customerID",
            "Churn",
        ]
    )

    pipeline = joblib.load(MODEL_FILE)

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    classifier = pipeline.named_steps[
        "classifier"
    ]

    print("Transforming customer features...")

    X_transformed = preprocessor.transform(X)

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    # Use a representative sample to keep SHAP
    # computation efficient.
    sample_size = min(
        1000,
        X_transformed.shape[0],
    )

    sample_indices = (
        pd.Series(
            range(X_transformed.shape[0])
        )
        .sample(
            n=sample_size,
            random_state=42,
        )
        .sort_values()
        .to_numpy()
    )

    X_sample = X_transformed[
        sample_indices
    ]

    print(
        f"SHAP sample size: {sample_size}"
    )

    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(
        classifier
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    # Handle SHAP versions that may return
    # different array structures.
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    mean_absolute_shap = (
        abs(shap_values)
        .mean(axis=0)
    )

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "MeanAbsoluteSHAP": mean_absolute_shap,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            by="MeanAbsoluteSHAP",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    importance_df.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    print("\nTop 15 Churn Drivers")
    print("--------------------")

    print(
        importance_df
        .head(15)
        .to_string(
            index=False
        )
    )

    print("\nCreating SHAP bar plot...")

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        BAR_PLOT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("Creating SHAP summary plot...")

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        SUMMARY_PLOT_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("\nSHAP feature importance saved to:")
    print(IMPORTANCE_FILE)

    print("\nSHAP bar plot saved to:")
    print(BAR_PLOT_FILE)

    print("\nSHAP summary plot saved to:")
    print(SUMMARY_PLOT_FILE)

    print(
        "\nSHAP explainability completed successfully."
    )


if __name__ == "__main__":
    main()