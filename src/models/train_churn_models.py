from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "telco_customer_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_FILE = MODEL_DIR / "churn_model.joblib"

METRICS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "churn_model_comparison.csv"
)


def build_preprocessor(
    numeric_columns,
    categorical_columns,
):
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_columns,
            ),
        ]
    )


def evaluate_model(
    name,
    pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
):
    print(f"\nTraining {name}...")

    pipeline.fit(
        X_train,
        y_train,
    )

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "ROC_AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    print(
        f"Accuracy : "
        f"{metrics['Accuracy']:.4f}"
    )
    print(
        f"Precision: "
        f"{metrics['Precision']:.4f}"
    )
    print(
        f"Recall   : "
        f"{metrics['Recall']:.4f}"
    )
    print(
        f"F1       : "
        f"{metrics['F1']:.4f}"
    )
    print(
        f"ROC-AUC  : "
        f"{metrics['ROC_AUC']:.4f}"
    )

    return metrics


def main():
    print("Loading customer churn feature data...")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Feature data not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print(
        f"Dataset Shape: {df.shape}"
    )

    if "Churn" not in df.columns:
        raise ValueError(
            "Churn target column not found."
        )

    if "customerID" not in df.columns:
        raise ValueError(
            "customerID column not found."
        )

    y = (
        df["Churn"]
        .map(
            {
                "No": 0,
                "Yes": 1,
            }
        )
    )

    if y.isna().any():
        raise ValueError(
            "Unexpected values found in Churn column."
        )

    X = df.drop(
        columns=[
            "customerID",
            "Churn",
        ]
    )

    numeric_columns = (
        X.select_dtypes(
            include=["number"]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            exclude=["number"]
        )
        .columns
        .tolist()
    )

    print(
        f"Predictor Columns: {X.shape[1]}"
    )
    print(
        f"Numeric Columns: "
        f"{len(numeric_columns)}"
    )
    print(
        f"Categorical Columns: "
        f"{len(categorical_columns)}"
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
    )

    print(
        f"Training Records: {len(X_train)}"
    )
    print(
        f"Testing Records: {len(X_test)}"
    )

    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print(
        f"XGBoost scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    base_preprocessor = build_preprocessor(
        numeric_columns,
        categorical_columns,
    )

    models = {
        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            ),
    }

    results = []

    for model_name, model in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    clone(base_preprocessor),
                ),
                (
                    "classifier",
                    model,
                ),
            ]
        )

        metrics = evaluate_model(
            model_name,
            pipeline,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="F1",
        ascending=False,
    ).reset_index(drop=True)

    print("\nModel Comparison")
    print("----------------")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    best_model_name = (
        results_df.iloc[0]["Model"]
    )

    print(
        f"\nSelected Final Model: "
        f"{best_model_name}"
    )

    print(
        "Selection Criterion: "
        "Highest F1-score"
    )

    final_model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns,
                    categorical_columns,
                ),
            ),
            (
                "classifier",
                models[best_model_name],
            ),
        ]
    )

    print(
        "\nTraining selected model "
        "on the complete dataset..."
    )

    final_model.fit(
        X,
        y,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        MODEL_FILE,
    )

    results_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    print(
        f"\nFinal churn model saved to:"
    )
    print(MODEL_FILE)

    print(
        "\nModel comparison saved to:"
    )
    print(METRICS_FILE)

    print(
        "\nChurn model training "
        "completed successfully."
    )


if __name__ == "__main__":
    main()