import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -------------------------------------------------
# File Paths
# -------------------------------------------------

DATA_PATH = Path(
    "data/processed/ltv_active_customers.csv"
)

MODEL_PATH = Path(
    "models/ltv_model.joblib"
)


def train_ltv_model():

    print("=" * 60)
    print("CUSTOMER LIFETIME VALUE - REGRESSION MODEL")
    print("=" * 60)

    # -------------------------------------------------
    # Load Dataset
    # -------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    print(f"\nActive Customers: {len(df)}")

    # -------------------------------------------------
    # Target Variable
    # -------------------------------------------------

    y = df["ProjectedLTV"]

    # -------------------------------------------------
    # Predictor Features
    # -------------------------------------------------
    # These columns are removed to prevent target leakage.
    #
    # customerID:
    #     Identifier only.
    #
    # Churn:
    #     LTV dataset already contains active customers only.
    #
    # ProjectedLTV:
    #     Target variable.
    #
    # CurrentRevenue and Projected12MonthRevenue:
    #     Directly used to calculate ProjectedLTV.
    #
    # TotalCharges:
    #     Directly contributes to our ProjectedLTV target.

    drop_columns = [
        "customerID",
        "Churn",
        "ProjectedLTV",
        "CurrentRevenue",
        "Projected12MonthRevenue",
        "TotalCharges"
    ]

    X = df.drop(columns=drop_columns)

    print(f"Predictor Columns: {X.shape[1]}")

    # -------------------------------------------------
    # Identify Numeric and Categorical Features
    # -------------------------------------------------

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    print(f"Numeric Features: {len(numeric_features)}")
    print(f"Categorical Features: {len(categorical_features)}")

    # -------------------------------------------------
    # Preprocessing
    # -------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "numeric",
                "passthrough",
                numeric_features
            )
        ]
    )

    # -------------------------------------------------
    # Random Forest Regression Model
    # -------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    # -------------------------------------------------
    # Create Complete ML Pipeline
    # -------------------------------------------------

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # -------------------------------------------------
    # Train/Test Split
    # -------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("\nTraining Records:", len(X_train))
    print("Testing Records:", len(X_test))

    # -------------------------------------------------
    # Train Model
    # -------------------------------------------------

    print("\nTraining Random Forest Regressor...")

    pipeline.fit(
        X_train,
        y_train
    )

    # -------------------------------------------------
    # Generate Predictions
    # -------------------------------------------------

    predictions = pipeline.predict(X_test)

    # -------------------------------------------------
    # Model Evaluation
    # -------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # -------------------------------------------------
    # Display Model Performance
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("LTV MODEL PERFORMANCE")
    print("=" * 60)

    print(f"\nMAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    # -------------------------------------------------
    # Create Models Folder
    # -------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # Save Trained Pipeline
    # -------------------------------------------------

    joblib.dump(
        pipeline,
        MODEL_PATH
    )

    print(
        f"\nTrained LTV model saved to: {MODEL_PATH}"
    )

    print(
        "\nLTV model training completed successfully."
    )


if __name__ == "__main__":
    train_ltv_model()