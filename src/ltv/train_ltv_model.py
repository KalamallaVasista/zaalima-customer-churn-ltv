import pandas as pd
import numpy as np
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


DATA_PATH = Path(
    "data/processed/ltv_active_customers.csv"
)


def train_ltv_model():
    print("=" * 60)
    print("CUSTOMER LIFETIME VALUE - REGRESSION MODEL")
    print("=" * 60)

    # Load LTV dataset
    df = pd.read_csv(DATA_PATH)

    print(f"\nActive Customers: {len(df)}")

    # -------------------------------------------------
    # Target Variable
    # -------------------------------------------------

    y = df["ProjectedLTV"]

    # -------------------------------------------------
    # Predictor Features
    # -------------------------------------------------
    # Remove:
    # - customerID because it is only an identifier
    # - Churn because LTV data contains active customers only
    # - ProjectedLTV because it is the target
    # - CurrentRevenue and Projected12MonthRevenue because
    #   they are directly used to calculate ProjectedLTV
    # - TotalCharges because it is directly represented
    #   in the LTV target calculation

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
    # Complete Pipeline
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

    pipeline.fit(X_train, y_train)

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
    # Display Results
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("LTV MODEL PERFORMANCE")
    print("=" * 60)

    print(f"\nMAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    print("\nLTV model training completed successfully.")


if __name__ == "__main__":
    train_ltv_model()