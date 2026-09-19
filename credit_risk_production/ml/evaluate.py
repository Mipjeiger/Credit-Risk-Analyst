import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Configuration
BASE_PATH = Path(__file__).resolve().parents[2]
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit_risk_ml")
DATA_PATH = (
    BASE_PATH
    / "credit_risk_production"
    / "database"
    / "data"
    / "merged_credit_risk_data.parquet"
)
ARTIFACT_PATH = (
    BASE_PATH
    / "credit_risk_production"
    / "database"
    / "LLM"
    / "outputs_llm"
    / "model_artifacts_mlflow"
)
TARGET = "Approved_Flag"


def run():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    bundle = joblib.load(ARTIFACT_PATH / "model_bundle.joblib")
    df = pd.read_parquet(DATA_PATH)

    X = df[bundle["feature_columns"]].copy()
    y = df[TARGET]

    # Train-Test Split First (Prevents Data Leakage)
    X_train, X_test, _y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fit LabelEncoders
    for col, le in bundle["label_encoders"].items():
        if col in X.columns:
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))

    # Scale the features with StandardScaler
    scaler = bundle["scaler"]
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # MLFlow Logging
    with mlflow.start_run(run_name="evaluation"):
        for name, model in bundle["models"].items():
            preds = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, preds)
            report = classification_report(
                y_test, preds, output_dict=True, zero_division=0
            )

            # Save metrics and confusion matrix to MLFlow
            np.savez(Path(ARTIFACT_PATH) / f"cm_{name.replace(' ', '_')}.npz", cm=cm)
            mlflow.log_dict(report, f"reports/{name}.json")
            mlflow.log_metric(f"{name}_macro_f1", report["macro avg"]["f1-score"])

        print("✅ Evaluation logged")


if __name__ == "__main__":
    run()
