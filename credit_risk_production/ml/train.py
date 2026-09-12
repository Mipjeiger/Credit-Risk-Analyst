import os
import json
import pandas as pd
import numpy as np
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path

"""Train all models and log to MLflow"""

# Configurations
BASE_PATH = Path(__file__).resolve().parents[2]
MLFLOW_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit_risk_ml")
DATA_PATH = BASE_PATH / "credit_risk_production" / "database" / "data" / "merged_credit_risk_data.parquet"
ARTIFACT_PATH = BASE_PATH / "credit_risk_production" / "database" / "LLM" / "outputs_llm" / "model_artifacts_mlflow"
TARGET       = "Approved_Flag"

def build_models():
    """Build configured ML classifiers."""
    return {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }

def build_param_distributions():
    return {
        'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'saga'],
                'max_iter': [1000, 2000]
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 300],    
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan'],
            },
            'Decision Tree': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
    }

def run():
    """Run the training pipeline on MLFlow"""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Label encode categorical features
    le_map = {}
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_map[col] = le

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                              random_state=42, stratify=y)

    # Train and log models
    param_distributions = build_param_distributions()
    models = build_models()
    metrics_row = []
    best = {"f1": -1, "name": None, "model": None}

    # Train each model with hyperparameter tuning and log to MLflow
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Hyperparameter tuning
            search = RandomizedSearchCV(model,
                                        param_distributions=param_distributions[name],
                                        n_iter=10,
                                        scoring='f1',
                                        cv=5,
                                        random_state=42,
                                        n_jobs=-1,
                                        verbose=1)
            search.fit(X_train, y_train)

            # Get predictions
            preds = search.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
                "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
                "f1": f1_score(y_test, preds, average="weighted", zero_division=0)
            }
            try:
                proba = search.predict_proba(X_test)
                metrics["roc_auc"] = roc_auc_score(y_test, proba, multi_class='ovr', average='weighted')
            except Exception:
                metrics["roc_auc"] = float('nan')

            # Log metrics and parameters
            mlflow.log_params(search.get_params())
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if not np.isnan(v)})
            mlflow.sklearn.log_model(model, artifact_path=f"models/{name}")

            metrics_row.append({"Model": name, **metrics})

            if metrics["f1"] > best["f1"]:
                best = {"f1": metrics["f1"], "name": name, "model": model}

    # Persist the bundle saving
    Path(ARTIFACT_PATH).mkdir(parents=True, exist_ok=True)
    bundle = {
        "models": models,
        "scaler": scaler,
        "label_encoders": le_map,
        "feature_columns": list(X.columns),
        "class_labels": sorted(y.unique().tolist())
    }
    joblib.dump(bundle, ARTIFACT_PATH / "model_bundle.joblib")
    pd.DataFrame(metrics_row).to_csv(Path(ARTIFACT_PATH) / "model_metrics.csv", index=False)

    # Log models to MLflow
    with mlflow.start_run(run_name="bundle_registration"):
        mlflow.log_artifact(str(Path(ARTIFACT_PATH) / "model_bundle.joblib"))
        mlflow.log_artifact(str(Path(ARTIFACT_PATH) / "model_metrics.csv"))
        mlflow.log_metric("best_f1", best["f1"])
        mlflow.set_tag("best_model", best["name"])

    print(f"✅ Trained {len(models)} models. Best: {best['name']} (F1={best['f1']:.4f})")

if __name__ == "__main__":
    run()