import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# Configuration
BASE_PATH = Path("/app") # Path by docker container -> back to base as /app
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit_risk_ml")
DATA_PATH = BASE_PATH / "database" / "data" / "merged_credit_risk_data.parquet"
ARTIFACT_PATH = BASE_PATH / "credit_risk_production" / "database" / "LLM" / "outputs_llm" / "model_artifacts_mlflow"
TARGET = "Approved_Flag"

"""Train all models with RandomizedSearchCV, log & register them to MLflow."""

def registry_name(model_name: str) -> str:
    """Registry names must match ^[a-zA-Z0-9._-]+$"""
    return "credit_risk_" + model_name.lower().replace(" ", "_").replace("-", "_")

def build_models():
    return {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest":       RandomForestClassifier(random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
        "XGBoost":             XGBClassifier(eval_metric="mlogloss", random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
    }

def build_param_distributions():
    return {
        'Logistic Regression': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [100, 200]
        },
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],    
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGBoost': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'K-Nearest Neighbors': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'metric': ['minkowski'],
        },
        'Decision Tree': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }

def run():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = pd.read_parquet(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Encode target (y) first -> it's uniqe categorical by seen (p1, p2, p3, p4)
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)

    # 1. Train-Test Split First (Prevents Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 2. Fit LabelEncoders ONLY on X_train
    le_map = {}
    X_train = X_train.copy()
    X_test = X_test .copy()

    # Encode categorical features using LabelEncoder
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        test_vals = X_test[col].astype(str)

        # Encode test values by handling unseen categories: if unseen, assign a default value (e.g., 0)
        X_test[col] = test_vals.map(lambda s: le.transform([s])[0] if s in le.classes_ else 0)
        le_map[col] = le

    # 3. Fit StandardScaler ONLY on X_train
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Infer signature: log the schema of the input and output data for the model
    signature = infer_signature(X_train_scaled, y_train)
    input_example = X_train_scaled.head(3)

    param_distributions = build_param_distributions()
    models = build_models()

    metrics_rows, tuned_models = [], {}
    best = {"f1": -1.0, "name": None, "model": None}

    # Train and log each model with RandomizedSearchCV
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions[name],
                n_iter=10, 
                scoring="f1_weighted", 
                cv=5,
                random_state=42, 
                n_jobs=-1, 
                verbose=2, 
                refit=True,
            )
            search.fit(X_train_scaled, y_train)
            tuned = search.best_estimator_

            preds = tuned.predict(X_test_scaled)
            metrics = {
                "accuracy":  accuracy_score(y_test, preds),
                "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
                "recall":    recall_score(y_test, preds, average="weighted", zero_division=0),
                "f1":        f1_score(y_test, preds, average="weighted", zero_division=0),
                "cv_best_f1": search.best_score_,
            }
            try:
                proba = tuned.predict_proba(X_test_scaled)
                metrics["roc_auc"] = roc_auc_score(
                    y_test, proba, multi_class="ovr", average="weighted"
                )
            except Exception:
                metrics["roc_auc"] = float("nan")

            # Log parameters, metrics, and the tuned model to MLflow
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if not np.isnan(v)})

            # Mlflow logging for model registration -> Check on UI Mlflow
            mlflow.sklearn.log_model(
                sk_model=tuned,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registry_name(name),
            )
            
            metrics_rows.append({"Model": name, **metrics})
            tuned_models[name] = tuned

            if metrics["f1"] > best["f1"]:
                best = {"f1": metrics["f1"], "name": name, "model": tuned}

    # Bundle of TUNED models
    ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
    bundle = {
        "models": tuned_models,
        "scaler": scaler,
        "label_encoders": le_map,
        "feature_columns": list(X.columns),
        "class_labels": sorted(y.unique().tolist()),
    }
    bundle_path  = ARTIFACT_PATH / "model_bundle.joblib"
    metrics_path = ARTIFACT_PATH / "model_metrics.csv"
    joblib.dump(bundle, bundle_path)
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    with mlflow.start_run(run_name="bundle_registration"):
        mlflow.log_artifact(str(bundle_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_metric("best_f1", best["f1"])
        mlflow.set_tag("best_model", best["name"])
        mlflow.sklearn.log_model(
            sk_model=best["model"],
            artifact_path="best_model",
            signature=signature,
            input_example=input_example,
            registered_model_name="credit_risk_best",
        )

    print(f"✅ Trained {len(tuned_models)} models. Best: {best['name']} (F1={best['f1']:.4f}) ⚡")

if __name__ == "__main__":
    run()