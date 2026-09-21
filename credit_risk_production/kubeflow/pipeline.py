from typing import List
from kfp import dsl
from kfp.dsl import component, Output, Input, Artifact, Metrics

# =========================
# Component 1: Ingest Data
# =========================
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "pyarrow"]
)
def ingest(data_path: str, out_data: Output[Artifact]):
    import shutil
    import os
    os.makedirs(os.path.dirname(out_data.path), exist_ok=True)
    shutil.copy(data_path, out_data.path)

# =========================
# Component 2: Train Model
# =========================
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn", "xgboost", "mlflow", "joblib", "pyarrow"]
)
def train(
    data: Input[Artifact], 
    model_type: str,
    n_estimators: int, 
    mlflow_uri: str, 
    out_model: Output[Artifact],
    out_test_data: Output[Artifact]
):
    import joblib
    import mlflow
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("credit_risk_ml")

    df = pd.read_parquet(data.path)
    X = df.drop(columns=["Approved_Flag"]).select_dtypes(include=["number"])
    y = df["Approved_Flag"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Export test dataset for evaluation
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_parquet(out_test_data.path, index=False)

    # Dynamic model selector
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=n_estimators, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=n_estimators, random_state=42),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "knn": KNeighborsClassifier(),
        "xgboost": XGBClassifier(n_estimators=n_estimators, random_state=42, eval_metric="logloss")
    }

    if model_type not in models:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(models.keys())}")

    # append model to models dictionary
    model = models[model_type]

    with mlflow.start_run(run_name=f"kubeflow_train_{model_type}") as run:
        model.fit(X_train, y_train)

        # Save model binary directly to artifact path
        joblib.dump(model, out_model.path)

        # Log parameters & model artifact to MLflow
        mlflow.log_param("model_type", model_type)
        if hasattr(model, "n_estimators"):
            mlflow.log_param("n_estimators", n_estimators)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Log active run ID for reference
        out_model.metadata["mlflow_run_id"] = run.info.run_id

# ========================
# Component 3: Evaluate Model
# ========================
@component(
    base_image="python:3.11-slim",
    packages_to_install=["pandas", "scikit-learn", "joblib", "mlflow", "pyarrow"]
)
def evaluate(
    model_artifact: Input[Artifact],
    test_data: Input[Artifact],
    mlflow_uri: str,
    kfp_metrics: Output[Metrics]
) -> float:
    import joblib
    import mlflow
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )

    mlflow.set_tracking_uri(mlflow_uri)
    model = joblib.load(model_artifact.path)

    df = pd.read_parquet(test_data.path)
    X_test = df.drop(columns=["Approved_Flag"])
    y_test = df["Approved_Flag"]

    # Define y predictions
    y_pred = model.predict(X_test)

    # Check if target is binary or multi-class for metrics calculation
    unique_classes = np.unique(y_test)
    is_binary = len(unique_classes) == 2
    average_mode = "binary" if is_binary else "weighted"

    # Compute predicted probabilities for ROC AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

        if is_binary:
            roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
    else:
        roc_auc = float("nan")  # Model does not support probability predictions

    # Evaluate metrics model
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=average_mode, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=average_mode, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average=average_mode, zero_division=0)),
        "roc_auc": roc_auc
    }

    # Log to MLflow run
    run_id = model_artifact.metadata.get("mlflow_run_id")
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)

    # Log metrics to Kubeflow UI
    for k, v in metrics.items():
        kfp_metrics.log_metric(k, v)

    return metrics["f1"]

# ========================
# Component 4: Register Model
# ========================
@component(
    base_image="python:3.11-slim",
    packages_to_install=["mlflow"]
)
def select_and_register_best_model(
    mlflow_uri: str,
    model_name: str,
    model_types: List[str],
    eval_metrics: List[float],
    model_artifacts: Input[Artifact],
    threshold: float = 0.70
) -> str:
    import mlflow
    from mlflow.tracking import MlflowClient

    # Map model types to evaluation metrics scores
    model_scores = dict(zip(model_types, eval_metrics))
    best_model_type = max(model_scores, key=model_scores.get)
    best_score = model_scores[best_model_type]

    print(f"Model Evaluation Summary: {model_scores}")
    print(f"Best Candidate Model: '{best_model_type}' with F1-Score: {best_score:.4f}")

    # Build logic eval metric threshold
    if best_score < threshold:
        return f"All models rejected. Best score ({best_score:.4f}) is below threshold ({threshold})."

    # Fetch corresponding MLflow run for the best model
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    exp = client.get_experiment_by_name("credit_risk_ml")

    # Query MLflow run for the best model
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"params.model_type = '{best_model_type}'",
        order_by=["metrics.f1 DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No matching MLflow run found for the best model.")

    best_run_id = runs[0].info.run_id
    model_uri = f"runs:/{best_run_id}/model"

    # Register best model to MLflow Model Registry
    res = mlflow.register_model(model_uri, model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=res.version,
        stage="Production",
        archive_existing_versions=True
    )

    return f"✅ Successfully registered best model '{best_model_type}' (v{res.version}) with F1: {best_score:.4f} to Production!"

# ========================
# Multi-Model Kubeflow Pipeline Definition
# ========================
@dsl.pipeline(
    name="credit-risk_training",
    description="End-to-end Train -> Evaluate -> Register ML Pipeline for Credit Risk"
)
def credit_risk_pipeline(
    data_path: str,
    mlflow_uri: str,
    candidate_models: List[str] = [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
        "decision_tree",
        "knn",
        "xgboost"
    ],
    n_estimators: int = 200,
    registration_threshold: float = 0.70
):
    # Step 1: Ingest data
    ingest_task = ingest(data_path=data_path)

    eval_scores = []
    trained_models = []

   # Step 2 & 3: Train and Evaluate all models in parallel
    with dsl.ParallelFor(candidate_models) as model_type:
        train_task = train(
            data=ingest_task.outputs["out_data"],
            model_type=model_type,
            n_estimators=n_estimators,
            mlflow_uri=mlflow_uri
        )

        evaluate_task = evaluate(
            model_artifact=train_task.outputs["out_model"],
            test_data=train_task.outputs["out_test_data"],
            mlflow_uri=mlflow_uri
        )

        # Collect metrics and artifacts across parallel runs
        eval_scores.append(evaluate_task.output)
        trained_models.append(train_task.outputs["out_model"])

    # Step 4: Compare candidate metrics and register the winner
    select_and_register_best_model(
        mlflow_uri=mlflow_uri,
        model_name="credit_risk_model",
        model_types=candidate_models,
        eval_metrics=eval_scores,
        model_artifacts=trained_models[0], # Passes collected run artifacts
        threshold=registration_threshold
    )