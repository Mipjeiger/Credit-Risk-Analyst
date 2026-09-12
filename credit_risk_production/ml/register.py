import os
import mlflow
from mlflow.tracking import MlflowClient

"""Register the best model in MLFlow Model Registry."""

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "credit_risk_ml")
MODEL_NAME = "credit_risk_bundle"

def run():
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    # define client and experiment
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="tags.best_model != ''",
        order_by=["metrics.best_f1 DESC"],
        max_results=1
    )

    if not runs:
        raise RuntimeError("❌ No best model run found")

    best = runs[0]
    model_uri = f"runs:/{best.info.run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)

    # Transition the model to Production stage
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"✅ Registered {MODEL_NAME} v{result.version} → Production")

if __name__ == "__main__":
    run()