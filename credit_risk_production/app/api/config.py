import os
from pathlib import Path
from llmops.rag_loader import LLM_PATH

class Settings:
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_MODEL_NAME: str =  os.getenv("MLFLOW_MODEL_NAME", "credit_risk_bundle")
    MLFLOW_STAGE: str =  os.getenv("MLFLOW_STAGE", "Production")

    # Credit: Credit risk RAG settings dependencies
    RAG_BASE_DIR: Path = Path(os.getenv("RAG_BASE_DIR", str(LLM_PATH)))

    # Fraud: Fraud risk RAG settings dependencies
    FRAUD_RAG_BASE_DIR: Path = Path(os.getenv("FRAUD_RAG_BASE_DIR", "/data_science/database/LLM"))
    FRAUD_BUNDLE_PATH: Path = Path(os.getenv("FRAUD_BUNDLE_PATH", "/data_science/models/fraud_models/model_bundle.joblib"))

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()