import os
from pathlib import Path
from llmops.rag_loader import LLM_PATH

class Settings:
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_MODEL_NAME: str =  os.getenv("MLFLOW_MODEL_NAME", "credit_risk_bundle")
    MLFLOW_STAGE: str =  os.getenv("MLFLOW_STAGE", "Production")
    RAG_BASE_DIR: Path = LLM_PATH
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()