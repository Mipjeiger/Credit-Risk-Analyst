import os


class Settings:
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    MLFLOW_MODEL_NAME: str = os.getenv("MLFLOW_MODEL_NAME", "credit_risk_bundle")
    MLFLOW_STAGE: str = os.getenv("MLFLOW_STAGE", "Production")
    RAG_BASE_DIR: str = os.getenv("RAG_BASE_DIR", "./database/LLM")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()