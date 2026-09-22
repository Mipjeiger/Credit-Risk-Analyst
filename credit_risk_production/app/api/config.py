import os


class Settings:
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "credit_risk_bundle")
    MLFLOW_STAGE = os.getenv("MLFLOW_STAGE", "Production")
    RAG_BASE_DIR = os.getenv("RAG_BASE_DIR", "./database/LLM")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
