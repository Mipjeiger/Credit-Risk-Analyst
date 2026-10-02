from functools import lru_cache
from pathlib import Path
from app.api.config import settings
from llmops.rag_loader import CreditRiskRAG
from llmops.fraud_rag_loader import FraudRiskRAG

@lru_cache(maxsize=1)
def get_rag() -> CreditRiskRAG:
    """Get the RAG instance for credit risk."""
    return CreditRiskRAG(base_dir=Path(settings.RAG_BASE_DIR))

@lru_cache(maxsize=1)
def get_fraud_rag() -> FraudRiskRAG:
    """Get the RAG instance for fraud risk."""
    return FraudRiskRAG(
        base_dir=Path(settings.FRAUD_RAG_BASE_DIR),
        bundle_path=Path(settings.FRAUD_BUNDLE_PATH) if getattr(settings, "FRAUD_BUNDLE_PATH", None) else None
    )