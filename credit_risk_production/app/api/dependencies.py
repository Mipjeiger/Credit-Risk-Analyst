from functools import lru_cache
from llmops.rag_loader import CreditRiskRAG
from app.api.config import settings

@lru_cache(maxsize=1)
def get_rag() -> CreditRiskRAG:
    """Get the RAG instance for credit risk."""
    return CreditRiskRAG(base_dir=settings.RAG_BASE_DIR)