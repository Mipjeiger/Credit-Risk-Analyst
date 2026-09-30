from functools import lru_cache
from pathlib import Path
from app.api.config import settings
from llmops.rag_loader import CreditRiskRAG

@lru_cache(maxsize=1)
def get_rag() -> CreditRiskRAG:
    """Get the RAG instance for credit risk."""
    return CreditRiskRAG(base_dir=Path(settings.RAG_BASE_DIR))