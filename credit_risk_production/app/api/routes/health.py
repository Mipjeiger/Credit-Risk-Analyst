from fastapi import APIRouter, Depends

from app.api.dependencies import get_rag
from app.api.metrics import MODEL_LOADED
from app.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health(rag=Depends(get_rag)):  # noqa: B008
    MODEL_LOADED.set(1)
    return HealthResponse(status="ok", model_loaded=True, rag_loaded=rag is not None)


@router.get("/ready")
def ready():
    return {"status": "ready"}
