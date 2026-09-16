from fastapi import APIRouter, Depends
from app.api.schemas import HealthResponse
from app.api.dependencies import get_rag
from app.api.metrics import MODEL_LOADED

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def health(rag=Depends(get_rag)):
    MODEL_LOADED.set(1)
    return HealthResponse(status="ok", model_loaded=True, rag_loaded=rag is not None)

# Endpoint router
@router.get("/ready")
def ready():
    return {"status": "ready"}