from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import CustomerFeatures, DecideResponse
from llmops.rag_loader import CreditRiskRAG

router = APIRouter(tags=["llm"])

@router.post("/llm", response_model=DecideResponse)
def llm_query(
    payload: CustomerFeatures,
    rag: Annotated[CreditRiskRAG, Depends(CreditRiskRAG)],
) -> DecideResponse:
    """Endpoint for LLM-based retrieval queries and responses."""
    try:
        result = rag.decide(payload.features)
        return DecideResponse.model_validate(result)

    except KeyError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Missing or invalid applicant feature: {exc}",
        ) from exc
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Unable to process the credit-risk decision.",
        )