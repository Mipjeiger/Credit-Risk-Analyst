import logging
import traceback
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from app.api.schemas import CustomerFeatures, DecideResponse
from app.api.dependencies import get_rag
from app.api.schemas import ChatRequest, ChatResponse, CustomerFeatures, DecideResponse
from llmops.rag_loader import CreditRiskRAG

# Configuration router & logging
logger = logging.getLogger(__name__)
router = APIRouter(tags=["llm"])

# System prompt template
DEFAULT_CHATBOT_SYSTEM_PROMPT = (
    "You are a senior credit-risk analyst assistant for a bank/fintech. "
    "Answer questions about credit policy, scorecards, and fraud typologies "
    "in plain business language. Be concise and do not invent numbers or "
    "policy references that are not provided to you."
)

# -------------------------------------------
# LLM Engine deiciding explanations
# -------------------------------------------
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
    
    except Exception as exc:  # noqa: BLE001  # noqa: BLE001
        logger.error("LLM /llm failed:\n%s", traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=(
                f"Unable to process the credit-risk decision: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

@router.post("/llm/chatbot", response_model=DecideResponse)
