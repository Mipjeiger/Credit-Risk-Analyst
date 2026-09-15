import time
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from app.api.schemas import CustomerFeatures, DecideResponse
from app.api.dependencies import get_rag
from app.api.metrics import REQUEST_COUNT, REQUEST_LATENCY, ML_PREDICTIONS, LLM_FALLBACKS, RISK_SCORE_HIST

router = APIRouter(tags=["decide"])

# Endpoint
@router.post("/decide", response_model=DecideResponse)
def decide(payload: CustomerFeatures, rag=Depends(get_rag)):
    start = time.time()

    try:
        row = pd.Series(payload.features)
        result = rag.decide(row)

        RISK_SCORE_HIST.observe(result.get("risk_score", 0.0))
        ML_PREDICTIONS.labels(
            model_name=result.get("model_used", "unknown"),
            decision_route=result.get("decision_route", "unknown")
        ).inc()

        if result.get("provider") == "groq":
            LLM_FALLBACKS.labels(
                from_provider="huggingface",
                to_provider="groq"
            ).inc()

        REQUEST_COUNT.labels(endpoint="/decide", method="POST", status="200").inc()

        return DecideResponse(
            decision_route=result["decision_route"],
            provider=result["provider"],
            model_used=result.get("model_used", "unknown"),
            risk_score=result.get("risk_score", 0.0),
            risk_score_100=result.get("risk_score_100", 0.0),
            recommended_action=result.get("recommended_action", ""),
            key_drivers=result.get("key_drivers", []),
            policy_refs=result.get("policy_refs", []),
            fraud_indicators=result.get("fraud_indicators", []),
            confidence=float(result.get("confidence", 0.0) or 0.0)
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/decide", method="POST", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        REQUEST_LATENCY.labels(endpoint="/decide").observe(time.time() - start)