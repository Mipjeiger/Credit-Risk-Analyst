import logging
import time
from typing import Annotated, Literal
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from app.api.dependencies import get_rag
from app.api.metrics import (
    ML_PREDICTIONS,
    MODEL_LOADED,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    RISK_SCORE_HIST,
)
from app.api.schemas import CustomerFeatures, PredictResponse
from llmops.rag_loader import CreditRiskRAG
from cache.feature_cache import get_or_score
from cache.redis_client import incr_velocity, upsert_profile, read_screen, upsert_screen
from cache.exposure import approve_credit, record_decision

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Define allowed model choices matching on model files keys
ModelType = Literal[
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "K-Nearest Neighbors",
    "Decision Tree",
]

router = APIRouter(tags=["predict"])

@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: CustomerFeatures,
    rag: Annotated[CreditRiskRAG, Depends(get_rag)],
    model_name: ModelType | None = None,
):
    start = time.time()

    try:
        MODEL_LOADED.set(1)

        # --- 1. Rate limiting check ---
        customer_id = str(payload.features.get("customer_id") or payload.features.get("ProspectID") or "unknown")
        rate = incr_velocity(customer_id, "predict_1m", ttl=60)

        if rate > 100:
            REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="429").inc()
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # --- 2. Screen check ---
        screen_bits = read_screen(customer_id)
        high_risk_screen = bool(screen_bits & (1 << 1)) # has recent delinquency

        # --- 3. Profile upsert ---
        if customer_id != "unknown":
            try:
                upsert_profile(customer_id, payload.features, ttl_seconds=86400)  # 1 day TTL
                upsert_screen(customer_id, payload.features, ttl=3600) 
            except Exception as exc:
                logger.warning(f"[predict] profile upsert skipped: {exc}")

        # --- 4. Score with caching ---
        cache_prefix = f"Score:{model_name or 'default'}"

        def _score(features: dict) -> dict:
            row = pd.Series(features)
            return rag.score(row, model_name=model_name)

        ml = get_or_score(
            prefix=cache_prefix,
            payload=payload.features,
            scorer=_score,
            ttl=600  # cache for 10 minutes
        )
        ml["screen_high_risk"] = high_risk_screen # Get the features score

        # --- 5. Metrics + aggregation ---
        RISK_SCORE_HIST.observe(ml["primary_risk_probability"])
        ML_PREDICTIONS.labels(
            model_name=ml.get("model_used", model_name or "default"),
            decision_route="PREDICT"
        ).inc()

        if customer_id != "unknown":
            hour_key = time.strftime("%Y-%m-%dT%H")
            try:
                # Use primary risk probability returned by rag.score()
                risk_val = float(ml.get("risk_score", ml.get("risk_score", 0.0)))
                record_decision(customer_id, risk_val, hour_key)
            except Exception as exc:
                logger.warning(f"❌ [predict] record_decision failed: {exc}")

        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()
        return PredictResponse(**{k: v for k, v in ml.items() if k in PredictResponse.model_fields})

    except HTTPException:
        raise

    except EOFError as e:
        logger.error(f"[predict] Non-interactive input error (EOFError): {e}")
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="500").inc()
        raise HTTPException(
            status_code=500, 
            detail="Execution failed due to non-interactive environment (EOFError)."
        ) from e

    except Exception as e:
        REQUEST_COUNT.labels(
            endpoint="/predict", method="POST", status="500"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e)) from e

    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)

@router.get("/models")
def list_models(rag=Depends(get_rag)):
    return {"models": sorted(rag.models.keys())}