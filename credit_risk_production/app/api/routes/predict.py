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

# Define allowed model choices matching on model files keys
ModelType = Literal[
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "K-Nearest Neighbors",
    "Decision Tree"
]

router = APIRouter(tags=["predict"])

@router.post("/predict", response_model=PredictResponse)
def predict(
    payload: CustomerFeatures, 
    rag: Annotated[object, Depends(get_rag)],
    model_name: ModelType | None = None,
):
    start = time.time()

    try:
        # Mark model as loaded during inference execution
        MODEL_LOADED.set(1)

        row = pd.Series(payload.features)

        # Predict scoring using selected model
        ml = rag.score(row, model_name=model_name)

        # Record metrics
        RISK_SCORE_HIST.observe(ml["primary_risk_probability"])
        ML_PREDICTIONS.labels(
            model_name=ml.get("model_used", model_name),
            decision_route="PREDICT",
        ).inc()

        # Track success with matching status="200" label key
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()
        return PredictResponse(**ml)

    except (KeyError, ValueError) as e:
        REQUEST_COUNT.labels(endpoint="/predict", method="POST", status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start)