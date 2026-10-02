from typing import Optional, Literal
from fastapi import APIRouter, Depends, HTTPException
from app.api.schemas import CustomerFeatures, PredictResponse
from app.api.dependencies import get_fraud_rag

router = APIRouter(tags=["Fraud"])

@router.get("/fraud/models")
def list_fraud_models(rag=Depends(get_fraud_rag)):
    return {"models": rag.model_names()}

@router.post("/fraud/predict", response_model=PredictResponse)
def fraud_predict(
    payload: CustomerFeatures,
    model_name: Optional[str] = None,
    rag=Depends(get_fraud_rag)
):
    try:
        result = rag.score(payload.features, model_name=model_name)
        return PredictResponse(**result)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))