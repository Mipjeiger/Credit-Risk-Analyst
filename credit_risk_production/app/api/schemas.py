from typing import Any

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    features: dict[str, Any] = Field(..., description="Customer feature row as dict")


class PredictResponse(BaseModel):
    model_used: str
    predicted_flag: int
    class_probabilities: dict[str, float]
    primary_risk_probability: float


class DecideResponse(BaseModel):
    decision_route: str
    provider: str
    model_used: str
    risk_score: float
    risk_score_100: float
    recommended_action: str
    key_driver: list[str] = []
    policy_refs: list[str] = []
    fraud_indicators: list[str] = []
    confidence: float = 0.0


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    rag_loaded: bool
