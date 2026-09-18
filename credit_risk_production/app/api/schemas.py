from typing import Dict, List, Any
from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    features: Dict[str, Any] = Field(..., description="Customer feature row as dict")

class PredictResponse(BaseModel):
    model_used: str
    predicted_flag: int
    class_probabilities: Dict[str, float]
    primary_risk_probability: float

class DecideResponse(BaseModel):
    decision_route: str
    provider: str
    model_used: str
    risk_score: float
    risk_score_100: float
    recommended_action: str
    key_driver: List[str] = []
    policy_refs: List[str] = []
    fraud_indicators: List[str] = []
    confidence: float = 0.0

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    rag_loaded: bool