from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field

class CustomerFeatures(BaseModel):
    features: dict[str, Any] = Field(..., description="Customer feature row as dict")

class PredictResponse(BaseModel):
    model_used: str
    predicted_flag: int
    class_probabilities: dict[str, float]
    primary_risk_probability: float | None = None
    primary_fraud_probability: float | None = None

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

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        examples=[
            [
                {"role": "user", 
                 "content": "What are the top fraud typologies? (example -> you can try with ur own prompt)"}
            ]
        ],
    )
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    system_prompt: Optional[str] = Field(
        None,
        examples=["You are a senior fraud risk analyst. Be concise. (example -> you can try with ur own prompt)"],
    )

class ChatResponse(BaseModel):
    provider: str
    text: str