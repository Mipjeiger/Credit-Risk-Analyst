from typing import Optional, Literal
from fastapi import APIRouter, Depends, HTTPException
from app.api.schemas import CustomerFeatures, PredictResponse
from app.api.dependencies import get_fraud_rag

router = APIRouter()