import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")
TIMEOUT = 30

def _post(path, payload):
    r = requests.post(f"{API_BASE_URL}/{path}", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def health():
    return requests.get(f"{API_BASE_URL}/health", timeout=5).json()

def predict(features: dict, model_name: str | None = None):
    """Calls POST /predict?model_name=... with the given features."""
    params = {"model_name": model_name} if model_name else {}
    r = requests.post(
        f"{API_BASE_URL}/predict",
        params=params,
        json={"features": features},
        timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json()

def decide(features: dict):
    """Calls POST /decide - RAG + LLM decision route"""
    r = requests.post(
        f"{API_BASE_URL}/decide",
        json={"features": features},
        timeout=TIMEOUT
    )
    r.raise_for_status()
    return r.json()

def explain(features: dict, prediction: dict):
    """Calls POST /llm/explain - LLM explanation of the decision"""
    r = requests.post(
        f"{API_BASE_URL}/llm/explain",
        json={"features": features, "prediction": prediction},
        timeout=60
    )
    r.raise_for_status()
    return r.json()