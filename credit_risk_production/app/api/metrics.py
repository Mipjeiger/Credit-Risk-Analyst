from prometheus_client import Counter, Histogram, Gauge

"""
Prometheus metrics for the Credit Risk API.

Label contract (must match call sites exactly):

  REQUEST_COUNT    -> endpoint, method, status
  REQUEST_LATENCY  -> endpoint
  ML_PREDICTIONS   -> model_name, decision_route
  LLM_FALLBACKS    -> from_provider, to_provider
  RISK_SCORE_HIST  -> (no labels)
  MODEL_LOADED     -> (no labels)
"""

# --- HTTP layer -------------------------------------------------------------
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["endpoint", "method", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
)

# --- ML / decision layer ----------------------------------------------------
ML_PREDICTIONS = Counter(
    "ml_predictions_total",
    "Number of ML predictions served",
    ["model_name", "decision_route"],
)

LLM_FALLBACKS = Counter(
    "llm_fallback_total",
    "LLM provider fallback events",
    ["from_provider", "to_provider"],
)

RISK_SCORE_HIST = Histogram(
    "ml_risk_score",
    "Distribution of primary risk probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# --- Service health ---------------------------------------------------------
MODEL_LOADED = Gauge(
    "model_loaded",
    "1 if the ML model/RAG is loaded, 0 otherwise",
)