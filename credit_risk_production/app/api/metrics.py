from prometheus_client import Counter, Histogram, Gauge

"""Dependencies for Prometheus metrics collection in the API."""

REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["endpoint", "method", "status"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds", "Request latency", ["endpoint"]
)
ML_PREDICTIONS = Counter(
    "ml_predictions_total", "ML predictions", ["model_name", "decision_route"]
)
LLM_FALLBACKS = Counter(
    "llm_fallback_total", "LLM provider fallback events", ["from_provider", "to_provider"]
)
RISK_SCORE_HIST = Histogram(
    "risk_score_value", "Distribution of risk scores", buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)
MODEL_LOADED = Gauge("model_loaded", "1 if model loaded")