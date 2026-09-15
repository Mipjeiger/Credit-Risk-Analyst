from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.api.routes import health, predict, decide
from app.api.config import settings

# Build FastAPI app
app = FastAPI(
    title="Credit Risk API",
    version="1.0.0",
    description="ML risk scoring + LLM-powered RAG decisions",
)

# Include API routes
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(decide.router)

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/")
def root():
    return {"service": "Credit Risk API",
            "version": "1.0.0"}