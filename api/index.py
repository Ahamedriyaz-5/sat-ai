import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.endpoints.analyze import router as analyze_router
from backend.api.endpoints.copernicus import router as copernicus_router

app = FastAPI(
    title="SatQuery AI Serverless API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include lightweight serverless routers
app.include_router(analyze_router, prefix="/api/v1")
app.include_router(copernicus_router, prefix="/api/v1")

@app.get("/api")
@app.get("/api/v1")
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SatQuery AI Serverless",
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
    }
