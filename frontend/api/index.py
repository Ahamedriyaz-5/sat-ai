import os
import sys

# Add both project root and frontend directory to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(CURRENT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.api.endpoints.analyze import router as analyze_router
    from backend.api.endpoints.copernicus import router as copernicus_router
    HAS_BACKEND = True
except Exception as e:
    HAS_BACKEND = False
    BACKEND_ERR = str(e)

app = FastAPI(
    title="SatQuery AI Serverless API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if HAS_BACKEND:
    app.include_router(analyze_router, prefix="/api/v1")
    app.include_router(copernicus_router, prefix="/api/v1")

@app.get("/api")
@app.get("/api/v1")
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "SatQuery AI Serverless",
        "has_backend": HAS_BACKEND,
        "backend_err": None if HAS_BACKEND else BACKEND_ERR,
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
    }
