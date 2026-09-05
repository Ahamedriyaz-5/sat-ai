import os
import sys

# Search up directory hierarchy for 'backend' module root
curr = os.path.dirname(os.path.abspath(__file__))
found_root = None
for _ in range(5):
    if os.path.exists(os.path.join(curr, "backend")):
        found_root = curr
        if curr not in sys.path:
            sys.path.insert(0, curr)
        break
    parent = os.path.dirname(curr)
    if parent == curr:
        break
    curr = parent

if not found_root and os.path.exists("/var/task"):
    if "/var/task" not in sys.path:
        sys.path.insert(0, "/var/task")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.api.endpoints.analyze import router as analyze_router
    from backend.api.endpoints.copernicus import router as copernicus_router
    HAS_BACKEND = True
    BACKEND_ERR = None
except Exception as e:
    HAS_BACKEND = False
    BACKEND_ERR = f"{type(e).__name__}: {str(e)}"

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
    task_contents = []
    if os.path.exists("/var/task"):
        try:
            task_contents = os.listdir("/var/task")
        except Exception:
            pass

    return {
        "status": "healthy",
        "service": "SatQuery AI Serverless",
        "has_backend": HAS_BACKEND,
        "backend_err": BACKEND_ERR,
        "found_root": found_root,
        "sys_path": sys.path[:5],
        "task_contents": task_contents,
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
    }
