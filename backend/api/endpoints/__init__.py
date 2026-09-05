"""
Endpoints package initialization.
Heavy ML modules are safely imported with try-except to support lightweight serverless deployments (e.g. Vercel).
"""

__all__ = []

try:
    from backend.api.endpoints.vqa import router as vqa_router
    __all__.append("vqa_router")
except Exception:
    vqa_router = None

try:
    from backend.api.endpoints.caption import router as caption_router
    __all__.append("caption_router")
except Exception:
    caption_router = None

try:
    from backend.api.endpoints.grounding import router as grounding_router
    __all__.append("grounding_router")
except Exception:
    grounding_router = None

try:
    from backend.api.endpoints.change import router as change_router
    __all__.append("change_router")
except Exception:
    change_router = None

try:
    from backend.api.endpoints.optical_sar import router as optical_sar_router
    __all__.append("optical_sar_router")
except Exception:
    optical_sar_router = None

try:
    from backend.api.endpoints.agent import router as agent_router
    __all__.append("agent_router")
except Exception:
    agent_router = None

try:
    from backend.api.endpoints.land_cover import router as land_cover_router
    __all__.append("land_cover_router")
except Exception:
    land_cover_router = None

from backend.api.endpoints.analyze import router as analyze_router
__all__.append("analyze_router")

from backend.api.endpoints.copernicus import router as copernicus_router
__all__.append("copernicus_router")
