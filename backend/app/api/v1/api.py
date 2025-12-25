"""
Main API router for v1 endpoints.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import files, text, voice, synthesis, session

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(text.router, prefix="/text", tags=["text"])
api_router.include_router(voice.router, prefix="/voice", tags=["voice"])
api_router.include_router(synthesis.router, prefix="/synthesis", tags=["synthesis"])
api_router.include_router(session.router, prefix="/session", tags=["session"])


@api_router.get("/status")
async def api_status():
    """API status endpoint."""
    return {"status": "API v1 running", "version": "1.0.0"}