"""
Main API router for v1 endpoints.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import files, text, voice, synthesis, session, voice_model_training, quality_monitoring, post_processing, performance_optimization

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(files.router, prefix="/files", tags=["files"])
api_router.include_router(text.router, prefix="/text", tags=["text"])
api_router.include_router(voice.router, prefix="/voice", tags=["voice"])
api_router.include_router(synthesis.router, prefix="/synthesis", tags=["synthesis"])
api_router.include_router(session.router, prefix="/session", tags=["session"])
api_router.include_router(voice_model_training.router, prefix="/voice-models", tags=["voice-models"])
api_router.include_router(quality_monitoring.router, prefix="/quality-monitoring", tags=["quality-monitoring"])
api_router.include_router(performance_optimization.router, prefix="/performance", tags=["performance"])


@api_router.get("/status")
async def api_status():
    """API status endpoint."""
    return {"status": "API v1 running", "version": "1.0.0"}