"""
FastAPI main application entry point for Voice Style Replication system.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.database import engine, Base
from app.core.logging_config import setup_logging, get_logger
from app.core.error_handling import (
    validation_exception_handler,
    file_processing_exception_handler,
    synthesis_exception_handler,
    system_exception_handler,
    ErrorCategory
)
from app.api.v1.api import api_router
# from app.middleware.performance_middleware import setup_performance_middleware
from app.middleware.session_middleware import setup_session_middleware
# from app.services.cleanup_service import schedule_cleanup_task
from app.services.real_voice_synthesis_service import initialize_voice_synthesis_service

# Import models to ensure they are registered with Base
from app.models import (
    ReferenceAudio, ProcessingStatus, VoiceProfile, VoiceModel, 
    UserSession, RequestTracker, FileAccessControl
)
from app.models.synthesis import SynthesisTask, SynthesisResult, BatchSynthesisTask

# Set up logging first
setup_logging()
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Voice Style Replication API")
    
    try:
        # Initialize database
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
        
        # Start performance monitoring (disabled temporarily to prevent blocking)
        # from app.services.performance_monitoring_service import performance_monitor
        # performance_monitor.start_monitoring()
        logger.info("Performance monitoring disabled temporarily")
        
        # Start cleanup task scheduler
        # schedule_cleanup_task()
        # logger.info("Cleanup task scheduler started")
        
        # Initialize real voice synthesis service
        logger.info("Initializing real voice synthesis service...")
        voice_service_ready = await initialize_voice_synthesis_service()
        if voice_service_ready:
            logger.info("Real voice synthesis service initialized successfully")
        else:
            logger.warning("Voice synthesis service initialization failed - synthesis may not work properly")
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.critical(f"Failed to start application: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Voice Style Replication API")
    
    try:
        # from app.services.performance_monitoring_service import performance_monitor
        # performance_monitor.stop_monitoring()
        logger.info("Performance monitoring was disabled")
        
        logger.info("Application shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


app = FastAPI(
    title="Voice Style Replication API",
    description="High-fidelity voice cloning and synthesis system with comprehensive error handling and monitoring",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up session management middleware
setup_session_middleware(app)

# Set up performance monitoring middleware
# setup_performance_middleware(app)

# Add custom exception handlers
app.add_exception_handler(422, validation_exception_handler)
app.add_exception_handler(ValueError, file_processing_exception_handler)
app.add_exception_handler(RuntimeError, synthesis_exception_handler)
app.add_exception_handler(Exception, system_exception_handler)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint for health check."""
    logger.info("Health check requested")
    return {
        "message": "Voice Style Replication API", 
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint."""
    logger.info("Comprehensive health check requested")
    
    try:
        # Check database connection
        from app.core.database import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_status = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        db_status = "unhealthy"
    
    # Check performance monitoring
    try:
        # Performance monitoring is disabled temporarily
        perf_status = "disabled"
    except Exception as e:
        logger.error(f"Performance monitoring health check failed: {str(e)}")
        perf_status = "unhealthy"
    
    health_status = {
        "status": "healthy" if db_status == "healthy" and perf_status == "healthy" else "degraded",
        "database": db_status,
        "performance_monitoring": perf_status,
        "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
    }
    
    logger.info(f"Health check completed: {health_status}")
    return health_status


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests."""
    import time
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": request.client.host if request.client else None
        }
    )
    
    # Process request
    response = await call_next(request)
    
    # Log response
    duration = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} - {duration*1000:.2f}ms",
        extra={
            "request_id": getattr(request.state, "request_id", None),
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2)
        }
    )
    
    return response