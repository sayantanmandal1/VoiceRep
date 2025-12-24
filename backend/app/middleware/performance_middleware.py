"""
Performance monitoring middleware for FastAPI application.
"""

import time
import logging
from typing import Callable, Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.services.performance_monitoring_service import performance_monitor

logger = logging.getLogger(__name__)


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor API endpoint performance."""
    
    def __init__(self, app, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/docs", "/redoc", "/openapi.json", "/health", "/",
            "/api/v1/performance"  # Avoid monitoring the monitoring endpoints
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and monitor performance."""
        # Skip monitoring for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        # Start timing
        start_time = time.time()
        operation_id = f"api_request_{int(time.time() * 1000)}_{id(request)}"
        
        # Extract request metadata
        metadata = {
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown")
        }
        
        # Start performance monitoring
        performance_monitor.start_operation(
            "api_request",
            operation_id,
            metadata
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Determine if request was successful
            success = 200 <= response.status_code < 400
            
            # Add response metadata
            response_metadata = {
                "status_code": response.status_code,
                "response_time_ms": processing_time * 1000,
                "response_size": response.headers.get("content-length", "unknown")
            }
            metadata.update(response_metadata)
            
            # End performance monitoring
            performance_monitor.end_operation(
                operation_id,
                success=success,
                additional_metadata=metadata
            )
            
            # Add performance headers to response
            response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
            response.headers["X-Request-ID"] = operation_id
            
            # Log slow requests
            if processing_time > 5.0:  # Log requests taking more than 5 seconds
                logger.warning(
                    f"Slow API request: {request.method} {request.url.path} "
                    f"took {processing_time:.2f}s (status: {response.status_code})"
                )
            
            return response
            
        except Exception as e:
            # Calculate processing time for failed requests
            processing_time = time.time() - start_time
            
            # End performance monitoring with error
            performance_monitor.end_operation(
                operation_id,
                success=False,
                error_message=str(e),
                additional_metadata={
                    **metadata,
                    "error_type": type(e).__name__,
                    "processing_time_ms": processing_time * 1000
                }
            )
            
            # Log the error
            logger.error(
                f"API request failed: {request.method} {request.url.path} "
                f"after {processing_time:.2f}s - {str(e)}"
            )
            
            # Re-raise the exception to let FastAPI handle it
            raise


class ResourceLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce resource limits and queue management."""
    
    def __init__(self, app, max_concurrent_requests: int = 50):
        super().__init__(app)
        self.max_concurrent_requests = max_concurrent_requests
        self.active_requests = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with resource limits."""
        # Check if we're at the concurrent request limit
        if self.active_requests >= self.max_concurrent_requests:
            # Get current system performance
            summary = performance_monitor.get_performance_summary()
            current_resources = summary.get('current_resources')
            
            # Check if system is under high load
            if current_resources:
                cpu_usage = current_resources.get('cpu_percent', 0)
                memory_usage = current_resources.get('memory_percent', 0)
                
                if cpu_usage > 90 or memory_usage > 90:
                    return JSONResponse(
                        status_code=503,
                        content={
                            "error": "Service temporarily unavailable",
                            "message": "System is under high load. Please try again later.",
                            "retry_after": 30,
                            "current_load": {
                                "cpu_percent": cpu_usage,
                                "memory_percent": memory_usage,
                                "active_requests": self.active_requests
                            }
                        }
                    )
        
        # Increment active request counter
        self.active_requests += 1
        
        try:
            # Process the request
            response = await call_next(request)
            return response
        finally:
            # Always decrement the counter
            self.active_requests -= 1


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Middleware to add appropriate cache control headers."""
    
    def __init__(self, app):
        super().__init__(app)
        self.cache_rules = {
            "/api/v1/performance/metrics": {"max_age": 5, "public": False},
            "/api/v1/performance/queue": {"max_age": 1, "public": False},
            "/api/v1/files": {"max_age": 300, "public": False},  # 5 minutes for file info
            "/health": {"max_age": 30, "public": True},
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add cache control headers based on endpoint."""
        response = await call_next(request)
        
        # Find matching cache rule
        cache_rule = None
        for path_pattern, rule in self.cache_rules.items():
            if request.url.path.startswith(path_pattern):
                cache_rule = rule
                break
        
        if cache_rule:
            # Add cache control headers
            cache_control_parts = [f"max-age={cache_rule['max_age']}"]
            
            if cache_rule.get('public', False):
                cache_control_parts.append("public")
            else:
                cache_control_parts.append("private")
            
            response.headers["Cache-Control"] = ", ".join(cache_control_parts)
        else:
            # Default: no cache for API endpoints
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        return response


def setup_performance_middleware(app):
    """
    Set up all performance-related middleware for the FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    # Add middleware in reverse order (last added is executed first)
    
    # Cache control (outermost)
    app.add_middleware(CacheControlMiddleware)
    
    # Resource limits
    app.add_middleware(ResourceLimitMiddleware, max_concurrent_requests=50)
    
    # Performance monitoring (innermost, closest to the actual request processing)
    app.add_middleware(PerformanceMonitoringMiddleware)
    
    logger.info("Performance monitoring middleware configured")