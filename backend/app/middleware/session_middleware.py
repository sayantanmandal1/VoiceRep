"""
Session management middleware for request tracking and data isolation.
"""

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.orm import Session
import uuid
import logging
from typing import Optional

from app.core.database import get_db
from app.services.session_service import SessionService, RequestTrackingService
from app.models.session import UserSession

logger = logging.getLogger(__name__)


class SessionMiddleware(BaseHTTPMiddleware):
    """Middleware for handling user sessions and request tracking."""
    
    def __init__(self, app, require_session: bool = True):
        super().__init__(app)
        self.require_session = require_session
        self.excluded_paths = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
    
    async def dispatch(self, request: Request, call_next):
        """Process request with session management."""
        # Skip session handling for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Get database session
        db = next(get_db())
        session_service = SessionService(db)
        request_service = RequestTrackingService(db)
        
        try:
            # Extract session token from header or create new session
            session_token = request.headers.get("X-Session-Token")
            user_session = None
            
            if session_token:
                user_session = session_service.get_session(session_token)
            
            # Create new session if none exists or token is invalid
            if not user_session:
                if self.require_session:
                    # Always create session for authenticated endpoints
                    user_identifier = self._get_user_identifier(request)
                    user_session = session_service.create_session(
                        user_identifier=user_identifier,
                        user_agent=request.headers.get("User-Agent"),
                        ip_address=self._get_client_ip(request)
                    )
                elif not self.require_session:
                    # Create anonymous session for non-authenticated endpoints
                    user_identifier = self._get_user_identifier(request)
                    user_session = session_service.create_session(
                        user_identifier=user_identifier,
                        user_agent=request.headers.get("User-Agent"),
                        ip_address=self._get_client_ip(request)
                    )
            
            # Add session to request state
            if user_session:
                request.state.session = user_session
                request.state.session_service = session_service
                request.state.request_service = request_service
                
                # Start request tracking
                request_id = str(uuid.uuid4())
                request.state.request_id = request_id
                
                request_tracker = request_service.start_request(
                    session_id=user_session.id,
                    request_id=request_id,
                    request_type=self._get_request_type(request),
                    endpoint=request.url.path,
                    method=request.method,
                    user_agent=request.headers.get("User-Agent"),
                    ip_address=self._get_client_ip(request)
                )
                request.state.request_tracker = request_tracker
            
            # Process request
            response = await call_next(request)
            
            # Add session token to response headers
            if user_session:
                response.headers["X-Session-Token"] = user_session.session_token
                
                # Update request status based on response
                if hasattr(request.state, 'request_id'):
                    if response.status_code < 400:
                        request_service.update_request_status(
                            request.state.request_id, 
                            "completed"
                        )
                    else:
                        request_service.update_request_status(
                            request.state.request_id, 
                            "failed",
                            f"HTTP {response.status_code}"
                        )
            
            return response
            
        except Exception as e:
            logger.error(f"Session middleware error: {e}")
            
            # Update request status as failed
            if hasattr(request.state, 'request_id'):
                request_service.update_request_status(
                    request.state.request_id,
                    "failed", 
                    str(e)
                )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Session management error"}
            )
        finally:
            db.close()
    
    def _get_user_identifier(self, request: Request) -> str:
        """Generate user identifier from request."""
        # Use IP address as base identifier for anonymous sessions
        ip = self._get_client_ip(request)
        user_agent = request.headers.get("User-Agent", "unknown")
        return f"{ip}_{hash(user_agent) % 10000}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to client host
        if hasattr(request.client, 'host'):
            return request.client.host
        
        return "unknown"
    
    def _get_request_type(self, request: Request) -> str:
        """Determine request type based on endpoint."""
        path = request.url.path.lower()
        
        if "/files" in path or "/upload" in path:
            return "upload"
        elif "/synthesis" in path or "/synthesize" in path:
            return "synthesis"
        elif "/voice" in path or "/analyze" in path:
            return "analysis"
        else:
            return "general"


def setup_session_middleware(app):
    """Set up session middleware for the application."""
    app.add_middleware(SessionMiddleware, require_session=True)
    logger.info("Session middleware configured")


# Dependency for getting current session
def get_current_session(request: Request) -> Optional[UserSession]:
    """Dependency to get current user session."""
    return getattr(request.state, 'session', None)


def require_session(request: Request) -> UserSession:
    """Dependency that requires a valid session."""
    session = get_current_session(request)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid session required"
        )
    return session