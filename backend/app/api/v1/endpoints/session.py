"""
Session management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.core.database import get_db
from app.services.session_service import SessionService, RequestTrackingService, FileAccessService
from app.middleware.session_middleware import get_current_session, require_session
from app.models.session import UserSession, RequestTracker
from app.schemas.session import (
    SessionResponse, SessionCreateRequest, SessionExtendRequest,
    RequestTrackerResponse, SessionStatsResponse
)

router = APIRouter()


@router.post("/create", response_model=SessionResponse)
async def create_session(
    request: SessionCreateRequest,
    req: Request,
    db: Session = Depends(get_db)
):
    """Create a new user session."""
    session_service = SessionService(db)
    
    user_identifier = request.user_identifier or f"anonymous_{datetime.utcnow().timestamp()}"
    user_agent = req.headers.get("User-Agent")
    ip_address = req.headers.get("X-Forwarded-For", req.client.host if req.client else "unknown")
    
    session = session_service.create_session(
        user_identifier=user_identifier,
        user_agent=user_agent,
        ip_address=ip_address
    )
    
    return SessionResponse.from_orm(session)


@router.get("/current", response_model=SessionResponse)
async def get_current_session_info(
    session: UserSession = Depends(get_current_session)
):
    """Get current session information."""
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No active session"
        )
    
    return SessionResponse.from_orm(session)


@router.post("/extend", response_model=SessionResponse)
async def extend_session(
    request: SessionExtendRequest,
    session: UserSession = Depends(require_session),
    db: Session = Depends(get_db)
):
    """Extend current session expiration."""
    session_service = SessionService(db)
    
    success = session_service.extend_session(
        session.session_token, 
        request.hours
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to extend session"
        )
    
    # Refresh session data
    updated_session = session_service.get_session(session.session_token)
    return SessionResponse.from_orm(updated_session)


@router.post("/invalidate")
async def invalidate_session(
    session: UserSession = Depends(require_session),
    db: Session = Depends(get_db)
):
    """Invalidate current session."""
    session_service = SessionService(db)
    
    success = session_service.invalidate_session(session.session_token)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to invalidate session"
        )
    
    return {"message": "Session invalidated successfully"}


@router.get("/requests", response_model=List[RequestTrackerResponse])
async def get_session_requests(
    session: UserSession = Depends(require_session),
    db: Session = Depends(get_db)
):
    """Get all requests for current session."""
    requests = db.query(RequestTracker).filter(
        RequestTracker.session_id == session.id
    ).order_by(RequestTracker.started_at.desc()).all()
    
    return [RequestTrackerResponse.from_orm(req) for req in requests]


@router.get("/stats", response_model=SessionStatsResponse)
async def get_session_stats(
    session: UserSession = Depends(require_session),
    db: Session = Depends(get_db)
):
    """Get session statistics."""
    request_service = RequestTrackingService(db)
    
    # Count requests by status
    requests = db.query(RequestTracker).filter(
        RequestTracker.session_id == session.id
    ).all()
    
    stats = {
        "total_requests": len(requests),
        "completed_requests": len([r for r in requests if r.status == "completed"]),
        "failed_requests": len([r for r in requests if r.status == "failed"]),
        "pending_requests": len([r for r in requests if r.status == "pending"]),
        "processing_requests": len([r for r in requests if r.status == "processing"]),
        "session_duration": (datetime.utcnow() - session.created_at).total_seconds(),
        "last_activity": session.last_activity,
        "expires_at": session.expires_at
    }
    
    return SessionStatsResponse(**stats)


@router.delete("/cleanup")
async def cleanup_session_data(
    session: UserSession = Depends(require_session),
    db: Session = Depends(get_db)
):
    """Clean up temporary files for current session."""
    request_service = RequestTrackingService(db)
    
    # Get all requests for this session
    requests = db.query(RequestTracker).filter(
        RequestTracker.session_id == session.id
    ).all()
    
    cleanup_count = 0
    for request in requests:
        try:
            request_service.cleanup_request_files(request.request_id)
            cleanup_count += 1
        except Exception as e:
            # Log error but continue with other requests
            pass
    
    return {
        "message": f"Cleaned up temporary files for {cleanup_count} requests",
        "cleaned_requests": cleanup_count
    }