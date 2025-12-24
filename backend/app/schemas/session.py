"""
Session management schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class SessionCreateRequest(BaseModel):
    """Request schema for creating a new session."""
    user_identifier: Optional[str] = Field(None, description="Optional user identifier")


class SessionExtendRequest(BaseModel):
    """Request schema for extending session."""
    hours: int = Field(24, ge=1, le=168, description="Hours to extend session (1-168)")


class SessionResponse(BaseModel):
    """Response schema for session information."""
    id: str
    session_token: str
    user_identifier: str
    is_active: bool
    expires_at: datetime
    last_activity: datetime
    data_namespace: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class RequestTrackerResponse(BaseModel):
    """Response schema for request tracker information."""
    id: str
    request_id: str
    request_type: str
    status: str
    endpoint: str
    method: str
    started_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]
    result_path: Optional[str]
    
    class Config:
        from_attributes = True


class SessionStatsResponse(BaseModel):
    """Response schema for session statistics."""
    total_requests: int
    completed_requests: int
    failed_requests: int
    pending_requests: int
    processing_requests: int
    session_duration: float  # seconds
    last_activity: datetime
    expires_at: datetime


class FileAccessRequest(BaseModel):
    """Request schema for granting file access."""
    file_path: str
    access_type: str = Field(..., pattern="^(read|write|delete)$")
    expires_in_hours: Optional[int] = Field(None, ge=1, le=168)


class FileAccessResponse(BaseModel):
    """Response schema for file access information."""
    id: str
    file_path: str
    access_type: str
    granted_at: datetime
    expires_at: Optional[datetime]
    is_valid: bool
    
    class Config:
        from_attributes = True