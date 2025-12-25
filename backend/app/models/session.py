"""
Session and user management models.
"""

from sqlalchemy import Column, String, DateTime, Boolean, Text, func
from sqlalchemy.orm import relationship
from app.models.base import BaseModel
from datetime import datetime, timedelta
import secrets


class UserSession(BaseModel):
    """Model for user sessions with data isolation."""
    
    __tablename__ = "user_sessions"
    
    session_token = Column(String, unique=True, nullable=False, index=True)
    user_identifier = Column(String, nullable=False)  # Can be IP-based or user-provided ID
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime, nullable=False)
    last_activity = Column(DateTime, server_default=func.now())
    
    # Session metadata for tracking
    user_agent = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    
    # Data isolation namespace
    data_namespace = Column(String, nullable=False, unique=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.session_token:
            self.session_token = secrets.token_urlsafe(32)
        if not self.data_namespace:
            self.data_namespace = f"session_{self.id}"
        if not self.expires_at:
            self.expires_at = datetime.utcnow() + timedelta(hours=24)
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at
    
    def extend_session(self, hours: int = 24):
        """Extend session expiration."""
        self.expires_at = datetime.utcnow() + timedelta(hours=hours)
        self.last_activity = datetime.utcnow()


class RequestTracker(BaseModel):
    """Model for tracking individual requests within sessions."""
    
    __tablename__ = "request_trackers"
    
    session_id = Column(String, nullable=False, index=True)
    request_id = Column(String, unique=True, nullable=False)
    request_type = Column(String, nullable=False)  # 'upload', 'synthesis', 'analysis'
    status = Column(String, default='pending')  # 'pending', 'processing', 'completed', 'failed'
    
    # Request metadata
    endpoint = Column(String, nullable=False)
    method = Column(String, nullable=False)
    user_agent = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    
    # Processing details
    started_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Resource tracking
    file_paths = Column(Text, nullable=True)  # JSON array of file paths created
    temp_files = Column(Text, nullable=True)  # JSON array of temporary files
    result_path = Column(String, nullable=True)
    
    def mark_completed(self, result_path: str = None):
        """Mark request as completed."""
        self.status = 'completed'
        self.completed_at = datetime.utcnow()
        if result_path:
            self.result_path = result_path
    
    def mark_failed(self, error_message: str):
        """Mark request as failed."""
        self.status = 'failed'
        self.completed_at = datetime.utcnow()
        self.error_message = error_message


class FileAccessControl(BaseModel):
    """Model for controlling file access permissions."""
    
    __tablename__ = "file_access_control"
    
    file_path = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    access_type = Column(String, nullable=False)  # 'read', 'write', 'delete'
    granted_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime, nullable=True)
    
    def is_access_valid(self) -> bool:
        """Check if file access is still valid."""
        if self.expires_at:
            return datetime.utcnow() <= self.expires_at
        return True