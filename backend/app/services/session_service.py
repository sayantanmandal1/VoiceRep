"""
Session management service for user sessions and data isolation.
"""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from datetime import datetime, timedelta
import json
import os
import shutil
import logging
from pathlib import Path

from app.models.session import UserSession, RequestTracker, FileAccessControl
from app.core.database import get_db
from app.core.config import settings

logger = logging.getLogger(__name__)


class SessionService:
    """Service for managing user sessions and data isolation."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_session(
        self, 
        user_identifier: str,
        user_agent: str = None,
        ip_address: str = None
    ) -> UserSession:
        """Create a new user session with isolated data namespace."""
        session = UserSession(
            user_identifier=user_identifier,
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        
        # Create isolated directories for this session
        self._create_session_directories(session.data_namespace)
        
        logger.info(f"Created session {session.id} for user {user_identifier}")
        return session
    
    def get_session(self, session_token: str) -> Optional[UserSession]:
        """Get session by token if valid and not expired."""
        session = self.db.query(UserSession).filter(
            UserSession.session_token == session_token,
            UserSession.is_active == True
        ).first()
        
        if not session:
            return None
        
        if session.is_expired():
            self.invalidate_session(session_token)
            return None
        
        # Update last activity
        session.last_activity = datetime.utcnow()
        self.db.commit()
        
        return session
    
    def invalidate_session(self, session_token: str) -> bool:
        """Invalidate a session and clean up its data."""
        session = self.db.query(UserSession).filter(
            UserSession.session_token == session_token
        ).first()
        
        if not session:
            return False
        
        # Mark session as inactive
        session.is_active = False
        
        # Clean up session data
        self._cleanup_session_data(session)
        
        self.db.commit()
        logger.info(f"Invalidated session {session.id}")
        return True
    
    def extend_session(self, session_token: str, hours: int = 24) -> bool:
        """Extend session expiration time."""
        session = self.get_session(session_token)
        if not session:
            return False
        
        session.extend_session(hours)
        self.db.commit()
        return True
    
    def cleanup_expired_sessions(self):
        """Clean up all expired sessions."""
        expired_sessions = self.db.query(UserSession).filter(
            or_(
                UserSession.expires_at < datetime.utcnow(),
                UserSession.is_active == False
            )
        ).all()
        
        for session in expired_sessions:
            self._cleanup_session_data(session)
            self.db.delete(session)
        
        self.db.commit()
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _create_session_directories(self, data_namespace: str):
        """Create isolated directories for session data."""
        base_dirs = [
            settings.UPLOAD_DIR,
            settings.RESULTS_DIR,
            settings.MODELS_DIR
        ]
        
        for base_dir in base_dirs:
            session_dir = Path(base_dir) / data_namespace
            session_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (session_dir / "temp").mkdir(exist_ok=True)
            (session_dir / "processed").mkdir(exist_ok=True)
    
    def _cleanup_session_data(self, session: UserSession):
        """Clean up all data associated with a session."""
        # Remove session directories
        base_dirs = [
            settings.UPLOAD_DIR,
            settings.RESULTS_DIR,
            settings.MODELS_DIR
        ]
        
        for base_dir in base_dirs:
            session_dir = Path(base_dir) / session.data_namespace
            if session_dir.exists():
                try:
                    shutil.rmtree(session_dir)
                    logger.info(f"Removed session directory: {session_dir}")
                except Exception as e:
                    logger.error(f"Failed to remove session directory {session_dir}: {e}")
        
        # Clean up request trackers
        self.db.query(RequestTracker).filter(
            RequestTracker.session_id == session.id
        ).delete()
        
        # Clean up file access controls
        self.db.query(FileAccessControl).filter(
            FileAccessControl.session_id == session.id
        ).delete()


class RequestTrackingService:
    """Service for tracking requests within sessions."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def start_request(
        self,
        session_id: str,
        request_id: str,
        request_type: str,
        endpoint: str,
        method: str,
        user_agent: str = None,
        ip_address: str = None
    ) -> RequestTracker:
        """Start tracking a new request."""
        tracker = RequestTracker(
            session_id=session_id,
            request_id=request_id,
            request_type=request_type,
            endpoint=endpoint,
            method=method,
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        self.db.add(tracker)
        self.db.commit()
        self.db.refresh(tracker)
        
        return tracker
    
    def update_request_status(
        self,
        request_id: str,
        status: str,
        error_message: str = None
    ):
        """Update request status."""
        tracker = self.db.query(RequestTracker).filter(
            RequestTracker.request_id == request_id
        ).first()
        
        if tracker:
            tracker.status = status
            if error_message:
                tracker.error_message = error_message
            if status in ['completed', 'failed']:
                tracker.completed_at = datetime.utcnow()
            
            self.db.commit()
    
    def add_file_to_request(self, request_id: str, file_path: str, is_temp: bool = False):
        """Add a file path to request tracking."""
        tracker = self.db.query(RequestTracker).filter(
            RequestTracker.request_id == request_id
        ).first()
        
        if not tracker:
            return
        
        if is_temp:
            temp_files = json.loads(tracker.temp_files or "[]")
            temp_files.append(file_path)
            tracker.temp_files = json.dumps(temp_files)
        else:
            file_paths = json.loads(tracker.file_paths or "[]")
            file_paths.append(file_path)
            tracker.file_paths = json.dumps(file_paths)
        
        self.db.commit()
    
    def complete_request(self, request_id: str, result_path: str = None):
        """Mark request as completed."""
        tracker = self.db.query(RequestTracker).filter(
            RequestTracker.request_id == request_id
        ).first()
        
        if tracker:
            tracker.mark_completed(result_path)
            self.db.commit()
    
    def fail_request(self, request_id: str, error_message: str):
        """Mark request as failed."""
        tracker = self.db.query(RequestTracker).filter(
            RequestTracker.request_id == request_id
        ).first()
        
        if tracker:
            tracker.mark_failed(error_message)
            self.db.commit()
    
    def cleanup_request_files(self, request_id: str):
        """Clean up temporary files for a request."""
        tracker = self.db.query(RequestTracker).filter(
            RequestTracker.request_id == request_id
        ).first()
        
        if not tracker or not tracker.temp_files:
            return
        
        temp_files = json.loads(tracker.temp_files)
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove temporary file {file_path}: {e}")
        
        # Clear temp files list
        tracker.temp_files = json.dumps([])
        self.db.commit()


class FileAccessService:
    """Service for managing file access permissions."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def grant_file_access(
        self,
        session_id: str,
        file_path: str,
        access_type: str,
        expires_in_hours: int = None
    ) -> FileAccessControl:
        """Grant file access to a session."""
        expires_at = None
        if expires_in_hours:
            expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        access_control = FileAccessControl(
            file_path=file_path,
            session_id=session_id,
            access_type=access_type,
            expires_at=expires_at
        )
        
        self.db.add(access_control)
        self.db.commit()
        self.db.refresh(access_control)
        
        return access_control
    
    def check_file_access(
        self,
        session_id: str,
        file_path: str,
        access_type: str
    ) -> bool:
        """Check if session has access to file."""
        access_control = self.db.query(FileAccessControl).filter(
            and_(
                FileAccessControl.session_id == session_id,
                FileAccessControl.file_path == file_path,
                FileAccessControl.access_type == access_type
            )
        ).first()
        
        if not access_control:
            return False
        
        return access_control.is_access_valid()
    
    def revoke_file_access(self, session_id: str, file_path: str):
        """Revoke all access to a file for a session."""
        self.db.query(FileAccessControl).filter(
            and_(
                FileAccessControl.session_id == session_id,
                FileAccessControl.file_path == file_path
            )
        ).delete()
        
        self.db.commit()
    
    def get_session_file_path(self, session: UserSession, base_dir: str, filename: str) -> str:
        """Get isolated file path for session."""
        session_dir = Path(base_dir) / session.data_namespace
        return str(session_dir / filename)