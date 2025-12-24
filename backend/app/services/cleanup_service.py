"""
Cleanup service for managing temporary files and expired data.
"""

import os
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.core.database import get_db
from app.core.config import settings
from app.models.session import UserSession, RequestTracker, FileAccessControl
from app.models.file import ReferenceAudio, ProcessingStatus
from app.models.synthesis import SynthesisTask, SynthesisResult
from app.services.session_service import SessionService

logger = logging.getLogger(__name__)


class CleanupService:
    """Service for cleaning up temporary files and expired data."""
    
    def __init__(self):
        self.db = next(get_db())
        self.session_service = SessionService(self.db)
    
    def run_full_cleanup(self):
        """Run complete cleanup process."""
        logger.info("Starting full cleanup process")
        
        try:
            # Clean up expired sessions
            self.cleanup_expired_sessions()
            
            # Clean up orphaned files
            self.cleanup_orphaned_files()
            
            # Clean up failed processing tasks
            self.cleanup_failed_tasks()
            
            # Clean up old temporary files
            self.cleanup_old_temp_files()
            
            # Clean up expired file access controls
            self.cleanup_expired_file_access()
            
            logger.info("Full cleanup process completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup process: {e}")
            raise
        finally:
            self.db.close()
    
    def cleanup_expired_sessions(self):
        """Clean up expired user sessions and their data."""
        logger.info("Cleaning up expired sessions")
        
        expired_sessions = self.db.query(UserSession).filter(
            or_(
                UserSession.expires_at < datetime.utcnow(),
                UserSession.is_active == False
            )
        ).all()
        
        cleanup_count = 0
        for session in expired_sessions:
            try:
                # Clean up session files and directories
                self._cleanup_session_files(session)
                
                # Clean up related database records
                self._cleanup_session_database_records(session)
                
                # Delete session
                self.db.delete(session)
                cleanup_count += 1
                
            except Exception as e:
                logger.error(f"Error cleaning up session {session.id}: {e}")
        
        self.db.commit()
        logger.info(f"Cleaned up {cleanup_count} expired sessions")
    
    def cleanup_orphaned_files(self):
        """Clean up files that are no longer referenced in database."""
        logger.info("Cleaning up orphaned files")
        
        # Get all file paths from database
        referenced_files = set()
        
        # Reference audio files
        audio_files = self.db.query(ReferenceAudio).all()
        for audio in audio_files:
            referenced_files.add(audio.file_path)
        
        # Synthesis result files
        synthesis_results = self.db.query(SynthesisResult).all()
        for result in synthesis_results:
            if result.file_path:
                referenced_files.add(result.file_path)
        
        # Request tracker files
        request_trackers = self.db.query(RequestTracker).all()
        for tracker in request_trackers:
            if tracker.file_paths:
                import json
                try:
                    file_paths = json.loads(tracker.file_paths)
                    referenced_files.update(file_paths)
                except json.JSONDecodeError:
                    pass
            
            if tracker.result_path:
                referenced_files.add(tracker.result_path)
        
        # Check filesystem for orphaned files
        cleanup_count = 0
        base_dirs = [settings.UPLOAD_DIR, settings.RESULTS_DIR, settings.MODELS_DIR]
        
        for base_dir in base_dirs:
            if not os.path.exists(base_dir):
                continue
                
            for root, dirs, files in os.walk(base_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    # Skip if file is referenced
                    if file_path in referenced_files:
                        continue
                    
                    # Skip recent files (less than 1 hour old)
                    try:
                        file_age = datetime.utcnow() - datetime.fromtimestamp(
                            os.path.getctime(file_path)
                        )
                        if file_age < timedelta(hours=1):
                            continue
                    except OSError:
                        pass
                    
                    # Remove orphaned file
                    try:
                        os.remove(file_path)
                        cleanup_count += 1
                        logger.debug(f"Removed orphaned file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to remove orphaned file {file_path}: {e}")
        
        logger.info(f"Cleaned up {cleanup_count} orphaned files")
    
    def cleanup_failed_tasks(self):
        """Clean up failed processing tasks and their associated files."""
        logger.info("Cleaning up failed tasks")
        
        # Clean up failed reference audio processing
        failed_audio = self.db.query(ReferenceAudio).filter(
            and_(
                ReferenceAudio.processing_status == ProcessingStatus.FAILED,
                ReferenceAudio.created_at < datetime.utcnow() - timedelta(hours=24)
            )
        ).all()
        
        for audio in failed_audio:
            try:
                # Remove file if it exists
                if os.path.exists(audio.file_path):
                    os.remove(audio.file_path)
                
                # Remove database record
                self.db.delete(audio)
                logger.debug(f"Cleaned up failed audio processing: {audio.id}")
                
            except Exception as e:
                logger.error(f"Error cleaning up failed audio {audio.id}: {e}")
        
        # Clean up failed synthesis tasks
        failed_synthesis = self.db.query(SynthesisTask).filter(
            and_(
                SynthesisTask.status == "failed",
                SynthesisTask.created_at < datetime.utcnow() - timedelta(hours=24)
            )
        ).all()
        
        for task in failed_synthesis:
            try:
                # Clean up associated result files
                results = self.db.query(SynthesisResult).filter(
                    SynthesisResult.synthesis_task_id == task.id
                ).all()
                
                for result in results:
                    if result.file_path and os.path.exists(result.file_path):
                        os.remove(result.file_path)
                    self.db.delete(result)
                
                # Remove task
                self.db.delete(task)
                logger.debug(f"Cleaned up failed synthesis task: {task.id}")
                
            except Exception as e:
                logger.error(f"Error cleaning up failed synthesis {task.id}: {e}")
        
        self.db.commit()
        logger.info("Completed cleanup of failed tasks")
    
    def cleanup_old_temp_files(self):
        """Clean up temporary files older than specified age."""
        logger.info("Cleaning up old temporary files")
        
        cleanup_count = 0
        temp_dirs = []
        
        # Add temp directories from all base directories
        base_dirs = [settings.UPLOAD_DIR, settings.RESULTS_DIR, settings.MODELS_DIR]
        for base_dir in base_dirs:
            temp_dir = Path(base_dir) / "temp"
            if temp_dir.exists():
                temp_dirs.append(temp_dir)
        
        # Clean up files older than 1 hour
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        for temp_dir in temp_dirs:
            try:
                for file_path in temp_dir.rglob("*"):
                    if not file_path.is_file():
                        continue
                    
                    # Check file age
                    try:
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_time:
                            file_path.unlink()
                            cleanup_count += 1
                            logger.debug(f"Removed old temp file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing temp file {file_path}: {e}")
                        
            except Exception as e:
                logger.error(f"Error cleaning temp directory {temp_dir}: {e}")
        
        logger.info(f"Cleaned up {cleanup_count} old temporary files")
    
    def cleanup_expired_file_access(self):
        """Clean up expired file access control records."""
        logger.info("Cleaning up expired file access controls")
        
        expired_access = self.db.query(FileAccessControl).filter(
            FileAccessControl.expires_at < datetime.utcnow()
        ).all()
        
        for access in expired_access:
            self.db.delete(access)
        
        self.db.commit()
        logger.info(f"Cleaned up {len(expired_access)} expired file access records")
    
    def _cleanup_session_files(self, session: UserSession):
        """Clean up all files associated with a session."""
        base_dirs = [settings.UPLOAD_DIR, settings.RESULTS_DIR, settings.MODELS_DIR]
        
        for base_dir in base_dirs:
            session_dir = Path(base_dir) / session.data_namespace
            if session_dir.exists():
                try:
                    shutil.rmtree(session_dir)
                    logger.debug(f"Removed session directory: {session_dir}")
                except Exception as e:
                    logger.error(f"Failed to remove session directory {session_dir}: {e}")
    
    def _cleanup_session_database_records(self, session: UserSession):
        """Clean up all database records associated with a session."""
        # Clean up request trackers
        self.db.query(RequestTracker).filter(
            RequestTracker.session_id == session.id
        ).delete()
        
        # Clean up file access controls
        self.db.query(FileAccessControl).filter(
            FileAccessControl.session_id == session.id
        ).delete()
        
        # Clean up reference audio files
        audio_files = self.db.query(ReferenceAudio).filter(
            ReferenceAudio.user_id == session.user_identifier
        ).all()
        
        for audio in audio_files:
            # Remove associated voice profiles and models
            from app.models.voice import VoiceProfile, VoiceModel
            
            if audio.voice_profile_id:
                voice_models = self.db.query(VoiceModel).filter(
                    VoiceModel.voice_profile_id == audio.voice_profile_id
                ).all()
                
                for model in voice_models:
                    # Remove model file if it exists
                    if model.model_path and os.path.exists(model.model_path):
                        try:
                            os.remove(model.model_path)
                        except Exception as e:
                            logger.error(f"Failed to remove model file {model.model_path}: {e}")
                    
                    self.db.delete(model)
                
                # Remove voice profile
                voice_profile = self.db.query(VoiceProfile).filter(
                    VoiceProfile.id == audio.voice_profile_id
                ).first()
                
                if voice_profile:
                    self.db.delete(voice_profile)
            
            # Remove audio file
            if os.path.exists(audio.file_path):
                try:
                    os.remove(audio.file_path)
                except Exception as e:
                    logger.error(f"Failed to remove audio file {audio.file_path}: {e}")
            
            self.db.delete(audio)
        
        # Clean up synthesis tasks and results
        synthesis_tasks = self.db.query(SynthesisTask).filter(
            SynthesisTask.user_id == session.user_identifier
        ).all()
        
        for task in synthesis_tasks:
            # Clean up results
            results = self.db.query(SynthesisResult).filter(
                SynthesisResult.synthesis_task_id == task.id
            ).all()
            
            for result in results:
                if result.file_path and os.path.exists(result.file_path):
                    try:
                        os.remove(result.file_path)
                    except Exception as e:
                        logger.error(f"Failed to remove result file {result.file_path}: {e}")
                
                self.db.delete(result)
            
            self.db.delete(task)


def schedule_cleanup_task():
    """Schedule periodic cleanup task."""
    import threading
    import time
    
    def cleanup_worker():
        while True:
            try:
                cleanup_service = CleanupService()
                cleanup_service.run_full_cleanup()
            except Exception as e:
                logger.error(f"Scheduled cleanup failed: {e}")
            
            # Wait 1 hour before next cleanup
            time.sleep(3600)
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    logger.info("Scheduled cleanup task started")