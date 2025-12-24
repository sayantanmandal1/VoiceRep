"""
File handling and validation service with session-based isolation.
"""

import os
import shutil
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, List
from fastapi import UploadFile, HTTPException
import librosa
import ffmpeg

from app.core.config import settings
from app.models.file import ProcessingStatus
from app.models.session import UserSession
from app.services.session_service import FileAccessService


class FileValidationService:
    """Service for file validation and processing with session isolation."""
    
    # Supported file formats
    SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.flac', '.m4a'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv'}
    SUPPORTED_FORMATS = SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS
    
    # MIME type mappings
    AUDIO_MIME_TYPES = {
        'audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/flac', 
        'audio/mp4', 'audio/m4a', 'audio/x-m4a'
    }
    VIDEO_MIME_TYPES = {
        'video/mp4', 'video/x-msvideo', 'video/quicktime', 
        'video/x-matroska'
    }
    SUPPORTED_MIME_TYPES = AUDIO_MIME_TYPES | VIDEO_MIME_TYPES
    
    def __init__(self):
        """Initialize file validation service."""
        self.max_file_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure upload directories exist."""
        for directory in [settings.UPLOAD_DIR, settings.RESULTS_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_session_file_path(self, session: UserSession, base_dir: str, filename: str) -> str:
        """Get isolated file path for session."""
        session_dir = Path(base_dir) / session.data_namespace
        session_dir.mkdir(parents=True, exist_ok=True)
        return str(session_dir / filename)
    
    def validate_file(self, file: UploadFile) -> Tuple[bool, Optional[str]]:
        """
        Validate uploaded file format and size.
        
        Args:
            file: The uploaded file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        if hasattr(file, 'size') and file.size and file.size > self.max_file_size:
            return False, f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE_MB}MB"
        
        # Check file extension
        if not file.filename:
            return False, "Filename is required"
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            return False, f"Unsupported file format. Supported formats: {', '.join(sorted(self.SUPPORTED_FORMATS))}"
        
        # Check MIME type if available
        if file.content_type and file.content_type not in self.SUPPORTED_MIME_TYPES:
            # Try to guess MIME type from filename
            guessed_type, _ = mimetypes.guess_type(file.filename)
            if guessed_type not in self.SUPPORTED_MIME_TYPES:
                return False, f"Invalid file type. Expected audio or video file."
        
        return True, None
    
    def validate_file_access(
        self, 
        session: UserSession, 
        file_path: str, 
        access_type: str,
        file_access_service: FileAccessService
    ) -> bool:
        """
        Validate that session has access to file.
        
        Args:
            session: User session
            file_path: Path to file
            access_type: Type of access (read, write, delete)
            file_access_service: File access service instance
            
        Returns:
            True if access is allowed
        """
        # Check if file is within session's namespace
        session_dir = Path(settings.UPLOAD_DIR) / session.data_namespace
        try:
            # Resolve paths to check if file is within session directory
            file_path_resolved = Path(file_path).resolve()
            session_dir_resolved = session_dir.resolve()
            
            # Check if file is within session directory
            if session_dir_resolved in file_path_resolved.parents or file_path_resolved == session_dir_resolved:
                return True
        except Exception:
            pass
        
        # Check explicit file access permissions
        return file_access_service.check_file_access(session.id, file_path, access_type)
    
    def is_audio_file(self, filename: str) -> bool:
        """Check if file is an audio format."""
        return Path(filename).suffix.lower() in self.SUPPORTED_AUDIO_FORMATS
    
    def is_video_file(self, filename: str) -> bool:
        """Check if file is a video format."""
        return Path(filename).suffix.lower() in self.SUPPORTED_VIDEO_FORMATS
    
    async def save_uploaded_file(
        self, 
        file: UploadFile, 
        file_id: str, 
        session: UserSession
    ) -> Tuple[str, int]:
        """
        Save uploaded file to session-isolated directory.
        
        Args:
            file: The uploaded file
            file_id: Unique identifier for the file
            session: User session for isolation
            
        Returns:
            Tuple of (file_path, file_size)
        """
        # Create filename with ID to avoid conflicts
        file_ext = Path(file.filename).suffix.lower()
        safe_filename = f"{file_id}{file_ext}"
        
        # Use session-isolated path
        file_path = self.get_session_file_path(session, settings.UPLOAD_DIR, safe_filename)
        
        # Save file
        file_size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):  # Read in 8KB chunks
                buffer.write(chunk)
                file_size += len(chunk)
        
        return str(file_path), file_size
    
    def extract_audio_metadata(self, file_path: str) -> Tuple[Optional[float], Optional[int]]:
        """
        Extract audio metadata (duration, sample rate).
        
        Args:
            file_path: Path to the audio/video file
            
        Returns:
            Tuple of (duration, sample_rate)
        """
        try:
            if self.is_audio_file(file_path):
                # Direct audio file
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                return duration, sr
            else:
                # Video file - extract audio info
                probe = ffmpeg.probe(file_path)
                audio_stream = next(
                    (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), 
                    None
                )
                if audio_stream:
                    duration = float(probe['format']['duration'])
                    sample_rate = int(audio_stream['sample_rate'])
                    return duration, sample_rate
                else:
                    return None, None
        except Exception:
            return None, None
    
    def extract_audio_from_video(
        self, 
        video_path: str, 
        output_path: str, 
        quality: str = "high"
    ) -> bool:
        """
        Extract audio from video file (legacy method - use Celery task for production).
        
        Args:
            video_path: Path to input video file
            output_path: Path for output audio file
            quality: Audio quality (low, medium, high)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Quality settings
            quality_settings = {
                "low": {"ar": 16000, "ab": "64k"},
                "medium": {"ar": 22050, "ab": "128k"},
                "high": {"ar": 44100, "ab": "320k"}
            }
            
            settings_dict = quality_settings.get(quality, quality_settings["high"])
            
            # Extract audio using ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(output_path, **settings_dict)
                .overwrite_output()
                .run(quiet=True)
            )
            return True
        except Exception:
            return False
    
    def start_audio_extraction_task(
        self, 
        video_path: str, 
        output_path: str, 
        quality: str = "high", 
        file_id: str = None,
        session_id: str = None
    ) -> str:
        """
        Start background audio extraction task with session context.
        
        Args:
            video_path: Path to input video file
            output_path: Path for output audio file
            quality: Audio quality preset
            file_id: Optional file ID for tracking
            session_id: Session ID for isolation
            
        Returns:
            Task ID for tracking progress
        """
        from app.tasks.audio_processing import extract_audio_from_video
        
        task = extract_audio_from_video.delay(
            video_file_path=video_path,
            output_path=output_path,
            quality=quality,
            file_id=file_id,
            session_id=session_id
        )
        
        return task.id
    
    def cleanup_file(self, file_path: str) -> None:
        """Remove file from disk."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass  # Ignore cleanup errors
    
    def cleanup_session_files(self, session: UserSession) -> None:
        """Clean up all files for a session."""
        base_dirs = [settings.UPLOAD_DIR, settings.RESULTS_DIR, settings.MODELS_DIR]
        
        for base_dir in base_dirs:
            session_dir = Path(base_dir) / session.data_namespace
            if session_dir.exists():
                try:
                    shutil.rmtree(session_dir)
                except Exception:
                    pass  # Ignore cleanup errors
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats."""
        return sorted(list(self.SUPPORTED_FORMATS))


# Global instance
file_validation_service = FileValidationService()