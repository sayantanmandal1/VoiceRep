"""
Pydantic schemas for file operations.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from app.models.file import ProcessingStatus


class FileUploadResponse(BaseModel):
    """Response schema for file upload."""
    file_id: str
    filename: str
    file_size: int
    duration: Optional[float] = None
    sample_rate: Optional[int] = None
    status: ProcessingStatus
    upload_timestamp: datetime
    
    class Config:
        from_attributes = True


class FileValidationError(BaseModel):
    """Schema for file validation errors."""
    error_type: str
    message: str
    supported_formats: Optional[list[str]] = None
    max_file_size_mb: Optional[int] = None


class AudioExtractionRequest(BaseModel):
    """Request schema for audio extraction from video."""
    video_file_id: str
    output_format: str = Field(default="wav", pattern="^(wav|mp3|flac)$")
    quality: str = Field(default="high", pattern="^(low|medium|high)$")


class FileProcessingStatus(BaseModel):
    """Schema for file processing status."""
    file_id: str
    status: ProcessingStatus
    progress: Optional[float] = None
    error_message: Optional[str] = None
    estimated_completion: Optional[datetime] = None


class AudioExtractionResponse(BaseModel):
    """Response schema for audio extraction task."""
    task_id: str
    file_id: str
    status: str
    message: str
    estimated_completion: Optional[datetime] = None


class TaskProgressResponse(BaseModel):
    """Schema for task progress information."""
    task_id: str
    status: str
    progress: Optional[int] = None
    current_step: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    file_id: Optional[str] = None