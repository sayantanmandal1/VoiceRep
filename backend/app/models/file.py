"""
File-related data models.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Enum as SQLEnum
from enum import Enum
from app.models.base import BaseModel


class ProcessingStatus(str, Enum):
    """File processing status enumeration."""
    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    READY = "ready"
    FAILED = "failed"


class ReferenceAudio(BaseModel):
    """Model for reference audio files uploaded by users."""
    
    __tablename__ = "reference_audio"
    
    user_id = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    duration = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    format = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)
    processing_status = Column(SQLEnum(ProcessingStatus), default=ProcessingStatus.UPLOADED)
    voice_profile_id = Column(String, nullable=True)
    error_message = Column(String, nullable=True)