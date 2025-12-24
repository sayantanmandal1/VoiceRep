"""
Pydantic schemas for text input and validation operations.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime
import re


class TextInputRequest(BaseModel):
    """Request schema for text input validation and processing."""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to synthesize")
    language: Optional[str] = Field(None, description="Target language code (auto-detected if not provided)")
    
    @validator('text')
    def validate_text_content(cls, v):
        """Validate text content."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class TextValidationResponse(BaseModel):
    """Response schema for text validation."""
    is_valid: bool
    text: str
    character_count: int
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None
    sanitized_text: str
    validation_errors: Optional[list[str]] = None


class TextProcessingRequest(BaseModel):
    """Request schema for text processing and synthesis preparation."""
    text: str
    reference_audio_id: str
    target_language: Optional[str] = None
    voice_settings: Optional[dict] = None


class TextProcessingResponse(BaseModel):
    """Response schema for text processing."""
    task_id: str
    text: str
    character_count: int
    detected_language: Optional[str]
    target_language: str
    status: str
    estimated_completion: Optional[datetime] = None


class LanguageDetectionResponse(BaseModel):
    """Response schema for language detection."""
    detected_language: str
    confidence: float
    supported_languages: list[str]
    is_cross_language: bool = False


class TextSanitizationResult(BaseModel):
    """Result of text sanitization process."""
    original_text: str
    sanitized_text: str
    removed_characters: list[str] = []
    character_count: int
    is_modified: bool = False