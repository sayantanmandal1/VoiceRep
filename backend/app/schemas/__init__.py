"""
Pydantic schemas package.
"""

from .file import (
    FileUploadResponse, FileValidationError, AudioExtractionRequest,
    FileProcessingStatus, AudioExtractionResponse, TaskProgressResponse
)
from .text import TextInputRequest, TextValidationResponse
from .voice import (
    VoiceProfileSchema, VoiceModelSchema, VoiceAnalysisRequest,
    VoiceAnalysisResponse, QualityMetrics, VoiceCharacteristics,
    VoiceAnalysisResult, FrequencyRange, ProsodyFeaturesSchema,
    EmotionalProfileSchema
)

__all__ = [
    # File schemas
    "FileUploadResponse", "FileValidationError", "AudioExtractionRequest",
    "FileProcessingStatus", "AudioExtractionResponse", "TaskProgressResponse",
    # Text schemas
    "TextInputRequest", "TextValidationResponse",
    # Voice schemas
    "VoiceProfileSchema", "VoiceModelSchema", "VoiceAnalysisRequest",
    "VoiceAnalysisResponse", "QualityMetrics", "VoiceCharacteristics",
    "VoiceAnalysisResult", "FrequencyRange", "ProsodyFeaturesSchema",
    "EmotionalProfileSchema"
]