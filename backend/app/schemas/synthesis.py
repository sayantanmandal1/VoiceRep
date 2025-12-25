"""
Pydantic schemas for speech synthesis operations.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class SynthesisStatus(str, Enum):
    """Speech synthesis task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VoiceSettings(BaseModel):
    """Schema for voice modification settings."""
    pitch_shift: float = Field(default=0.0, ge=-12.0, le=12.0, description="Pitch shift in semitones")
    speed_factor: float = Field(default=1.0, ge=0.5, le=2.0, description="Speed modification factor")
    emotion_intensity: float = Field(default=1.0, ge=0.1, le=2.0, description="Emotional intensity multiplier")
    volume_gain: float = Field(default=0.0, ge=-20.0, le=20.0, description="Volume gain in dB")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pitch_shift": 0.0,
                "speed_factor": 1.0,
                "emotion_intensity": 1.0,
                "volume_gain": 0.0
            }
        }


class SynthesisRequest(BaseModel):
    """Request schema for speech synthesis."""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to synthesize")
    voice_model_id: str = Field(..., description="ID of voice model to use")
    language: Optional[str] = Field(None, description="Target language (auto-detected if not provided)")
    voice_settings: Optional[VoiceSettings] = Field(default_factory=VoiceSettings, description="Voice modification settings")
    output_format: str = Field(default="wav", pattern="^(wav|mp3|flac)$", description="Output audio format")
    quality: str = Field(default="high", pattern="^(low|medium|high|ultra)$", description="Output quality preset")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, this is a test of voice synthesis.",
                "voice_model_id": "voice_model_123",
                "language": "english",
                "voice_settings": {
                    "pitch_shift": 0.0,
                    "speed_factor": 1.0,
                    "emotion_intensity": 1.0
                },
                "output_format": "wav",
                "quality": "high"
            }
        }


class CrossLanguageSynthesisRequest(BaseModel):
    """Request schema for cross-language speech synthesis."""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to synthesize")
    source_voice_model_id: str = Field(..., description="ID of source voice model")
    target_language: str = Field(..., description="Target language for synthesis")
    preserve_accent: bool = Field(default=True, description="Whether to preserve original accent characteristics")
    phonetic_adaptation: bool = Field(default=True, description="Whether to apply phonetic adaptation")
    output_format: str = Field(default="wav", pattern="^(wav|mp3|flac)$", description="Output audio format")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Bonjour, comment allez-vous?",
                "source_voice_model_id": "voice_model_123",
                "target_language": "french",
                "preserve_accent": True,
                "phonetic_adaptation": True,
                "output_format": "wav"
            }
        }


class BatchSynthesisRequest(BaseModel):
    """Request schema for batch speech synthesis."""
    requests: List[SynthesisRequest] = Field(..., min_items=1, max_items=10, description="List of synthesis requests")
    batch_name: Optional[str] = Field(None, description="Optional name for the batch")
    priority: int = Field(default=5, ge=1, le=10, description="Batch priority (1=highest, 10=lowest)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "requests": [
                    {
                        "text": "First text to synthesize",
                        "voice_model_id": "voice_model_123"
                    },
                    {
                        "text": "Second text to synthesize", 
                        "voice_model_id": "voice_model_456"
                    }
                ],
                "batch_name": "My batch synthesis",
                "priority": 5
            }
        }


class SynthesisResponse(BaseModel):
    """Response schema for synthesis task creation."""
    task_id: str = Field(..., description="Unique task identifier")
    status: SynthesisStatus = Field(..., description="Current task status")
    message: str = Field(..., description="Status message")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    queue_position: Optional[int] = Field(None, description="Position in processing queue")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "synthesis_task_123",
                "status": "pending",
                "message": "Synthesis task created successfully",
                "estimated_completion": "2024-01-01T12:30:00Z",
                "queue_position": 3
            }
        }


class SynthesisResult(BaseModel):
    """Schema for completed synthesis results."""
    task_id: str = Field(..., description="Task identifier")
    status: SynthesisStatus = Field(..., description="Final task status")
    output_url: Optional[str] = Field(None, description="URL to download synthesized audio")
    output_path: Optional[str] = Field(None, description="Server path to output file")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Synthesis metadata")
    error_message: Optional[str] = Field(None, description="Error message if synthesis failed")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")
    created_at: datetime = Field(..., description="Task creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Task completion timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "synthesis_task_123",
                "status": "completed",
                "output_url": "/api/v1/synthesis/download/synthesis_task_123",
                "metadata": {
                    "duration": 5.2,
                    "sample_rate": 22050,
                    "language": "english",
                    "quality_score": 0.92
                },
                "processing_time": 12.5,
                "created_at": "2024-01-01T12:00:00Z",
                "completed_at": "2024-01-01T12:00:12Z"
            }
        }


class SynthesisProgress(BaseModel):
    """Schema for synthesis progress updates."""
    task_id: str = Field(..., description="Task identifier")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    status: str = Field(..., description="Current status message")
    stage: Optional[str] = Field(None, description="Current processing stage")
    estimated_remaining: Optional[float] = Field(None, description="Estimated remaining time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "synthesis_task_123",
                "progress": 65,
                "status": "Applying voice conversion",
                "stage": "voice_conversion",
                "estimated_remaining": 8.5
            }
        }


class QualityAssessment(BaseModel):
    """Schema for synthesis quality assessment."""
    overall_quality: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    naturalness: float = Field(..., ge=0.0, le=1.0, description="Naturalness score")
    intelligibility: float = Field(..., ge=0.0, le=1.0, description="Intelligibility score")
    voice_similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity to reference voice")
    audio_quality: float = Field(..., ge=0.0, le=1.0, description="Technical audio quality")
    prosody_accuracy: float = Field(..., ge=0.0, le=1.0, description="Prosody preservation accuracy")
    
    class Config:
        json_schema_extra = {
            "example": {
                "overall_quality": 0.92,
                "naturalness": 0.89,
                "intelligibility": 0.95,
                "voice_similarity": 0.88,
                "audio_quality": 0.94,
                "prosody_accuracy": 0.87
            }
        }


class SynthesisMetadata(BaseModel):
    """Schema for detailed synthesis metadata."""
    text: str = Field(..., description="Original text")
    language: str = Field(..., description="Synthesis language")
    voice_model_id: str = Field(..., description="Voice model used")
    voice_settings: VoiceSettings = Field(..., description="Applied voice settings")
    duration: float = Field(..., description="Audio duration in seconds")
    sample_rate: int = Field(..., description="Audio sample rate")
    file_size: int = Field(..., description="Output file size in bytes")
    processing_time: float = Field(..., description="Total processing time")
    quality_assessment: QualityAssessment = Field(..., description="Quality metrics")
    cross_language: bool = Field(default=False, description="Whether cross-language synthesis was used")
    phonetic_adaptation: bool = Field(default=False, description="Whether phonetic adaptation was applied")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, this is a test.",
                "language": "english",
                "voice_model_id": "voice_model_123",
                "voice_settings": {
                    "pitch_shift": 0.0,
                    "speed_factor": 1.0,
                    "emotion_intensity": 1.0
                },
                "duration": 3.2,
                "sample_rate": 22050,
                "file_size": 141120,
                "processing_time": 8.5,
                "quality_assessment": {
                    "overall_quality": 0.92,
                    "naturalness": 0.89,
                    "intelligibility": 0.95,
                    "voice_similarity": 0.88,
                    "audio_quality": 0.94,
                    "prosody_accuracy": 0.87
                },
                "cross_language": False,
                "phonetic_adaptation": False
            }
        }


class BatchSynthesisResult(BaseModel):
    """Schema for batch synthesis results."""
    batch_id: str = Field(..., description="Batch identifier")
    status: SynthesisStatus = Field(..., description="Overall batch status")
    total_requests: int = Field(..., description="Total number of requests in batch")
    completed_requests: int = Field(..., description="Number of completed requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    results: List[SynthesisResult] = Field(..., description="Individual synthesis results")
    processing_time: float = Field(..., description="Total batch processing time")
    created_at: datetime = Field(..., description="Batch creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Batch completion timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "batch_id": "batch_123",
                "status": "completed",
                "total_requests": 5,
                "completed_requests": 4,
                "failed_requests": 1,
                "results": [],
                "processing_time": 45.2,
                "created_at": "2024-01-01T12:00:00Z",
                "completed_at": "2024-01-01T12:00:45Z"
            }
        }


class SynthesisStats(BaseModel):
    """Schema for synthesis statistics."""
    total_syntheses: int = Field(..., description="Total number of syntheses performed")
    successful_syntheses: int = Field(..., description="Number of successful syntheses")
    failed_syntheses: int = Field(..., description="Number of failed syntheses")
    average_processing_time: float = Field(..., description="Average processing time in seconds")
    average_quality_score: float = Field(..., description="Average quality score")
    total_audio_duration: float = Field(..., description="Total duration of synthesized audio")
    languages_supported: List[str] = Field(..., description="List of supported languages")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_syntheses": 1250,
                "successful_syntheses": 1198,
                "failed_syntheses": 52,
                "average_processing_time": 8.7,
                "average_quality_score": 0.89,
                "total_audio_duration": 3600.5,
                "languages_supported": ["english", "spanish", "french", "german"]
            }
        }