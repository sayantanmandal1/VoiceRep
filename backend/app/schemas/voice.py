"""
Pydantic schemas for voice analysis and modeling.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from app.models.voice import VoiceModelStatus


class FrequencyRange(BaseModel):
    """Schema for frequency range data."""
    min_hz: float
    max_hz: float
    mean_hz: float
    std_hz: float


class ProsodyFeaturesSchema(BaseModel):
    """Schema for prosody analysis features."""
    speech_rate: Optional[float] = None
    pause_frequency: Optional[float] = None
    emphasis_variance: Optional[float] = None
    syllable_duration_mean: Optional[float] = None
    syllable_duration_std: Optional[float] = None
    pause_duration_mean: Optional[float] = None
    pause_duration_std: Optional[float] = None
    stress_pattern_entropy: Optional[float] = None
    primary_stress_ratio: Optional[float] = None
    pitch_contour_complexity: Optional[float] = None
    pitch_range_semitones: Optional[float] = None
    declination_slope: Optional[float] = None
    excitement_score: Optional[float] = None
    calmness_score: Optional[float] = None
    confidence_score: Optional[float] = None
    pitch_contour: Optional[List[float]] = None
    energy_contour: Optional[List[float]] = None
    timing_features: Optional[Dict[str, Any]] = None


class EmotionalProfileSchema(BaseModel):
    """Schema for emotional voice characteristics."""
    valence: Optional[float] = Field(None, ge=-1.0, le=1.0)
    arousal: Optional[float] = Field(None, ge=-1.0, le=1.0)
    dominance: Optional[float] = Field(None, ge=-1.0, le=1.0)
    happiness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    sadness_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    anger_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    fear_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    surprise_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    disgust_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    breathiness: Optional[float] = Field(None, ge=0.0, le=1.0)
    roughness: Optional[float] = Field(None, ge=0.0, le=1.0)
    strain: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_interval: Optional[float] = None
    analysis_reliability: Optional[float] = Field(None, ge=0.0, le=1.0)


class VoiceProfileSchema(BaseModel):
    """Schema for complete voice profile data."""
    id: str
    reference_audio_id: str
    
    # Fundamental frequency
    fundamental_frequency: Optional[FrequencyRange] = None
    
    # Formant frequencies
    formant_frequencies: Optional[List[float]] = None
    
    # Spectral features
    spectral_centroid_mean: Optional[float] = None
    spectral_rolloff_mean: Optional[float] = None
    spectral_bandwidth_mean: Optional[float] = None
    zero_crossing_rate_mean: Optional[float] = None
    
    # MFCC features
    mfcc_features: Optional[List[List[float]]] = None
    
    # Prosody
    speech_rate: Optional[float] = None
    pause_frequency: Optional[float] = None
    emphasis_variance: Optional[float] = None
    
    # Energy and dynamics
    energy_mean: Optional[float] = None
    energy_variance: Optional[float] = None
    pitch_variance: Optional[float] = None
    
    # Quality metrics
    signal_to_noise_ratio: Optional[float] = None
    voice_activity_ratio: Optional[float] = None
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Metadata
    analysis_duration: Optional[float] = None
    sample_rate: Optional[int] = None
    total_frames: Optional[int] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class VoiceModelSchema(BaseModel):
    """Schema for voice model data."""
    id: str
    voice_profile_id: str
    reference_audio_id: str
    model_path: str
    config_path: Optional[str] = None
    voice_characteristics: Optional[Dict[str, Any]] = None
    model_type: str = "tortoise_tts"
    model_version: Optional[str] = None
    training_duration: Optional[float] = None
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    validation_loss: Optional[float] = None
    model_size_mb: Optional[float] = None
    inference_time_ms: Optional[float] = None
    status: VoiceModelStatus
    error_message: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class VoiceAnalysisRequest(BaseModel):
    """Request schema for voice analysis."""
    reference_audio_id: str
    analysis_options: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "extract_prosody": True,
            "extract_emotions": True,
            "create_model": True,
            "quality_threshold": 0.7
        }
    )


class VoiceAnalysisResponse(BaseModel):
    """Response schema for voice analysis task."""
    task_id: str
    reference_audio_id: str
    status: str
    message: str
    estimated_completion: Optional[datetime] = None
    voice_profile_id: Optional[str] = None


class QualityMetrics(BaseModel):
    """Schema for voice quality assessment."""
    signal_to_noise_ratio: float = Field(..., ge=0.0)
    spectral_similarity: Optional[float] = Field(None, ge=0.0, le=1.0)
    prosody_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0)
    overall_quality: float = Field(..., ge=0.0, le=1.0)
    voice_activity_ratio: float = Field(..., ge=0.0, le=1.0)
    frequency_response_quality: Optional[float] = Field(None, ge=0.0, le=1.0)
    temporal_consistency: Optional[float] = Field(None, ge=0.0, le=1.0)


class VoiceCharacteristics(BaseModel):
    """Schema for extracted voice characteristics."""
    timbre_features: Dict[str, float]
    pitch_characteristics: FrequencyRange
    prosody_features: ProsodyFeaturesSchema
    emotional_markers: EmotionalProfileSchema
    quality_metrics: QualityMetrics
    
    
class VoiceAnalysisResult(BaseModel):
    """Complete voice analysis result schema."""
    voice_profile: VoiceProfileSchema
    voice_characteristics: VoiceCharacteristics
    prosody_features: Optional[ProsodyFeaturesSchema] = None
    emotional_profile: Optional[EmotionalProfileSchema] = None
    processing_time: float
    analysis_metadata: Dict[str, Any]