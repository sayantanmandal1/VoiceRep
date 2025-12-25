"""
Voice analysis and modeling data models.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from enum import Enum
from app.models.base import BaseModel
import numpy as np
from typing import Dict, List, Any


class VoiceModelStatus(str, Enum):
    """Voice model processing status enumeration."""
    ANALYZING = "analyzing"
    READY = "ready"
    FAILED = "failed"


class VoiceProfile(BaseModel):
    """Model for storing voice characteristics and analysis results."""
    
    __tablename__ = "voice_profiles"
    
    reference_audio_id = Column(String, nullable=False, unique=True)
    
    # Fundamental frequency characteristics
    f0_mean = Column(Float, nullable=True)
    f0_std = Column(Float, nullable=True)
    f0_min = Column(Float, nullable=True)
    f0_max = Column(Float, nullable=True)
    
    # Formant frequencies (stored as JSON array)
    formant_frequencies = Column(JSON, nullable=True)
    
    # Spectral characteristics
    spectral_centroid_mean = Column(Float, nullable=True)
    spectral_rolloff_mean = Column(Float, nullable=True)
    spectral_bandwidth_mean = Column(Float, nullable=True)
    zero_crossing_rate_mean = Column(Float, nullable=True)
    
    # MFCC features (stored as JSON array)
    mfcc_features = Column(JSON, nullable=True)
    
    # Prosody features
    speech_rate = Column(Float, nullable=True)  # syllables per second
    pause_frequency = Column(Float, nullable=True)  # pauses per minute
    emphasis_variance = Column(Float, nullable=True)
    
    # Emotional and style markers
    energy_mean = Column(Float, nullable=True)
    energy_variance = Column(Float, nullable=True)
    pitch_variance = Column(Float, nullable=True)
    
    # Quality metrics
    signal_to_noise_ratio = Column(Float, nullable=True)
    voice_activity_ratio = Column(Float, nullable=True)
    
    # Analysis metadata
    analysis_duration = Column(Float, nullable=True)
    sample_rate = Column(Integer, nullable=True)
    total_frames = Column(Integer, nullable=True)
    
    # Quality score (0-1)
    quality_score = Column(Float, nullable=True)
    
    # Raw feature data path (for large arrays)
    features_file_path = Column(String, nullable=True)


class VoiceModel(BaseModel):
    """Model for storing trained voice models."""
    
    __tablename__ = "voice_models"
    
    voice_profile_id = Column(String, nullable=False)
    reference_audio_id = Column(String, nullable=False)
    
    # Model file paths
    model_path = Column(String, nullable=False)
    config_path = Column(String, nullable=True)
    
    # Model characteristics (JSON storage for flexibility)
    voice_characteristics = Column(JSON, nullable=True)
    
    # Model metadata
    model_type = Column(String, default="tortoise_tts")
    model_version = Column(String, nullable=True)
    training_duration = Column(Float, nullable=True)
    
    # Quality metrics
    quality_score = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    
    # Model size and performance
    model_size_mb = Column(Float, nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    
    # Status
    status = Column(SQLEnum(VoiceModelStatus), default=VoiceModelStatus.ANALYZING)
    error_message = Column(Text, nullable=True)


class ProsodyFeatures(BaseModel):
    """Model for detailed prosody analysis results."""
    
    __tablename__ = "prosody_features"
    
    voice_profile_id = Column(String, nullable=False)
    
    # Rhythm and timing
    syllable_duration_mean = Column(Float, nullable=True)
    syllable_duration_std = Column(Float, nullable=True)
    pause_duration_mean = Column(Float, nullable=True)
    pause_duration_std = Column(Float, nullable=True)
    
    # Stress patterns
    stress_pattern_entropy = Column(Float, nullable=True)
    primary_stress_ratio = Column(Float, nullable=True)
    
    # Intonation patterns
    pitch_contour_complexity = Column(Float, nullable=True)
    pitch_range_semitones = Column(Float, nullable=True)
    declination_slope = Column(Float, nullable=True)
    
    # Emotional prosody markers
    excitement_score = Column(Float, nullable=True)
    calmness_score = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Temporal features (stored as JSON arrays)
    pitch_contour = Column(JSON, nullable=True)
    energy_contour = Column(JSON, nullable=True)
    timing_features = Column(JSON, nullable=True)


class EmotionalProfile(BaseModel):
    """Model for emotional characteristics of voice."""
    
    __tablename__ = "emotional_profiles"
    
    voice_profile_id = Column(String, nullable=False)
    
    # Basic emotional dimensions
    valence = Column(Float, nullable=True)  # positive/negative emotion
    arousal = Column(Float, nullable=True)  # energy/activation level
    dominance = Column(Float, nullable=True)  # control/power
    
    # Specific emotional markers
    happiness_score = Column(Float, nullable=True)
    sadness_score = Column(Float, nullable=True)
    anger_score = Column(Float, nullable=True)
    fear_score = Column(Float, nullable=True)
    surprise_score = Column(Float, nullable=True)
    disgust_score = Column(Float, nullable=True)
    
    # Voice quality emotional indicators
    breathiness = Column(Float, nullable=True)
    roughness = Column(Float, nullable=True)
    strain = Column(Float, nullable=True)
    
    # Confidence metrics
    confidence_interval = Column(Float, nullable=True)
    analysis_reliability = Column(Float, nullable=True)