"""
Database models for speech synthesis operations.
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Text, Enum as SQLEnum, Boolean
from sqlalchemy.orm import relationship
from enum import Enum
from datetime import datetime
from app.models.base import BaseModel


class SynthesisTaskStatus(str, Enum):
    """Synthesis task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SynthesisTask(BaseModel):
    """Model for synthesis task tracking."""
    
    __tablename__ = "synthesis_tasks"
    
    # Task identification
    task_id = Column(String, nullable=False, unique=True)
    user_id = Column(String, nullable=True)  # For user tracking if implemented
    
    # Request parameters
    text = Column(Text, nullable=False)
    voice_model_id = Column(String, nullable=False)
    language = Column(String, nullable=True)
    voice_settings = Column(JSON, nullable=True)
    output_format = Column(String, default="wav")
    quality = Column(String, default="high")
    
    # Task status and progress
    status = Column(SQLEnum(SynthesisTaskStatus), default=SynthesisTaskStatus.PENDING)
    progress = Column(Integer, default=0)
    current_stage = Column(String, nullable=True)
    
    # Results
    output_path = Column(String, nullable=True)
    output_url = Column(String, nullable=True)
    
    # Metadata and metrics
    processing_time = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    synthesis_metadata = Column(JSON, nullable=True)
    
    # Error handling
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Cross-language synthesis flags
    is_cross_language = Column(Boolean, default=False)
    source_language = Column(String, nullable=True)
    target_language = Column(String, nullable=True)
    phonetic_adaptation = Column(Boolean, default=False)


class BatchSynthesisTask(BaseModel):
    """Model for batch synthesis operations."""
    
    __tablename__ = "batch_synthesis_tasks"
    
    # Batch identification
    batch_id = Column(String, nullable=False, unique=True)
    batch_name = Column(String, nullable=True)
    user_id = Column(String, nullable=True)
    
    # Batch parameters
    total_requests = Column(Integer, nullable=False)
    priority = Column(Integer, default=5)
    
    # Status tracking
    status = Column(SQLEnum(SynthesisTaskStatus), default=SynthesisTaskStatus.PENDING)
    completed_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    
    # Processing metrics
    processing_time = Column(Float, nullable=True)
    average_quality_score = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Results summary
    results_summary = Column(JSON, nullable=True)


class SynthesisResult(BaseModel):
    """Model for storing synthesis results and metadata."""
    
    __tablename__ = "synthesis_results"
    
    # Result identification
    task_id = Column(String, nullable=False)
    synthesis_task_id = Column(String, nullable=True)  # Link to SynthesisTask
    
    # Output information
    output_path = Column(String, nullable=False)
    output_url = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)
    
    # Audio characteristics
    sample_rate = Column(Integer, nullable=True)
    channels = Column(Integer, default=1)
    bit_depth = Column(Integer, default=16)
    
    # Quality metrics
    overall_quality = Column(Float, nullable=True)
    naturalness_score = Column(Float, nullable=True)
    intelligibility_score = Column(Float, nullable=True)
    voice_similarity_score = Column(Float, nullable=True)
    audio_quality_score = Column(Float, nullable=True)
    prosody_accuracy_score = Column(Float, nullable=True)
    
    # Processing information
    processing_time = Column(Float, nullable=True)
    model_used = Column(String, nullable=True)
    synthesis_method = Column(String, nullable=True)
    
    # Detailed metadata
    synthesis_metadata = Column(JSON, nullable=True)
    quality_assessment = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)


class VoiceModelOptimization(BaseModel):
    """Model for tracking voice model optimization tasks."""
    
    __tablename__ = "voice_model_optimizations"
    
    # Optimization identification
    optimization_id = Column(String, nullable=False, unique=True)
    voice_model_id = Column(String, nullable=False)
    
    # Optimization parameters
    optimization_type = Column(String, nullable=False)  # quantization, pruning, etc.
    optimization_settings = Column(JSON, nullable=True)
    
    # Status and progress
    status = Column(SQLEnum(SynthesisTaskStatus), default=SynthesisTaskStatus.PENDING)
    progress = Column(Integer, default=0)
    
    # Results
    optimized_model_path = Column(String, nullable=True)
    original_model_size = Column(Float, nullable=True)
    optimized_model_size = Column(Float, nullable=True)
    size_reduction_percent = Column(Float, nullable=True)
    
    # Performance metrics
    original_inference_time = Column(Float, nullable=True)
    optimized_inference_time = Column(Float, nullable=True)
    speed_improvement_percent = Column(Float, nullable=True)
    quality_retention_percent = Column(Float, nullable=True)
    
    # Processing information
    processing_time = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)


class SynthesisStatistics(BaseModel):
    """Model for storing synthesis system statistics."""
    
    __tablename__ = "synthesis_statistics"
    
    # Time period
    date = Column(DateTime, nullable=False)
    period_type = Column(String, nullable=False)  # daily, weekly, monthly
    
    # Synthesis counts
    total_syntheses = Column(Integer, default=0)
    successful_syntheses = Column(Integer, default=0)
    failed_syntheses = Column(Integer, default=0)
    cancelled_syntheses = Column(Integer, default=0)
    
    # Cross-language synthesis
    cross_language_syntheses = Column(Integer, default=0)
    
    # Performance metrics
    average_processing_time = Column(Float, nullable=True)
    median_processing_time = Column(Float, nullable=True)
    average_quality_score = Column(Float, nullable=True)
    
    # Audio metrics
    total_audio_duration = Column(Float, default=0.0)
    average_audio_duration = Column(Float, nullable=True)
    
    # Language distribution
    language_distribution = Column(JSON, nullable=True)
    
    # Quality distribution
    quality_distribution = Column(JSON, nullable=True)
    
    # Error analysis
    error_types = Column(JSON, nullable=True)
    
    # Resource usage
    average_cpu_usage = Column(Float, nullable=True)
    average_memory_usage = Column(Float, nullable=True)
    peak_concurrent_tasks = Column(Integer, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SynthesisQueue(BaseModel):
    """Model for managing synthesis task queue."""
    
    __tablename__ = "synthesis_queue"
    
    # Queue entry identification
    queue_id = Column(String, nullable=False, unique=True)
    task_id = Column(String, nullable=False)
    
    # Queue parameters
    priority = Column(Integer, default=5)
    queue_position = Column(Integer, nullable=True)
    estimated_processing_time = Column(Float, nullable=True)
    
    # Task information
    task_type = Column(String, nullable=False)  # synthesis, cross_language, batch
    text_length = Column(Integer, nullable=True)
    voice_model_id = Column(String, nullable=True)
    
    # Status
    status = Column(String, default="queued")  # queued, processing, completed, failed
    
    # Timestamps
    queued_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Processing node (for distributed processing)
    processing_node = Column(String, nullable=True)