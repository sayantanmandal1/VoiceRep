"""
API endpoints for real-time quality monitoring system.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
import logging
import numpy as np
from datetime import datetime

from app.services.real_time_quality_monitor import (
    real_time_quality_monitor, ProcessingStage, RealTimeMetrics,
    SimilarityMetrics, ImprovementRecommendation, OptimizationStrategy,
    ConfidenceScores
)
from app.schemas.voice import VoiceCharacteristics

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response schemas
from pydantic import BaseModel, Field


class StartMonitoringRequest(BaseModel):
    """Request to start quality monitoring session."""
    session_id: str
    quality_targets: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {
            'minimum_similarity': 0.95,
            'minimum_quality': 0.85,
            'minimum_confidence': 0.80,
            'target_processing_time': 30.0
        }
    )


class UpdateStageRequest(BaseModel):
    """Request to update processing stage."""
    session_id: str
    stage: str  # ProcessingStage enum value
    progress: float = Field(0.0, ge=0.0, le=1.0)


class QualityAssessmentRequest(BaseModel):
    """Request for real-time quality assessment."""
    session_id: str
    audio_file_id: str
    stage: str  # ProcessingStage enum value
    reference_audio_id: Optional[str] = None


class SimilarityCalculationRequest(BaseModel):
    """Request for detailed similarity calculation."""
    session_id: str
    reference_audio_id: str
    synthesized_audio_id: str


class RecommendationsRequest(BaseModel):
    """Request for improvement recommendations."""
    session_id: str
    voice_characteristics: Optional[VoiceCharacteristics] = None


class OptimizationRequest(BaseModel):
    """Request for optimization strategy."""
    session_id: str
    target_quality: float = Field(0.95, ge=0.0, le=1.0)


# Response schemas
class RealTimeMetricsResponse(BaseModel):
    """Response containing real-time metrics."""
    timestamp: datetime
    stage: str
    similarity_score: float
    quality_score: float
    confidence_score: float
    processing_time: float
    issues_detected: List[str]
    recommendations: List[str]
    stage_progress: float


class SimilarityMetricsResponse(BaseModel):
    """Response containing detailed similarity metrics."""
    overall_similarity: float
    pitch_similarity: float
    timbre_similarity: float
    prosody_similarity: float
    emotional_similarity: float
    spectral_similarity: float
    temporal_similarity: float
    confidence_interval: List[float]  # [lower, upper]
    breakdown: Dict[str, float]


class ImprovementRecommendationResponse(BaseModel):
    """Response containing improvement recommendation."""
    category: str
    priority: int
    issue_description: str
    recommended_action: str
    expected_improvement: float
    implementation_steps: List[str]
    estimated_time: float
    prerequisites: List[str]


class OptimizationStrategyResponse(BaseModel):
    """Response containing optimization strategy."""
    strategy_name: str
    target_quality: float
    current_quality: float
    improvement_potential: float
    recommended_steps: List[ImprovementRecommendationResponse]
    estimated_total_time: float
    success_probability: float


class ConfidenceScoresResponse(BaseModel):
    """Response containing confidence scores."""
    pitch_extraction: float
    formant_detection: float
    timbre_analysis: float
    prosody_extraction: float
    emotional_analysis: float
    overall_analysis: float
    voice_model_quality: float
    synthesis_quality: float
    characteristic_reliability: Dict[str, float]


@router.post("/start-session", response_model=Dict[str, str])
async def start_monitoring_session(request: StartMonitoringRequest):
    """
    Start a new quality monitoring session.
    
    Args:
        request: Session start request
        
    Returns:
        Session information
    """
    try:
        session_id = real_time_quality_monitor.start_monitoring_session(
            session_id=request.session_id,
            quality_targets=request.quality_targets
        )
        
        logger.info(f"Started quality monitoring session: {session_id}")
        
        return {
            "session_id": session_id,
            "status": "started",
            "message": "Quality monitoring session started successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to start monitoring session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring session: {str(e)}")


@router.post("/update-stage")
async def update_processing_stage(request: UpdateStageRequest):
    """
    Update the current processing stage.
    
    Args:
        request: Stage update request
        
    Returns:
        Update confirmation
    """
    try:
        # Convert string to ProcessingStage enum
        stage_mapping = {
            'audio_preprocessing': ProcessingStage.AUDIO_PREPROCESSING,
            'voice_analysis': ProcessingStage.VOICE_ANALYSIS,
            'model_training': ProcessingStage.MODEL_TRAINING,
            'synthesis': ProcessingStage.SYNTHESIS,
            'post_processing': ProcessingStage.POST_PROCESSING
        }
        
        stage = stage_mapping.get(request.stage)
        if not stage:
            raise HTTPException(status_code=400, detail=f"Invalid stage: {request.stage}")
        
        real_time_quality_monitor.update_processing_stage(
            session_id=request.session_id,
            stage=stage,
            progress=request.progress
        )
        
        return {
            "session_id": request.session_id,
            "stage": request.stage,
            "progress": request.progress,
            "status": "updated"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to update processing stage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update processing stage: {str(e)}")


@router.post("/assess-quality", response_model=RealTimeMetricsResponse)
async def assess_real_time_quality(request: QualityAssessmentRequest):
    """
    Perform real-time quality assessment.
    
    Args:
        request: Quality assessment request
        
    Returns:
        Real-time metrics
    """
    try:
        # Convert string to ProcessingStage enum
        stage_mapping = {
            'audio_preprocessing': ProcessingStage.AUDIO_PREPROCESSING,
            'voice_analysis': ProcessingStage.VOICE_ANALYSIS,
            'model_training': ProcessingStage.MODEL_TRAINING,
            'synthesis': ProcessingStage.SYNTHESIS,
            'post_processing': ProcessingStage.POST_PROCESSING
        }
        
        stage = stage_mapping.get(request.stage)
        if not stage:
            raise HTTPException(status_code=400, detail=f"Invalid stage: {request.stage}")
        
        # Load audio file
        # For now, we'll assume the audio_file_id is a direct path
        # In a real implementation, this would look up the file in a database
        try:
            import librosa
            audio, sample_rate = librosa.load(request.audio_file_id, sr=22050)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Audio file not found or invalid: {str(e)}")
        
        # Load reference audio if provided
        reference_audio = None
        if request.reference_audio_id:
            try:
                reference_audio, _ = librosa.load(request.reference_audio_id, sr=22050)
            except Exception as e:
                logger.warning(f"Could not load reference audio: {str(e)}")
                reference_audio = None
        
        # Perform quality assessment
        metrics = real_time_quality_monitor.assess_real_time_quality(
            session_id=request.session_id,
            audio=audio,
            sample_rate=sample_rate,
            stage=stage,
            reference_audio=reference_audio
        )
        
        return RealTimeMetricsResponse(
            timestamp=metrics.timestamp,
            stage=metrics.stage.value,
            similarity_score=metrics.similarity_score,
            quality_score=metrics.quality_score,
            confidence_score=metrics.confidence_score,
            processing_time=metrics.processing_time,
            issues_detected=metrics.issues_detected,
            recommendations=metrics.recommendations,
            stage_progress=metrics.stage_progress
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to assess quality: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to assess quality: {str(e)}")


@router.post("/calculate-similarity", response_model=SimilarityMetricsResponse)
async def calculate_detailed_similarity(request: SimilarityCalculationRequest):
    """
    Calculate detailed similarity metrics.
    
    Args:
        request: Similarity calculation request
        
    Returns:
        Detailed similarity metrics
    """
    try:
        # Load reference audio
        try:
            reference_audio, sample_rate = librosa.load(request.reference_audio_id, sr=22050)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Reference audio file not found: {str(e)}")
        
        # Load synthesized audio
        try:
            synthesized_audio, _ = librosa.load(request.synthesized_audio_id, sr=22050)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Synthesized audio file not found: {str(e)}")
        
        # Calculate similarity metrics
        similarity_metrics = real_time_quality_monitor.calculate_detailed_similarity_metrics(
            session_id=request.session_id,
            reference_audio=reference_audio,
            synthesized_audio=synthesized_audio,
            sample_rate=sample_rate
        )
        
        return SimilarityMetricsResponse(
            overall_similarity=similarity_metrics.overall_similarity,
            pitch_similarity=similarity_metrics.pitch_similarity,
            timbre_similarity=similarity_metrics.timbre_similarity,
            prosody_similarity=similarity_metrics.prosody_similarity,
            emotional_similarity=similarity_metrics.emotional_similarity,
            spectral_similarity=similarity_metrics.spectral_similarity,
            temporal_similarity=similarity_metrics.temporal_similarity,
            confidence_interval=[similarity_metrics.confidence_interval[0], similarity_metrics.confidence_interval[1]],
            breakdown=similarity_metrics.breakdown
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to calculate similarity: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate similarity: {str(e)}")


@router.post("/recommendations", response_model=List[ImprovementRecommendationResponse])
async def generate_improvement_recommendations(request: RecommendationsRequest):
    """
    Generate improvement recommendations.
    
    Args:
        request: Recommendations request
        
    Returns:
        List of improvement recommendations
    """
    try:
        recommendations = real_time_quality_monitor.generate_improvement_recommendations(
            session_id=request.session_id,
            voice_characteristics=request.voice_characteristics
        )
        
        return [
            ImprovementRecommendationResponse(
                category=rec.category,
                priority=rec.priority,
                issue_description=rec.issue_description,
                recommended_action=rec.recommended_action,
                expected_improvement=rec.expected_improvement,
                implementation_steps=rec.implementation_steps,
                estimated_time=rec.estimated_time,
                prerequisites=rec.prerequisites
            )
            for rec in recommendations
        ]
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@router.post("/optimization-strategy", response_model=OptimizationStrategyResponse)
async def generate_optimization_strategy(request: OptimizationRequest):
    """
    Generate optimization strategy.
    
    Args:
        request: Optimization request
        
    Returns:
        Optimization strategy
    """
    try:
        strategy = real_time_quality_monitor.generate_optimization_strategy(
            session_id=request.session_id,
            target_quality=request.target_quality
        )
        
        recommended_steps = [
            ImprovementRecommendationResponse(
                category=rec.category,
                priority=rec.priority,
                issue_description=rec.issue_description,
                recommended_action=rec.recommended_action,
                expected_improvement=rec.expected_improvement,
                implementation_steps=rec.implementation_steps,
                estimated_time=rec.estimated_time,
                prerequisites=rec.prerequisites
            )
            for rec in strategy.recommended_steps
        ]
        
        return OptimizationStrategyResponse(
            strategy_name=strategy.strategy_name,
            target_quality=strategy.target_quality,
            current_quality=strategy.current_quality,
            improvement_potential=strategy.improvement_potential,
            recommended_steps=recommended_steps,
            estimated_total_time=strategy.estimated_total_time,
            success_probability=strategy.success_probability
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate optimization strategy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate optimization strategy: {str(e)}")


@router.get("/confidence-scores/{session_id}", response_model=Optional[ConfidenceScoresResponse])
async def get_confidence_scores(session_id: str):
    """
    Get confidence scores for all extracted characteristics.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Confidence scores or None if not available
    """
    try:
        confidence_scores = real_time_quality_monitor.get_confidence_scores(session_id)
        
        if not confidence_scores:
            return None
        
        return ConfidenceScoresResponse(
            pitch_extraction=confidence_scores.pitch_extraction,
            formant_detection=confidence_scores.formant_detection,
            timbre_analysis=confidence_scores.timbre_analysis,
            prosody_extraction=confidence_scores.prosody_extraction,
            emotional_analysis=confidence_scores.emotional_analysis,
            overall_analysis=confidence_scores.overall_analysis,
            voice_model_quality=confidence_scores.voice_model_quality,
            synthesis_quality=confidence_scores.synthesis_quality,
            characteristic_reliability=confidence_scores.characteristic_reliability
        )
        
    except Exception as e:
        logger.error(f"Failed to get confidence scores: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get confidence scores: {str(e)}")


@router.get("/session-summary/{session_id}")
async def get_session_summary(session_id: str):
    """
    Get comprehensive session summary.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session summary
    """
    try:
        summary = real_time_quality_monitor.get_session_summary(session_id)
        return summary
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get session summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get session summary: {str(e)}")


@router.delete("/end-session/{session_id}")
async def end_monitoring_session(session_id: str):
    """
    End monitoring session and return final summary.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Final session summary
    """
    try:
        summary = real_time_quality_monitor.end_monitoring_session(session_id)
        
        logger.info(f"Ended quality monitoring session: {session_id}")
        
        return {
            "session_id": session_id,
            "status": "ended",
            "final_summary": summary
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to end monitoring session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to end monitoring session: {str(e)}")


@router.get("/active-sessions")
async def get_active_sessions():
    """
    Get list of active monitoring sessions.
    
    Returns:
        List of active session IDs
    """
    try:
        # Access the sessions directly from the monitor
        active_sessions = list(real_time_quality_monitor.sessions.keys())
        
        return {
            "active_sessions": active_sessions,
            "count": len(active_sessions)
        }
        
    except Exception as e:
        logger.error(f"Failed to get active sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get active sessions: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint for quality monitoring service.
    
    Returns:
        Service health status
    """
    return {
        "service": "real_time_quality_monitor",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(real_time_quality_monitor.sessions)
    }