"""
Voice analysis API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import logging

from app.core.database import get_db
from app.models.file import ReferenceAudio, ProcessingStatus
from app.models.voice import VoiceProfile, VoiceModel, VoiceModelStatus
from app.schemas.voice import (
    VoiceAnalysisRequest, VoiceAnalysisResponse, VoiceProfileSchema,
    VoiceModelSchema, VoiceAnalysisResult
)
from app.services.voice_analysis_service import VoiceAnalyzer
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analyze", response_model=VoiceAnalysisResponse)
async def analyze_voice(
    request: VoiceAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start voice analysis for a reference audio file.
    """
    # Check if reference audio exists
    reference_audio = db.query(ReferenceAudio).filter(
        ReferenceAudio.id == request.reference_audio_id
    ).first()
    
    if not reference_audio:
        raise HTTPException(status_code=404, detail="Reference audio not found")
    
    if reference_audio.processing_status != ProcessingStatus.READY:
        raise HTTPException(
            status_code=400, 
            detail=f"Reference audio is not ready for analysis. Status: {reference_audio.processing_status}"
        )
    
    # Check if analysis already exists
    existing_profile = db.query(VoiceProfile).filter(
        VoiceProfile.reference_audio_id == request.reference_audio_id
    ).first()
    
    if existing_profile:
        return VoiceAnalysisResponse(
            task_id=f"existing_{existing_profile.id}",
            reference_audio_id=request.reference_audio_id,
            status="completed",
            message="Voice analysis already exists",
            voice_profile_id=existing_profile.id
        )
    
    # Start background analysis task
    task_id = f"voice_analysis_{reference_audio.id}"
    
    background_tasks.add_task(
        _perform_voice_analysis,
        reference_audio.id,
        reference_audio.file_path,
        request.analysis_options,
        db
    )
    
    return VoiceAnalysisResponse(
        task_id=task_id,
        reference_audio_id=request.reference_audio_id,
        status="processing",
        message="Voice analysis started"
    )


@router.get("/profile/{reference_audio_id}", response_model=VoiceProfileSchema)
async def get_voice_profile(
    reference_audio_id: str,
    db: Session = Depends(get_db)
):
    """
    Get voice profile for a reference audio file.
    """
    voice_profile = db.query(VoiceProfile).filter(
        VoiceProfile.reference_audio_id == reference_audio_id
    ).first()
    
    if not voice_profile:
        raise HTTPException(status_code=404, detail="Voice profile not found")
    
    return voice_profile


@router.get("/model/{reference_audio_id}", response_model=VoiceModelSchema)
async def get_voice_model(
    reference_audio_id: str,
    db: Session = Depends(get_db)
):
    """
    Get voice model for a reference audio file.
    """
    voice_model = db.query(VoiceModel).filter(
        VoiceModel.reference_audio_id == reference_audio_id
    ).first()
    
    if not voice_model:
        raise HTTPException(status_code=404, detail="Voice model not found")
    
    return voice_model


@router.get("/profiles", response_model=List[VoiceProfileSchema])
async def list_voice_profiles(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    List all voice profiles.
    """
    profiles = db.query(VoiceProfile).offset(skip).limit(limit).all()
    return profiles


@router.delete("/profile/{reference_audio_id}")
async def delete_voice_profile(
    reference_audio_id: str,
    db: Session = Depends(get_db)
):
    """
    Delete voice profile and associated model.
    """
    # Delete voice profile
    voice_profile = db.query(VoiceProfile).filter(
        VoiceProfile.reference_audio_id == reference_audio_id
    ).first()
    
    if voice_profile:
        db.delete(voice_profile)
    
    # Delete voice model
    voice_model = db.query(VoiceModel).filter(
        VoiceModel.reference_audio_id == reference_audio_id
    ).first()
    
    if voice_model:
        # Clean up model files
        if os.path.exists(voice_model.model_path):
            os.remove(voice_model.model_path)
        if voice_model.config_path and os.path.exists(voice_model.config_path):
            os.remove(voice_model.config_path)
        
        db.delete(voice_model)
    
    db.commit()
    
    return {"message": "Voice profile and model deleted successfully"}


async def _perform_voice_analysis(
    reference_audio_id: str,
    audio_file_path: str,
    analysis_options: dict,
    db: Session
):
    """
    Background task to perform voice analysis.
    """
    try:
        logger.info(f"Starting voice analysis for audio {reference_audio_id}")
        
        # Initialize voice analyzer
        analyzer = VoiceAnalyzer()
        
        # Perform analysis
        analysis_result = analyzer.analyze_voice_characteristics(audio_file_path)
        
        # Create voice profile in database
        voice_profile = VoiceProfile(
            reference_audio_id=reference_audio_id,
            f0_mean=analysis_result.voice_profile.fundamental_frequency.mean_hz if analysis_result.voice_profile.fundamental_frequency else None,
            f0_std=analysis_result.voice_profile.fundamental_frequency.std_hz if analysis_result.voice_profile.fundamental_frequency else None,
            f0_min=analysis_result.voice_profile.fundamental_frequency.min_hz if analysis_result.voice_profile.fundamental_frequency else None,
            f0_max=analysis_result.voice_profile.fundamental_frequency.max_hz if analysis_result.voice_profile.fundamental_frequency else None,
            formant_frequencies=analysis_result.voice_profile.formant_frequencies,
            spectral_centroid_mean=analysis_result.voice_profile.spectral_centroid_mean,
            spectral_rolloff_mean=analysis_result.voice_profile.spectral_rolloff_mean,
            spectral_bandwidth_mean=analysis_result.voice_profile.spectral_bandwidth_mean,
            zero_crossing_rate_mean=analysis_result.voice_profile.zero_crossing_rate_mean,
            mfcc_features=analysis_result.voice_profile.mfcc_features,
            speech_rate=analysis_result.voice_profile.speech_rate,
            pause_frequency=analysis_result.voice_profile.pause_frequency,
            emphasis_variance=analysis_result.voice_profile.emphasis_variance,
            energy_mean=analysis_result.voice_profile.energy_mean,
            energy_variance=analysis_result.voice_profile.energy_variance,
            pitch_variance=analysis_result.voice_profile.pitch_variance,
            signal_to_noise_ratio=analysis_result.voice_profile.signal_to_noise_ratio,
            voice_activity_ratio=analysis_result.voice_profile.voice_activity_ratio,
            quality_score=analysis_result.voice_profile.quality_score,
            analysis_duration=analysis_result.voice_profile.analysis_duration,
            sample_rate=analysis_result.voice_profile.sample_rate,
            total_frames=analysis_result.voice_profile.total_frames
        )
        
        db.add(voice_profile)
        db.commit()
        db.refresh(voice_profile)
        
        # Create voice model if requested
        if analysis_options.get("create_model", True):
            model_data = analyzer.create_voice_model(analysis_result.voice_profile)
            
            # Create model directory if it doesn't exist
            model_dir = os.path.join(settings.MODELS_DIR, f"voice_{reference_audio_id}")
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, "model.json")
            
            # Save model data (placeholder - in production this would be actual model files)
            import json
            with open(model_path, 'w') as f:
                json.dump(model_data["characteristics"], f)
            
            voice_model = VoiceModel(
                voice_profile_id=voice_profile.id,
                reference_audio_id=reference_audio_id,
                model_path=model_path,
                voice_characteristics=model_data["characteristics"],
                model_type=model_data["model_type"],
                model_version=model_data["model_version"],
                training_duration=model_data["training_duration"],
                quality_score=model_data["quality_score"],
                model_size_mb=model_data["model_size_mb"],
                inference_time_ms=model_data["inference_time_ms"],
                status=VoiceModelStatus.READY
            )
            
            db.add(voice_model)
            db.commit()
        
        # Update reference audio status
        reference_audio = db.query(ReferenceAudio).filter(
            ReferenceAudio.id == reference_audio_id
        ).first()
        
        if reference_audio:
            reference_audio.voice_profile_id = voice_profile.id
            reference_audio.processing_status = ProcessingStatus.READY
            db.commit()
        
        logger.info(f"Voice analysis completed for audio {reference_audio_id}")
        
    except Exception as e:
        logger.error(f"Voice analysis failed for audio {reference_audio_id}: {str(e)}")
        
        # Update reference audio with error
        reference_audio = db.query(ReferenceAudio).filter(
            ReferenceAudio.id == reference_audio_id
        ).first()
        
        if reference_audio:
            reference_audio.processing_status = ProcessingStatus.FAILED
            reference_audio.error_message = str(e)
            db.commit()
        
        raise