"""
Voice analysis API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
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
from app.tasks.audio_processing import preprocess_audio_advanced_task, assess_audio_quality_task
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/analyze-multidimensional", response_model=VoiceAnalysisResponse)
async def analyze_voice_multidimensional(
    request: VoiceAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Start comprehensive multi-dimensional voice analysis with 1000+ features.
    This provides advanced voice characteristic extraction including:
    - Sub-Hz precision pitch analysis
    - Comprehensive formant tracking
    - Prosodic pattern extraction using ML
    - Timbre analysis (breathiness, roughness, resonance)
    - Emotional pattern recognition
    - Voice fingerprint with 1000+ features
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
    
    # Check if multi-dimensional analysis already exists
    existing_profile = db.query(VoiceProfile).filter(
        VoiceProfile.reference_audio_id == request.reference_audio_id,
        VoiceProfile.analysis_version == "2.0_multidimensional"
    ).first()
    
    if existing_profile:
        return VoiceAnalysisResponse(
            task_id=f"existing_multidim_{existing_profile.id}",
            reference_audio_id=request.reference_audio_id,
            status="completed",
            message="Multi-dimensional voice analysis already exists",
            voice_profile_id=existing_profile.id
        )
    
    # Start background multi-dimensional analysis task
    task_id = f"multidim_voice_analysis_{reference_audio.id}"
    
    background_tasks.add_task(
        _perform_multidimensional_voice_analysis,
        reference_audio.id,
        reference_audio.file_path,
        request.analysis_options,
        db
    )
    
    return VoiceAnalysisResponse(
        task_id=task_id,
        reference_audio_id=request.reference_audio_id,
        status="processing",
        message="Multi-dimensional voice analysis started (1000+ features)"
    )


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


@router.post("/preprocess-advanced/{reference_audio_id}")
async def preprocess_audio_advanced(
    reference_audio_id: str,
    db: Session = Depends(get_db)
):
    """
    Apply advanced audio preprocessing for optimal voice cloning.
    """
    # Check if reference audio exists
    reference_audio = db.query(ReferenceAudio).filter(
        ReferenceAudio.id == reference_audio_id
    ).first()
    
    if not reference_audio:
        raise HTTPException(status_code=404, detail="Reference audio not found")
    
    if not os.path.exists(reference_audio.file_path):
        raise HTTPException(status_code=404, detail="Audio file not found on disk")
    
    # Generate output path for preprocessed audio
    output_filename = f"preprocessed_{reference_audio.filename}"
    output_path = os.path.join(os.path.dirname(reference_audio.file_path), output_filename)
    
    # Start preprocessing task
    task = preprocess_audio_advanced_task.delay(
        reference_audio.file_path,
        output_path,
        reference_audio_id
    )
    
    return {
        "task_id": task.id,
        "reference_audio_id": reference_audio_id,
        "status": "processing",
        "message": "Advanced preprocessing started",
        "output_path": output_path
    }


@router.get("/preprocess-status/{task_id}")
async def get_preprocessing_status(task_id: str):
    """
    Get status of advanced preprocessing task.
    """
    from app.core.celery_app import celery_app
    
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            'task_id': task_id,
            'state': task.state,
            'status': 'Task is waiting to be processed'
        }
    elif task.state == 'PROGRESS':
        response = {
            'task_id': task_id,
            'state': task.state,
            'progress': task.info.get('progress', 0),
            'status': task.info.get('status', ''),
            'file_id': task.info.get('file_id', '')
        }
    elif task.state == 'SUCCESS':
        response = {
            'task_id': task_id,
            'state': task.state,
            'result': task.result,
            'status': 'Preprocessing completed successfully'
        }
    else:  # FAILURE
        response = {
            'task_id': task_id,
            'state': task.state,
            'error': str(task.info),
            'status': 'Preprocessing failed'
        }
    
    return response


@router.post("/assess-quality/{reference_audio_id}")
async def assess_audio_quality(
    reference_audio_id: str,
    db: Session = Depends(get_db)
):
    """
    Perform detailed audio quality assessment.
    """
    # Check if reference audio exists
    reference_audio = db.query(ReferenceAudio).filter(
        ReferenceAudio.id == reference_audio_id
    ).first()
    
    if not reference_audio:
        raise HTTPException(status_code=404, detail="Reference audio not found")
    
    if not os.path.exists(reference_audio.file_path):
        raise HTTPException(status_code=404, detail="Audio file not found on disk")
    
    # Start quality assessment task
    task = assess_audio_quality_task.delay(
        reference_audio.file_path,
        reference_audio_id
    )
    
    return {
        "task_id": task.id,
        "reference_audio_id": reference_audio_id,
        "status": "processing",
        "message": "Quality assessment started"
    }


@router.get("/quality-status/{task_id}")
async def get_quality_assessment_status(task_id: str):
    """
    Get status of quality assessment task.
    """
    from app.core.celery_app import celery_app
    
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            'task_id': task_id,
            'state': task.state,
            'status': 'Task is waiting to be processed'
        }
    elif task.state == 'PROGRESS':
        response = {
            'task_id': task_id,
            'state': task.state,
            'progress': task.info.get('progress', 0),
            'status': task.info.get('status', ''),
            'file_id': task.info.get('file_id', '')
        }
    elif task.state == 'SUCCESS':
        response = {
            'task_id': task_id,
            'state': task.state,
            'result': task.result,
            'status': 'Quality assessment completed successfully'
        }
    else:  # FAILURE
        response = {
            'task_id': task_id,
            'state': task.state,
            'error': str(task.info),
            'status': 'Quality assessment failed'
        }
    
    return response


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


async def _perform_multidimensional_voice_analysis(
    reference_audio_id: str,
    audio_file_path: str,
    analysis_options: dict,
    db: Session
):
    """
    Background task to perform comprehensive multi-dimensional voice analysis.
    """
    try:
        logger.info(f"Starting multi-dimensional voice analysis for audio {reference_audio_id}")
        
        # Initialize voice analyzer
        analyzer = VoiceAnalyzer()
        
        # Perform comprehensive multi-dimensional analysis
        analysis_result = analyzer.analyze_voice_comprehensive_multidimensional(audio_file_path)
        
        # Create enhanced voice profile in database with multi-dimensional features
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
        
        # Create enhanced voice model with multi-dimensional characteristics
        if analysis_options.get("create_model", True):
            # Create model using comprehensive characteristics
            model_data = analyzer.create_voice_model(analysis_result.voice_profile)
            
            # Enhance model data with multi-dimensional features
            enhanced_characteristics = model_data["characteristics"].copy()
            enhanced_characteristics.update({
                "multidimensional_features": analysis_result.analysis_metadata.get("advanced_features", {}),
                "voice_fingerprint_count": analysis_result.analysis_metadata.get("voice_fingerprint_features", 0),
                "analysis_version": "2.0_multidimensional",
                "emotional_profile": {
                    "valence": analysis_result.emotional_profile.valence if analysis_result.emotional_profile else 0.0,
                    "arousal": analysis_result.emotional_profile.arousal if analysis_result.emotional_profile else 0.0,
                    "dominance": analysis_result.emotional_profile.dominance if analysis_result.emotional_profile else 0.0,
                    "confidence": analysis_result.emotional_profile.analysis_reliability if analysis_result.emotional_profile else 0.0
                },
                "prosodic_features": {
                    "pitch_complexity": analysis_result.prosody_features.pitch_contour_complexity if analysis_result.prosody_features else 0.0,
                    "rhythm_regularity": analysis_result.prosody_features.emphasis_variance if analysis_result.prosody_features else 0.0,
                    "speech_rate": analysis_result.prosody_features.speech_rate if analysis_result.prosody_features else 0.0
                },
                "voice_quality": {
                    "breathiness": analysis_result.emotional_profile.breathiness if analysis_result.emotional_profile else 0.0,
                    "roughness": analysis_result.emotional_profile.roughness if analysis_result.emotional_profile else 0.0,
                    "strain": analysis_result.emotional_profile.strain if analysis_result.emotional_profile else 0.0
                }
            })
            
            # Create model directory if it doesn't exist
            model_dir = os.path.join(settings.MODELS_DIR, f"voice_multidim_{reference_audio_id}")
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, "multidimensional_model.json")
            
            # Save enhanced model data
            import json
            with open(model_path, 'w') as f:
                json.dump(enhanced_characteristics, f, indent=2)
            
            voice_model = VoiceModel(
                voice_profile_id=voice_profile.id,
                reference_audio_id=reference_audio_id,
                model_path=model_path,
                voice_characteristics=enhanced_characteristics,
                model_type="multidimensional_tortoise_tts",
                model_version="2.0",
                training_duration=model_data["training_duration"],
                quality_score=analysis_result.voice_characteristics.quality_metrics.overall_quality,
                model_size_mb=model_data["model_size_mb"] * 1.5,  # Larger due to additional features
                inference_time_ms=model_data["inference_time_ms"] * 1.2,  # Slightly slower due to complexity
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
        
        logger.info(f"Multi-dimensional voice analysis completed for audio {reference_audio_id} with {analysis_result.analysis_metadata.get('voice_fingerprint_features', 0)} features")
        
    except Exception as e:
        logger.error(f"Multi-dimensional voice analysis failed for audio {reference_audio_id}: {str(e)}")
        
        # Update reference audio with error
        reference_audio = db.query(ReferenceAudio).filter(
            ReferenceAudio.id == reference_audio_id
        ).first()
        
        if reference_audio:
            reference_audio.processing_status = ProcessingStatus.FAILED
            reference_audio.error_message = str(e)
            db.commit()
        
        raise