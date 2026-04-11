"""
API endpoints for speech synthesis operations using real voice cloning.
"""

import os
import uuid
import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import FileResponse
from pathlib import Path

from app.schemas.synthesis import (
    SynthesisRequest, SynthesisResponse, SynthesisResult, SynthesisProgress,
    CrossLanguageSynthesisRequest, BatchSynthesisRequest, BatchSynthesisResult,
    SynthesisStatus, SynthesisStats
)
from app.schemas.voice import VoiceModelSchema
from app.core.config import settings
from app.models.voice import VoiceModel, VoiceModelStatus
from app.services.voice_cloner import voice_cloner
from app.models.file import ReferenceAudio
from app.core.database import get_db
from app.middleware.session_middleware import require_session
from app.models.session import UserSession
from app.tasks.synthesis_tasks import (
    cross_language_synthesis_task, batch_synthesis_task, optimize_synthesis_model
)
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter()

# Real task states storage (in production, use Redis or database)
synthesis_tasks: Dict[str, Dict[str, Any]] = {}


def get_voice_model_by_id(voice_model_id: str, db: Session) -> Optional[VoiceModelSchema]:
    """Get voice model from database by ID."""
    try:
        # Handle different voice model ID formats
        if voice_model_id.startswith("voice_model_"):
            # Extract file ID from voice_model_id (format: voice_model_{file_id})
            file_id = voice_model_id.replace("voice_model_", "")
        else:
            # Assume it's a direct file ID
            file_id = voice_model_id
        
        logger.info(f"Looking for voice model with file_id: {file_id}")
        
        # Look up voice model by reference audio ID
        voice_model = db.query(VoiceModel).filter(
            VoiceModel.reference_audio_id == file_id
        ).first()
        
        if voice_model:
            logger.info(f"Found existing voice model for file {file_id}")
            return VoiceModelSchema(
                id=voice_model.id,
                voice_profile_id=voice_model.voice_profile_id,
                reference_audio_id=voice_model.reference_audio_id,
                model_path=voice_model.model_path,
                voice_characteristics=voice_model.voice_characteristics or {},
                model_type=voice_model.model_type or "xtts_v2",
                quality_score=voice_model.quality_score or 0.85,
                status=voice_model.status,
                created_at=voice_model.created_at
            )
        
        # If no voice model found, look for reference audio and create a temporary voice model
        reference_audio = db.query(ReferenceAudio).filter(
            ReferenceAudio.id == file_id
        ).first()
        
        if reference_audio:
            logger.info(f"Creating temporary voice model for file {file_id} at path: {reference_audio.file_path}")
            
            # Check if the file actually exists
            if not os.path.exists(reference_audio.file_path):
                logger.error(f"Reference audio file not found at path: {reference_audio.file_path}")
                return None
            
            return VoiceModelSchema(
                id=f"temp_voice_model_{file_id}",
                voice_profile_id=f"temp_profile_{file_id}",
                reference_audio_id=file_id,
                model_path=reference_audio.file_path,  # Use the reference audio file directly
                voice_characteristics={
                    "fundamental_frequency_range": {"min": 80, "max": 300, "mean": 150},
                    "formant_frequencies": [500, 1500, 2500, 3500],
                    "spectral_characteristics": {"centroid": 2000, "rolloff": 4000},
                    "prosody_parameters": {"speech_rate": 4.0, "pause_frequency": 10.0}
                },
                model_type="xtts_v2",
                quality_score=0.75,
                status=VoiceModelStatus.READY,
                created_at=datetime.now()
            )
        
        logger.error(f"No reference audio found for file_id: {file_id}")
        return None
    except Exception as e:
        logger.error(f"Error getting voice model: {str(e)}")
        return None


def get_reference_audio_path(file_id: str, db: Session) -> Optional[str]:
    """Get the file path for a reference audio file."""
    try:
        reference_audio = db.query(ReferenceAudio).filter(
            ReferenceAudio.id == file_id
        ).first()
        
        if reference_audio and os.path.exists(reference_audio.file_path):
            return reference_audio.file_path
        return None
    except Exception as e:
        logger.error(f"Error getting reference audio path: {str(e)}")
        return None


def run_enhanced_synthesis_task_sync(
    task_id: str,
    text: str,
    reference_audio_path: str,
    language: str,
    voice_settings: Optional[Dict[str, Any]] = None
):
    """Synchronous wrapper for the async synthesis task."""
    import asyncio
    
    try:
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async task
        loop.run_until_complete(
            run_enhanced_synthesis_task_async(
                task_id, text, reference_audio_path, language, voice_settings
            )
        )
    except Exception as e:
        logger.error(f"Synthesis task {task_id} failed: {str(e)}")
        if task_id in synthesis_tasks:
            synthesis_tasks[task_id]["status"] = str(e)
            synthesis_tasks[task_id]["stage"] = "failed"
            synthesis_tasks[task_id]["error"] = str(e)
            synthesis_tasks[task_id]["error_details"] = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "is_retryable": not isinstance(e, ValueError),
                "recovery_suggestions": _recovery_suggestions_for(e),
            }
    finally:
        # Clean up the event loop
        try:
            loop.close()
        except Exception:
            pass


def _recovery_suggestions_for(error: Exception) -> list:
    """Generate user-friendly recovery suggestions based on error type."""
    msg = str(error).lower()
    suggestions = []
    if "silent" in msg or "empty" in msg or "speech" in msg:
        suggestions.extend([
            "Upload a clip with clear spoken/sung vocals (not just instrumentals)",
            "Ensure the audio is at least 6 seconds of clear voice",
            "Try a different section of the video with louder vocals",
        ])
    elif "short" in msg or "minimum" in msg or "duration" in msg:
        suggestions.extend([
            "Upload a longer clip — at least 6 seconds of clear speech or singing",
            "Music videos with long instrumental sections may not have enough vocal content",
        ])
    else:
        suggestions.extend([
            "Check reference audio quality and format",
            "Ensure reference audio is at least 6 seconds long",
            "Try with shorter text segments",
            "Check system resources and try again later",
        ])
    return suggestions


async def run_enhanced_synthesis_task_async(
    task_id: str,
    text: str,
    reference_audio_path: str,
    language: str,
    voice_settings: Optional[Dict[str, Any]] = None
):
    """Run voice cloning synthesis using VoiceCloner (single XTTS v2 model)."""
    try:
        synthesis_tasks[task_id]["status"] = "processing"
        synthesis_tasks[task_id]["stage"] = "processing"
        synthesis_tasks[task_id]["progress"] = 10

        def progress_callback(progress: int, message: str):
            if task_id in synthesis_tasks:
                synthesis_tasks[task_id]["progress"] = progress
                synthesis_tasks[task_id]["status"] = message
                if progress > 50:
                    synthesis_tasks[task_id]["quality_metrics"] = {
                        "current_similarity": min(0.95, progress / 100.0 * 0.95),
                        "confidence_score": min(0.90, progress / 100.0 * 0.90),
                        "processing_stage": "synthesis" if progress < 90 else "post_processing"
                    }
                logger.info(f"Task {task_id}: {progress}% - {message}")

        output_path, metrics = await voice_cloner.clone_voice(
            text=text,
            reference_audio_path=reference_audio_path,
            language=language,
            progress_callback=progress_callback,
        )

        result = {
            "output_path": str(output_path),
            "processing_time": metrics.get("processing_time", 0),
            "quality_metrics": metrics,
            "similarity_score": metrics.get("similarity_score", 0.0),
            "confidence_score": metrics.get("similarity_score", 0.0),
            "synthesis_method": "xtts_v2_voice_cloner",
            "recommendations": []
        }

        synthesis_tasks[task_id]["status"] = "completed"
        synthesis_tasks[task_id]["stage"] = "completed"
        synthesis_tasks[task_id]["progress"] = 100
        synthesis_tasks[task_id]["result"] = result
        synthesis_tasks[task_id]["completed_at"] = datetime.now()
        synthesis_tasks[task_id]["final_quality_metrics"] = {
            "overall_similarity": metrics.get("similarity_score", 0.0),
            "quality_score": metrics.get("similarity_score", 0.0),
            "confidence_score": metrics.get("similarity_score", 0.0),
            "quality_level": metrics.get("quality_level", "unknown"),
            "ecapa_similarity": metrics.get("ecapa_similarity"),
            "resemblyzer_similarity": metrics.get("resemblyzer_similarity"),
            "processing_time": metrics.get("processing_time", 0),
            "synthesis_method": "xtts_v2_voice_cloner",
            "recommendations": []
        }

        logger.info(f"Voice cloning completed for task {task_id} — similarity={metrics.get('similarity_score')}")

    except Exception as e:
        logger.error(f"Voice cloning failed for task {task_id}: {str(e)}")

        from app.core.error_handling import error_recovery_manager, ErrorCategory
        error_info = error_recovery_manager.handle_error(
            e, ErrorCategory.SYNTHESIS,
            {"task_id": task_id, "text_length": len(text), "language": language}
        )

        if task_id in synthesis_tasks:
            synthesis_tasks[task_id]["status"] = f"Voice cloning failed: {str(e)}"
            synthesis_tasks[task_id]["stage"] = "failed"
            synthesis_tasks[task_id]["error"] = str(e)
            synthesis_tasks[task_id]["error_details"] = {
                "error_id": error_info.error_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "is_retryable": error_info.is_retryable,
                "recovery_suggestions": [
                    "Check reference audio quality and format",
                    "Ensure reference audio is at least 3 seconds long",
                    "Try with shorter text segments",
                    "Check system resources and try again later"
                ]
            }


@router.post("/synthesize", response_model=SynthesisResponse)
async def create_synthesis_task(
    request: SynthesisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a new speech synthesis task using real voice cloning.
    
    Args:
        request: Synthesis request parameters
        background_tasks: FastAPI background tasks
        db: Database session
    
    Returns:
        SynthesisResponse with task ID and status
    """
    try:
        # Validate voice model exists and is ready
        voice_model = get_voice_model_by_id(request.voice_model_id, db)
        if not voice_model:
            raise HTTPException(
                status_code=404,
                detail=f"Voice model not found: {request.voice_model_id}"
            )
        
        # Check reference audio file path
        reference_audio_path = get_reference_audio_path(voice_model.reference_audio_id, db)
        if not reference_audio_path:
            raise HTTPException(
                status_code=404,
                detail=f"Reference audio file not found for voice model: {request.voice_model_id}"
            )
        
        # Generate unique synthesis ID
        synthesis_id = f"synthesis_{uuid.uuid4().hex[:12]}"
        
        # Create task record with enhanced tracking
        synthesis_tasks[synthesis_id] = {
            'task_id': synthesis_id,
            'status': 'queued',
            'stage': 'queued',
            'progress': 0,
            'created_at': datetime.now(),
            'text': request.text,
            'voice_model_id': request.voice_model_id,
            'language': request.language,
            'reference_audio_path': reference_audio_path,
            'quality_metrics': {},
            'recommendations': []
        }
        
        # Start enhanced synthesis in background
        background_tasks.add_task(
            run_enhanced_synthesis_task_sync,
            synthesis_id,
            request.text,
            reference_audio_path,
            request.language,
            request.voice_settings.model_dump() if request.voice_settings else None
        )
        
        # Estimate completion time based on text length and model complexity
        estimated_seconds = max(30, len(request.text) * 0.5)  # Real synthesis takes longer
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        
        logger.info(f"Enhanced voice cloning task created: {synthesis_id}")
        
        return SynthesisResponse(
            task_id=synthesis_id,
            status=SynthesisStatus.PENDING,
            message="Enhanced voice cloning task created successfully",
            estimated_completion=estimated_completion,
            queue_position=1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create real synthesis task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create synthesis task: {str(e)}"
        )


@router.post("/synthesize/cross-language", response_model=SynthesisResponse)
async def create_cross_language_synthesis_task(
    request: CrossLanguageSynthesisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a cross-language speech synthesis task.
    
    Args:
        request: Cross-language synthesis request
        background_tasks: FastAPI background tasks
    
    Returns:
        SynthesisResponse with task ID and status
    """
    try:
        # Validate voice model
        voice_model = get_voice_model_by_id(request.source_voice_model_id, db)
        if not voice_model:
            raise HTTPException(
                status_code=404,
                detail=f"Voice model not found: {request.source_voice_model_id}"
            )
        
        if voice_model.status != VoiceModelStatus.READY:
            raise HTTPException(
                status_code=400,
                detail=f"Voice model not ready for synthesis: {voice_model.status}"
            )
        
        # Generate unique synthesis ID
        synthesis_id = f"cross_lang_{uuid.uuid4().hex[:12]}"
        
        # Convert voice model to dict
        voice_model_data = voice_model.model_dump()
        
        # Create Celery task
        task = cross_language_synthesis_task.apply_async(
            args=[
                request.text,
                voice_model_data,
                request.target_language,
                synthesis_id
            ],
            task_id=synthesis_id
        )
        
        # Estimate completion time (cross-language takes longer)
        estimated_seconds = max(15, len(request.text) * 0.15)
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        
        return SynthesisResponse(
            task_id=task.id,
            status=SynthesisStatus.PENDING,
            message="Cross-language synthesis task created successfully",
            estimated_completion=estimated_completion,
            queue_position=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create cross-language synthesis task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create cross-language synthesis task: {str(e)}"
        )


@router.post("/synthesize/batch", response_model=SynthesisResponse)
async def create_batch_synthesis_task(
    request: BatchSynthesisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a batch speech synthesis task.
    
    Args:
        request: Batch synthesis request
        background_tasks: FastAPI background tasks
    
    Returns:
        SynthesisResponse with batch task ID
    """
    try:
        # Validate all voice models
        for synthesis_request in request.requests:
            voice_model = get_voice_model_by_id(synthesis_request.voice_model_id, db)
            if not voice_model:
                raise HTTPException(
                    status_code=404,
                    detail=f"Voice model not found: {synthesis_request.voice_model_id}"
                )
        
        # Generate unique batch ID
        batch_id = f"batch_{uuid.uuid4().hex[:12]}"
        
        # Prepare synthesis requests for Celery
        synthesis_requests = []
        for req in request.requests:
            voice_model = get_voice_model_by_id(req.voice_model_id, db)
            synthesis_requests.append({
                "text": req.text,
                "voice_model_data": voice_model.model_dump(),
                "language": req.language,
                "voice_settings": req.voice_settings.model_dump() if req.voice_settings else None
            })
        
        # Create Celery task
        task = batch_synthesis_task.apply_async(
            args=[synthesis_requests, batch_id],
            task_id=batch_id,
            priority=request.priority
        )
        
        # Estimate completion time for batch
        total_text_length = sum(len(req.text) for req in request.requests)
        estimated_seconds = max(20, total_text_length * 0.1)
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        
        return SynthesisResponse(
            task_id=task.id,
            status=SynthesisStatus.PENDING,
            message=f"Batch synthesis task created with {len(request.requests)} requests",
            estimated_completion=estimated_completion,
            queue_position=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create batch synthesis task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create batch synthesis task: {str(e)}"
        )


@router.get("/status/{task_id}", response_model=SynthesisProgress)
async def get_synthesis_status(task_id: str):
    """
    Get the status of a real synthesis task.
    
    Args:
        task_id: Synthesis task ID
    
    Returns:
        SynthesisProgress with current status
    """
    try:
        # Check if task exists
        if task_id not in synthesis_tasks:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {task_id}"
            )
        
        task_state = synthesis_tasks[task_id]
        
        return SynthesisProgress(
            task_id=task_id,
            progress=task_state.get('progress', 0),
            status=task_state.get('status', 'Processing'),
            stage=task_state.get('stage', 'processing'),
            estimated_remaining=task_state.get('estimated_remaining'),
            quality_metrics=task_state.get('quality_metrics', {}),
            recommendations=task_state.get('recommendations', [])
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get synthesis status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get synthesis status: {str(e)}"
        )


@router.get("/result/{task_id}", response_model=SynthesisResult)
async def get_synthesis_result(task_id: str):
    """
    Get the result of a completed real synthesis task.
    
    Args:
        task_id: Synthesis task ID
    
    Returns:
        SynthesisResult with output information
    """
    try:
        # Check if task exists
        if task_id not in synthesis_tasks:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {task_id}"
            )
        
        task_state = synthesis_tasks[task_id]
        
        if task_state['stage'] in ['queued', 'processing']:
            raise HTTPException(
                status_code=202,
                detail="Synthesis task is still in progress"
            )
        elif task_state['stage'] == 'completed':
            result = task_state.get('result', {})
            final_quality_metrics = task_state.get('final_quality_metrics', {})
            
            return SynthesisResult(
                task_id=task_id,
                status=SynthesisStatus.COMPLETED,
                output_url=f"/api/v1/synthesis/download/{task_id}",
                output_path=result.get('output_path'),
                metadata={
                    **result,
                    "quality_metrics": final_quality_metrics,
                    "recommendations": final_quality_metrics.get('recommendations', [])
                },
                processing_time=result.get('processing_time'),
                created_at=task_state.get('created_at', datetime.now()),
                completed_at=task_state.get('completed_at', datetime.now())
            )
        else:
            # Failed
            error_details = task_state.get('error_details', {})
            return SynthesisResult(
                task_id=task_id,
                status=SynthesisStatus.FAILED,
                error_message=task_state.get('error', 'Synthesis failed'),
                metadata={
                    "error_details": error_details,
                    "recovery_suggestions": error_details.get('recovery_suggestions', [])
                },
                created_at=task_state.get('created_at', datetime.now()),
                completed_at=datetime.now()
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get synthesis result: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get synthesis result: {str(e)}"
        )


@router.get("/download/{task_id}")
async def download_synthesized_audio(
    task_id: str,
    format: str = Query(default="wav", regex="^(wav|mp3|flac)$", description="Output audio format")
):
    """
    Download real synthesized audio file.
    
    Args:
        task_id: Synthesis task ID
        format: Output format (wav, mp3, flac)
    
    Returns:
        FileResponse with real synthesized audio file
    """
    try:
        # Check if task exists and is completed
        if task_id not in synthesis_tasks:
            raise HTTPException(
                status_code=404,
                detail="Synthesis task not found"
            )
        
        task_state = synthesis_tasks[task_id]
        
        if task_state['stage'] != 'completed':
            raise HTTPException(
                status_code=404,
                detail="Synthesis not completed or failed"
            )
        
        # Get the real output file path
        result = task_state.get('result', {})
        output_path = result.get('output_path')
        
        if not output_path or not os.path.exists(output_path):
            raise HTTPException(
                status_code=404,
                detail="Synthesized audio file not found"
            )
        
        # Generate download filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthesized_{timestamp}.{format}"
        
        # Determine media type
        media_types = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg", 
            "flac": "audio/flac"
        }
        media_type = media_types.get(format, "audio/wav")
        
        # Return the real synthesized audio file
        return FileResponse(
            path=output_path,
            media_type=media_type,
            filename=filename,
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download synthesized audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download audio: {str(e)}"
        )


@router.post("/optimize/{voice_model_id}")
async def optimize_voice_model(
    voice_model_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Optimize a voice model for faster synthesis.
    
    Args:
        voice_model_id: Voice model ID to optimize
        background_tasks: FastAPI background tasks
    
    Returns:
        Task information
    """
    try:
        # Validate voice model exists
        voice_model = get_voice_model_by_id(voice_model_id, db)
        if not voice_model:
            raise HTTPException(
                status_code=404,
                detail=f"Voice model not found: {voice_model_id}"
            )
        
        # Create optimization task
        task = optimize_synthesis_model.apply_async(
            args=[voice_model_id, {}]
        )
        
        return {
            "task_id": task.id,
            "voice_model_id": voice_model_id,
            "status": "Optimization task created",
            "message": "Voice model optimization started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create optimization task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create optimization task: {str(e)}"
        )


@router.get("/quality/{task_id}")
async def get_quality_metrics(task_id: str):
    """
    Get detailed quality metrics for a synthesis task.
    
    Args:
        task_id: Synthesis task ID
    
    Returns:
        Detailed quality metrics and recommendations
    """
    try:
        # Check if task exists
        if task_id not in synthesis_tasks:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {task_id}"
            )
        
        task_state = synthesis_tasks[task_id]
        
        # Get quality metrics from different stages
        current_metrics = task_state.get('quality_metrics', {})
        final_metrics = task_state.get('final_quality_metrics', {})
        
        # Combine metrics
        quality_data = {
            "task_id": task_id,
            "status": task_state.get('stage', 'unknown'),
            "current_metrics": current_metrics,
            "final_metrics": final_metrics,
            "recommendations": final_metrics.get('recommendations', []),
            "similarity_breakdown": final_metrics.get('similarity_breakdown', {}),
            "confidence_scores": {
                "overall": final_metrics.get('confidence_score', 0.0),
                "pitch": final_metrics.get('pitch_confidence', 0.0),
                "timbre": final_metrics.get('timbre_confidence', 0.0),
                "prosody": final_metrics.get('prosody_confidence', 0.0)
            },
            "processing_info": {
                "synthesis_method": final_metrics.get('synthesis_method', 'unknown'),
                "processing_time": final_metrics.get('processing_time', 0.0),
                "models_used": final_metrics.get('models_used', [])
            }
        }
        
        return quality_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get quality metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get quality metrics: {str(e)}"
        )


@router.post("/feedback/{task_id}")
async def submit_quality_feedback(
    task_id: str,
    feedback: Dict[str, Any]
):
    """
    Submit quality feedback for a synthesis task.
    
    Args:
        task_id: Synthesis task ID
        feedback: User feedback on quality
    
    Returns:
        Feedback acknowledgment and improvement suggestions
    """
    try:
        # Check if task exists
        if task_id not in synthesis_tasks:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {task_id}"
            )
        
        task_state = synthesis_tasks[task_id]
        
        # Store feedback
        if 'feedback' not in task_state:
            task_state['feedback'] = []
        
        feedback_entry = {
            "timestamp": datetime.now(),
            "user_rating": feedback.get('rating', 0),
            "quality_issues": feedback.get('issues', []),
            "user_comments": feedback.get('comments', ''),
            "similarity_rating": feedback.get('similarity_rating', 0)
        }
        
        task_state['feedback'].append(feedback_entry)
        
        # Generate improvement suggestions based on feedback
        suggestions = []
        
        if feedback.get('rating', 5) < 3:
            suggestions.extend([
                "Consider using higher quality reference audio (>10 seconds, clear speech)",
                "Ensure reference audio has minimal background noise",
                "Try breaking long text into shorter segments"
            ])
        
        if feedback.get('similarity_rating', 5) < 3:
            suggestions.extend([
                "Use reference audio with more emotional variation",
                "Ensure reference audio matches target language",
                "Consider providing multiple reference samples"
            ])
        
        issues = feedback.get('issues', [])
        if 'robotic_sound' in issues:
            suggestions.append("Try using ensemble synthesis for more natural results")
        if 'wrong_pitch' in issues:
            suggestions.append("Adjust pitch settings or use reference audio with similar pitch")
        if 'poor_pronunciation' in issues:
            suggestions.append("Ensure text is properly formatted and uses correct language")
        
        return {
            "task_id": task_id,
            "feedback_received": True,
            "improvement_suggestions": suggestions,
            "next_steps": [
                "Review quality metrics for specific areas of improvement",
                "Consider re-running synthesis with suggested optimizations",
                "Contact support if issues persist"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}"
        )


@router.get("/stats", response_model=SynthesisStats)
async def get_synthesis_statistics():
    """
    Get synthesis statistics and metrics.
    
    Returns:
        SynthesisStats with system statistics
    """
    try:
        # This would query actual database for statistics
        # For now, return mock data
        return SynthesisStats(
            total_syntheses=1250,
            successful_syntheses=1198,
            failed_syntheses=52,
            average_processing_time=8.7,
            average_quality_score=0.89,
            total_audio_duration=3600.5,
            languages_supported=["english", "spanish", "french", "german", "italian", "portuguese"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get synthesis statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.delete("/task/{task_id}")
async def cancel_synthesis_task(task_id: str):
    """
    Cancel a pending or in-progress real synthesis task.
    
    Args:
        task_id: Synthesis task ID to cancel
    
    Returns:
        Cancellation confirmation
    """
    try:
        # Check if task exists
        if task_id not in synthesis_tasks:
            raise HTTPException(
                status_code=404,
                detail=f"Task not found: {task_id}"
            )
        
        task_state = synthesis_tasks[task_id]
        
        if task_state['stage'] in ['completed', 'failed']:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel completed task (stage: {task_state['stage']})"
            )
        
        # Mark task as cancelled
        task_state['stage'] = 'cancelled'
        task_state['status'] = 'Task cancelled by user'
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Real synthesis task cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel synthesis task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {str(e)}"
        )