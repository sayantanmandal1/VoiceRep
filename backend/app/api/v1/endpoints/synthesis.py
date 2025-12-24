"""
API endpoints for speech synthesis operations.
"""

import os
import uuid
import logging
from typing import Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import FileResponse
from celery.result import AsyncResult

from app.schemas.synthesis import (
    SynthesisRequest, SynthesisResponse, SynthesisResult, SynthesisProgress,
    CrossLanguageSynthesisRequest, BatchSynthesisRequest, BatchSynthesisResult,
    SynthesisStatus, SynthesisStats
)
from app.schemas.voice import VoiceModelSchema
from app.tasks.synthesis_tasks import (
    synthesize_speech_task, cross_language_synthesis_task, 
    batch_synthesis_task, optimize_synthesis_model
)
from app.core.config import settings
from app.models.voice import VoiceModel, VoiceModelStatus

logger = logging.getLogger(__name__)

router = APIRouter()


# Mock database functions (replace with actual database operations)
def get_voice_model_by_id(voice_model_id: str) -> Optional[VoiceModelSchema]:
    """Get voice model from database by ID."""
    # This is a placeholder - in production, query actual database
    # For now, return a mock voice model
    return VoiceModelSchema(
        id=voice_model_id,
        voice_profile_id="profile_123",
        reference_audio_id="audio_123",
        model_path=f"/models/{voice_model_id}.pt",
        voice_characteristics={
            "fundamental_frequency_range": {"min": 80, "max": 300, "mean": 150},
            "formant_frequencies": [500, 1500, 2500, 3500],
            "spectral_characteristics": {"centroid": 2000, "rolloff": 4000},
            "prosody_parameters": {"speech_rate": 4.0, "pause_frequency": 10.0}
        },
        model_type="tortoise_tts",
        quality_score=0.85,
        status=VoiceModelStatus.READY,
        created_at=datetime.now()
    )


@router.post("/synthesize", response_model=SynthesisResponse)
async def create_synthesis_task(
    request: SynthesisRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new speech synthesis task.
    
    Args:
        request: Synthesis request parameters
        background_tasks: FastAPI background tasks
    
    Returns:
        SynthesisResponse with task ID and status
    """
    try:
        # Validate voice model exists and is ready
        voice_model = get_voice_model_by_id(request.voice_model_id)
        if not voice_model:
            raise HTTPException(
                status_code=404,
                detail=f"Voice model not found: {request.voice_model_id}"
            )
        
        if voice_model.status != VoiceModelStatus.READY:
            raise HTTPException(
                status_code=400,
                detail=f"Voice model not ready for synthesis: {voice_model.status}"
            )
        
        # Generate unique synthesis ID
        synthesis_id = f"synthesis_{uuid.uuid4().hex[:12]}"
        
        # Convert voice model to dict for Celery task
        voice_model_data = voice_model.model_dump()
        
        # Create Celery task
        task = synthesize_speech_task.apply_async(
            args=[
                request.text,
                voice_model_data,
                request.language,
                request.voice_settings.model_dump() if request.voice_settings else None,
                synthesis_id
            ],
            task_id=synthesis_id
        )
        
        # Estimate completion time (rough estimate based on text length)
        estimated_seconds = max(10, len(request.text) * 0.1)
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        
        return SynthesisResponse(
            task_id=task.id,
            status=SynthesisStatus.PENDING,
            message="Synthesis task created successfully",
            estimated_completion=estimated_completion,
            queue_position=None  # Would need queue inspection for actual position
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create synthesis task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create synthesis task: {str(e)}"
        )


@router.post("/synthesize/cross-language", response_model=SynthesisResponse)
async def create_cross_language_synthesis_task(
    request: CrossLanguageSynthesisRequest,
    background_tasks: BackgroundTasks
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
        voice_model = get_voice_model_by_id(request.source_voice_model_id)
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
    background_tasks: BackgroundTasks
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
            voice_model = get_voice_model_by_id(synthesis_request.voice_model_id)
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
            voice_model = get_voice_model_by_id(req.voice_model_id)
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
    Get the status of a synthesis task.
    
    Args:
        task_id: Synthesis task ID
    
    Returns:
        SynthesisProgress with current status
    """
    try:
        # Get task result from Celery
        task_result = AsyncResult(task_id)
        
        if task_result.state == 'PENDING':
            return SynthesisProgress(
                task_id=task_id,
                progress=0,
                status="Task is pending in queue",
                stage="queued"
            )
        elif task_result.state == 'PROGRESS':
            info = task_result.info or {}
            return SynthesisProgress(
                task_id=task_id,
                progress=info.get('progress', 0),
                status=info.get('status', 'Processing'),
                stage=info.get('stage', 'processing')
            )
        elif task_result.state == 'SUCCESS':
            return SynthesisProgress(
                task_id=task_id,
                progress=100,
                status="Synthesis completed successfully",
                stage="completed"
            )
        elif task_result.state == 'FAILURE':
            info = task_result.info or {}
            return SynthesisProgress(
                task_id=task_id,
                progress=0,
                status=f"Synthesis failed: {info.get('error', 'Unknown error')}",
                stage="failed"
            )
        else:
            return SynthesisProgress(
                task_id=task_id,
                progress=0,
                status=f"Unknown task state: {task_result.state}",
                stage="unknown"
            )
            
    except Exception as e:
        logger.error(f"Failed to get synthesis status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get synthesis status: {str(e)}"
        )


@router.get("/result/{task_id}", response_model=SynthesisResult)
async def get_synthesis_result(task_id: str):
    """
    Get the result of a completed synthesis task.
    
    Args:
        task_id: Synthesis task ID
    
    Returns:
        SynthesisResult with output information
    """
    try:
        # Get task result from Celery
        task_result = AsyncResult(task_id)
        
        if task_result.state == 'PENDING':
            raise HTTPException(
                status_code=202,
                detail="Synthesis task is still pending"
            )
        elif task_result.state == 'PROGRESS':
            raise HTTPException(
                status_code=202,
                detail="Synthesis task is still in progress"
            )
        elif task_result.state == 'SUCCESS':
            result = task_result.result
            
            return SynthesisResult(
                task_id=task_id,
                status=SynthesisStatus.COMPLETED,
                output_url=f"/api/v1/synthesis/download/{task_id}",
                output_path=result.get('output_path'),
                metadata=result.get('metadata'),
                processing_time=result.get('metadata', {}).get('processing_time'),
                created_at=datetime.now(),  # Would be from database
                completed_at=datetime.now()
            )
        elif task_result.state == 'FAILURE':
            info = task_result.info or {}
            return SynthesisResult(
                task_id=task_id,
                status=SynthesisStatus.FAILED,
                error_message=info.get('error', 'Unknown error'),
                created_at=datetime.now(),
                completed_at=datetime.now()
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Unknown task state: {task_result.state}"
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
    Download synthesized audio file with optional format conversion.
    
    Args:
        task_id: Synthesis task ID
        format: Output format (wav, mp3, flac)
    
    Returns:
        FileResponse with audio file
    """
    try:
        # Get task result
        task_result = AsyncResult(task_id)
        
        if task_result.state != 'SUCCESS':
            raise HTTPException(
                status_code=404,
                detail="Synthesis not completed or failed"
            )
        
        result = task_result.result
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
        
        # For now, return the original file (format conversion would be implemented here)
        # In a full implementation, you would use ffmpeg or similar to convert formats
        return FileResponse(
            path=output_path,
            media_type=media_type,
            filename=filename,
            headers={
                "Accept-Ranges": "bytes",  # Enable streaming support
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
    background_tasks: BackgroundTasks
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
        voice_model = get_voice_model_by_id(voice_model_id)
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
    Cancel a pending or in-progress synthesis task.
    
    Args:
        task_id: Synthesis task ID to cancel
    
    Returns:
        Cancellation confirmation
    """
    try:
        # Get task result
        task_result = AsyncResult(task_id)
        
        if task_result.state in ['SUCCESS', 'FAILURE']:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel completed task (state: {task_result.state})"
            )
        
        # Revoke the task
        task_result.revoke(terminate=True)
        
        return {
            "task_id": task_id,
            "status": "cancelled",
            "message": "Synthesis task cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel synthesis task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel task: {str(e)}"
        )