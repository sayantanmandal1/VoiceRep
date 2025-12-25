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
from app.services.real_voice_synthesis_service import advanced_voice_cloning_service
from app.models.file import ReferenceAudio
from app.core.database import get_db
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


async def run_real_synthesis_task(
    task_id: str,
    text: str,
    reference_audio_path: str,
    language: str,
    voice_settings: Optional[Dict[str, Any]] = None
):
    """Run real voice synthesis in background."""
    try:
        # Update task status
        synthesis_tasks[task_id]["status"] = "processing"
        synthesis_tasks[task_id]["stage"] = "processing"
        synthesis_tasks[task_id]["progress"] = 10
        
        # Create output directory
        output_dir = Path(settings.RESULTS_DIR) if hasattr(settings, 'RESULTS_DIR') else Path("results")
        output_dir.mkdir(exist_ok=True)
        
        # Generate output path
        output_filename = f"synthesis_{task_id}.wav"
        output_path = output_dir / output_filename
        
        # Progress callback
        def progress_callback(progress: int, message: str):
            if task_id in synthesis_tasks:
                synthesis_tasks[task_id]["progress"] = progress
                synthesis_tasks[task_id]["status"] = message
                logger.info(f"Task {task_id}: {progress}% - {message}")
        
        # Perform advanced voice cloning synthesis
        result = await advanced_voice_cloning_service.synthesize_speech(
            text=text,
            reference_audio_path=reference_audio_path,
            output_path=str(output_path),
            language=language,
            progress_callback=progress_callback
        )
        
        # Update task as completed
        synthesis_tasks[task_id]["status"] = "completed"
        synthesis_tasks[task_id]["stage"] = "completed"
        synthesis_tasks[task_id]["progress"] = 100
        synthesis_tasks[task_id]["result"] = result
        synthesis_tasks[task_id]["completed_at"] = datetime.now()
        
        logger.info(f"Advanced voice cloning completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"Advanced voice cloning failed for task {task_id}: {str(e)}")
        if task_id in synthesis_tasks:
            synthesis_tasks[task_id]["status"] = f"Voice cloning failed: {str(e)}"
            synthesis_tasks[task_id]["stage"] = "failed"
            synthesis_tasks[task_id]["error"] = str(e)


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
        
        # Get reference audio file path
        reference_audio_path = get_reference_audio_path(voice_model.reference_audio_id, db)
        if not reference_audio_path:
            raise HTTPException(
                status_code=404,
                detail=f"Reference audio file not found for voice model: {request.voice_model_id}"
            )
        
        # Check if advanced voice cloning service is ready
        if not advanced_voice_cloning_service.is_model_ready():
            # Try to initialize the service
            logger.info("Advanced voice cloning model not ready, initializing...")
            success = await advanced_voice_cloning_service.initialize_model()
            if not success:
                raise HTTPException(
                    status_code=503,
                    detail="Advanced voice cloning service is not available. Please try again later."
                )
        
        # Generate unique synthesis ID
        synthesis_id = f"synthesis_{uuid.uuid4().hex[:12]}"
        
        # Create task record
        synthesis_tasks[synthesis_id] = {
            'task_id': synthesis_id,
            'status': 'queued',
            'stage': 'queued',
            'progress': 0,
            'created_at': datetime.now(),
            'text': request.text,
            'voice_model_id': request.voice_model_id,
            'language': request.language,
            'reference_audio_path': reference_audio_path
        }
        
        # Start real synthesis in background
        background_tasks.add_task(
            run_real_synthesis_task,
            synthesis_id,
            request.text,
            reference_audio_path,
            request.language,
            request.voice_settings.model_dump() if request.voice_settings else None
        )
        
        # Estimate completion time based on text length and model complexity
        estimated_seconds = max(30, len(request.text) * 0.5)  # Real synthesis takes longer
        estimated_completion = datetime.now() + timedelta(seconds=estimated_seconds)
        
        logger.info(f"Advanced voice cloning task created: {synthesis_id}")
        
        return SynthesisResponse(
            task_id=synthesis_id,
            status=SynthesisStatus.PENDING,
            message="Advanced voice cloning task created successfully",
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
            estimated_remaining=task_state.get('estimated_remaining')
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
            
            return SynthesisResult(
                task_id=task_id,
                status=SynthesisStatus.COMPLETED,
                output_url=f"/api/v1/synthesis/download/{task_id}",
                output_path=result.get('output_path'),
                metadata=result,
                processing_time=result.get('processing_time'),
                created_at=task_state.get('created_at', datetime.now()),
                completed_at=task_state.get('completed_at', datetime.now())
            )
        else:
            # Failed
            return SynthesisResult(
                task_id=task_id,
                status=SynthesisStatus.FAILED,
                error_message=task_state.get('error', 'Synthesis failed'),
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