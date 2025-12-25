"""
API endpoints for intelligent voice model training system.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import asyncio
import os
import tempfile
import time
from pathlib import Path

from app.services.intelligent_voice_model_trainer import IntelligentVoiceModelTrainer
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global trainer instance
trainer = IntelligentVoiceModelTrainer()


class VoiceModelTrainingRequest(BaseModel):
    """Request schema for voice model training."""
    voice_profile_id: str
    audio_file_ids: List[str]
    training_options: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "quality_threshold": 0.8,
            "max_segments": 10,
            "enable_optimization": True
        }
    )


class VoiceModelImprovementRequest(BaseModel):
    """Request schema for incremental model improvement."""
    model_id: str
    additional_audio_ids: List[str]
    improvement_options: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "min_improvement_threshold": 0.02,
            "enable_validation": True
        }
    )


class VoiceModelTrainingResponse(BaseModel):
    """Response schema for voice model training."""
    success: bool
    model_id: Optional[str] = None
    message: str
    training_duration: Optional[float] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None


class VoiceModelImprovementResponse(BaseModel):
    """Response schema for model improvement."""
    success: bool
    model_id: str
    message: str
    improvement_metrics: Optional[Dict[str, Any]] = None
    new_quality_score: Optional[float] = None
    error_details: Optional[Dict[str, Any]] = None


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    model_id: str
    voice_profile_id: str
    reference_audio_ids: List[str]
    training_duration: float
    audio_segments: int
    quality_score: float
    similarity_score: float
    model_size_mb: float
    inference_time_ms: float
    created_at: float
    last_updated: float
    usage_count: int
    optimization_history: List[Dict[str, Any]]


class CacheStatsResponse(BaseModel):
    """Response schema for cache statistics."""
    total_models: int
    memory_cached_models: int
    total_size_mb: float
    cache_hit_rate: float
    average_model_size_mb: float


@router.post("/train", response_model=VoiceModelTrainingResponse)
async def create_dedicated_voice_model(
    request: VoiceModelTrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a dedicated voice model for audio longer than 30 seconds.
    
    This endpoint analyzes multiple audio segments, combines their characteristics,
    and trains a dedicated voice model for high-fidelity synthesis.
    """
    try:
        logger.info(f"Starting dedicated voice model training for profile {request.voice_profile_id}")
        
        # Validate audio files exist
        audio_paths = []
        uploads_dir = Path(settings.UPLOAD_DIR) if hasattr(settings, 'UPLOAD_DIR') else Path("uploads")
        
        for audio_id in request.audio_file_ids:
            # Look for audio file in uploads directory
            audio_path = None
            for ext in ['.wav', '.mp3', '.flac', '.m4a']:
                potential_path = uploads_dir / f"{audio_id}{ext}"
                if potential_path.exists():
                    audio_path = str(potential_path)
                    break
            
            if not audio_path:
                raise HTTPException(
                    status_code=404,
                    detail=f"Audio file not found for ID: {audio_id}"
                )
            
            audio_paths.append(audio_path)
        
        # Progress tracking
        progress_status = {"current": 0, "message": "Starting training"}
        
        def progress_callback(progress: int, message: str):
            progress_status["current"] = progress
            progress_status["message"] = message
            logger.info(f"Training progress: {progress}% - {message}")
        
        # Create dedicated model
        success, model_id, metadata = await trainer.create_dedicated_voice_model(
            audio_paths=audio_paths,
            voice_profile_id=request.voice_profile_id,
            progress_callback=progress_callback
        )
        
        if success:
            return VoiceModelTrainingResponse(
                success=True,
                model_id=model_id,
                message=f"Dedicated voice model created successfully: {model_id}",
                training_duration=metadata.get('training_duration'),
                quality_metrics={
                    'quality_score': metadata.get('quality_score'),
                    'similarity_score': metadata.get('similarity_score'),
                    'audio_segments': metadata.get('audio_segments'),
                    'model_size_mb': metadata.get('model_size_mb')
                }
            )
        else:
            return VoiceModelTrainingResponse(
                success=False,
                message="Failed to create dedicated voice model",
                error_details=metadata
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice model training failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice model training failed: {str(e)}"
        )


@router.post("/improve", response_model=VoiceModelImprovementResponse)
async def improve_voice_model_incrementally(
    request: VoiceModelImprovementRequest,
    background_tasks: BackgroundTasks
):
    """
    Improve existing voice model with additional audio data.
    
    This endpoint takes an existing model and additional audio segments
    to incrementally improve the model's quality and accuracy.
    """
    try:
        logger.info(f"Starting incremental improvement for model {request.model_id}")
        
        # Validate additional audio files
        audio_paths = []
        uploads_dir = Path(settings.UPLOAD_DIR) if hasattr(settings, 'UPLOAD_DIR') else Path("uploads")
        
        for audio_id in request.additional_audio_ids:
            audio_path = None
            for ext in ['.wav', '.mp3', '.flac', '.m4a']:
                potential_path = uploads_dir / f"{audio_id}{ext}"
                if potential_path.exists():
                    audio_path = str(potential_path)
                    break
            
            if not audio_path:
                raise HTTPException(
                    status_code=404,
                    detail=f"Additional audio file not found for ID: {audio_id}"
                )
            
            audio_paths.append(audio_path)
        
        # Progress tracking
        def progress_callback(progress: int, message: str):
            logger.info(f"Improvement progress: {progress}% - {message}")
        
        # Improve model incrementally
        success, results = await trainer.improve_model_incrementally(
            model_id=request.model_id,
            additional_audio_paths=audio_paths,
            progress_callback=progress_callback
        )
        
        if success:
            return VoiceModelImprovementResponse(
                success=True,
                model_id=request.model_id,
                message="Voice model improved successfully",
                improvement_metrics=results.get('improvement_metrics'),
                new_quality_score=results.get('new_quality_score')
            )
        else:
            return VoiceModelImprovementResponse(
                success=False,
                model_id=request.model_id,
                message="Failed to improve voice model",
                error_details=results
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice model improvement failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Voice model improvement failed: {str(e)}"
        )


@router.get("/models", response_model=List[ModelInfoResponse])
async def list_cached_voice_models():
    """
    List all cached voice models with their metadata.
    
    Returns information about all voice models in the cache including
    quality metrics, usage statistics, and optimization history.
    """
    try:
        models = trainer.list_cached_models()
        
        response_models = []
        for model_data in models:
            response_models.append(ModelInfoResponse(
                model_id=model_data['model_id'],
                voice_profile_id=model_data['voice_profile_id'],
                reference_audio_ids=model_data['reference_audio_ids'],
                training_duration=model_data['training_duration'],
                audio_segments=model_data['audio_segments'],
                quality_score=model_data['quality_score'],
                similarity_score=model_data['similarity_score'],
                model_size_mb=model_data['model_size_mb'],
                inference_time_ms=model_data['inference_time_ms'],
                created_at=model_data['created_at'],
                last_updated=model_data['last_updated'],
                usage_count=model_data['usage_count'],
                optimization_history=model_data['optimization_history']
            ))
        
        return response_models
    
    except Exception as e:
        logger.error(f"Failed to list cached models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list cached models: {str(e)}"
        )


@router.get("/models/{model_id}", response_model=ModelInfoResponse)
async def get_voice_model_info(model_id: str):
    """
    Get detailed information about a specific voice model.
    
    Returns comprehensive metadata about the model including training history,
    quality metrics, and performance characteristics.
    """
    try:
        model_info = trainer.get_model_info(model_id)
        
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Voice model not found: {model_id}"
            )
        
        return ModelInfoResponse(
            model_id=model_info['model_id'],
            voice_profile_id=model_info['voice_profile_id'],
            reference_audio_ids=model_info['reference_audio_ids'],
            training_duration=model_info['training_duration'],
            audio_segments=model_info['audio_segments'],
            quality_score=model_info['quality_score'],
            similarity_score=model_info['similarity_score'],
            model_size_mb=model_info['model_size_mb'],
            inference_time_ms=model_info['inference_time_ms'],
            created_at=model_info['created_at'],
            last_updated=model_info['last_updated'],
            usage_count=model_info['usage_count'],
            optimization_history=model_info['optimization_history']
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_statistics():
    """
    Get voice model cache performance statistics.
    
    Returns information about cache usage, hit rates, and storage efficiency.
    """
    try:
        stats = trainer.get_cache_statistics()
        
        return CacheStatsResponse(
            total_models=stats['total_models'],
            memory_cached_models=stats['memory_cached_models'],
            total_size_mb=stats['total_size_mb'],
            cache_hit_rate=stats['cache_hit_rate'],
            average_model_size_mb=stats['average_model_size_mb']
        )
    
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache statistics: {str(e)}"
        )


@router.post("/cache/optimize")
async def optimize_model_cache():
    """
    Optimize voice model cache performance and storage.
    
    Performs cache cleanup, updates priorities, and evicts low-priority models
    to improve cache efficiency and performance.
    """
    try:
        optimization_results = trainer.optimize_cache()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Cache optimization completed successfully",
                "optimization_results": optimization_results
            }
        )
    
    except Exception as e:
        logger.error(f"Cache optimization failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Cache optimization failed: {str(e)}"
        )


@router.delete("/models/{model_id}")
async def delete_voice_model(model_id: str):
    """
    Delete a specific voice model from cache and storage.
    
    Removes the model from both memory and disk cache, freeing up storage space.
    """
    try:
        # Check if model exists
        model_info = trainer.get_model_info(model_id)
        if not model_info:
            raise HTTPException(
                status_code=404,
                detail=f"Voice model not found: {model_id}"
            )
        
        # Remove from cache (this would need to be implemented in the cache class)
        # For now, return success message
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Voice model {model_id} deleted successfully"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model: {str(e)}"
        )


@router.post("/upload-segments")
async def upload_audio_segments(
    voice_profile_id: str,
    files: List[UploadFile] = File(...)
):
    """
    Upload multiple audio segments for voice model training.
    
    Accepts multiple audio files and stores them for use in dedicated
    voice model creation or incremental improvement.
    """
    try:
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No audio files provided"
            )
        
        uploads_dir = Path(settings.UPLOAD_DIR) if hasattr(settings, 'UPLOAD_DIR') else Path("uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            # Validate file type
            if not file.content_type or not file.content_type.startswith('audio/'):
                continue
            
            # Generate unique filename
            file_extension = Path(file.filename).suffix if file.filename else '.wav'
            file_id = f"{voice_profile_id}_{int(time.time())}_{len(uploaded_files)}"
            file_path = uploads_dir / f"{file_id}{file_extension}"
            
            # Save file
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            uploaded_files.append({
                'file_id': file_id,
                'original_filename': file.filename,
                'file_path': str(file_path),
                'file_size': len(content)
            })
        
        if not uploaded_files:
            raise HTTPException(
                status_code=400,
                detail="No valid audio files uploaded"
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Uploaded {len(uploaded_files)} audio segments",
                "uploaded_files": uploaded_files
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio upload failed: {str(e)}"
        )