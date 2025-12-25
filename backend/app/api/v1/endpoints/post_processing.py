"""
API endpoints for advanced audio post-processing operations.
"""

import os
import uuid
import logging
import asyncio
import tempfile
from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse
from pathlib import Path
import soundfile as sf
import numpy as np

from app.services.advanced_audio_post_processor import AdvancedAudioPostProcessor
from app.core.config import settings
from app.core.database import get_db
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

router = APIRouter()

# Task storage for post-processing operations
post_processing_tasks: Dict[str, Dict[str, Any]] = {}


@router.post("/enhance-audio")
async def enhance_audio(
    synthesized_audio: UploadFile = File(..., description="Synthesized audio file to enhance"),
    reference_audio: UploadFile = File(..., description="Reference audio file for matching"),
    preserve_characteristics: bool = Form(True, description="Whether to preserve voice characteristics"),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    Enhance synthesized audio using advanced post-processing techniques.
    
    This endpoint applies the complete post-processing pipeline including:
    - Spectral matching and frequency alignment (Requirement 6.1)
    - Artifact removal and audio smoothing (Requirement 6.2)
    - Voice characteristic preservation during enhancement (Requirement 6.3)
    - Dynamic range compression matching (Requirement 6.4)
    - Consistency maintenance for volume and quality (Requirement 6.5)
    """
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Validate file formats
        if not synthesized_audio.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
            raise HTTPException(status_code=400, detail="Synthesized audio must be in WAV, MP3, FLAC, or M4A format")
        
        if not reference_audio.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
            raise HTTPException(status_code=400, detail="Reference audio must be in WAV, MP3, FLAC, or M4A format")
        
        # Initialize task status
        post_processing_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting audio post-processing",
            "created_at": datetime.now(),
            "result": None,
            "error": None
        }
        
        # Start background processing
        background_tasks.add_task(
            process_audio_enhancement,
            task_id,
            synthesized_audio,
            reference_audio,
            preserve_characteristics
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Audio post-processing started"
        }
        
    except Exception as e:
        logger.error(f"Error starting audio enhancement: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start audio enhancement: {str(e)}")


@router.get("/enhance-audio/{task_id}/status")
async def get_enhancement_status(task_id: str):
    """Get the status of an audio enhancement task."""
    if task_id not in post_processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = post_processing_tasks[task_id]
    
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "message": task["message"],
        "created_at": task["created_at"],
        "result": task["result"],
        "error": task["error"]
    }


@router.get("/enhance-audio/{task_id}/download")
async def download_enhanced_audio(task_id: str):
    """Download the enhanced audio file."""
    if task_id not in post_processing_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = post_processing_tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Task not completed yet")
    
    if not task["result"] or "output_path" not in task["result"]:
        raise HTTPException(status_code=404, detail="Enhanced audio file not found")
    
    output_path = task["result"]["output_path"]
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Enhanced audio file not found on disk")
    
    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=f"enhanced_audio_{task_id}.wav"
    )


@router.post("/spectral-matching")
async def apply_spectral_matching(
    synthesized_audio: UploadFile = File(..., description="Synthesized audio file"),
    reference_audio: UploadFile = File(..., description="Reference audio file for matching"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Apply spectral matching and frequency alignment to synthesized audio.
    
    This endpoint specifically implements Requirement 6.1:
    - Spectral matching and frequency alignment system
    """
    try:
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        post_processing_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting spectral matching",
            "created_at": datetime.now(),
            "result": None,
            "error": None
        }
        
        # Start background processing
        background_tasks.add_task(
            process_spectral_matching,
            task_id,
            synthesized_audio,
            reference_audio
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Spectral matching started"
        }
        
    except Exception as e:
        logger.error(f"Error starting spectral matching: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start spectral matching: {str(e)}")


@router.post("/remove-artifacts")
async def remove_synthesis_artifacts(
    audio_file: UploadFile = File(..., description="Audio file with potential artifacts"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Remove synthesis artifacts and smooth audio.
    
    This endpoint specifically implements Requirement 6.2:
    - Artifact removal and audio smoothing algorithms
    """
    try:
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        post_processing_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting artifact removal",
            "created_at": datetime.now(),
            "result": None,
            "error": None
        }
        
        # Start background processing
        background_tasks.add_task(
            process_artifact_removal,
            task_id,
            audio_file
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Artifact removal started"
        }
        
    except Exception as e:
        logger.error(f"Error starting artifact removal: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start artifact removal: {str(e)}")


@router.post("/calculate-similarity")
async def calculate_audio_similarity(
    synthesized_audio: UploadFile = File(..., description="Synthesized audio file"),
    reference_audio: UploadFile = File(..., description="Reference audio file"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Calculate detailed similarity score between synthesized and reference audio.
    
    Returns similarity metrics including spectral, temporal, prosodic, and timbre similarity.
    """
    try:
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        post_processing_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Calculating similarity score",
            "created_at": datetime.now(),
            "result": None,
            "error": None
        }
        
        # Start background processing
        background_tasks.add_task(
            process_similarity_calculation,
            task_id,
            synthesized_audio,
            reference_audio
        )
        
        return {
            "task_id": task_id,
            "status": "processing",
            "message": "Similarity calculation started"
        }
        
    except Exception as e:
        logger.error(f"Error starting similarity calculation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start similarity calculation: {str(e)}")


# Background task functions

async def process_audio_enhancement(
    task_id: str,
    synthesized_audio: UploadFile,
    reference_audio: UploadFile,
    preserve_characteristics: bool
):
    """Background task for complete audio enhancement."""
    try:
        # Update progress
        post_processing_tasks[task_id]["progress"] = 10
        post_processing_tasks[task_id]["message"] = "Loading audio files"
        
        # Save uploaded files to temporary locations
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as synth_temp:
            synth_temp.write(await synthesized_audio.read())
            synth_temp_path = synth_temp.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as ref_temp:
            ref_temp.write(await reference_audio.read())
            ref_temp_path = ref_temp.name
        
        try:
            # Load audio data
            post_processing_tasks[task_id]["progress"] = 20
            post_processing_tasks[task_id]["message"] = "Loading and preprocessing audio"
            
            import librosa
            synth_audio, synth_sr = librosa.load(synth_temp_path, sr=None, mono=True)
            ref_audio, ref_sr = librosa.load(ref_temp_path, sr=None, mono=True)
            
            # Initialize post-processor
            post_processor = AdvancedAudioPostProcessor(sample_rate=22050)
            
            # Process audio enhancement
            post_processing_tasks[task_id]["progress"] = 30
            post_processing_tasks[task_id]["message"] = "Applying advanced post-processing"
            
            enhanced_audio, processing_metrics = post_processor.enhance_synthesis_quality(
                synth_audio, ref_audio, ref_sr, preserve_characteristics
            )
            
            # Save enhanced audio
            post_processing_tasks[task_id]["progress"] = 90
            post_processing_tasks[task_id]["message"] = "Saving enhanced audio"
            
            # Create output directory if it doesn't exist
            output_dir = Path(settings.RESULTS_DIR) / "enhanced_audio"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"enhanced_{task_id}.wav"
            sf.write(str(output_path), enhanced_audio, 22050)
            
            # Complete task
            post_processing_tasks[task_id]["status"] = "completed"
            post_processing_tasks[task_id]["progress"] = 100
            post_processing_tasks[task_id]["message"] = "Audio enhancement completed successfully"
            post_processing_tasks[task_id]["result"] = {
                "output_path": str(output_path),
                "similarity_score": processing_metrics["final_similarity_score"],
                "processing_metrics": processing_metrics,
                "target_achieved": processing_metrics["final_similarity_score"] > 0.95
            }
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(synth_temp_path)
                os.unlink(ref_temp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error in audio enhancement task {task_id}: {e}")
        post_processing_tasks[task_id]["status"] = "failed"
        post_processing_tasks[task_id]["error"] = str(e)
        post_processing_tasks[task_id]["message"] = f"Audio enhancement failed: {str(e)}"


async def process_spectral_matching(
    task_id: str,
    synthesized_audio: UploadFile,
    reference_audio: UploadFile
):
    """Background task for spectral matching."""
    try:
        # Update progress
        post_processing_tasks[task_id]["progress"] = 10
        post_processing_tasks[task_id]["message"] = "Loading audio files"
        
        # Save uploaded files to temporary locations
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as synth_temp:
            synth_temp.write(await synthesized_audio.read())
            synth_temp_path = synth_temp.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as ref_temp:
            ref_temp.write(await reference_audio.read())
            ref_temp_path = ref_temp.name
        
        try:
            # Load audio data
            post_processing_tasks[task_id]["progress"] = 30
            post_processing_tasks[task_id]["message"] = "Processing spectral matching"
            
            import librosa
            synth_audio, synth_sr = librosa.load(synth_temp_path, sr=None, mono=True)
            ref_audio, ref_sr = librosa.load(ref_temp_path, sr=None, mono=True)
            
            # Resample if needed
            if ref_sr != 22050:
                ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=22050)
            if synth_sr != 22050:
                synth_audio = librosa.resample(synth_audio, orig_sr=synth_sr, target_sr=22050)
            
            # Initialize post-processor
            post_processor = AdvancedAudioPostProcessor(sample_rate=22050)
            
            # Apply spectral matching
            post_processing_tasks[task_id]["progress"] = 70
            spectral_result = post_processor.apply_spectral_matching(synth_audio, ref_audio)
            
            # Save matched audio
            post_processing_tasks[task_id]["progress"] = 90
            post_processing_tasks[task_id]["message"] = "Saving matched audio"
            
            output_dir = Path(settings.RESULTS_DIR) / "spectral_matched"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"spectral_matched_{task_id}.wav"
            sf.write(str(output_path), spectral_result.matched_audio, 22050)
            
            # Complete task
            post_processing_tasks[task_id]["status"] = "completed"
            post_processing_tasks[task_id]["progress"] = 100
            post_processing_tasks[task_id]["message"] = "Spectral matching completed successfully"
            post_processing_tasks[task_id]["result"] = {
                "output_path": str(output_path),
                "frequency_alignment_score": spectral_result.frequency_alignment_score,
                "spectral_distance": spectral_result.spectral_distance,
                "enhancement_applied": spectral_result.enhancement_applied,
                "processing_time": spectral_result.processing_time
            }
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(synth_temp_path)
                os.unlink(ref_temp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error in spectral matching task {task_id}: {e}")
        post_processing_tasks[task_id]["status"] = "failed"
        post_processing_tasks[task_id]["error"] = str(e)
        post_processing_tasks[task_id]["message"] = f"Spectral matching failed: {str(e)}"


async def process_artifact_removal(task_id: str, audio_file: UploadFile):
    """Background task for artifact removal."""
    try:
        # Update progress
        post_processing_tasks[task_id]["progress"] = 10
        post_processing_tasks[task_id]["message"] = "Loading audio file"
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(await audio_file.read())
            temp_path = temp_file.name
        
        try:
            # Load audio data
            post_processing_tasks[task_id]["progress"] = 30
            post_processing_tasks[task_id]["message"] = "Removing artifacts"
            
            import librosa
            audio, sr = librosa.load(temp_path, sr=None, mono=True)
            
            # Resample if needed
            if sr != 22050:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
            
            # Initialize post-processor
            post_processor = AdvancedAudioPostProcessor(sample_rate=22050)
            
            # Remove artifacts
            post_processing_tasks[task_id]["progress"] = 70
            artifact_result = post_processor.remove_synthesis_artifacts(audio)
            
            # Save cleaned audio
            post_processing_tasks[task_id]["progress"] = 90
            post_processing_tasks[task_id]["message"] = "Saving cleaned audio"
            
            output_dir = Path(settings.RESULTS_DIR) / "artifact_removed"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"cleaned_{task_id}.wav"
            sf.write(str(output_path), artifact_result.cleaned_audio, 22050)
            
            # Complete task
            post_processing_tasks[task_id]["status"] = "completed"
            post_processing_tasks[task_id]["progress"] = 100
            post_processing_tasks[task_id]["message"] = "Artifact removal completed successfully"
            post_processing_tasks[task_id]["result"] = {
                "output_path": str(output_path),
                "artifacts_detected": artifact_result.artifacts_detected,
                "artifacts_removed": artifact_result.artifacts_removed,
                "quality_improvement": artifact_result.quality_improvement,
                "processing_time": artifact_result.processing_time
            }
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error in artifact removal task {task_id}: {e}")
        post_processing_tasks[task_id]["status"] = "failed"
        post_processing_tasks[task_id]["error"] = str(e)
        post_processing_tasks[task_id]["message"] = f"Artifact removal failed: {str(e)}"


async def process_similarity_calculation(
    task_id: str,
    synthesized_audio: UploadFile,
    reference_audio: UploadFile
):
    """Background task for similarity calculation."""
    try:
        # Update progress
        post_processing_tasks[task_id]["progress"] = 10
        post_processing_tasks[task_id]["message"] = "Loading audio files"
        
        # Save uploaded files to temporary locations
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as synth_temp:
            synth_temp.write(await synthesized_audio.read())
            synth_temp_path = synth_temp.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as ref_temp:
            ref_temp.write(await reference_audio.read())
            ref_temp_path = ref_temp.name
        
        try:
            # Load audio data
            post_processing_tasks[task_id]["progress"] = 30
            post_processing_tasks[task_id]["message"] = "Calculating similarity"
            
            import librosa
            synth_audio, synth_sr = librosa.load(synth_temp_path, sr=None, mono=True)
            ref_audio, ref_sr = librosa.load(ref_temp_path, sr=None, mono=True)
            
            # Resample if needed
            if ref_sr != 22050:
                ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=22050)
            if synth_sr != 22050:
                synth_audio = librosa.resample(synth_audio, orig_sr=synth_sr, target_sr=22050)
            
            # Initialize post-processor
            post_processor = AdvancedAudioPostProcessor(sample_rate=22050)
            
            # Calculate similarity
            post_processing_tasks[task_id]["progress"] = 70
            similarity_score = post_processor.calculate_similarity_score(synth_audio, ref_audio)
            
            # Calculate detailed similarity metrics
            spectral_sim = post_processor._calculate_spectral_similarity(synth_audio, ref_audio)
            temporal_sim = post_processor._calculate_temporal_similarity(synth_audio, ref_audio)
            prosodic_sim = post_processor._calculate_prosodic_similarity(synth_audio, ref_audio)
            timbre_sim = post_processor._calculate_timbre_similarity(synth_audio, ref_audio)
            
            # Complete task
            post_processing_tasks[task_id]["status"] = "completed"
            post_processing_tasks[task_id]["progress"] = 100
            post_processing_tasks[task_id]["message"] = "Similarity calculation completed successfully"
            post_processing_tasks[task_id]["result"] = {
                "overall_similarity": similarity_score,
                "spectral_similarity": spectral_sim,
                "temporal_similarity": temporal_sim,
                "prosodic_similarity": prosodic_sim,
                "timbre_similarity": timbre_sim,
                "target_achieved": similarity_score > 0.95
            }
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(synth_temp_path)
                os.unlink(ref_temp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error in similarity calculation task {task_id}: {e}")
        post_processing_tasks[task_id]["status"] = "failed"
        post_processing_tasks[task_id]["error"] = str(e)
        post_processing_tasks[task_id]["message"] = f"Similarity calculation failed: {str(e)}"