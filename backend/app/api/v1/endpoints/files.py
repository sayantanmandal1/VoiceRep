"""
File upload and management API endpoints with session-based isolation.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, Request
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.core.config import settings
from app.models.file import ReferenceAudio, ProcessingStatus
from app.models.session import UserSession
from app.schemas.file import (
    FileUploadResponse, 
    FileValidationError, 
    FileProcessingStatus, 
    AudioExtractionRequest, 
    AudioExtractionResponse, 
    TaskProgressResponse
)
from app.services.file_service import file_validation_service
from app.services.session_service import FileAccessService, RequestTrackingService
from app.middleware.session_middleware import require_session
from app.core.celery_app import celery_app

router = APIRouter()


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    session: UserSession = Depends(require_session),
    db: Session = Depends(get_db)
):
    """
    Upload audio or video file for voice cloning with session isolation.
    
    Accepts audio formats: .mp3, .wav, .flac, .m4a
    Accepts video formats: .mp4, .avi, .mov, .mkv
    Maximum file size: 100MB
    """
    # Get services
    file_access_service = FileAccessService(db)
    request_service = RequestTrackingService(db)
    
    # Validate file
    is_valid, error_message = file_validation_service.validate_file(file)
    if not is_valid:
        # Update request status
        if hasattr(request.state, 'request_id'):
            request_service.fail_request(request.state.request_id, error_message)
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_type": "validation_error",
                "message": error_message,
                "supported_formats": file_validation_service.get_supported_formats(),
                "max_file_size_mb": 100
            }
        )
    
    # Create database record with session isolation
    reference_audio = ReferenceAudio(
        user_id=session.user_identifier,
        filename=file.filename,
        file_path="",  # Will be updated after saving
        format=file.content_type or "unknown",
        file_size=0,  # Will be updated after saving
        processing_status=ProcessingStatus.UPLOADED
    )
    
    db.add(reference_audio)
    db.commit()
    db.refresh(reference_audio)
    
    try:
        # Save file to session-isolated directory
        file_path, file_size = await file_validation_service.save_uploaded_file(
            file, reference_audio.id, session
        )
        
        # Update database record with file info
        reference_audio.file_path = file_path
        reference_audio.file_size = file_size
        
        # Grant file access to session
        file_access_service.grant_file_access(
            session_id=session.id,
            file_path=file_path,
            access_type="read",
            expires_in_hours=24
        )
        
        # Track file in request
        if hasattr(request.state, 'request_id'):
            request_service.add_file_to_request(request.state.request_id, file_path)
        
        # Extract metadata if possible
        duration, sample_rate = file_validation_service.extract_audio_metadata(file_path)
        reference_audio.duration = duration
        reference_audio.sample_rate = sample_rate
        
        db.commit()
        db.refresh(reference_audio)
        
        return FileUploadResponse(
            id=reference_audio.id,  # Frontend expects 'id'
            file_id=reference_audio.id,  # Keep for backward compatibility
            filename=reference_audio.filename,
            file_size=reference_audio.file_size,
            duration=reference_audio.duration,
            sample_rate=reference_audio.sample_rate,
            status=reference_audio.processing_status,
            upload_timestamp=reference_audio.created_at
        )
        
    except Exception as e:
        # Cleanup on error
        db.delete(reference_audio)
        db.commit()
        
        # Update request status
        if hasattr(request.state, 'request_id'):
            request_service.fail_request(request.state.request_id, f"Failed to save file: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )


@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats."""
    return {
        "audio_formats": sorted(list(file_validation_service.SUPPORTED_AUDIO_FORMATS)),
        "video_formats": sorted(list(file_validation_service.SUPPORTED_VIDEO_FORMATS)),
        "all_formats": file_validation_service.get_supported_formats(),
        "max_file_size_mb": 100
    }


@router.get("/{file_id}/status", response_model=FileProcessingStatus)
async def get_file_status(
    file_id: str, 
    session: UserSession = Depends(require_session),
    db: Session = Depends(get_db)
):
    """Get processing status of uploaded file with session validation."""
    reference_audio = db.query(ReferenceAudio).filter(
        ReferenceAudio.id == file_id,
        ReferenceAudio.user_id == session.user_identifier
    ).first()
    
    if not reference_audio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or access denied"
        )
    
    return FileProcessingStatus(
        file_id=reference_audio.id,
        status=reference_audio.processing_status,
        error_message=reference_audio.error_message
    )


@router.delete("/{file_id}")
async def delete_file(
    file_id: str, 
    session: UserSession = Depends(require_session),
    db: Session = Depends(get_db)
):
    """Delete uploaded file and its data with session validation."""
    file_access_service = FileAccessService(db)
    
    reference_audio = db.query(ReferenceAudio).filter(
        ReferenceAudio.id == file_id,
        ReferenceAudio.user_id == session.user_identifier
    ).first()
    
    if not reference_audio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or access denied"
        )
    
    # Validate file access
    if not file_validation_service.validate_file_access(
        session, reference_audio.file_path, "delete", file_access_service
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to file"
        )
    
    # Remove file from disk
    file_validation_service.cleanup_file(reference_audio.file_path)
    
    # Revoke file access
    file_access_service.revoke_file_access(session.id, reference_audio.file_path)
    
    # Remove from database
    db.delete(reference_audio)
    db.commit()
    
    return {"message": "File deleted successfully"}


@router.get("/", response_model=List[FileUploadResponse])
async def list_files(
    session: UserSession = Depends(require_session),
    db: Session = Depends(get_db)
):
    """List all uploaded files for the current session."""
    files = db.query(ReferenceAudio).filter(
        ReferenceAudio.user_id == session.user_identifier
    ).all()
    
    return [
        FileUploadResponse(
            file_id=file.id,
            filename=file.filename,
            file_size=file.file_size,
            duration=file.duration,
            sample_rate=file.sample_rate,
            status=file.processing_status,
            upload_timestamp=file.created_at
        )
        for file in files
    ]


@router.post("/{file_id}/extract-audio", response_model=AudioExtractionResponse)
async def extract_audio(
    file_id: str,
    request_data: AudioExtractionRequest,
    request: Request,
    session: UserSession = Depends(require_session),
    db: Session = Depends(get_db)
):
    """
    Extract audio from uploaded video file with session isolation.
    
    Starts a background task to extract audio from video.
    Use the returned task_id to track progress.
    """
    file_access_service = FileAccessService(db)
    request_service = RequestTrackingService(db)
    
    # Verify file exists and belongs to session
    reference_audio = db.query(ReferenceAudio).filter(
        ReferenceAudio.id == file_id,
        ReferenceAudio.user_id == session.user_identifier
    ).first()
    
    if not reference_audio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or access denied"
        )
    
    # Validate file access
    if not file_validation_service.validate_file_access(
        session, reference_audio.file_path, "read", file_access_service
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to file"
        )
    
    # Check if file is a video
    if not file_validation_service.is_video_file(reference_audio.file_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File is not a video format. Audio extraction only works with video files."
        )
    
    # Update status to extracting
    reference_audio.processing_status = ProcessingStatus.EXTRACTING
    db.commit()
    
    # Generate session-isolated output path
    output_filename = f"{file_id}_extracted.{request_data.output_format}"
    output_path = file_validation_service.get_session_file_path(
        session, settings.RESULTS_DIR, output_filename
    )
    
    # Grant access to output file
    file_access_service.grant_file_access(
        session_id=session.id,
        file_path=output_path,
        access_type="write",
        expires_in_hours=24
    )
    
    # Track output file in request
    if hasattr(request.state, 'request_id'):
        request_service.add_file_to_request(request.state.request_id, output_path)
    
    # Start background extraction task with session context
    task_id = file_validation_service.start_audio_extraction_task(
        video_path=reference_audio.file_path,
        output_path=output_path,
        quality=request_data.quality,
        file_id=file_id,
        session_id=session.id
    )
    
    return AudioExtractionResponse(
        task_id=task_id,
        file_id=file_id,
        status="PENDING",
        message="Audio extraction task started"
    )


@router.get("/tasks/{task_id}/progress", response_model=TaskProgressResponse)
async def get_task_progress(
    task_id: str,
    session: UserSession = Depends(require_session)
):
    """
    Get progress of a background task with session validation.
    
    Returns current status, progress percentage, and any results or errors.
    """
    from celery.result import AsyncResult
    
    task = AsyncResult(task_id, app=celery_app)
    
    response = {
        "task_id": task_id,
        "status": task.state,
        "progress": None,
        "current_step": None,
        "result": None,
        "error": None,
        "file_id": None
    }
    
    if task.state == 'PENDING':
        response["current_step"] = "Task is waiting to start"
    elif task.state == 'PROGRESS':
        if task.info:
            # Validate session access to task
            task_session_id = task.info.get('session_id')
            if task_session_id and task_session_id != session.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to task"
                )
            
            response["progress"] = task.info.get('progress', 0)
            response["current_step"] = task.info.get('status', 'Processing')
            response["file_id"] = task.info.get('file_id')
    elif task.state == 'SUCCESS':
        response["progress"] = 100
        response["current_step"] = "Completed"
        response["result"] = task.result
        if task.result:
            response["file_id"] = task.result.get('file_id')
    elif task.state == 'FAILURE':
        response["error"] = str(task.info) if task.info else "Task failed"
        if isinstance(task.info, dict):
            response["error"] = task.info.get('error', str(task.info))
            response["file_id"] = task.info.get('file_id')
    
    return TaskProgressResponse(**response)