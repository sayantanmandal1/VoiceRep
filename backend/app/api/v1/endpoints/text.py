"""
Text input and processing API endpoints.
"""

from fastapi import APIRouter, HTTPException, status
from typing import Optional

from app.schemas.text import (
    TextInputRequest,
    TextValidationResponse,
    TextProcessingRequest,
    TextProcessingResponse,
    LanguageDetectionResponse
)
from app.services.text_service import text_processing_service

router = APIRouter()


@router.post("/validate", response_model=TextValidationResponse)
async def validate_text(request: TextInputRequest):
    """
    Validate text input for synthesis.
    
    Validates text length, Unicode content, and detects language.
    Supports all Unicode characters including special characters and punctuation.
    """
    try:
        validation_result = text_processing_service.validate_text_input(
            text=request.text,
            target_language=request.language
        )
        
        return validation_result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text validation failed: {str(e)}"
        )


@router.post("/detect-language", response_model=LanguageDetectionResponse)
async def detect_language(request: TextInputRequest):
    """
    Detect language of input text.
    
    Returns detected language with confidence score and indicates
    if cross-language synthesis will be required.
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )
        
        # Sanitize text first
        sanitization_result = text_processing_service.sanitize_text(request.text)
        
        # Detect language
        detection_result = text_processing_service.detect_language(
            sanitization_result.sanitized_text
        )
        
        return detection_result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Language detection failed: {str(e)}"
        )


@router.get("/supported-languages")
async def get_supported_languages():
    """
    Get list of supported languages for cross-language synthesis.
    
    Returns language codes and their display names.
    """
    return {
        "supported_languages": text_processing_service.get_supported_languages(),
        "total_count": len(text_processing_service.SUPPORTED_LANGUAGES)
    }


@router.post("/prepare", response_model=dict)
async def prepare_text_for_synthesis(request: TextProcessingRequest):
    """
    Prepare text for speech synthesis.
    
    Validates, sanitizes, and prepares text with language detection
    for use in voice synthesis pipeline.
    """
    try:
        prepared_text = text_processing_service.prepare_text_for_synthesis(
            text=request.text,
            target_language=request.target_language
        )
        
        return {
            "status": "success",
            "prepared_text": prepared_text,
            "reference_audio_id": request.reference_audio_id,
            "voice_settings": request.voice_settings or {}
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text preparation failed: {str(e)}"
        )


@router.get("/limits")
async def get_text_limits():
    """
    Get text input limits and constraints.
    
    Returns maximum text length and other validation rules.
    """
    return {
        "max_text_length": text_processing_service.max_text_length,
        "supported_unicode": True,
        "preserves_punctuation": True,
        "preserves_special_characters": True,
        "validation_rules": [
            "Text cannot be empty or only whitespace",
            f"Maximum length: {text_processing_service.max_text_length} characters",
            "All Unicode characters are supported",
            "Special characters and punctuation are preserved",
            "Control characters are removed during sanitization"
        ]
    }