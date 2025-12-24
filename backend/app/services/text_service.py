"""
Text processing and validation service.
"""

import re
import unicodedata
from typing import Tuple, Optional, List
from langdetect import detect, detect_langs, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

from app.core.config import settings
from app.schemas.text import (
    TextValidationResponse, 
    LanguageDetectionResponse, 
    TextSanitizationResult
)


class TextProcessingService:
    """Service for text validation, sanitization, and language detection."""
    
    # Supported languages for cross-language synthesis
    SUPPORTED_LANGUAGES = {
        'en': 'English',
        'es': 'Spanish', 
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh': 'Chinese',
        'ar': 'Arabic',
        'hi': 'Hindi'
    }
    
    def __init__(self):
        """Initialize text processing service."""
        # Set seed for consistent language detection
        DetectorFactory.seed = 0
        self.max_text_length = settings.MAX_TEXT_LENGTH
    
    def validate_text_input(self, text: str, target_language: Optional[str] = None) -> TextValidationResponse:
        """
        Validate text input for synthesis.
        
        Args:
            text: Input text to validate
            target_language: Optional target language code
            
        Returns:
            TextValidationResponse with validation results
        """
        validation_errors = []
        
        # Basic validation
        if not text or not text.strip():
            validation_errors.append("Text cannot be empty")
            return TextValidationResponse(
                is_valid=False,
                text=text,
                character_count=0,
                sanitized_text="",
                validation_errors=validation_errors
            )
        
        # Length validation
        if len(text) > self.max_text_length:
            validation_errors.append(f"Text exceeds maximum length of {self.max_text_length} characters")
        
        # Sanitize text
        sanitization_result = self.sanitize_text(text)
        
        # Language detection
        detected_language = None
        language_confidence = None
        
        try:
            detection_result = self.detect_language(sanitization_result.sanitized_text)
            detected_language = detection_result.detected_language
            language_confidence = detection_result.confidence
        except Exception:
            # Language detection failed, but this is not a validation error
            pass
        
        # Check if target language is supported
        if target_language and target_language not in self.SUPPORTED_LANGUAGES:
            validation_errors.append(f"Unsupported target language: {target_language}")
        
        is_valid = len(validation_errors) == 0
        
        return TextValidationResponse(
            is_valid=is_valid,
            text=text,
            character_count=len(sanitization_result.sanitized_text),
            detected_language=detected_language,
            language_confidence=language_confidence,
            sanitized_text=sanitization_result.sanitized_text,
            validation_errors=validation_errors if validation_errors else None
        )
    
    def sanitize_text(self, text: str) -> TextSanitizationResult:
        """
        Sanitize text while preserving Unicode characters and punctuation.
        
        Args:
            text: Input text to sanitize
            
        Returns:
            TextSanitizationResult with sanitization details
        """
        original_text = text
        removed_characters = []
        
        # Normalize Unicode characters
        normalized_text = unicodedata.normalize('NFC', text)
        
        # Remove control characters but preserve printable Unicode
        sanitized_chars = []
        for char in normalized_text:
            category = unicodedata.category(char)
            # Keep letters, numbers, punctuation, symbols, and whitespace
            # Remove control characters (Cc, Cf) except for common whitespace
            if category.startswith(('L', 'N', 'P', 'S', 'Z')) or char in '\n\r\t ':
                sanitized_chars.append(char)
            else:
                removed_characters.append(char)
        
        sanitized_text = ''.join(sanitized_chars)
        
        # Clean up excessive whitespace
        sanitized_text = re.sub(r'\s+', ' ', sanitized_text).strip()
        
        return TextSanitizationResult(
            original_text=original_text,
            sanitized_text=sanitized_text,
            removed_characters=removed_characters,
            character_count=len(sanitized_text),
            is_modified=(original_text != sanitized_text)
        )
    
    def detect_language(self, text: str) -> LanguageDetectionResponse:
        """
        Detect language of input text.
        
        Args:
            text: Text to analyze for language detection
            
        Returns:
            LanguageDetectionResponse with detection results
        """
        try:
            # Get detailed language detection results
            lang_probs = detect_langs(text)
            
            if not lang_probs:
                raise LangDetectException("No language detected")
            
            # Get the most likely language
            top_detection = lang_probs[0]
            detected_language = top_detection.lang
            confidence = top_detection.prob
            
            # Check if it's a cross-language scenario
            # This would be determined by comparing with reference audio language
            # For now, we'll mark as cross-language if not English
            is_cross_language = detected_language != 'en'
            
            return LanguageDetectionResponse(
                detected_language=detected_language,
                confidence=confidence,
                supported_languages=list(self.SUPPORTED_LANGUAGES.keys()),
                is_cross_language=is_cross_language
            )
            
        except LangDetectException:
            # Fallback to English if detection fails
            return LanguageDetectionResponse(
                detected_language='en',
                confidence=0.5,
                supported_languages=list(self.SUPPORTED_LANGUAGES.keys()),
                is_cross_language=False
            )
    
    def prepare_text_for_synthesis(self, text: str, target_language: Optional[str] = None) -> dict:
        """
        Prepare text for speech synthesis.
        
        Args:
            text: Input text
            target_language: Optional target language
            
        Returns:
            Dictionary with prepared text and metadata
        """
        # Validate and sanitize
        validation_result = self.validate_text_input(text, target_language)
        
        if not validation_result.is_valid:
            raise ValueError(f"Text validation failed: {validation_result.validation_errors}")
        
        # Use detected language if target not specified
        final_language = target_language or validation_result.detected_language or 'en'
        
        return {
            'text': validation_result.sanitized_text,
            'character_count': validation_result.character_count,
            'detected_language': validation_result.detected_language,
            'target_language': final_language,
            'is_cross_language': validation_result.detected_language != final_language,
            'language_confidence': validation_result.language_confidence
        }
    
    def get_supported_languages(self) -> dict:
        """Get list of supported languages."""
        return self.SUPPORTED_LANGUAGES.copy()
    
    def is_language_supported(self, language_code: str) -> bool:
        """Check if language is supported for synthesis."""
        return language_code in self.SUPPORTED_LANGUAGES


# Global instance
text_processing_service = TextProcessingService()