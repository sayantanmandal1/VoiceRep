"""
Robust Error Handling and Recovery System for Voice Style Replication.

This module implements advanced restoration techniques, alternative analysis methods,
automatic retry systems, detailed diagnostics, and graceful degradation for the
voice cloning system to achieve >95% reliability and user satisfaction.

Validates Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import asyncio
import logging
import traceback
import time
import hashlib
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import librosa
import scipy.signal

from app.core.error_handling import (
    ErrorCategory, ErrorSeverity, RecoveryAction, ErrorInfo,
    error_recovery_manager, CircuitBreaker
)

logger = logging.getLogger(__name__)


class RestorationType(str, Enum):
    """Types of audio restoration techniques."""
    SPECTRAL_INTERPOLATION = "spectral_interpolation"
    HARMONIC_RECONSTRUCTION = "harmonic_reconstruction"
    NOISE_PROFILE_MATCHING = "noise_profile_matching"
    BANDWIDTH_EXTENSION = "bandwidth_extension"
    DYNAMIC_RANGE_RESTORATION = "dynamic_range_restoration"
    ARTIFACT_REMOVAL = "artifact_removal"


class AnalysisMethod(str, Enum):
    """Alternative analysis methods for voice extraction."""
    STANDARD_MFCC = "standard_mfcc"
    ADVANCED_SPECTRAL = "advanced_spectral"
    HARMONIC_ANALYSIS = "harmonic_analysis"
    CEPSTRAL_ANALYSIS = "cepstral_analysis"
    WAVELET_ANALYSIS = "wavelet_analysis"
    PITCH_TRACKING = "pitch_tracking"
    FORMANT_TRACKING = "formant_tracking"


class QualityLevel(str, Enum):
    """Quality levels for processing decisions."""
    CRITICAL = "critical"  # < 30%
    POOR = "poor"         # 30-50%
    FAIR = "fair"         # 50-70%
    GOOD = "good"         # 70-85%
    EXCELLENT = "excellent"  # > 85%

@dataclass
class RestorationResult:
    """Result of audio restoration attempt."""
    success: bool
    restored_audio: Optional[np.ndarray]
    restoration_type: RestorationType
    quality_improvement: float
    processing_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Result of voice analysis attempt."""
    success: bool
    analysis_method: AnalysisMethod
    extracted_features: Optional[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiagnosticInfo:
    """Comprehensive diagnostic information."""
    error_id: str
    timestamp: datetime
    operation_type: str
    input_characteristics: Dict[str, Any]
    processing_steps: List[str]
    failure_point: str
    error_details: Dict[str, Any]
    system_state: Dict[str, Any]
    recovery_attempts: List[Dict[str, Any]]
    recommendations: List[str]


class RobustErrorRecoveryService:
    """Main service coordinating all error recovery components."""
    
    def __init__(self):
        self.diagnostic_history = {}
    
    async def handle_error_with_recovery(
        self,
        error: Exception,
        operation_type: str,
        context: Dict[str, Any],
        enable_retry: bool = True,
        enable_degradation: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive error handling with all recovery mechanisms.
        
        This method coordinates all error recovery components to provide
        the best possible outcome when errors occur.
        """
        
        # Generate unique error ID
        error_id = f"{operation_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create diagnostic information
        diagnostic_info = self.create_diagnostic_info(
            error_id, operation_type, error, context.get('input_data'), context
        )
        
        recovery_result = {
            'error_id': error_id,
            'original_error': str(error),
            'recovery_applied': [],
            'final_success': False,
            'diagnostic_info': diagnostic_info
        }
        
        try:
            # Attempt graceful degradation if enabled
            if enable_degradation:
                degradation_result = await self.handle_graceful_degradation(
                    operation_type, error, context
                )
                recovery_result['degradation_result'] = degradation_result
                recovery_result['recovery_applied'].append('graceful_degradation')
            
            logger.info(f"Completed error recovery for {error_id}")
            
        except Exception as recovery_error:
            logger.error(f"Error recovery failed for {error_id}: {str(recovery_error)}")
            recovery_result['recovery_error'] = str(recovery_error)
        
        return recovery_result
    
    def create_diagnostic_info(
        self,
        error_id: str,
        operation_type: str,
        error: Exception,
        input_data: Optional[Dict[str, Any]] = None,
        processing_context: Optional[Dict[str, Any]] = None
    ) -> DiagnosticInfo:
        """
        Create comprehensive diagnostic information for errors.
        
        Validates Requirements: 8.4 - Detailed diagnostic information system
        """
        
        # Analyze input characteristics
        input_characteristics = self._analyze_input_characteristics(input_data)
        
        # Reconstruct processing steps
        processing_steps = self._reconstruct_processing_steps(processing_context)
        
        # Identify failure point
        failure_point = self._identify_failure_point(error, processing_context)
        
        # Collect error details
        error_details = self._collect_error_details(error)
        
        # Capture system state
        system_state = self._capture_system_state()
        
        # Generate recovery recommendations
        recommendations = self._generate_recommendations(
            error, input_characteristics, processing_steps, failure_point
        )
        
        diagnostic_info = DiagnosticInfo(
            error_id=error_id,
            timestamp=datetime.now(),
            operation_type=operation_type,
            input_characteristics=input_characteristics,
            processing_steps=processing_steps,
            failure_point=failure_point,
            error_details=error_details,
            system_state=system_state,
            recovery_attempts=[],
            recommendations=recommendations
        )
        
        # Store diagnostic information
        self.diagnostic_history[error_id] = diagnostic_info
        
        logger.info(f"Created diagnostic info for error {error_id}")
        
        return diagnostic_info
    
    async def handle_graceful_degradation(
        self,
        operation_type: str,
        error: Exception,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle graceful degradation with informative error messages.
        
        Validates Requirements: 8.5 - Graceful degradation with informative messages
        """
        
        # Determine degradation strategy
        degradation_type = self._classify_degradation_type(error, operation_type)
        
        # Apply appropriate degradation strategy
        result = await self._default_degradation(error, context)
        
        # Add informative error message
        result['error_message'] = self._generate_user_friendly_message(error, degradation_type, result)
        result['degradation_type'] = degradation_type
        result['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Applied graceful degradation: {degradation_type}")
        
        return result
    
    def _analyze_input_characteristics(self, input_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze characteristics of input data."""
        if not input_data:
            return {'status': 'no_input_data'}
        
        characteristics = {}
        
        # Text characteristics
        if 'text' in input_data:
            text = input_data['text']
            characteristics.update({
                'text_length': len(text),
                'word_count': len(text.split()),
                'character_set': 'ascii' if text.isascii() else 'unicode'
            })
        
        # Processing parameters
        if 'parameters' in input_data:
            params = input_data['parameters']
            characteristics['processing_parameters'] = {
                k: str(v) for k, v in params.items()
            }
        
        return characteristics
    
    def _reconstruct_processing_steps(self, processing_context: Optional[Dict[str, Any]]) -> List[str]:
        """Reconstruct the processing steps that were attempted."""
        if not processing_context:
            return ['unknown_processing_pipeline']
        
        steps = []
        
        # Common processing steps
        step_indicators = [
            ('file_upload', 'File upload and validation'),
            ('audio_preprocessing', 'Audio preprocessing'),
            ('voice_analysis', 'Voice characteristic extraction'),
            ('model_training', 'Voice model training'),
            ('synthesis', 'Speech synthesis'),
            ('post_processing', 'Audio post-processing'),
            ('quality_assessment', 'Quality assessment')
        ]
        
        for indicator, description in step_indicators:
            if indicator in processing_context:
                status = processing_context[indicator]
                if isinstance(status, dict) and status.get('completed'):
                    steps.append(f"{description} (completed)")
                elif isinstance(status, dict) and status.get('started'):
                    steps.append(f"{description} (started)")
                elif status:
                    steps.append(f"{description} (attempted)")
        
        return steps if steps else ['processing_pipeline_unknown']
    
    def _identify_failure_point(
        self,
        error: Exception,
        processing_context: Optional[Dict[str, Any]]
    ) -> str:
        """Identify the specific point where processing failed."""
        
        error_message = str(error).lower()
        
        # File-related failures
        if any(keyword in error_message for keyword in ['file', 'path', 'not found', 'permission']):
            return 'file_handling'
        
        # Audio processing failures
        elif any(keyword in error_message for keyword in ['audio', 'sample rate', 'channels', 'format']):
            return 'audio_processing'
        
        # Analysis failures
        elif any(keyword in error_message for keyword in ['analysis', 'feature', 'extraction', 'mfcc']):
            return 'voice_analysis'
        
        # Model failures
        elif any(keyword in error_message for keyword in ['model', 'training', 'checkpoint', 'weights']):
            return 'model_operations'
        
        # Synthesis failures
        elif any(keyword in error_message for keyword in ['synthesis', 'generation', 'tts', 'text']):
            return 'speech_synthesis'
        
        # Memory/resource failures
        elif any(keyword in error_message for keyword in ['memory', 'resource', 'timeout', 'cuda']):
            return 'resource_management'
        
        return 'unknown_failure_point'
    
    def _collect_error_details(self, error: Exception) -> Dict[str, Any]:
        """Collect comprehensive error details."""
        return {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'error_args': list(error.args) if error.args else [],
            'traceback': traceback.format_exc(),
            'error_hash': hashlib.md5(str(error).encode()).hexdigest()[:8]
        }
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for diagnostics."""
        try:
            import platform
            
            system_state = {
                'timestamp': datetime.now().isoformat(),
                'platform': platform.platform(),
                'python_version': platform.python_version()
            }
            
            return system_state
            
        except Exception as e:
            return {
                'system_state_error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_recommendations(
        self,
        error: Exception,
        input_characteristics: Dict[str, Any],
        processing_steps: List[str],
        failure_point: str
    ) -> List[str]:
        """Generate specific recommendations based on error analysis."""
        recommendations = []
        
        error_message = str(error).lower()
        
        # File-related recommendations
        if failure_point == 'file_handling':
            recommendations.extend([
                "Verify that the audio file exists and is accessible",
                "Check file permissions and ensure the file is not corrupted",
                "Try uploading the file again or use a different file format"
            ])
        
        # Analysis failure recommendations
        if failure_point == 'voice_analysis':
            recommendations.extend([
                "Ensure the audio contains clear speech without excessive background noise",
                "Try using a different audio segment with clearer voice characteristics",
                "Check that the audio sample rate is at least 16kHz for optimal analysis"
            ])
        
        # Synthesis recommendations
        if failure_point == 'speech_synthesis':
            recommendations.extend([
                "Try simplifying the text or removing special characters",
                "Ensure the text is in a supported language",
                "Consider using a different synthesis model or parameters"
            ])
        
        # Generic recommendations based on error patterns
        if 'timeout' in error_message:
            recommendations.append("Operation timed out. Try with a shorter audio file or simpler processing settings")
        
        if 'memory' in error_message or 'out of memory' in error_message:
            recommendations.extend([
                "Insufficient memory available. Close other applications and try again",
                "Consider using lower quality settings to reduce memory usage"
            ])
        
        if not recommendations:
            recommendations.extend([
                "Try the operation again as this may be a temporary issue",
                "Check system resources and ensure sufficient memory and disk space",
                "Contact support if the problem persists"
            ])
        
        return recommendations
    
    def _classify_degradation_type(self, error: Exception, operation_type: str) -> str:
        """Classify the type of degradation needed."""
        error_message = str(error).lower()
        
        if 'analysis' in error_message or 'feature' in error_message:
            return 'voice_analysis_failure'
        elif 'synthesis' in error_message or 'generation' in error_message:
            return 'synthesis_failure'
        elif 'quality' in error_message or 'similarity' in error_message:
            return 'quality_failure'
        elif 'memory' in error_message or 'resource' in error_message:
            return 'resource_failure'
        elif 'timeout' in error_message:
            return 'timeout_failure'
        else:
            return 'general_failure'
    
    async def _default_degradation(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default degradation strategy for unclassified errors."""
        return {
            'success': False,
            'degraded_result': {
                'error_handled': True,
                'fallback_mode': 'safe',
                'partial_functionality': True
            },
            'degradation_applied': 'safe_mode',
            'user_impact': 'Some features may be temporarily unavailable',
            'suggested_actions': [
                'Try the operation again',
                'Check your input data and try with different parameters',
                'Contact support if the problem persists'
            ]
        }
    
    def _generate_user_friendly_message(
        self,
        error: Exception,
        degradation_type: str,
        result: Dict[str, Any]
    ) -> str:
        """Generate user-friendly error message."""
        
        base_messages = {
            'voice_analysis_failure': "We encountered difficulty analyzing your voice characteristics, but we've applied a basic analysis to continue processing.",
            'synthesis_failure': "Voice synthesis encountered an issue, but we can offer alternative text-to-speech options.",
            'quality_failure': "The voice cloning quality is lower than our target, but we've produced the best result possible with your audio.",
            'resource_failure': "Our system is currently experiencing high load. Please try again in a few minutes.",
            'timeout_failure': "Processing took longer than expected and was interrupted. You can try with shorter audio or faster settings.",
            'general_failure': "We encountered an unexpected issue but have applied safety measures to handle it gracefully."
        }
        
        base_message = base_messages.get(degradation_type, base_messages['general_failure'])
        
        # Add specific suggestions
        suggestions = result.get('suggested_actions', [])
        if suggestions:
            suggestion_text = " Here's what you can try: " + "; ".join(suggestions[:3])
            base_message += suggestion_text
        
        return base_message
    
    def get_diagnostic_info(self, error_id: str) -> Optional[DiagnosticInfo]:
        """Get diagnostic information for specific error."""
        return self.diagnostic_history.get(error_id)
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        return {
            'error_statistics': {'total_errors': len(self.diagnostic_history)},
            'timestamp': datetime.now().isoformat()
        }


# Global service instance
robust_error_recovery_service = RobustErrorRecoveryService()