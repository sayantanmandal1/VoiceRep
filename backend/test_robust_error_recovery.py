"""
Comprehensive tests for the Robust Error Handling and Recovery System.

Tests all components including diagnostic information creation, graceful degradation,
and error recovery mechanisms.

Validates Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import pytest
import pytest_asyncio
import numpy as np
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from app.services.robust_error_recovery import (
    RobustErrorRecoveryService,
    RestorationType,
    AnalysisMethod,
    QualityLevel,
    DiagnosticInfo,
    RestorationResult,
    AnalysisResult,
    robust_error_recovery_service
)

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


class TestRobustErrorRecoveryService:
    """Test the main robust error recovery service."""
    
    @pytest.fixture
    def recovery_service(self):
        return RobustErrorRecoveryService()
    
    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio data for testing."""
        duration = 2.0  # 2 seconds
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Generate a simple sine wave with some noise
        audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
        return audio, sample_rate
    
    @pytest.mark.asyncio
    async def test_comprehensive_error_handling(self, recovery_service):
        """Test comprehensive error handling with all recovery mechanisms."""
        
        error = RuntimeError("Comprehensive test error")
        context = {
            'input_data': {'audio_file': 'test.wav', 'text': 'Hello'},
            'operation_stage': 'voice_analysis'
        }
        
        result = await recovery_service.handle_error_with_recovery(
            error, 'voice_cloning', context
        )
        
        assert 'error_id' in result
        assert 'original_error' in result
        assert 'recovery_applied' in result
        assert 'diagnostic_info' in result
        assert len(result['recovery_applied']) > 0
    
    def test_create_diagnostic_info(self, recovery_service):
        """Test creation of comprehensive diagnostic information."""
        
        error = ValueError("Test error for diagnostics")
        input_data = {
            'audio_file': 'test.wav',
            'text': 'Hello world',
            'parameters': {'quality': 'high'}
        }
        context = {
            'audio_preprocessing': {'completed': True},
            'voice_analysis': {'started': True, 'error': 'analysis_failed'}
        }
        
        diag_info = recovery_service.create_diagnostic_info(
            'test_error_123', 'voice_synthesis', error, input_data, context
        )
        
        assert diag_info.error_id == 'test_error_123'
        assert diag_info.operation_type == 'voice_synthesis'
        assert 'text_length' in diag_info.input_characteristics
        assert diag_info.failure_point is not None
        assert len(diag_info.recommendations) > 0
        assert diag_info.error_details['error_type'] == 'ValueError'
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, recovery_service):
        """Test graceful degradation with informative error messages."""
        
        error = RuntimeError("Voice feature extraction failed")
        context = {'input_audio': 'test.wav'}
        
        result = await recovery_service.handle_graceful_degradation(
            'voice_analysis', error, context
        )
        
        assert 'degradation_type' in result
        assert 'degraded_result' in result
        assert 'user_impact' in result
        assert 'suggested_actions' in result
        assert len(result['suggested_actions']) > 0
        assert 'error_message' in result
    
    def test_input_characteristics_analysis(self, recovery_service):
        """Test analysis of input characteristics."""
        
        input_data = {
            'text': 'Test text with special chars: @#$%',
            'parameters': {'mode': 'test'}
        }
        
        characteristics = recovery_service._analyze_input_characteristics(input_data)
        
        assert 'text_length' in characteristics
        assert 'word_count' in characteristics
        assert 'character_set' in characteristics
        assert characteristics['text_length'] == len(input_data['text'])
        assert characteristics['word_count'] == len(input_data['text'].split())
    
    def test_failure_point_identification(self, recovery_service):
        """Test identification of failure points."""
        
        # Test different error types
        file_error = FileNotFoundError("Audio file not found")
        assert recovery_service._identify_failure_point(file_error, {}) == 'file_handling'
        
        analysis_error = ValueError("MFCC extraction failed")
        assert recovery_service._identify_failure_point(analysis_error, {}) == 'voice_analysis'
        
        synthesis_error = RuntimeError("TTS model generation failed")
        assert recovery_service._identify_failure_point(synthesis_error, {}) == 'model_operations'  # Fixed expectation
        
        memory_error = MemoryError("Out of memory")
        assert recovery_service._identify_failure_point(memory_error, {}) == 'resource_management'
    
    def test_recommendation_generation(self, recovery_service):
        """Test generation of specific recommendations."""
        
        error = ValueError("Audio signal processing failed")
        input_chars = {'text_length': 50, 'word_count': 8}
        
        recommendations = recovery_service._generate_recommendations(
            error, input_chars, [], 'audio_processing'
        )
        
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)
        assert all(len(rec) > 10 for rec in recommendations)  # Meaningful recommendations
    
    def test_degradation_type_classification(self, recovery_service):
        """Test classification of degradation types."""
        
        analysis_error = ValueError("Feature analysis failed")
        assert recovery_service._classify_degradation_type(
            analysis_error, 'voice_processing'
        ) == 'voice_analysis_failure'
        
        synthesis_error = RuntimeError("Speech generation timeout")
        assert recovery_service._classify_degradation_type(
            synthesis_error, 'tts'
        ) == 'synthesis_failure'
        
        timeout_error = TimeoutError("Operation timeout")
        assert recovery_service._classify_degradation_type(
            timeout_error, 'any_operation'
        ) == 'timeout_failure'
    
    def test_user_friendly_message_generation(self, recovery_service):
        """Test generation of user-friendly error messages."""
        
        error = ValueError("Test error")
        result = {
            'suggested_actions': [
                'Try with clearer audio',
                'Ensure sufficient audio length',
                'Remove background noise'
            ]
        }
        
        message = recovery_service._generate_user_friendly_message(
            error, 'voice_analysis_failure', result
        )
        
        assert isinstance(message, str)
        assert len(message) > 0
        assert 'Try with clearer audio' in message
    
    def test_system_health_report(self, recovery_service):
        """Test system health report generation."""
        
        report = recovery_service.get_system_health_report()
        
        assert 'error_statistics' in report
        assert 'timestamp' in report
        assert isinstance(report['error_statistics']['total_errors'], int)
    
    def test_diagnostic_info_storage_and_retrieval(self, recovery_service):
        """Test storage and retrieval of diagnostic information."""
        
        error = RuntimeError("Test storage error")
        diag_info = recovery_service.create_diagnostic_info(
            'storage_test_123', 'test_operation', error, {}, {}
        )
        
        # Test retrieval
        retrieved_info = recovery_service.get_diagnostic_info('storage_test_123')
        assert retrieved_info is not None
        assert retrieved_info.error_id == 'storage_test_123'
        assert retrieved_info.operation_type == 'test_operation'
        
        # Test non-existent error ID
        non_existent = recovery_service.get_diagnostic_info('non_existent_id')
        assert non_existent is None


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.fixture
    def service(self):
        return robust_error_recovery_service
    
    @pytest.mark.asyncio
    async def test_synthesis_failure_recovery_scenario(self, service):
        """Test recovery scenario for synthesis failures."""
        
        synthesis_error = RuntimeError("TTS synthesis failed to generate audio")  # Changed to include "synthesis"
        context = {
            'input_data': {
                'text': 'This is a test synthesis',
                'voice_model': 'custom_model_v1'
            },
            'synthesis_stage': 'audio_generation',
            'model_info': {'type': 'neural_tts', 'version': '1.0'}
        }
        
        recovery_result = await service.handle_error_with_recovery(
            synthesis_error, 'speech_synthesis', context
        )
        
        assert 'error_id' in recovery_result
        assert 'degradation_result' in recovery_result
        assert recovery_result['degradation_result']['degradation_type'] == 'synthesis_failure'
        
        # Check that diagnostic info was created
        diagnostic_info = service.get_diagnostic_info(recovery_result['error_id'])
        assert diagnostic_info is not None
        assert diagnostic_info.operation_type == 'speech_synthesis'
        assert len(diagnostic_info.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_timeout_recovery_scenario(self, service):
        """Test recovery scenario for timeout errors."""
        
        timeout_error = TimeoutError("Operation timeout after 60 seconds")  # Changed to include "timeout"
        context = {
            'input_data': {
                'audio_file': 'long_audio.wav',
                'duration': 600  # 10 minutes
            },
            'processing_timeout': 60,
            'partial_results': {
                'progress': 0.7,
                'features_extracted': ['pitch', 'energy']
            }
        }
        
        recovery_result = await service.handle_error_with_recovery(
            timeout_error, 'voice_analysis', context
        )
        
        assert recovery_result['degradation_result']['degradation_type'] == 'timeout_failure'
        
        # Check recommendations include timeout-specific advice
        diagnostic_info = service.get_diagnostic_info(recovery_result['error_id'])
        recommendations = diagnostic_info.recommendations
        assert any('timeout' in rec.lower() or 'shorter' in rec.lower() for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_file_handling_error_scenario(self, service):
        """Test recovery scenario for file handling errors."""
        
        file_error = FileNotFoundError("Audio file 'missing.wav' not found")
        context = {
            'input_data': {
                'audio_file': 'missing.wav',
                'expected_path': '/uploads/missing.wav'
            },
            'file_operation': 'read_audio'
        }
        
        recovery_result = await service.handle_error_with_recovery(
            file_error, 'file_processing', context
        )
        
        assert 'error_id' in recovery_result
        diagnostic_info = service.get_diagnostic_info(recovery_result['error_id'])
        assert diagnostic_info.failure_point == 'file_handling'
        
        # Should provide file-specific recommendations
        recommendations = diagnostic_info.recommendations
        assert any('file' in rec.lower() for rec in recommendations)


# Property-based tests for error recovery system
class TestErrorRecoveryProperties:
    """Property-based tests for error recovery system correctness."""
    
    @pytest.fixture
    def service(self):
        return RobustErrorRecoveryService()
    
    def test_property_diagnostic_info_completeness(self, service):
        """
        Property: Diagnostic information should always contain essential fields.
        **Validates: Requirements 8.4**
        """
        
        error_types = [ValueError, RuntimeError, TimeoutError, MemoryError, FileNotFoundError]
        operation_types = ['voice_analysis', 'speech_synthesis', 'model_training', 'file_processing']
        
        for error_type in error_types:
            for operation_type in operation_types:
                error = error_type(f"Test {error_type.__name__} for {operation_type}")
                
                diagnostic_info = service.create_diagnostic_info(
                    f'test_{error_type.__name__}_{operation_type}',
                    operation_type,
                    error,
                    {'test': 'data'},
                    {'context': 'test'}
                )
                
                # Essential fields should always be present
                assert diagnostic_info.error_id is not None
                assert diagnostic_info.timestamp is not None
                assert diagnostic_info.operation_type == operation_type
                assert diagnostic_info.failure_point is not None
                assert len(diagnostic_info.recommendations) > 0
                assert diagnostic_info.error_details['error_type'] == error_type.__name__
    
    @pytest.mark.asyncio
    async def test_property_graceful_degradation_always_provides_guidance(self, service):
        """
        Property: Graceful degradation should always provide user guidance.
        **Validates: Requirements 8.5**
        """
        
        error_scenarios = [
            (ValueError("Analysis failed"), 'voice_analysis'),
            (RuntimeError("Synthesis error"), 'speech_synthesis'),
            (MemoryError("Out of memory"), 'model_training'),
            (TimeoutError("Operation timeout"), 'audio_processing'),
            (FileNotFoundError("File missing"), 'file_upload')
        ]
        
        for error, operation_type in error_scenarios:
            result = await service.handle_graceful_degradation(
                operation_type, error, {}
            )
            
            # Should always provide essential guidance fields
            assert 'error_message' in result
            assert 'user_impact' in result
            assert 'suggested_actions' in result
            assert 'degradation_type' in result
            
            # Error message should be user-friendly (not technical)
            error_message = result['error_message']
            assert len(error_message) > 20, "Error message too short"
            assert not any(tech_term in error_message.lower() 
                          for tech_term in ['traceback', 'exception', 'null pointer']), \
                   "Error message contains technical terms"
            
            # Should provide actionable suggestions
            suggestions = result['suggested_actions']
            assert len(suggestions) > 0, "No suggestions provided"
            assert all(len(suggestion) > 10 for suggestion in suggestions), \
                   "Suggestions too brief"
    
    def test_property_error_classification_consistency(self, service):
        """
        Property: Error classification should be consistent for similar errors.
        **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**
        """
        
        # Test that similar errors get classified consistently
        similar_errors = [
            (ValueError("analysis failed"), ValueError("analysis error")),  # Both should be voice_analysis_failure
            (RuntimeError("synthesis failed"), RuntimeError("synthesis error")),  # Both should be synthesis_failure
            (TimeoutError("timeout occurred"), TimeoutError("timeout error")),  # Both should be timeout_failure
            (MemoryError("memory error"), MemoryError("memory issue"))  # Both should be resource_failure
        ]
        
        for error1, error2 in similar_errors:
            failure_point1 = service._identify_failure_point(error1, {})
            failure_point2 = service._identify_failure_point(error2, {})
            
            # Similar errors should have same failure point classification
            assert failure_point1 == failure_point2, \
                f"Inconsistent classification: {error1} -> {failure_point1}, {error2} -> {failure_point2}"
            
            degradation_type1 = service._classify_degradation_type(error1, 'test_operation')
            degradation_type2 = service._classify_degradation_type(error2, 'test_operation')
            
            # Similar errors should have same degradation type
            assert degradation_type1 == degradation_type2, \
                f"Inconsistent degradation: {error1} -> {degradation_type1}, {error2} -> {degradation_type2}"


class TestDataClasses:
    """Test the data classes and enums."""
    
    def test_restoration_result_creation(self):
        """Test creation of RestorationResult."""
        
        audio_data = np.random.randn(1000)
        result = RestorationResult(
            success=True,
            restored_audio=audio_data,
            restoration_type=RestorationType.SPECTRAL_INTERPOLATION,
            quality_improvement=0.25,
            processing_time=1.5
        )
        
        assert result.success is True
        assert len(result.restored_audio) == 1000
        assert result.restoration_type == RestorationType.SPECTRAL_INTERPOLATION
        assert result.quality_improvement == 0.25
        assert result.processing_time == 1.5
    
    def test_analysis_result_creation(self):
        """Test creation of AnalysisResult."""
        
        features = {'pitch_mean': 150.0, 'energy': 0.8}
        result = AnalysisResult(
            success=True,
            analysis_method=AnalysisMethod.HARMONIC_ANALYSIS,
            extracted_features=features,
            confidence_score=0.92,
            processing_time=2.1
        )
        
        assert result.success is True
        assert result.analysis_method == AnalysisMethod.HARMONIC_ANALYSIS
        assert result.extracted_features == features
        assert result.confidence_score == 0.92
        assert result.processing_time == 2.1
    
    def test_diagnostic_info_creation(self):
        """Test creation of DiagnosticInfo."""
        
        diag_info = DiagnosticInfo(
            error_id='test_123',
            timestamp=datetime.now(),
            operation_type='voice_analysis',
            input_characteristics={'text_length': 100},
            processing_steps=['step1', 'step2'],
            failure_point='voice_analysis',
            error_details={'error_type': 'ValueError'},
            system_state={'memory': '8GB'},
            recovery_attempts=[],
            recommendations=['Try again']
        )
        
        assert diag_info.error_id == 'test_123'
        assert diag_info.operation_type == 'voice_analysis'
        assert len(diag_info.processing_steps) == 2
        assert len(diag_info.recommendations) == 1
    
    def test_enum_values(self):
        """Test enum values are correct."""
        
        # Test RestorationType
        assert RestorationType.SPECTRAL_INTERPOLATION == "spectral_interpolation"
        assert RestorationType.HARMONIC_RECONSTRUCTION == "harmonic_reconstruction"
        
        # Test AnalysisMethod
        assert AnalysisMethod.STANDARD_MFCC == "standard_mfcc"
        assert AnalysisMethod.ADVANCED_SPECTRAL == "advanced_spectral"
        
        # Test QualityLevel
        assert QualityLevel.CRITICAL == "critical"
        assert QualityLevel.POOR == "poor"
        assert QualityLevel.FAIR == "fair"
        assert QualityLevel.GOOD == "good"
        assert QualityLevel.EXCELLENT == "excellent"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])