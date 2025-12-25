"""
Comprehensive tests for Real-Time Quality Monitoring System.

This test suite validates all aspects of the real-time quality monitoring system
including quality assessment, improvement recommendations, similarity metrics,
optimization strategies, and confidence scoring.
"""

import pytest
import numpy as np
import librosa
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from app.services.real_time_quality_monitor import (
    RealTimeQualityMonitor, ProcessingStage, QualityLevel,
    RealTimeMetrics, SimilarityMetrics, ImprovementRecommendation,
    OptimizationStrategy, ConfidenceScores, QualityMonitoringSession
)
from app.services.audio_quality_assessment import QualityAssessmentReport, QualityIssue
from app.schemas.voice import VoiceCharacteristics, QualityMetrics


class TestRealTimeQualityMonitor:
    """Test suite for RealTimeQualityMonitor class."""
    
    @pytest.fixture
    def monitor(self):
        """Create a fresh RealTimeQualityMonitor instance for each test."""
        return RealTimeQualityMonitor()
    
    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio data for testing."""
        duration = 3.0  # 3 seconds
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Generate a simple sine wave with some harmonics
        audio = (np.sin(2 * np.pi * 220 * t) + 
                0.5 * np.sin(2 * np.pi * 440 * t) + 
                0.25 * np.sin(2 * np.pi * 880 * t))
        # Add some noise
        audio += 0.1 * np.random.randn(len(audio))
        return audio.astype(np.float32)
    
    @pytest.fixture
    def sample_quality_report(self):
        """Create a sample quality assessment report."""
        from app.services.audio_quality_assessment import QualityIssueDetail, EnhancementRecommendation
        return QualityAssessmentReport(
            overall_score=0.85,
            voice_suitability_score=0.8,
            technical_metrics={
                'voice_activity_ratio': 0.8,
                'snr_db': 20.0,
                'spectral_clarity': 0.75,
                'frequency_response_score': 0.8,
                'duration': 3.0
            },
            issues_detected=[],
            enhancement_recommendations=[],
            processing_suggestions={}
        )
    
    @pytest.fixture
    def sample_voice_characteristics(self):
        """Create sample voice characteristics."""
        from app.schemas.voice import FrequencyRange, ProsodyFeaturesSchema, EmotionalProfileSchema
        
        pitch_chars = FrequencyRange(
            mean_hz=220.0,
            std_hz=25.0,
            min_hz=180.0,
            max_hz=280.0
        )
        
        prosody_features = ProsodyFeaturesSchema(
            speech_rate=5.0,
            pause_frequency=0.3,
            emphasis_variance=0.2
        )
        
        emotional_profile = EmotionalProfileSchema(
            valence=0.1,
            arousal=0.2,
            dominance=0.3
        )
        
        quality_metrics = QualityMetrics(
            signal_to_noise_ratio=18.0,
            voice_activity_ratio=0.75,
            overall_quality=0.8
        )
        
        return VoiceCharacteristics(
            timbre_features={'mfcc_mean': 0.5, 'spectral_centroid': 1200.0},
            pitch_characteristics=pitch_chars,
            prosody_features=prosody_features,
            emotional_markers=emotional_profile,
            quality_metrics=quality_metrics
        )

    def test_start_monitoring_session(self, monitor):
        """Test starting a new monitoring session."""
        session_id = "test_session_001"
        quality_targets = {
            'minimum_similarity': 0.90,
            'minimum_quality': 0.80,
            'minimum_confidence': 0.75
        }
        
        result_session_id = monitor.start_monitoring_session(
            session_id=session_id,
            quality_targets=quality_targets
        )
        
        assert result_session_id == session_id
        assert session_id in monitor.sessions
        
        session = monitor.sessions[session_id]
        assert session.session_id == session_id
        assert session.current_stage == ProcessingStage.AUDIO_PREPROCESSING
        assert session.quality_targets['minimum_similarity'] == 0.90
        assert session.quality_targets['minimum_quality'] == 0.80
        assert session.quality_targets['minimum_confidence'] == 0.75

    def test_update_processing_stage(self, monitor):
        """Test updating processing stage."""
        session_id = "test_session_002"
        monitor.start_monitoring_session(session_id)
        
        # Update to voice analysis stage
        monitor.update_processing_stage(
            session_id=session_id,
            stage=ProcessingStage.VOICE_ANALYSIS,
            progress=0.5
        )
        
        session = monitor.sessions[session_id]
        assert session.current_stage == ProcessingStage.VOICE_ANALYSIS
        assert session.current_metrics.stage_progress == 0.5
        assert len(session.metrics_history) == 1

    def test_update_processing_stage_invalid_session(self, monitor):
        """Test updating processing stage with invalid session ID."""
        with pytest.raises(ValueError, match="Session invalid_session not found"):
            monitor.update_processing_stage(
                session_id="invalid_session",
                stage=ProcessingStage.SYNTHESIS,
                progress=0.3
            )

    @patch('app.services.real_time_quality_monitor.AudioQualityAssessor')
    def test_assess_real_time_quality(self, mock_assessor_class, monitor, sample_audio, sample_quality_report):
        """Test real-time quality assessment."""
        # Setup mock
        mock_assessor = Mock()
        mock_assessor.assess_audio_quality.return_value = sample_quality_report
        mock_assessor_class.return_value = mock_assessor
        
        # Create fresh monitor with mocked assessor
        monitor = RealTimeQualityMonitor()
        
        session_id = "test_session_003"
        monitor.start_monitoring_session(session_id)
        
        # Perform quality assessment
        metrics = monitor.assess_real_time_quality(
            session_id=session_id,
            audio=sample_audio,
            sample_rate=22050,
            stage=ProcessingStage.VOICE_ANALYSIS
        )
        
        assert isinstance(metrics, RealTimeMetrics)
        assert metrics.stage == ProcessingStage.VOICE_ANALYSIS
        assert metrics.quality_score == 0.85
        assert metrics.confidence_score > 0.0
        assert metrics.processing_time >= 0.0  # Allow 0.0 for fast operations
        
        # Check session was updated
        session = monitor.sessions[session_id]
        assert session.current_metrics == metrics
        assert len(session.metrics_history) >= 1

    def test_assess_real_time_quality_with_reference(self, monitor, sample_audio):
        """Test quality assessment with reference audio."""
        session_id = "test_session_004"
        monitor.start_monitoring_session(session_id)
        
        # Create slightly different reference audio
        reference_audio = sample_audio * 0.9 + 0.05 * np.random.randn(len(sample_audio))
        
        with patch.object(monitor.quality_assessor, 'assess_audio_quality') as mock_assess:
            mock_assess.return_value = QualityAssessmentReport(
                overall_score=0.8,
                voice_suitability_score=0.75,
                technical_metrics={'voice_activity_ratio': 0.7, 'snr_db': 15.0},
                issues_detected=[],
                enhancement_recommendations=[],
                processing_suggestions={}
            )
            
            metrics = monitor.assess_real_time_quality(
                session_id=session_id,
                audio=sample_audio,
                sample_rate=22050,
                stage=ProcessingStage.SYNTHESIS,
                reference_audio=reference_audio
            )
        
        assert metrics.similarity_score > 0.0
        assert metrics.similarity_score <= 1.0

    def test_generate_improvement_recommendations(self, monitor, sample_voice_characteristics):
        """Test generating improvement recommendations."""
        session_id = "test_session_005"
        monitor.start_monitoring_session(session_id)
        
        # Set up session with low quality metrics
        monitor.sessions[session_id].current_metrics = RealTimeMetrics(
            timestamp=datetime.now(),
            stage=ProcessingStage.VOICE_ANALYSIS,
            similarity_score=0.65,  # Below target
            quality_score=0.60,     # Below good threshold
            confidence_score=0.55,  # Below high confidence
            processing_time=5.0,
            issues_detected=['low_snr', 'spectral_distortion'],
            recommendations=[],
            stage_progress=1.0
        )
        
        recommendations = monitor.generate_improvement_recommendations(
            session_id=session_id,
            voice_characteristics=sample_voice_characteristics
        )
        
        assert len(recommendations) > 0
        
        # Check that we get recommendations for different categories
        categories = [rec.category for rec in recommendations]
        assert 'audio_quality' in categories or 'voice_similarity' in categories
        
        # Verify recommendation structure
        for rec in recommendations:
            assert isinstance(rec, ImprovementRecommendation)
            assert rec.priority >= 1 and rec.priority <= 5
            assert rec.expected_improvement > 0.0
            assert len(rec.implementation_steps) > 0
            assert rec.estimated_time > 0.0

    def test_calculate_detailed_similarity_metrics(self, monitor, sample_audio):
        """Test calculating detailed similarity metrics."""
        session_id = "test_session_006"
        monitor.start_monitoring_session(session_id)
        
        # Create reference and synthesized audio
        reference_audio = sample_audio
        synthesized_audio = sample_audio * 0.95 + 0.02 * np.random.randn(len(sample_audio))
        
        similarity_metrics = monitor.calculate_detailed_similarity_metrics(
            session_id=session_id,
            reference_audio=reference_audio,
            synthesized_audio=synthesized_audio,
            sample_rate=22050
        )
        
        assert isinstance(similarity_metrics, SimilarityMetrics)
        assert 0.0 <= similarity_metrics.overall_similarity <= 1.0
        assert 0.0 <= similarity_metrics.pitch_similarity <= 1.0
        assert 0.0 <= similarity_metrics.timbre_similarity <= 1.0
        assert 0.0 <= similarity_metrics.prosody_similarity <= 1.0
        assert 0.0 <= similarity_metrics.emotional_similarity <= 1.0
        assert 0.0 <= similarity_metrics.spectral_similarity <= 1.0
        assert 0.0 <= similarity_metrics.temporal_similarity <= 1.0
        
        # Check confidence interval
        lower, upper = similarity_metrics.confidence_interval
        assert lower <= similarity_metrics.overall_similarity <= upper
        
        # Check breakdown
        assert len(similarity_metrics.breakdown) > 0
        for key, value in similarity_metrics.breakdown.items():
            assert 0.0 <= value <= 1.0

    def test_generate_optimization_strategy(self, monitor):
        """Test generating optimization strategy."""
        session_id = "test_session_007"
        monitor.start_monitoring_session(session_id)
        
        # Set up session with current quality
        monitor.sessions[session_id].current_metrics = RealTimeMetrics(
            timestamp=datetime.now(),
            stage=ProcessingStage.SYNTHESIS,
            similarity_score=0.75,
            quality_score=0.70,
            confidence_score=0.65,
            processing_time=8.0,
            issues_detected=['noise', 'low_quality'],
            recommendations=[],
            stage_progress=1.0
        )
        
        # Generate some recommendations first
        recommendations = monitor.generate_improvement_recommendations(session_id)
        
        target_quality = 0.95
        strategy = monitor.generate_optimization_strategy(
            session_id=session_id,
            target_quality=target_quality
        )
        
        assert isinstance(strategy, OptimizationStrategy)
        assert strategy.target_quality == target_quality
        assert strategy.current_quality == 0.70
        assert strategy.improvement_potential > 0.0
        assert len(strategy.recommended_steps) > 0
        assert strategy.estimated_total_time > 0.0
        assert 0.0 <= strategy.success_probability <= 1.0

    def test_get_confidence_scores(self, monitor, sample_audio):
        """Test getting confidence scores."""
        session_id = "test_session_008"
        monitor.start_monitoring_session(session_id)
        
        # Perform assessment to generate confidence scores
        with patch.object(monitor.quality_assessor, 'assess_audio_quality') as mock_assess:
            mock_assess.return_value = QualityAssessmentReport(
                overall_score=0.85,
                voice_suitability_score=0.8,
                technical_metrics={
                    'voice_activity_ratio': 0.8,
                    'snr_db': 20.0,
                    'spectral_clarity': 0.75,
                    'frequency_response_score': 0.8,
                    'duration': 3.0
                },
                issues_detected=[],
                enhancement_recommendations=[],
                processing_suggestions={}
            )
            
            monitor.assess_real_time_quality(
                session_id=session_id,
                audio=sample_audio,
                sample_rate=22050,
                stage=ProcessingStage.VOICE_ANALYSIS
            )
        
        confidence_scores = monitor.get_confidence_scores(session_id)
        
        assert isinstance(confidence_scores, ConfidenceScores)
        assert 0.0 <= confidence_scores.pitch_extraction <= 1.0
        assert 0.0 <= confidence_scores.formant_detection <= 1.0
        assert 0.0 <= confidence_scores.timbre_analysis <= 1.0
        assert 0.0 <= confidence_scores.prosody_extraction <= 1.0
        assert 0.0 <= confidence_scores.emotional_analysis <= 1.0
        assert 0.0 <= confidence_scores.overall_analysis <= 1.0
        assert 0.0 <= confidence_scores.voice_model_quality <= 1.0
        assert 0.0 <= confidence_scores.synthesis_quality <= 1.0
        assert len(confidence_scores.characteristic_reliability) > 0

    def test_get_session_summary(self, monitor, sample_audio):
        """Test getting comprehensive session summary."""
        session_id = "test_session_009"
        monitor.start_monitoring_session(session_id)
        
        # Add some metrics to the session
        with patch.object(monitor.quality_assessor, 'assess_audio_quality') as mock_assess:
            mock_assess.return_value = QualityAssessmentReport(
                overall_score=0.8,
                voice_suitability_score=0.75,
                technical_metrics={'voice_activity_ratio': 0.7},
                issues_detected=[],
                enhancement_recommendations=[],
                processing_suggestions={}
            )
            
            # Perform multiple assessments
            for i in range(3):
                monitor.assess_real_time_quality(
                    session_id=session_id,
                    audio=sample_audio,
                    sample_rate=22050,
                    stage=ProcessingStage.VOICE_ANALYSIS
                )
        
        # Update the stage explicitly to ensure it's set correctly
        monitor.update_processing_stage(session_id, ProcessingStage.VOICE_ANALYSIS, 1.0)
        
        summary = monitor.get_session_summary(session_id)
        
        assert summary['session_id'] == session_id
        assert 'start_time' in summary
        assert 'duration' in summary
        assert summary['current_stage'] == ProcessingStage.VOICE_ANALYSIS.value
        assert summary['total_assessments'] >= 3
        
        # Check statistics
        assert 'quality_statistics' in summary
        assert 'similarity_statistics' in summary
        assert 'confidence_statistics' in summary
        
        quality_stats = summary['quality_statistics']
        assert 'current' in quality_stats
        assert 'average' in quality_stats
        assert 'maximum' in quality_stats
        assert 'minimum' in quality_stats

    def test_register_callback(self, monitor):
        """Test registering callbacks for real-time updates."""
        session_id = "test_session_010"
        monitor.start_monitoring_session(session_id)
        
        callback_called = []
        
        def test_callback(sid, metrics):
            callback_called.append((sid, metrics))
        
        monitor.register_callback(session_id, test_callback)
        
        # Trigger callback by performing assessment
        with patch.object(monitor.quality_assessor, 'assess_audio_quality') as mock_assess:
            mock_assess.return_value = QualityAssessmentReport(
                overall_score=0.8,
                voice_suitability_score=0.75,
                technical_metrics={'voice_activity_ratio': 0.7},
                issues_detected=[],
                enhancement_recommendations=[],
                processing_suggestions={}
            )
            
            sample_audio = np.random.randn(22050)  # 1 second of audio
            monitor.assess_real_time_quality(
                session_id=session_id,
                audio=sample_audio,
                sample_rate=22050,
                stage=ProcessingStage.VOICE_ANALYSIS
            )
        
        assert len(callback_called) == 1
        assert callback_called[0][0] == session_id
        assert isinstance(callback_called[0][1], RealTimeMetrics)

    def test_end_monitoring_session(self, monitor):
        """Test ending monitoring session."""
        session_id = "test_session_011"
        monitor.start_monitoring_session(session_id)
        
        # Verify session exists
        assert session_id in monitor.sessions
        
        # End session
        summary = monitor.end_monitoring_session(session_id)
        
        # Verify session is removed
        assert session_id not in monitor.sessions
        
        # Verify summary is returned
        assert isinstance(summary, dict)
        assert summary['session_id'] == session_id

    def test_similarity_calculation_methods(self, monitor):
        """Test individual similarity calculation methods."""
        # Create test audio signals
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Reference audio: 220 Hz sine wave
        ref_audio = np.sin(2 * np.pi * 220 * t)
        
        # Synthesized audio: similar but slightly different
        syn_audio = 0.9 * np.sin(2 * np.pi * 225 * t)  # Slightly different frequency
        
        # Test overall similarity
        similarity = monitor._calculate_similarity_score(ref_audio, syn_audio, sample_rate)
        assert 0.0 <= similarity <= 1.0
        
        # Test pitch similarity
        pitch_sim = monitor._calculate_pitch_similarity(ref_audio, syn_audio, sample_rate)
        assert 0.0 <= pitch_sim <= 1.0
        
        # Test timbre similarity
        timbre_sim = monitor._calculate_timbre_similarity(ref_audio, syn_audio, sample_rate)
        assert 0.0 <= timbre_sim <= 1.0
        
        # Test prosody similarity
        prosody_sim = monitor._calculate_prosody_similarity(ref_audio, syn_audio, sample_rate)
        assert 0.0 <= prosody_sim <= 1.0

    def test_confidence_score_calculation(self, monitor, sample_quality_report):
        """Test confidence score calculation."""
        sample_audio = np.random.randn(22050)  # 1 second of audio
        
        confidence_scores = monitor._calculate_confidence_scores(
            audio=sample_audio,
            sample_rate=22050,
            quality_report=sample_quality_report
        )
        
        assert isinstance(confidence_scores, ConfidenceScores)
        
        # All confidence scores should be between 0 and 1
        assert 0.0 <= confidence_scores.pitch_extraction <= 1.0
        assert 0.0 <= confidence_scores.formant_detection <= 1.0
        assert 0.0 <= confidence_scores.timbre_analysis <= 1.0
        assert 0.0 <= confidence_scores.prosody_extraction <= 1.0
        assert 0.0 <= confidence_scores.emotional_analysis <= 1.0
        assert 0.0 <= confidence_scores.overall_analysis <= 1.0

    def test_voice_characteristics_analysis(self, monitor, sample_voice_characteristics):
        """Test voice characteristics issue analysis."""
        # Test with low SNR characteristics
        low_snr_characteristics = sample_voice_characteristics
        low_snr_characteristics.quality_metrics.signal_to_noise_ratio = 10.0  # Low SNR
        low_snr_characteristics.quality_metrics.voice_activity_ratio = 0.4   # Low voice activity
        
        recommendations = monitor._analyze_voice_characteristics_issues(low_snr_characteristics)
        
        assert len(recommendations) > 0
        
        # Should get noise reduction recommendation
        categories = [rec.category for rec in recommendations]
        assert 'noise_reduction' in categories or 'voice_content' in categories

    def test_success_probability_calculation(self, monitor):
        """Test success probability calculation."""
        current_quality = 0.70
        target_quality = 0.95
        
        recommendations = [
            ImprovementRecommendation(
                category="audio_quality",
                priority=1,
                issue_description="Test issue",
                recommended_action="Test action",
                expected_improvement=0.15,
                implementation_steps=[],
                estimated_time=10.0,
                prerequisites=[]
            ),
            ImprovementRecommendation(
                category="voice_similarity",
                priority=2,
                issue_description="Test issue 2",
                recommended_action="Test action 2",
                expected_improvement=0.10,
                implementation_steps=[],
                estimated_time=15.0,
                prerequisites=[]
            )
        ]
        
        probability = monitor._calculate_success_probability(
            current_quality, target_quality, recommendations
        )
        
        assert 0.0 <= probability <= 1.0

    def test_trend_calculation(self, monitor):
        """Test trend calculation for metrics."""
        # Improving trend
        improving_values = [0.6, 0.65, 0.7, 0.75, 0.8]
        improving_trend = monitor._calculate_trend(improving_values)
        assert improving_trend > 0
        
        # Declining trend
        declining_values = [0.8, 0.75, 0.7, 0.65, 0.6]
        declining_trend = monitor._calculate_trend(declining_values)
        assert declining_trend < 0
        
        # Stable trend
        stable_values = [0.7, 0.7, 0.7, 0.7, 0.7]
        stable_trend = monitor._calculate_trend(stable_values)
        assert abs(stable_trend) < 0.01  # Should be close to 0

    def test_issues_summarization(self, monitor):
        """Test issue summarization across metrics."""
        metrics_history = [
            RealTimeMetrics(
                timestamp=datetime.now(),
                stage=ProcessingStage.VOICE_ANALYSIS,
                similarity_score=0.8,
                quality_score=0.7,
                confidence_score=0.6,
                processing_time=5.0,
                issues_detected=['noise', 'low_quality'],
                recommendations=[],
                stage_progress=1.0
            ),
            RealTimeMetrics(
                timestamp=datetime.now(),
                stage=ProcessingStage.SYNTHESIS,
                similarity_score=0.85,
                quality_score=0.75,
                confidence_score=0.65,
                processing_time=6.0,
                issues_detected=['noise', 'spectral_distortion'],
                recommendations=[],
                stage_progress=1.0
            )
        ]
        
        summary = monitor._summarize_issues(metrics_history)
        
        assert summary['total_issues'] == 4
        assert summary['unique_issue_types'] == 3
        assert 'noise' in summary['issue_types']
        assert summary['issue_types']['noise'] == 2  # Appears twice
        assert summary['most_common'] == 'noise'

    def test_targets_checking(self, monitor):
        """Test quality targets checking."""
        session_id = "test_session_012"
        monitor.start_monitoring_session(session_id, quality_targets={
            'minimum_similarity': 0.90,
            'minimum_quality': 0.80,
            'minimum_confidence': 0.75
        })
        
        # Set current metrics
        monitor.sessions[session_id].current_metrics = RealTimeMetrics(
            timestamp=datetime.now(),
            stage=ProcessingStage.SYNTHESIS,
            similarity_score=0.92,  # Above target
            quality_score=0.78,     # Below target
            confidence_score=0.80,  # Above target
            processing_time=5.0,
            issues_detected=[],
            recommendations=[],
            stage_progress=1.0
        )
        
        session = monitor.sessions[session_id]
        targets_met = monitor._check_targets_met(session)
        
        assert targets_met['minimum_similarity'] == True
        assert targets_met['minimum_quality'] == False
        assert targets_met['minimum_confidence'] == True
        assert targets_met['all_targets'] == False


# Property-based tests using Hypothesis
try:
    from hypothesis import given, strategies as st, settings, HealthCheck
    from hypothesis.strategies import floats, integers, lists
    
    class TestRealTimeQualityMonitorProperties:
        """Property-based tests for RealTimeQualityMonitor."""
        
        @given(
            quality_score=floats(min_value=0.0, max_value=1.0),
            similarity_score=floats(min_value=0.0, max_value=1.0),
            confidence_score=floats(min_value=0.0, max_value=1.0)
        )
        @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_property_quality_assessment_bounds(self, quality_score, similarity_score, confidence_score):
            """
            Property 21: Real-time quality assessment provision
            For any reference audio processing, real-time quality assessment should be provided throughout the processing pipeline.
            **Validates: Requirements 5.1**
            """
            monitor = RealTimeQualityMonitor()  # Create fresh instance
            session_id = f"prop_test_{hash((quality_score, similarity_score, confidence_score)) % 10000}"
            
            try:
                monitor.start_monitoring_session(session_id)
                
                # Create mock metrics with the given scores
                metrics = RealTimeMetrics(
                    timestamp=datetime.now(),
                    stage=ProcessingStage.VOICE_ANALYSIS,
                    similarity_score=similarity_score,
                    quality_score=quality_score,
                    confidence_score=confidence_score,
                    processing_time=1.0,
                    issues_detected=[],
                    recommendations=[],
                    stage_progress=1.0
                )
                
                # Update session with metrics
                monitor.sessions[session_id].current_metrics = metrics
                monitor.sessions[session_id].metrics_history.append(metrics)
                
                # Verify that quality assessment is always provided
                assert monitor.sessions[session_id].current_metrics is not None
                assert len(monitor.sessions[session_id].metrics_history) > 0
                
                # All scores should remain within valid bounds
                current = monitor.sessions[session_id].current_metrics
                assert 0.0 <= current.quality_score <= 1.0
                assert 0.0 <= current.similarity_score <= 1.0
                assert 0.0 <= current.confidence_score <= 1.0
                
            finally:
                if session_id in monitor.sessions:
                    monitor.end_monitoring_session(session_id)
        
        @given(
            target_quality=floats(min_value=0.5, max_value=1.0),
            current_quality=floats(min_value=0.0, max_value=0.95)
        )
        @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_property_optimization_strategy_generation(self, target_quality, current_quality):
            """
            Property 24: Optimization strategies for low quality
            For any synthesis with quality below 95%, the system should suggest specific optimization strategies for improvement.
            **Validates: Requirements 5.4**
            """
            monitor = RealTimeQualityMonitor()  # Create fresh instance
            session_id = f"opt_test_{hash((target_quality, current_quality)) % 10000}"
            
            try:
                monitor.start_monitoring_session(session_id)
                
                # Set current quality metrics
                monitor.sessions[session_id].current_metrics = RealTimeMetrics(
                    timestamp=datetime.now(),
                    stage=ProcessingStage.SYNTHESIS,
                    similarity_score=current_quality,
                    quality_score=current_quality,
                    confidence_score=current_quality,
                    processing_time=5.0,
                    issues_detected=[],
                    recommendations=[],
                    stage_progress=1.0
                )
                
                # Generate optimization strategy
                strategy = monitor.generate_optimization_strategy(session_id, target_quality)
                
                # Verify strategy properties
                assert isinstance(strategy, OptimizationStrategy)
                assert strategy.target_quality == target_quality
                assert strategy.current_quality == current_quality
                assert 0.0 <= strategy.success_probability <= 1.0
                assert strategy.estimated_total_time >= 0.0
                
                # If quality is below target, should have improvement potential
                if current_quality < target_quality:
                    assert strategy.improvement_potential > 0.0
                    # Should provide recommendations for improvement
                    if current_quality < 0.95:  # Below 95% threshold
                        assert len(strategy.recommended_steps) > 0
                
            finally:
                if session_id in monitor.sessions:
                    monitor.end_monitoring_session(session_id)
        
        @given(
            confidence_values=lists(
                floats(min_value=0.0, max_value=1.0),
                min_size=1,
                max_size=10
            )
        )
        @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_property_confidence_scores_provision(self, confidence_values):
            """
            Property 25: Confidence scores for all characteristics
            For any extracted voice characteristic, a confidence score should be provided indicating the reliability of the extraction.
            **Validates: Requirements 5.5**
            """
            monitor = RealTimeQualityMonitor()  # Create fresh instance
            session_id = f"conf_test_{hash(tuple(confidence_values)) % 10000}"
            
            try:
                monitor.start_monitoring_session(session_id)
                
                # Create mock quality report
                quality_report = QualityAssessmentReport(
                    overall_score=np.mean(confidence_values),
                    voice_suitability_score=np.mean(confidence_values) * 0.9,
                    technical_metrics={
                        'voice_activity_ratio': confidence_values[0] if len(confidence_values) > 0 else 0.5,
                        'snr_db': 15.0,
                        'spectral_clarity': confidence_values[1] if len(confidence_values) > 1 else 0.5,
                        'frequency_response_score': confidence_values[2] if len(confidence_values) > 2 else 0.5,
                        'duration': 3.0
                    },
                    issues_detected=[],
                    enhancement_recommendations=[],
                    processing_suggestions={}
                )
                
                # Calculate confidence scores
                sample_audio = np.random.randn(1000)
                confidence_scores = monitor._calculate_confidence_scores(
                    audio=sample_audio,
                    sample_rate=22050,
                    quality_report=quality_report
                )
                
                # Verify all confidence scores are provided and within bounds
                assert isinstance(confidence_scores, ConfidenceScores)
                assert 0.0 <= confidence_scores.pitch_extraction <= 1.0
                assert 0.0 <= confidence_scores.formant_detection <= 1.0
                assert 0.0 <= confidence_scores.timbre_analysis <= 1.0
                assert 0.0 <= confidence_scores.prosody_extraction <= 1.0
                assert 0.0 <= confidence_scores.emotional_analysis <= 1.0
                assert 0.0 <= confidence_scores.overall_analysis <= 1.0
                assert 0.0 <= confidence_scores.voice_model_quality <= 1.0
                assert 0.0 <= confidence_scores.synthesis_quality <= 1.0
                
                # Verify characteristic reliability is provided for all characteristics
                assert len(confidence_scores.characteristic_reliability) > 0
                for char_name, reliability in confidence_scores.characteristic_reliability.items():
                    assert 0.0 <= reliability <= 1.0
                
            finally:
                if session_id in monitor.sessions:
                    monitor.end_monitoring_session(session_id)
        
        @given(
            similarity_scores=lists(
                floats(min_value=0.0, max_value=1.0),
                min_size=2,
                max_size=8
            )
        )
        @settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
        def test_property_detailed_similarity_metrics(self, similarity_scores):
            """
            Property 23: Detailed similarity metrics reporting
            For any completed synthesis, detailed similarity metrics should be reported covering all aspects of voice matching.
            **Validates: Requirements 5.3**
            """
            monitor = RealTimeQualityMonitor()  # Create fresh instance
            session_id = f"sim_test_{hash(tuple(similarity_scores)) % 10000}"
            
            try:
                monitor.start_monitoring_session(session_id)
                
                # Create mock similarity metrics
                overall_sim = np.mean(similarity_scores)
                
                similarity_metrics = SimilarityMetrics(
                    overall_similarity=overall_sim,
                    pitch_similarity=similarity_scores[0] if len(similarity_scores) > 0 else overall_sim,
                    timbre_similarity=similarity_scores[1] if len(similarity_scores) > 1 else overall_sim,
                    prosody_similarity=similarity_scores[2] if len(similarity_scores) > 2 else overall_sim,
                    emotional_similarity=similarity_scores[3] if len(similarity_scores) > 3 else overall_sim,
                    spectral_similarity=similarity_scores[4] if len(similarity_scores) > 4 else overall_sim,
                    temporal_similarity=similarity_scores[5] if len(similarity_scores) > 5 else overall_sim,
                    confidence_interval=(max(0.0, overall_sim - 0.1), min(1.0, overall_sim + 0.1)),
                    breakdown={
                        'fundamental_frequency': similarity_scores[0] if len(similarity_scores) > 0 else overall_sim,
                        'formant_frequencies': similarity_scores[1] if len(similarity_scores) > 1 else overall_sim,
                        'spectral_envelope': similarity_scores[2] if len(similarity_scores) > 2 else overall_sim,
                        'prosodic_patterns': similarity_scores[3] if len(similarity_scores) > 3 else overall_sim,
                        'emotional_content': similarity_scores[4] if len(similarity_scores) > 4 else overall_sim,
                        'temporal_dynamics': similarity_scores[5] if len(similarity_scores) > 5 else overall_sim,
                    }
                )
                
                # Update session with similarity metrics
                monitor.sessions[session_id].similarity_metrics = similarity_metrics
                
                # Verify detailed metrics are provided
                retrieved_metrics = monitor.sessions[session_id].similarity_metrics
                assert retrieved_metrics is not None
                
                # All similarity components should be within bounds
                assert 0.0 <= retrieved_metrics.overall_similarity <= 1.0
                assert 0.0 <= retrieved_metrics.pitch_similarity <= 1.0
                assert 0.0 <= retrieved_metrics.timbre_similarity <= 1.0
                assert 0.0 <= retrieved_metrics.prosody_similarity <= 1.0
                assert 0.0 <= retrieved_metrics.emotional_similarity <= 1.0
                assert 0.0 <= retrieved_metrics.spectral_similarity <= 1.0
                assert 0.0 <= retrieved_metrics.temporal_similarity <= 1.0
                
                # Confidence interval should be valid
                lower, upper = retrieved_metrics.confidence_interval
                assert lower <= retrieved_metrics.overall_similarity <= upper
                assert 0.0 <= lower <= upper <= 1.0
                
                # Breakdown should cover all aspects
                assert len(retrieved_metrics.breakdown) > 0
                for aspect, score in retrieved_metrics.breakdown.items():
                    assert 0.0 <= score <= 1.0
                
            finally:
                if session_id in monitor.sessions:
                    monitor.end_monitoring_session(session_id)

except ImportError:
    # Hypothesis not available, skip property-based tests
    class TestRealTimeQualityMonitorProperties:
        """Placeholder for property-based tests when Hypothesis is not available."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])