"""
Real-Time Quality Monitoring System for Voice Cloning Pipeline.

This module provides comprehensive real-time quality assessment, improvement
recommendations, similarity metrics reporting, and confidence scoring for
all voice cloning operations.
"""

import numpy as np
import librosa
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
import json
from datetime import datetime, timedelta

from app.services.audio_quality_assessment import (
    AudioQualityAssessor, QualityAssessmentReport, QualityIssue, 
    EnhancementRecommendation
)
from app.schemas.voice import QualityMetrics, VoiceCharacteristics

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Processing stages for quality monitoring."""
    AUDIO_PREPROCESSING = "audio_preprocessing"
    VOICE_ANALYSIS = "voice_analysis"
    MODEL_TRAINING = "model_training"
    SYNTHESIS = "synthesis"
    POST_PROCESSING = "post_processing"


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class RealTimeMetrics:
    """Real-time quality metrics during processing."""
    timestamp: datetime
    stage: ProcessingStage
    similarity_score: float
    quality_score: float
    confidence_score: float
    processing_time: float
    issues_detected: List[str]
    recommendations: List[str]
    stage_progress: float  # 0.0 to 1.0


@dataclass
class SimilarityMetrics:
    """Detailed similarity metrics between reference and synthesized audio."""
    overall_similarity: float
    pitch_similarity: float
    timbre_similarity: float
    prosody_similarity: float
    emotional_similarity: float
    spectral_similarity: float
    temporal_similarity: float
    confidence_interval: Tuple[float, float]
    breakdown: Dict[str, float]


@dataclass
class ImprovementRecommendation:
    """Specific improvement recommendation with actionable steps."""
    category: str
    priority: int  # 1 (highest) to 5 (lowest)
    issue_description: str
    recommended_action: str
    expected_improvement: float
    implementation_steps: List[str]
    estimated_time: float
    prerequisites: List[str]


@dataclass
class OptimizationStrategy:
    """Optimization strategy for improving quality."""
    strategy_name: str
    target_quality: float
    current_quality: float
    improvement_potential: float
    recommended_steps: List[ImprovementRecommendation]
    estimated_total_time: float
    success_probability: float


@dataclass
class ConfidenceScores:
    """Confidence scores for all extracted characteristics."""
    pitch_extraction: float
    formant_detection: float
    timbre_analysis: float
    prosody_extraction: float
    emotional_analysis: float
    overall_analysis: float
    voice_model_quality: float
    synthesis_quality: float
    characteristic_reliability: Dict[str, float]


@dataclass
class QualityMonitoringSession:
    """Session data for quality monitoring."""
    session_id: str
    start_time: datetime
    current_stage: ProcessingStage
    metrics_history: List[RealTimeMetrics] = field(default_factory=list)
    current_metrics: Optional[RealTimeMetrics] = None
    similarity_metrics: Optional[SimilarityMetrics] = None
    confidence_scores: Optional[ConfidenceScores] = None
    recommendations: List[ImprovementRecommendation] = field(default_factory=list)
    optimization_strategy: Optional[OptimizationStrategy] = None
    quality_targets: Dict[str, float] = field(default_factory=dict)
    callbacks: List[Callable] = field(default_factory=list)


class RealTimeQualityMonitor:
    """
    Real-time quality monitoring system that provides continuous assessment,
    recommendations, and optimization strategies during voice cloning operations.
    """
    
    def __init__(self):
        """Initialize the real-time quality monitor."""
        self.sessions: Dict[str, QualityMonitoringSession] = {}
        self.session_lock = Lock()
        self.quality_assessor = AudioQualityAssessor()
        
        # Quality thresholds
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 0.95,
            QualityLevel.GOOD: 0.85,
            QualityLevel.ACCEPTABLE: 0.70,
            QualityLevel.POOR: 0.50,
            QualityLevel.UNACCEPTABLE: 0.30
        }
        
        # Similarity thresholds
        self.similarity_thresholds = {
            'target_similarity': 0.95,
            'minimum_acceptable': 0.70,
            'excellent_threshold': 0.98
        }
        
        # Confidence thresholds
        self.confidence_thresholds = {
            'high_confidence': 0.90,
            'medium_confidence': 0.70,
            'low_confidence': 0.50
        }
    
    def start_monitoring_session(self, session_id: str, 
                               quality_targets: Optional[Dict[str, float]] = None) -> str:
        """
        Start a new quality monitoring session.
        
        Args:
            session_id: Unique session identifier
            quality_targets: Optional quality targets for the session
            
        Returns:
            Session ID for tracking
        """
        with self.session_lock:
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, updating...")
            
            default_targets = {
                'minimum_similarity': 0.95,
                'minimum_quality': 0.85,
                'minimum_confidence': 0.80,
                'target_processing_time': 30.0  # seconds
            }
            
            if quality_targets:
                default_targets.update(quality_targets)
            
            session = QualityMonitoringSession(
                session_id=session_id,
                start_time=datetime.now(),
                current_stage=ProcessingStage.AUDIO_PREPROCESSING,
                quality_targets=default_targets
            )
            
            self.sessions[session_id] = session
            
        logger.info(f"Started quality monitoring session: {session_id}")
        return session_id
    
    def update_processing_stage(self, session_id: str, stage: ProcessingStage, 
                              progress: float = 0.0) -> None:
        """
        Update the current processing stage.
        
        Args:
            session_id: Session identifier
            stage: Current processing stage
            progress: Stage progress (0.0 to 1.0)
        """
        with self.session_lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.sessions[session_id]
            session.current_stage = stage
            
            # Create stage update metrics
            metrics = RealTimeMetrics(
                timestamp=datetime.now(),
                stage=stage,
                similarity_score=0.0,  # Will be updated with actual data
                quality_score=0.0,
                confidence_score=0.0,
                processing_time=0.0,
                issues_detected=[],
                recommendations=[],
                stage_progress=progress
            )
            
            session.current_metrics = metrics
            session.metrics_history.append(metrics)
        
        logger.info(f"Session {session_id}: Updated to stage {stage.value} ({progress*100:.1f}%)")
    
    def assess_real_time_quality(self, session_id: str, audio: np.ndarray, 
                                sample_rate: int, stage: ProcessingStage,
                                reference_audio: Optional[np.ndarray] = None) -> RealTimeMetrics:
        """
        Perform real-time quality assessment during processing.
        
        Args:
            session_id: Session identifier
            audio: Audio data to assess
            sample_rate: Sample rate
            stage: Current processing stage
            reference_audio: Optional reference audio for comparison
            
        Returns:
            Real-time metrics
        """
        start_time = time.time()
        
        with self.session_lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.sessions[session_id]
        
        # Perform quality assessment
        quality_report = self.quality_assessor.assess_audio_quality(audio, sample_rate)
        
        # Calculate similarity if reference is provided
        similarity_score = 0.0
        if reference_audio is not None:
            similarity_score = self._calculate_similarity_score(
                audio, reference_audio, sample_rate
            )
        
        # Calculate confidence scores
        confidence_scores = self._calculate_confidence_scores(
            audio, sample_rate, quality_report
        )
        
        # Detect issues and generate recommendations
        issues = [issue.issue_type.value for issue in quality_report.issues_detected]
        recommendations = [rec.description for rec in quality_report.enhancement_recommendations[:3]]
        
        processing_time = time.time() - start_time
        
        # Create metrics
        metrics = RealTimeMetrics(
            timestamp=datetime.now(),
            stage=stage,
            similarity_score=similarity_score,
            quality_score=quality_report.overall_score,
            confidence_score=confidence_scores.overall_analysis,
            processing_time=processing_time,
            issues_detected=issues,
            recommendations=recommendations,
            stage_progress=1.0  # Assume complete for this assessment
        )
        
        # Update session
        with self.session_lock:
            session.current_metrics = metrics
            session.metrics_history.append(metrics)
            session.confidence_scores = confidence_scores
        
        # Trigger callbacks
        self._trigger_callbacks(session_id, metrics)
        
        logger.info(f"Session {session_id}: Quality assessment - Score: {quality_report.overall_score:.3f}, "
                   f"Similarity: {similarity_score:.3f}, Confidence: {confidence_scores.overall_analysis:.3f}")
        
        return metrics
    
    def generate_improvement_recommendations(self, session_id: str, 
                                          voice_characteristics: Optional[VoiceCharacteristics] = None) -> List[ImprovementRecommendation]:
        """
        Generate improvement recommendations for insufficient characteristics.
        
        Args:
            session_id: Session identifier
            voice_characteristics: Optional voice characteristics for analysis
            
        Returns:
            List of improvement recommendations
        """
        with self.session_lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.sessions[session_id]
        
        recommendations = []
        
        # Analyze current metrics
        if session.current_metrics:
            current_quality = session.current_metrics.quality_score
            current_similarity = session.current_metrics.similarity_score
            current_confidence = session.current_metrics.confidence_score
            
            # Quality-based recommendations
            if current_quality < self.quality_thresholds[QualityLevel.GOOD]:
                recommendations.append(ImprovementRecommendation(
                    category="audio_quality",
                    priority=1,
                    issue_description=f"Audio quality is below target ({current_quality:.2f} < {self.quality_thresholds[QualityLevel.GOOD]})",
                    recommended_action="Apply advanced audio preprocessing and enhancement",
                    expected_improvement=min(0.3, self.quality_thresholds[QualityLevel.GOOD] - current_quality),
                    implementation_steps=[
                        "Apply noise reduction with voice preservation",
                        "Enhance spectral content and frequency response",
                        "Optimize dynamic range and compression",
                        "Validate improvements with quality assessment"
                    ],
                    estimated_time=15.0,
                    prerequisites=[]
                ))
            
            # Similarity-based recommendations
            if current_similarity < self.similarity_thresholds['target_similarity']:
                recommendations.append(ImprovementRecommendation(
                    category="voice_similarity",
                    priority=1,
                    issue_description=f"Voice similarity is below target ({current_similarity:.2f} < {self.similarity_thresholds['target_similarity']})",
                    recommended_action="Improve voice characteristic extraction and model training",
                    expected_improvement=min(0.2, self.similarity_thresholds['target_similarity'] - current_similarity),
                    implementation_steps=[
                        "Increase reference audio duration if possible",
                        "Apply advanced voice analysis techniques",
                        "Use ensemble synthesis methods",
                        "Fine-tune model parameters for voice characteristics"
                    ],
                    estimated_time=25.0,
                    prerequisites=["audio_quality"]
                ))
            
            # Confidence-based recommendations
            if current_confidence < self.confidence_thresholds['high_confidence']:
                recommendations.append(ImprovementRecommendation(
                    category="analysis_confidence",
                    priority=2,
                    issue_description=f"Analysis confidence is below optimal ({current_confidence:.2f} < {self.confidence_thresholds['high_confidence']})",
                    recommended_action="Improve feature extraction reliability and validation",
                    expected_improvement=min(0.15, self.confidence_thresholds['high_confidence'] - current_confidence),
                    implementation_steps=[
                        "Apply multiple analysis methods for cross-validation",
                        "Increase analysis precision and resolution",
                        "Validate extracted features against known patterns",
                        "Use ensemble analysis techniques"
                    ],
                    estimated_time=10.0,
                    prerequisites=[]
                ))
        
        # Voice characteristics-based recommendations
        if voice_characteristics:
            recommendations.extend(self._analyze_voice_characteristics_issues(voice_characteristics))
        
        # Update session recommendations
        with self.session_lock:
            session.recommendations = recommendations
        
        logger.info(f"Session {session_id}: Generated {len(recommendations)} improvement recommendations")
        return recommendations
    
    def calculate_detailed_similarity_metrics(self, session_id: str, 
                                            reference_audio: np.ndarray,
                                            synthesized_audio: np.ndarray,
                                            sample_rate: int) -> SimilarityMetrics:
        """
        Calculate detailed similarity metrics between reference and synthesized audio.
        
        Args:
            session_id: Session identifier
            reference_audio: Reference audio
            synthesized_audio: Synthesized audio
            sample_rate: Sample rate
            
        Returns:
            Detailed similarity metrics
        """
        logger.info(f"Session {session_id}: Calculating detailed similarity metrics")
        
        # Overall similarity
        overall_similarity = self._calculate_similarity_score(
            synthesized_audio, reference_audio, sample_rate
        )
        
        # Pitch similarity
        pitch_similarity = self._calculate_pitch_similarity(
            reference_audio, synthesized_audio, sample_rate
        )
        
        # Timbre similarity
        timbre_similarity = self._calculate_timbre_similarity(
            reference_audio, synthesized_audio, sample_rate
        )
        
        # Prosody similarity
        prosody_similarity = self._calculate_prosody_similarity(
            reference_audio, synthesized_audio, sample_rate
        )
        
        # Emotional similarity
        emotional_similarity = self._calculate_emotional_similarity(
            reference_audio, synthesized_audio, sample_rate
        )
        
        # Spectral similarity
        spectral_similarity = self._calculate_spectral_similarity(
            reference_audio, synthesized_audio, sample_rate
        )
        
        # Temporal similarity
        temporal_similarity = self._calculate_temporal_similarity(
            reference_audio, synthesized_audio, sample_rate
        )
        
        # Calculate confidence interval
        confidence_interval = self._calculate_similarity_confidence_interval(
            overall_similarity, [pitch_similarity, timbre_similarity, prosody_similarity,
                               emotional_similarity, spectral_similarity, temporal_similarity]
        )
        
        # Create detailed breakdown
        breakdown = {
            'fundamental_frequency': pitch_similarity,
            'formant_frequencies': timbre_similarity * 0.8,  # Formants are part of timbre
            'spectral_envelope': spectral_similarity,
            'prosodic_patterns': prosody_similarity,
            'emotional_content': emotional_similarity,
            'temporal_dynamics': temporal_similarity,
            'voice_quality': (timbre_similarity + spectral_similarity) / 2,
            'speaking_style': (prosody_similarity + emotional_similarity) / 2
        }
        
        similarity_metrics = SimilarityMetrics(
            overall_similarity=overall_similarity,
            pitch_similarity=pitch_similarity,
            timbre_similarity=timbre_similarity,
            prosody_similarity=prosody_similarity,
            emotional_similarity=emotional_similarity,
            spectral_similarity=spectral_similarity,
            temporal_similarity=temporal_similarity,
            confidence_interval=confidence_interval,
            breakdown=breakdown
        )
        
        # Update session
        with self.session_lock:
            if session_id in self.sessions:
                self.sessions[session_id].similarity_metrics = similarity_metrics
        
        logger.info(f"Session {session_id}: Similarity metrics - Overall: {overall_similarity:.3f}, "
                   f"Pitch: {pitch_similarity:.3f}, Timbre: {timbre_similarity:.3f}")
        
        return similarity_metrics
    
    def generate_optimization_strategy(self, session_id: str, 
                                     target_quality: float = 0.95) -> OptimizationStrategy:
        """
        Generate optimization strategy for improving quality to target level.
        
        Args:
            session_id: Session identifier
            target_quality: Target quality score
            
        Returns:
            Optimization strategy
        """
        with self.session_lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.sessions[session_id]
        
        current_quality = 0.0
        if session.current_metrics:
            current_quality = session.current_metrics.quality_score
        
        improvement_potential = target_quality - current_quality
        
        # Generate prioritized recommendations
        recommendations = session.recommendations or []
        if not recommendations:
            recommendations = self.generate_improvement_recommendations(session_id)
        
        # Sort by priority and expected improvement
        sorted_recommendations = sorted(
            recommendations, 
            key=lambda x: (x.priority, -x.expected_improvement)
        )
        
        # Calculate total estimated time
        total_time = sum(rec.estimated_time for rec in sorted_recommendations)
        
        # Estimate success probability based on current state and recommendations
        success_probability = self._calculate_success_probability(
            current_quality, target_quality, recommendations
        )
        
        strategy = OptimizationStrategy(
            strategy_name=f"Quality Optimization to {target_quality:.2f}",
            target_quality=target_quality,
            current_quality=current_quality,
            improvement_potential=improvement_potential,
            recommended_steps=sorted_recommendations,
            estimated_total_time=total_time,
            success_probability=success_probability
        )
        
        # Update session
        with self.session_lock:
            session.optimization_strategy = strategy
        
        logger.info(f"Session {session_id}: Generated optimization strategy - "
                   f"Target: {target_quality:.2f}, Current: {current_quality:.2f}, "
                   f"Success probability: {success_probability:.2f}")
        
        return strategy
    
    def get_confidence_scores(self, session_id: str) -> Optional[ConfidenceScores]:
        """
        Get confidence scores for all extracted characteristics.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Confidence scores or None if not available
        """
        with self.session_lock:
            if session_id not in self.sessions:
                return None
            
            return self.sessions[session_id].confidence_scores
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive session summary.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary dictionary
        """
        with self.session_lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.sessions[session_id]
        
        # Calculate session statistics
        quality_scores = [m.quality_score for m in session.metrics_history if m.quality_score > 0]
        similarity_scores = [m.similarity_score for m in session.metrics_history if m.similarity_score > 0]
        confidence_scores = [m.confidence_score for m in session.metrics_history if m.confidence_score > 0]
        
        summary = {
            'session_id': session_id,
            'start_time': session.start_time.isoformat(),
            'duration': (datetime.now() - session.start_time).total_seconds(),
            'current_stage': session.current_stage.value,
            'total_assessments': len(session.metrics_history),
            'quality_statistics': {
                'current': quality_scores[-1] if quality_scores else 0.0,
                'average': np.mean(quality_scores) if quality_scores else 0.0,
                'maximum': np.max(quality_scores) if quality_scores else 0.0,
                'minimum': np.min(quality_scores) if quality_scores else 0.0,
                'trend': self._calculate_trend(quality_scores) if len(quality_scores) > 1 else 0.0
            },
            'similarity_statistics': {
                'current': similarity_scores[-1] if similarity_scores else 0.0,
                'average': np.mean(similarity_scores) if similarity_scores else 0.0,
                'maximum': np.max(similarity_scores) if similarity_scores else 0.0,
                'minimum': np.min(similarity_scores) if similarity_scores else 0.0,
                'trend': self._calculate_trend(similarity_scores) if len(similarity_scores) > 1 else 0.0
            },
            'confidence_statistics': {
                'current': confidence_scores[-1] if confidence_scores else 0.0,
                'average': np.mean(confidence_scores) if confidence_scores else 0.0,
                'maximum': np.max(confidence_scores) if confidence_scores else 0.0,
                'minimum': np.min(confidence_scores) if confidence_scores else 0.0,
                'trend': self._calculate_trend(confidence_scores) if len(confidence_scores) > 1 else 0.0
            },
            'issues_summary': self._summarize_issues(session.metrics_history),
            'recommendations_count': len(session.recommendations),
            'quality_targets': session.quality_targets,
            'targets_met': self._check_targets_met(session),
            'similarity_metrics': session.similarity_metrics.__dict__ if session.similarity_metrics else None,
            'confidence_scores': session.confidence_scores.__dict__ if session.confidence_scores else None,
            'optimization_strategy': session.optimization_strategy.__dict__ if session.optimization_strategy else None
        }
        
        return summary
    
    def register_callback(self, session_id: str, callback: Callable[[str, RealTimeMetrics], None]) -> None:
        """
        Register a callback for real-time updates.
        
        Args:
            session_id: Session identifier
            callback: Callback function to register
        """
        with self.session_lock:
            if session_id not in self.sessions:
                raise ValueError(f"Session {session_id} not found")
            
            self.sessions[session_id].callbacks.append(callback)
    
    def end_monitoring_session(self, session_id: str) -> Dict[str, Any]:
        """
        End monitoring session and return final summary.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Final session summary
        """
        summary = self.get_session_summary(session_id)
        
        with self.session_lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
        
        logger.info(f"Ended quality monitoring session: {session_id}")
        return summary
    
    # Private helper methods
    
    def _calculate_similarity_score(self, audio1: np.ndarray, audio2: np.ndarray, 
                                  sample_rate: int) -> float:
        """Calculate overall similarity score between two audio signals."""
        try:
            # Ensure same length
            min_len = min(len(audio1), len(audio2))
            audio1 = audio1[:min_len]
            audio2 = audio2[:min_len]
            
            if min_len == 0:
                return 0.0
            
            # Spectral similarity
            spec1 = np.abs(librosa.stft(audio1))
            spec2 = np.abs(librosa.stft(audio2))
            
            # Align spectrograms
            min_frames = min(spec1.shape[1], spec2.shape[1])
            spec1 = spec1[:, :min_frames]
            spec2 = spec2[:, :min_frames]
            
            # Calculate correlation
            if spec1.size > 0 and spec2.size > 0:
                correlation = np.corrcoef(spec1.flatten(), spec2.flatten())[0, 1]
                similarity = max(0.0, correlation) if not np.isnan(correlation) else 0.0
            else:
                similarity = 0.0
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating similarity score: {e}")
            return 0.0
    
    def _calculate_pitch_similarity(self, ref_audio: np.ndarray, syn_audio: np.ndarray, 
                                  sample_rate: int) -> float:
        """Calculate pitch similarity between reference and synthesized audio."""
        try:
            # Extract pitch from both signals
            f0_ref, _, _ = librosa.pyin(ref_audio, fmin=80, fmax=400, sr=sample_rate)
            f0_syn, _, _ = librosa.pyin(syn_audio, fmin=80, fmax=400, sr=sample_rate)
            
            # Remove NaN values
            f0_ref_clean = f0_ref[~np.isnan(f0_ref)]
            f0_syn_clean = f0_syn[~np.isnan(f0_syn)]
            
            if len(f0_ref_clean) == 0 or len(f0_syn_clean) == 0:
                return 0.0
            
            # Calculate statistics
            ref_mean = np.mean(f0_ref_clean)
            syn_mean = np.mean(f0_syn_clean)
            ref_std = np.std(f0_ref_clean)
            syn_std = np.std(f0_syn_clean)
            
            # Similarity based on mean and variance
            mean_similarity = 1.0 - abs(ref_mean - syn_mean) / max(ref_mean, syn_mean)
            std_similarity = 1.0 - abs(ref_std - syn_std) / max(ref_std, syn_std, 1.0)
            
            return float(max(0.0, (mean_similarity + std_similarity) / 2))
            
        except Exception as e:
            logger.warning(f"Error calculating pitch similarity: {e}")
            return 0.0
    
    def _calculate_timbre_similarity(self, ref_audio: np.ndarray, syn_audio: np.ndarray, 
                                   sample_rate: int) -> float:
        """Calculate timbre similarity using MFCC features."""
        try:
            # Extract MFCC features
            mfcc_ref = librosa.feature.mfcc(y=ref_audio, sr=sample_rate, n_mfcc=13)
            mfcc_syn = librosa.feature.mfcc(y=syn_audio, sr=sample_rate, n_mfcc=13)
            
            # Calculate mean MFCC vectors
            mfcc_ref_mean = np.mean(mfcc_ref, axis=1)
            mfcc_syn_mean = np.mean(mfcc_syn, axis=1)
            
            # Calculate cosine similarity
            dot_product = np.dot(mfcc_ref_mean, mfcc_syn_mean)
            norm_ref = np.linalg.norm(mfcc_ref_mean)
            norm_syn = np.linalg.norm(mfcc_syn_mean)
            
            if norm_ref > 0 and norm_syn > 0:
                similarity = dot_product / (norm_ref * norm_syn)
                return float(max(0.0, similarity))
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating timbre similarity: {e}")
            return 0.0
    
    def _calculate_prosody_similarity(self, ref_audio: np.ndarray, syn_audio: np.ndarray, 
                                    sample_rate: int) -> float:
        """Calculate prosody similarity based on rhythm and energy patterns."""
        try:
            # Extract energy contours
            energy_ref = librosa.feature.rms(y=ref_audio)[0]
            energy_syn = librosa.feature.rms(y=syn_audio)[0]
            
            # Align lengths
            min_len = min(len(energy_ref), len(energy_syn))
            energy_ref = energy_ref[:min_len]
            energy_syn = energy_syn[:min_len]
            
            if min_len == 0:
                return 0.0
            
            # Calculate correlation
            correlation = np.corrcoef(energy_ref, energy_syn)[0, 1]
            similarity = max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating prosody similarity: {e}")
            return 0.0
    
    def _calculate_emotional_similarity(self, ref_audio: np.ndarray, syn_audio: np.ndarray, 
                                      sample_rate: int) -> float:
        """Calculate emotional similarity based on spectral characteristics."""
        try:
            # Extract spectral features for emotional analysis
            centroid_ref = librosa.feature.spectral_centroid(y=ref_audio, sr=sample_rate)[0]
            centroid_syn = librosa.feature.spectral_centroid(y=syn_audio, sr=sample_rate)[0]
            
            rolloff_ref = librosa.feature.spectral_rolloff(y=ref_audio, sr=sample_rate)[0]
            rolloff_syn = librosa.feature.spectral_rolloff(y=syn_audio, sr=sample_rate)[0]
            
            # Calculate similarity for each feature
            centroid_sim = 1.0 - abs(np.mean(centroid_ref) - np.mean(centroid_syn)) / max(np.mean(centroid_ref), np.mean(centroid_syn))
            rolloff_sim = 1.0 - abs(np.mean(rolloff_ref) - np.mean(rolloff_syn)) / max(np.mean(rolloff_ref), np.mean(rolloff_syn))
            
            return float(max(0.0, (centroid_sim + rolloff_sim) / 2))
            
        except Exception as e:
            logger.warning(f"Error calculating emotional similarity: {e}")
            return 0.0
    
    def _calculate_spectral_similarity(self, ref_audio: np.ndarray, syn_audio: np.ndarray, 
                                     sample_rate: int) -> float:
        """Calculate spectral similarity using power spectral density."""
        try:
            # Calculate power spectral density
            from scipy import signal
            
            freqs_ref, psd_ref = signal.welch(ref_audio, fs=sample_rate, nperseg=1024)
            freqs_syn, psd_syn = signal.welch(syn_audio, fs=sample_rate, nperseg=1024)
            
            # Align frequency ranges
            min_len = min(len(psd_ref), len(psd_syn))
            psd_ref = psd_ref[:min_len]
            psd_syn = psd_syn[:min_len]
            
            # Normalize PSDs
            psd_ref_norm = psd_ref / np.sum(psd_ref)
            psd_syn_norm = psd_syn / np.sum(psd_syn)
            
            # Calculate correlation
            correlation = np.corrcoef(psd_ref_norm, psd_syn_norm)[0, 1]
            similarity = max(0.0, correlation) if not np.isnan(correlation) else 0.0
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error calculating spectral similarity: {e}")
            return 0.0
    
    def _calculate_temporal_similarity(self, ref_audio: np.ndarray, syn_audio: np.ndarray, 
                                     sample_rate: int) -> float:
        """Calculate temporal similarity based on onset patterns."""
        try:
            # Detect onsets
            onsets_ref = librosa.onset.onset_detect(y=ref_audio, sr=sample_rate, units='time')
            onsets_syn = librosa.onset.onset_detect(y=syn_audio, sr=sample_rate, units='time')
            
            if len(onsets_ref) == 0 or len(onsets_syn) == 0:
                return 0.0
            
            # Calculate onset rate similarity
            rate_ref = len(onsets_ref) / (len(ref_audio) / sample_rate)
            rate_syn = len(onsets_syn) / (len(syn_audio) / sample_rate)
            
            rate_similarity = 1.0 - abs(rate_ref - rate_syn) / max(rate_ref, rate_syn)
            
            return float(max(0.0, rate_similarity))
            
        except Exception as e:
            logger.warning(f"Error calculating temporal similarity: {e}")
            return 0.0
    
    def _calculate_similarity_confidence_interval(self, overall_similarity: float, 
                                                component_similarities: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for similarity score."""
        if not component_similarities:
            return (overall_similarity, overall_similarity)
        
        # Calculate standard error
        std_error = np.std(component_similarities) / np.sqrt(len(component_similarities))
        
        # 95% confidence interval
        margin = 1.96 * std_error
        lower_bound = max(0.0, overall_similarity - margin)
        upper_bound = min(1.0, overall_similarity + margin)
        
        return (float(lower_bound), float(upper_bound))
    
    def _calculate_confidence_scores(self, audio: np.ndarray, sample_rate: int, 
                                   quality_report: QualityAssessmentReport) -> ConfidenceScores:
        """Calculate confidence scores for all extracted characteristics."""
        
        # Base confidence on quality metrics
        base_confidence = quality_report.overall_score
        
        # Pitch extraction confidence (based on voice activity and SNR)
        voice_activity = quality_report.technical_metrics.get('voice_activity_ratio', 0.5)
        snr = quality_report.technical_metrics.get('snr_db', 10.0)
        pitch_confidence = min(1.0, base_confidence * voice_activity * min(1.0, snr / 15.0))
        
        # Formant detection confidence (based on spectral clarity)
        spectral_clarity = quality_report.technical_metrics.get('spectral_clarity', 0.5)
        formant_confidence = min(1.0, base_confidence * spectral_clarity)
        
        # Timbre analysis confidence (based on frequency response)
        freq_response = quality_report.technical_metrics.get('frequency_response_score', 0.5)
        timbre_confidence = min(1.0, base_confidence * freq_response)
        
        # Prosody extraction confidence (based on duration and voice activity)
        duration = quality_report.technical_metrics.get('duration', 0.0)
        duration_factor = min(1.0, duration / 3.0)  # 3 seconds is good for prosody
        prosody_confidence = min(1.0, base_confidence * voice_activity * duration_factor)
        
        # Emotional analysis confidence (lower baseline due to complexity)
        emotional_confidence = min(1.0, base_confidence * 0.8 * voice_activity)
        
        # Overall analysis confidence
        overall_confidence = np.mean([
            pitch_confidence, formant_confidence, timbre_confidence,
            prosody_confidence, emotional_confidence
        ])
        
        # Voice model quality (based on overall analysis quality)
        model_confidence = min(1.0, overall_confidence * base_confidence)
        
        # Synthesis quality (placeholder - would be updated during synthesis)
        synthesis_confidence = 0.8  # Default value
        
        # Characteristic reliability breakdown
        characteristic_reliability = {
            'fundamental_frequency': pitch_confidence,
            'formant_frequencies': formant_confidence,
            'spectral_features': timbre_confidence,
            'prosodic_features': prosody_confidence,
            'emotional_features': emotional_confidence,
            'voice_quality': base_confidence,
            'temporal_features': prosody_confidence * 0.9
        }
        
        return ConfidenceScores(
            pitch_extraction=pitch_confidence,
            formant_detection=formant_confidence,
            timbre_analysis=timbre_confidence,
            prosody_extraction=prosody_confidence,
            emotional_analysis=emotional_confidence,
            overall_analysis=overall_confidence,
            voice_model_quality=model_confidence,
            synthesis_quality=synthesis_confidence,
            characteristic_reliability=characteristic_reliability
        )
    
    def _analyze_voice_characteristics_issues(self, voice_characteristics: VoiceCharacteristics) -> List[ImprovementRecommendation]:
        """Analyze voice characteristics for potential issues and recommendations."""
        recommendations = []
        
        # Check pitch characteristics
        pitch_range = voice_characteristics.pitch_characteristics
        if pitch_range.std_hz > 50:  # High pitch variability
            recommendations.append(ImprovementRecommendation(
                category="pitch_stability",
                priority=2,
                issue_description="High pitch variability detected",
                recommended_action="Apply pitch stabilization and smoothing",
                expected_improvement=0.1,
                implementation_steps=[
                    "Apply pitch tracking with higher precision",
                    "Use temporal smoothing for pitch contour",
                    "Validate pitch extraction with multiple methods"
                ],
                estimated_time=8.0,
                prerequisites=[]
            ))
        
        # Check quality metrics
        quality = voice_characteristics.quality_metrics
        if quality.signal_to_noise_ratio < 15.0:
            recommendations.append(ImprovementRecommendation(
                category="noise_reduction",
                priority=1,
                issue_description=f"Low SNR detected ({quality.signal_to_noise_ratio:.1f} dB)",
                recommended_action="Apply advanced noise reduction",
                expected_improvement=0.2,
                implementation_steps=[
                    "Apply spectral subtraction noise reduction",
                    "Use Wiener filtering for voice preservation",
                    "Validate noise reduction effectiveness"
                ],
                estimated_time=12.0,
                prerequisites=[]
            ))
        
        if quality.voice_activity_ratio < 0.6:
            recommendations.append(ImprovementRecommendation(
                category="voice_content",
                priority=2,
                issue_description=f"Low voice activity ratio ({quality.voice_activity_ratio:.2f})",
                recommended_action="Trim silence and enhance voice segments",
                expected_improvement=0.15,
                implementation_steps=[
                    "Apply voice activity detection",
                    "Trim silent segments",
                    "Enhance voice segment quality"
                ],
                estimated_time=5.0,
                prerequisites=[]
            ))
        
        return recommendations
    
    def _calculate_success_probability(self, current_quality: float, target_quality: float, 
                                     recommendations: List[ImprovementRecommendation]) -> float:
        """Calculate probability of successfully reaching target quality."""
        if current_quality >= target_quality:
            return 1.0
        
        improvement_needed = target_quality - current_quality
        total_expected_improvement = sum(rec.expected_improvement for rec in recommendations)
        
        if total_expected_improvement == 0:
            return 0.1  # Low probability without recommendations
        
        # Base probability on improvement potential
        base_probability = min(1.0, total_expected_improvement / improvement_needed)
        
        # Adjust based on current quality (higher quality = higher success probability)
        quality_factor = current_quality
        
        # Adjust based on number of high-priority recommendations
        high_priority_count = sum(1 for rec in recommendations if rec.priority <= 2)
        priority_factor = min(1.0, 0.5 + high_priority_count * 0.1)
        
        success_probability = base_probability * quality_factor * priority_factor
        
        return float(min(1.0, max(0.1, success_probability)))
    
    def _trigger_callbacks(self, session_id: str, metrics: RealTimeMetrics) -> None:
        """Trigger registered callbacks with new metrics."""
        with self.session_lock:
            if session_id in self.sessions:
                callbacks = self.sessions[session_id].callbacks.copy()
        
        for callback in callbacks:
            try:
                callback(session_id, metrics)
            except Exception as e:
                logger.warning(f"Callback error for session {session_id}: {e}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in values (positive = improving, negative = declining)."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        else:
            return 0.0
    
    def _summarize_issues(self, metrics_history: List[RealTimeMetrics]) -> Dict[str, Any]:
        """Summarize issues detected across all metrics."""
        all_issues = []
        for metrics in metrics_history:
            all_issues.extend(metrics.issues_detected)
        
        if not all_issues:
            return {'total_issues': 0, 'issue_types': {}, 'most_common': None}
        
        # Count issue types
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        most_common = max(issue_counts.items(), key=lambda x: x[1]) if issue_counts else None
        
        return {
            'total_issues': len(all_issues),
            'unique_issue_types': len(issue_counts),
            'issue_types': issue_counts,
            'most_common': most_common[0] if most_common else None,
            'most_common_count': most_common[1] if most_common else 0
        }
    
    def _check_targets_met(self, session: QualityMonitoringSession) -> Dict[str, bool]:
        """Check if quality targets have been met."""
        targets_met = {}
        
        if session.current_metrics:
            current = session.current_metrics
            targets = session.quality_targets
            
            targets_met['minimum_similarity'] = current.similarity_score >= targets.get('minimum_similarity', 0.95)
            targets_met['minimum_quality'] = current.quality_score >= targets.get('minimum_quality', 0.85)
            targets_met['minimum_confidence'] = current.confidence_score >= targets.get('minimum_confidence', 0.80)
            
            # Overall target achievement
            targets_met['all_targets'] = all(targets_met.values())
        
        return targets_met


# Global instance
real_time_quality_monitor = RealTimeQualityMonitor()