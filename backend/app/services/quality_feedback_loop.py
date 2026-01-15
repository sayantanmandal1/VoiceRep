"""
Quality Feedback Loop for Perfect Voice Cloning.

This module implements a quality feedback system that:
1. Logs and analyzes quality metrics over time
2. Tracks model performance per voice profile
3. Builds adaptive parameter tuning based on historical quality
4. Provides voice profile quality improvement suggestions

Requirements: 8.1, 8.2, 8.3, 8.4, 10.4
"""

import logging
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Quality dimensions tracked by the feedback loop."""
    SPEAKER_SIMILARITY = "speaker_similarity"
    PROSODY_MATCH = "prosody_match"
    TIMBRE_MATCH = "timbre_match"
    NATURALNESS = "naturalness"
    EMOTION_MATCH = "emotion_match"
    ARTIFACT_SCORE = "artifact_score"
    OVERALL_QUALITY = "overall_quality"


@dataclass
class QualityMetricEntry:
    """Single quality metric entry for logging."""
    timestamp: str
    voice_profile_id: str
    model_type: str
    dimension: str
    value: float
    text_length: int
    synthesis_time: float
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformanceStats:
    """Performance statistics for a specific model."""
    model_type: str
    total_syntheses: int = 0
    average_quality: float = 0.0
    quality_std: float = 0.0
    average_synthesis_time: float = 0.0
    success_rate: float = 1.0
    quality_by_dimension: Dict[str, float] = field(default_factory=dict)
    quality_trend: str = "stable"  # improving, declining, stable
    last_updated: str = ""


@dataclass
class VoiceProfilePerformance:
    """Performance tracking for a specific voice profile."""
    voice_profile_id: str
    total_syntheses: int = 0
    average_quality: float = 0.0
    best_model: str = ""
    model_performance: Dict[str, ModelPerformanceStats] = field(default_factory=dict)
    quality_history: List[float] = field(default_factory=list)
    recommended_parameters: Dict[str, Any] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)


@dataclass
class AdaptiveParameters:
    """Adaptive parameters tuned based on feedback."""
    voice_profile_id: str
    model_weights: Dict[str, float] = field(default_factory=dict)
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    synthesis_parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    last_updated: str = ""


@dataclass
class QualityFeedbackReport:
    """Comprehensive quality feedback report."""
    generated_at: str
    total_syntheses: int
    average_quality: float
    quality_trend: str
    top_performing_models: List[str]
    improvement_areas: List[str]
    recommendations: List[str]
    voice_profile_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class QualityFeedbackLoop:
    """
    Implements a quality feedback loop for continuous improvement of voice cloning.
    
    This system:
    1. Logs quality metrics from each synthesis
    2. Analyzes trends and patterns
    3. Adapts parameters based on historical performance
    4. Provides actionable improvement suggestions
    
    Requirements: 8.1, 8.2, 8.3, 8.4, 10.4
    """
    
    # Quality thresholds
    EXCELLENT_QUALITY = 0.95
    GOOD_QUALITY = 0.85
    ACCEPTABLE_QUALITY = 0.75
    POOR_QUALITY = 0.60
    
    # Trend analysis window
    TREND_WINDOW_SIZE = 20
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_history_entries: int = 10000
    ):
        """
        Initialize the quality feedback loop.
        
        Args:
            storage_path: Path for persistent storage
            max_history_entries: Maximum entries to keep in memory
        """
        self.storage_path = storage_path or Path("./quality_feedback")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.max_history_entries = max_history_entries
        
        # In-memory storage
        self._quality_history: List[QualityMetricEntry] = []
        self._voice_profile_performance: Dict[str, VoiceProfilePerformance] = {}
        self._model_performance: Dict[str, ModelPerformanceStats] = {}
        self._adaptive_parameters: Dict[str, AdaptiveParameters] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load existing data
        self._load_persistent_data()
        
        logger.info("Quality Feedback Loop initialized")
    
    def log_quality_metrics(
        self,
        voice_profile_id: str,
        model_type: str,
        quality_metrics: Dict[str, float],
        text_length: int,
        synthesis_time: float,
        parameters: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log quality metrics from a synthesis operation.
        
        Args:
            voice_profile_id: ID of the voice profile used
            model_type: Type of model used for synthesis
            quality_metrics: Dictionary of quality dimension scores
            text_length: Length of synthesized text
            synthesis_time: Time taken for synthesis
            parameters: Synthesis parameters used
        """
        with self._lock:
            timestamp = datetime.now().isoformat()
            
            # Log each quality dimension
            for dimension, value in quality_metrics.items():
                entry = QualityMetricEntry(
                    timestamp=timestamp,
                    voice_profile_id=voice_profile_id,
                    model_type=model_type,
                    dimension=dimension,
                    value=value,
                    text_length=text_length,
                    synthesis_time=synthesis_time,
                    parameters=parameters or {}
                )
                self._quality_history.append(entry)
            
            # Trim history if needed
            if len(self._quality_history) > self.max_history_entries:
                self._quality_history = self._quality_history[-self.max_history_entries:]
            
            # Update performance tracking
            self._update_voice_profile_performance(
                voice_profile_id, model_type, quality_metrics, synthesis_time
            )
            self._update_model_performance(model_type, quality_metrics, synthesis_time)
            
            # Periodically save to disk
            if len(self._quality_history) % 100 == 0:
                self._save_persistent_data()
        
        logger.debug(f"Logged quality metrics for profile {voice_profile_id}")
    
    def get_adaptive_parameters(
        self,
        voice_profile_id: str
    ) -> AdaptiveParameters:
        """
        Get adaptive parameters tuned for a specific voice profile.
        
        Args:
            voice_profile_id: ID of the voice profile
            
        Returns:
            AdaptiveParameters tuned based on historical performance
        """
        with self._lock:
            if voice_profile_id in self._adaptive_parameters:
                return self._adaptive_parameters[voice_profile_id]
            
            # Generate new adaptive parameters
            params = self._generate_adaptive_parameters(voice_profile_id)
            self._adaptive_parameters[voice_profile_id] = params
            
            return params
    
    def get_model_recommendation(
        self,
        voice_profile_id: str,
        text_characteristics: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float]:
        """
        Get model recommendation based on historical performance.
        
        Args:
            voice_profile_id: ID of the voice profile
            text_characteristics: Optional characteristics of text to synthesize
            
        Returns:
            Tuple of (recommended_model, confidence)
        """
        with self._lock:
            if voice_profile_id in self._voice_profile_performance:
                profile_perf = self._voice_profile_performance[voice_profile_id]
                
                if profile_perf.best_model:
                    # Calculate confidence based on number of samples
                    confidence = min(1.0, profile_perf.total_syntheses / 50)
                    return profile_perf.best_model, confidence
            
            # Default recommendation
            return "xtts_v2", 0.3
    
    def get_improvement_suggestions(
        self,
        voice_profile_id: str
    ) -> List[str]:
        """
        Get improvement suggestions for a voice profile.
        
        Args:
            voice_profile_id: ID of the voice profile
            
        Returns:
            List of improvement suggestions
        """
        with self._lock:
            suggestions = []
            
            if voice_profile_id in self._voice_profile_performance:
                profile_perf = self._voice_profile_performance[voice_profile_id]
                
                # Analyze quality dimensions
                for model_type, model_stats in profile_perf.model_performance.items():
                    for dim, score in model_stats.quality_by_dimension.items():
                        if score < self.GOOD_QUALITY:
                            suggestion = self._generate_dimension_suggestion(dim, score)
                            if suggestion and suggestion not in suggestions:
                                suggestions.append(suggestion)
                
                # Add general suggestions based on overall quality
                if profile_perf.average_quality < self.ACCEPTABLE_QUALITY:
                    suggestions.append(
                        "Consider providing longer or higher quality reference audio"
                    )
                
                if profile_perf.total_syntheses < 10:
                    suggestions.append(
                        "More synthesis attempts will improve parameter tuning"
                    )
            else:
                suggestions.append("No synthesis history available for this profile")
            
            return suggestions
    
    def generate_feedback_report(
        self,
        time_range_hours: int = 24
    ) -> QualityFeedbackReport:
        """
        Generate a comprehensive quality feedback report.
        
        Args:
            time_range_hours: Hours of history to include
            
        Returns:
            QualityFeedbackReport with analysis and recommendations
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            
            # Filter recent entries
            recent_entries = [
                e for e in self._quality_history
                if datetime.fromisoformat(e.timestamp) > cutoff_time
            ]
            
            if not recent_entries:
                return QualityFeedbackReport(
                    generated_at=datetime.now().isoformat(),
                    total_syntheses=0,
                    average_quality=0.0,
                    quality_trend="unknown",
                    top_performing_models=[],
                    improvement_areas=[],
                    recommendations=["No recent synthesis data available"]
                )
            
            # Calculate overall statistics
            overall_scores = [
                e.value for e in recent_entries 
                if e.dimension == QualityDimension.OVERALL_QUALITY.value
            ]
            
            avg_quality = np.mean(overall_scores) if overall_scores else 0.0
            
            # Determine quality trend
            quality_trend = self._analyze_quality_trend(overall_scores)
            
            # Find top performing models
            model_scores = defaultdict(list)
            for entry in recent_entries:
                if entry.dimension == QualityDimension.OVERALL_QUALITY.value:
                    model_scores[entry.model_type].append(entry.value)
            
            model_avg_scores = {
                model: np.mean(scores) 
                for model, scores in model_scores.items()
            }
            
            top_models = sorted(
                model_avg_scores.keys(),
                key=lambda m: model_avg_scores[m],
                reverse=True
            )[:3]
            
            # Identify improvement areas
            dimension_scores = defaultdict(list)
            for entry in recent_entries:
                dimension_scores[entry.dimension].append(entry.value)
            
            improvement_areas = [
                dim for dim, scores in dimension_scores.items()
                if np.mean(scores) < self.GOOD_QUALITY
            ]
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                avg_quality, quality_trend, improvement_areas
            )
            
            # Voice profile summaries
            profile_summaries = {}
            for profile_id, perf in self._voice_profile_performance.items():
                profile_summaries[profile_id] = {
                    "total_syntheses": perf.total_syntheses,
                    "average_quality": perf.average_quality,
                    "best_model": perf.best_model
                }
            
            return QualityFeedbackReport(
                generated_at=datetime.now().isoformat(),
                total_syntheses=len(recent_entries),
                average_quality=float(avg_quality),
                quality_trend=quality_trend,
                top_performing_models=top_models,
                improvement_areas=improvement_areas,
                recommendations=recommendations,
                voice_profile_summaries=profile_summaries
            )
    
    def _update_voice_profile_performance(
        self,
        voice_profile_id: str,
        model_type: str,
        quality_metrics: Dict[str, float],
        synthesis_time: float
    ) -> None:
        """Update performance tracking for a voice profile."""
        if voice_profile_id not in self._voice_profile_performance:
            self._voice_profile_performance[voice_profile_id] = VoiceProfilePerformance(
                voice_profile_id=voice_profile_id
            )
        
        profile_perf = self._voice_profile_performance[voice_profile_id]
        profile_perf.total_syntheses += 1
        
        # Update overall quality
        overall_quality = quality_metrics.get(
            QualityDimension.OVERALL_QUALITY.value, 0.0
        )
        profile_perf.quality_history.append(overall_quality)
        
        # Keep only recent history
        if len(profile_perf.quality_history) > 100:
            profile_perf.quality_history = profile_perf.quality_history[-100:]
        
        profile_perf.average_quality = np.mean(profile_perf.quality_history)
        
        # Update model performance within profile
        if model_type not in profile_perf.model_performance:
            profile_perf.model_performance[model_type] = ModelPerformanceStats(
                model_type=model_type
            )
        
        model_stats = profile_perf.model_performance[model_type]
        model_stats.total_syntheses += 1
        
        # Update running average
        n = model_stats.total_syntheses
        model_stats.average_quality = (
            (model_stats.average_quality * (n - 1) + overall_quality) / n
        )
        model_stats.average_synthesis_time = (
            (model_stats.average_synthesis_time * (n - 1) + synthesis_time) / n
        )
        
        # Update quality by dimension
        for dim, value in quality_metrics.items():
            if dim not in model_stats.quality_by_dimension:
                model_stats.quality_by_dimension[dim] = value
            else:
                model_stats.quality_by_dimension[dim] = (
                    (model_stats.quality_by_dimension[dim] * (n - 1) + value) / n
                )
        
        model_stats.last_updated = datetime.now().isoformat()
        
        # Determine best model
        best_model = max(
            profile_perf.model_performance.keys(),
            key=lambda m: profile_perf.model_performance[m].average_quality
        )
        profile_perf.best_model = best_model
    
    def _update_model_performance(
        self,
        model_type: str,
        quality_metrics: Dict[str, float],
        synthesis_time: float
    ) -> None:
        """Update global performance tracking for a model."""
        if model_type not in self._model_performance:
            self._model_performance[model_type] = ModelPerformanceStats(
                model_type=model_type
            )
        
        model_stats = self._model_performance[model_type]
        model_stats.total_syntheses += 1
        
        overall_quality = quality_metrics.get(
            QualityDimension.OVERALL_QUALITY.value, 0.0
        )
        
        n = model_stats.total_syntheses
        model_stats.average_quality = (
            (model_stats.average_quality * (n - 1) + overall_quality) / n
        )
        model_stats.average_synthesis_time = (
            (model_stats.average_synthesis_time * (n - 1) + synthesis_time) / n
        )
        
        model_stats.last_updated = datetime.now().isoformat()
    
    def _generate_adaptive_parameters(
        self,
        voice_profile_id: str
    ) -> AdaptiveParameters:
        """Generate adaptive parameters based on historical performance."""
        params = AdaptiveParameters(
            voice_profile_id=voice_profile_id,
            last_updated=datetime.now().isoformat()
        )
        
        # Default model weights
        params.model_weights = {
            "xtts_v2": 0.35,
            "styletts2": 0.30,
            "openvoice": 0.20,
            "bark": 0.15
        }
        
        # Default quality thresholds
        params.quality_thresholds = {
            "minimum_acceptable": 0.75,
            "target_quality": 0.90,
            "regeneration_threshold": 0.85
        }
        
        # Default synthesis parameters
        params.synthesis_parameters = {
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
        
        # Adjust based on historical performance
        if voice_profile_id in self._voice_profile_performance:
            profile_perf = self._voice_profile_performance[voice_profile_id]
            
            # Adjust model weights based on performance
            for model_type, model_stats in profile_perf.model_performance.items():
                if model_type in params.model_weights:
                    # Increase weight for better performing models
                    quality_factor = model_stats.average_quality / 0.9
                    params.model_weights[model_type] *= quality_factor
            
            # Normalize weights
            total_weight = sum(params.model_weights.values())
            if total_weight > 0:
                params.model_weights = {
                    k: v / total_weight 
                    for k, v in params.model_weights.items()
                }
            
            # Adjust confidence based on data
            params.confidence = min(1.0, profile_perf.total_syntheses / 50)
        
        return params
    
    def _generate_dimension_suggestion(
        self,
        dimension: str,
        score: float
    ) -> Optional[str]:
        """Generate improvement suggestion for a quality dimension."""
        suggestions = {
            QualityDimension.SPEAKER_SIMILARITY.value: (
                "Improve speaker similarity by using longer reference audio "
                "or ensuring reference audio is clean and representative"
            ),
            QualityDimension.PROSODY_MATCH.value: (
                "Improve prosody matching by using reference audio with "
                "varied intonation patterns"
            ),
            QualityDimension.TIMBRE_MATCH.value: (
                "Improve timbre matching by ensuring reference audio "
                "captures the full vocal range"
            ),
            QualityDimension.NATURALNESS.value: (
                "Improve naturalness by adjusting synthesis temperature "
                "or using ensemble synthesis"
            ),
            QualityDimension.EMOTION_MATCH.value: (
                "Improve emotion matching by using reference audio "
                "with similar emotional content"
            ),
            QualityDimension.ARTIFACT_SCORE.value: (
                "Reduce artifacts by enabling post-processing "
                "or using higher quality models"
            )
        }
        
        return suggestions.get(dimension)
    
    def _analyze_quality_trend(
        self,
        quality_scores: List[float]
    ) -> str:
        """Analyze quality trend from recent scores."""
        if len(quality_scores) < self.TREND_WINDOW_SIZE:
            return "insufficient_data"
        
        recent = quality_scores[-self.TREND_WINDOW_SIZE:]
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        
        diff = second_half - first_half
        
        if diff > 0.05:
            return "improving"
        elif diff < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _generate_recommendations(
        self,
        avg_quality: float,
        quality_trend: str,
        improvement_areas: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if avg_quality < self.ACCEPTABLE_QUALITY:
            recommendations.append(
                "Overall quality is below acceptable threshold. "
                "Consider reviewing reference audio quality."
            )
        
        if quality_trend == "declining":
            recommendations.append(
                "Quality trend is declining. Review recent changes "
                "to synthesis parameters or reference audio."
            )
        
        if QualityDimension.SPEAKER_SIMILARITY.value in improvement_areas:
            recommendations.append(
                "Speaker similarity needs improvement. "
                "Try using longer or cleaner reference audio."
            )
        
        if QualityDimension.NATURALNESS.value in improvement_areas:
            recommendations.append(
                "Naturalness scores are low. "
                "Enable micro-expression injection for more natural speech."
            )
        
        if not recommendations:
            recommendations.append(
                "Quality metrics are within acceptable ranges. "
                "Continue monitoring for any changes."
            )
        
        return recommendations
    
    def _load_persistent_data(self) -> None:
        """Load persistent data from storage."""
        try:
            history_file = self.storage_path / "quality_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self._quality_history = [
                        QualityMetricEntry(**entry) for entry in data
                    ]
                logger.info(f"Loaded {len(self._quality_history)} quality history entries")
        except Exception as e:
            logger.warning(f"Failed to load persistent data: {e}")
    
    def _save_persistent_data(self) -> None:
        """Save data to persistent storage."""
        try:
            history_file = self.storage_path / "quality_history.json"
            with open(history_file, 'w') as f:
                data = [asdict(entry) for entry in self._quality_history[-1000:]]
                json.dump(data, f)
            logger.debug("Saved quality history to disk")
        except Exception as e:
            logger.warning(f"Failed to save persistent data: {e}")


# Global instance
quality_feedback_loop = QualityFeedbackLoop()


async def initialize_quality_feedback_service():
    """Initialize the quality feedback loop service."""
    logger.info("Quality Feedback Loop service initialized")
    return True
