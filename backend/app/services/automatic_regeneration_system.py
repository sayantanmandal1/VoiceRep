"""
Automatic Regeneration System for Perfect Voice Cloning.

This module implements an intelligent regeneration system that automatically
detects when synthesized audio quality falls below the 95% threshold and
triggers regeneration with adjusted parameters to achieve optimal results.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class RegenerationAttempt:
    """Record of a regeneration attempt."""
    attempt_number: int
    quality_score: float
    similarity_score: float
    adjustments_applied: Dict[str, Any]
    weak_areas_addressed: List[str]
    success: bool


@dataclass
class RegenerationResult:
    """Result of the regeneration process."""
    audio: np.ndarray
    sample_rate: int
    final_quality: float
    final_similarity: float
    attempts_made: int
    attempts_history: List[RegenerationAttempt]
    success: bool
    best_attempt: int
    improvement_achieved: float


class ParameterAdjuster:
    """
    Intelligent parameter adjustment based on quality weaknesses.
    
    Analyzes quality metrics to determine optimal parameter adjustments
    for regeneration attempts.
    """
    
    # Adjustment strategies for different weak areas
    ADJUSTMENT_STRATEGIES = {
        'speaker_identity': {
            'temperature': -0.05,  # Lower temperature for more consistent output
            'speaker_embedding_weight': 0.1,  # Increase speaker embedding influence
            'repetition_penalty': 0.05
        },
        'prosody': {
            'speed_factor': 0.02,  # Slight speed adjustment
            'pitch_shift': 0.0,  # Keep pitch stable
            'prosody_weight': 0.1
        },
        'timbre': {
            'spectral_matching_strength': 0.1,
            'formant_shift': 0.0,
            'timbre_weight': 0.1
        },
        'emotion': {
            'energy_scale': 0.05,
            'pitch_variance_scale': 0.05,
            'emotion_weight': 0.1
        },
        'spectral': {
            'frequency_response_correction': 0.1,
            'spectral_smoothing': 0.05
        },
        'naturalness': {
            'vocoder_quality': 'high',
            'denoising_strength': 0.1
        },
        'artifacts': {
            'artifact_removal_strength': 0.2,
            'smoothing_factor': 0.1
        }
    }

    def compute_adjustments(
        self,
        weak_areas: List[str],
        current_params: Dict[str, Any],
        attempt_number: int
    ) -> Dict[str, Any]:
        """
        Compute parameter adjustments based on weak areas.
        
        Args:
            weak_areas: List of areas needing improvement
            current_params: Current synthesis parameters
            attempt_number: Current attempt number (for scaling)
            
        Returns:
            Dictionary of adjusted parameters
        """
        adjustments = current_params.copy()
        
        # Scale adjustments based on attempt number (more aggressive later)
        scale = 1.0 + (attempt_number - 1) * 0.5
        
        for area in weak_areas:
            if area in self.ADJUSTMENT_STRATEGIES:
                strategy = self.ADJUSTMENT_STRATEGIES[area]
                for param, adjustment in strategy.items():
                    if isinstance(adjustment, (int, float)):
                        current_value = adjustments.get(param, 0)
                        adjustments[param] = current_value + adjustment * scale
                    else:
                        adjustments[param] = adjustment
        
        return adjustments
    
    def get_priority_adjustments(
        self,
        quality_metrics: Dict[str, float]
    ) -> List[str]:
        """
        Determine priority order for adjustments based on metrics.
        
        Args:
            quality_metrics: Dictionary of quality metric scores
            
        Returns:
            Ordered list of areas to prioritize
        """
        # Sort by lowest score (most improvement needed)
        sorted_metrics = sorted(
            quality_metrics.items(),
            key=lambda x: x[1]
        )
        
        return [metric for metric, score in sorted_metrics if score < 0.95]


class AutomaticRegenerationSystem:
    """
    Automatic regeneration system for achieving perfect voice cloning.
    
    Implements:
    1. Quality threshold checking (>95% required)
    2. Weakness identification from quality metrics
    3. Parameter adjustment based on weaknesses
    4. Iterative regeneration with improvements
    5. Best-effort fallback with quality tracking
    
    Target: Achieve >95% similarity or return best attempt.
    """
    
    QUALITY_THRESHOLD = 0.95  # 95% similarity required
    MAX_ATTEMPTS = 5  # Maximum regeneration attempts
    MIN_IMPROVEMENT = 0.02  # Minimum improvement to continue
    
    def __init__(self):
        """Initialize automatic regeneration system."""
        self.parameter_adjuster = ParameterAdjuster()
        self._quality_metrics = None
        
        logger.info("Automatic Regeneration System initialized")
    
    async def regenerate_until_quality(
        self,
        synthesis_function: Callable,
        reference_audio: np.ndarray,
        text: str,
        initial_params: Dict[str, Any],
        quality_function: Callable,
        progress_callback: Optional[Callable] = None
    ) -> RegenerationResult:
        """
        Regenerate synthesis until quality threshold is met.
        
        Args:
            synthesis_function: Async function to synthesize audio
            reference_audio: Reference audio for comparison
            text: Text to synthesize
            initial_params: Initial synthesis parameters
            quality_function: Function to compute quality metrics
            progress_callback: Optional progress callback
            
        Returns:
            RegenerationResult with best output
        """
        attempts_history = []
        best_audio = None
        best_quality = 0.0
        best_similarity = 0.0
        best_attempt = 0
        current_params = initial_params.copy()
        
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            if progress_callback:
                progress_callback(
                    int(20 + (attempt - 1) * 15),
                    f"Synthesis attempt {attempt}/{self.MAX_ATTEMPTS}"
                )
            
            try:
                # Synthesize audio
                synthesized_audio, sample_rate = await synthesis_function(
                    text=text,
                    params=current_params
                )
                
                # Compute quality metrics
                quality_metrics = quality_function(
                    synthesized_audio=synthesized_audio,
                    reference_audio=reference_audio
                )
                
                current_quality = quality_metrics.overall_quality
                current_similarity = quality_metrics.overall_similarity
                
                # Record attempt
                attempt_record = RegenerationAttempt(
                    attempt_number=attempt,
                    quality_score=current_quality,
                    similarity_score=current_similarity,
                    adjustments_applied=current_params.copy(),
                    weak_areas_addressed=quality_metrics.weak_areas,
                    success=current_similarity >= self.QUALITY_THRESHOLD
                )
                attempts_history.append(attempt_record)
                
                # Track best result
                if current_similarity > best_similarity:
                    best_audio = synthesized_audio
                    best_quality = current_quality
                    best_similarity = current_similarity
                    best_attempt = attempt
                
                logger.info(
                    f"Attempt {attempt}: similarity={current_similarity:.3f}, "
                    f"quality={current_quality:.3f}"
                )
                
                # Check if threshold met
                if current_similarity >= self.QUALITY_THRESHOLD:
                    logger.info(f"Quality threshold met on attempt {attempt}")
                    return RegenerationResult(
                        audio=synthesized_audio,
                        sample_rate=sample_rate,
                        final_quality=current_quality,
                        final_similarity=current_similarity,
                        attempts_made=attempt,
                        attempts_history=attempts_history,
                        success=True,
                        best_attempt=attempt,
                        improvement_achieved=current_similarity - (
                            attempts_history[0].similarity_score if attempts_history else 0
                        )
                    )
                
                # Check if improvement is stalling
                if attempt > 1:
                    prev_similarity = attempts_history[-2].similarity_score
                    improvement = current_similarity - prev_similarity
                    
                    if improvement < self.MIN_IMPROVEMENT:
                        logger.info(
                            f"Improvement stalling ({improvement:.3f}), "
                            f"returning best attempt"
                        )
                        break
                
                # Compute adjustments for next attempt
                current_params = self.parameter_adjuster.compute_adjustments(
                    weak_areas=quality_metrics.weak_areas,
                    current_params=current_params,
                    attempt_number=attempt
                )
                
            except Exception as e:
                logger.error(f"Regeneration attempt {attempt} failed: {e}")
                attempts_history.append(RegenerationAttempt(
                    attempt_number=attempt,
                    quality_score=0.0,
                    similarity_score=0.0,
                    adjustments_applied=current_params.copy(),
                    weak_areas_addressed=[],
                    success=False
                ))
        
        # Return best result
        if best_audio is None:
            raise RuntimeError("All regeneration attempts failed")
        
        return RegenerationResult(
            audio=best_audio,
            sample_rate=sample_rate if 'sample_rate' in dir() else 22050,
            final_quality=best_quality,
            final_similarity=best_similarity,
            attempts_made=len(attempts_history),
            attempts_history=attempts_history,
            success=best_similarity >= self.QUALITY_THRESHOLD,
            best_attempt=best_attempt,
            improvement_achieved=best_similarity - (
                attempts_history[0].similarity_score if attempts_history else 0
            )
        )
    
    def should_regenerate(
        self,
        quality_metrics: Any
    ) -> Tuple[bool, List[str]]:
        """
        Determine if regeneration is needed.
        
        Args:
            quality_metrics: Quality metrics object
            
        Returns:
            Tuple of (should_regenerate, reasons)
        """
        reasons = []
        
        if quality_metrics.overall_similarity < self.QUALITY_THRESHOLD:
            reasons.append(
                f"Overall similarity {quality_metrics.overall_similarity:.2f} "
                f"below threshold {self.QUALITY_THRESHOLD}"
            )
        
        if quality_metrics.speaker_similarity < 0.90:
            reasons.append("Speaker identity mismatch")
        
        if quality_metrics.naturalness_score < 0.85:
            reasons.append("Low naturalness score")
        
        if quality_metrics.artifact_score < 0.90:
            reasons.append("Artifacts detected")
        
        return len(reasons) > 0, reasons
    
    def get_improvement_suggestions(
        self,
        attempts_history: List[RegenerationAttempt]
    ) -> List[str]:
        """
        Generate improvement suggestions based on attempt history.
        
        Args:
            attempts_history: List of regeneration attempts
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        if not attempts_history:
            return ["No attempts recorded"]
        
        # Analyze trends
        similarities = [a.similarity_score for a in attempts_history]
        
        if len(similarities) > 1:
            trend = similarities[-1] - similarities[0]
            if trend < 0:
                suggestions.append(
                    "Quality decreased over attempts - consider different approach"
                )
            elif trend < 0.05:
                suggestions.append(
                    "Limited improvement - try longer reference audio"
                )
        
        # Common weak areas
        all_weak_areas = []
        for attempt in attempts_history:
            all_weak_areas.extend(attempt.weak_areas_addressed)
        
        if all_weak_areas:
            from collections import Counter
            common_issues = Counter(all_weak_areas).most_common(3)
            for issue, count in common_issues:
                if count >= len(attempts_history) // 2:
                    suggestions.append(f"Persistent issue: {issue}")
        
        # Best attempt analysis
        best = max(attempts_history, key=lambda x: x.similarity_score)
        if best.similarity_score < self.QUALITY_THRESHOLD:
            gap = self.QUALITY_THRESHOLD - best.similarity_score
            suggestions.append(
                f"Best attempt {gap:.1%} below threshold - "
                f"consider improving reference audio quality"
            )
        
        return suggestions


# Global instance
_regeneration_system: Optional[AutomaticRegenerationSystem] = None


def get_regeneration_system() -> AutomaticRegenerationSystem:
    """Get or create global regeneration system instance."""
    global _regeneration_system
    if _regeneration_system is None:
        _regeneration_system = AutomaticRegenerationSystem()
    return _regeneration_system
