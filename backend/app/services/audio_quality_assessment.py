"""
Audio Quality Assessment and Enhancement Recommendation System.

This module provides comprehensive audio quality analysis and intelligent
enhancement recommendations for optimal voice cloning performance.
"""

import numpy as np
import librosa
import scipy.signal
import scipy.stats
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QualityIssue(Enum):
    """Types of audio quality issues."""
    LOW_SNR = "low_signal_to_noise_ratio"
    CLIPPING = "clipping_detected"
    COMPRESSION_ARTIFACTS = "compression_artifacts"
    POOR_FREQUENCY_RESPONSE = "poor_frequency_response"
    LOW_DYNAMIC_RANGE = "low_dynamic_range"
    NOISE_INTERFERENCE = "noise_interference"
    SPECTRAL_DISTORTION = "spectral_distortion"
    INSUFFICIENT_VOICE_CONTENT = "insufficient_voice_content"
    RECORDING_QUALITY = "poor_recording_quality"


@dataclass
class QualityIssueDetail:
    """Detailed information about a quality issue."""
    issue_type: QualityIssue
    severity: float  # 0.0 (minor) to 1.0 (severe)
    description: str
    recommended_action: str
    technical_details: Dict[str, Any]


@dataclass
class EnhancementRecommendation:
    """Enhancement recommendation with priority and expected improvement."""
    enhancement_type: str
    priority: int  # 1 (highest) to 5 (lowest)
    expected_improvement: float  # 0.0 to 1.0
    description: str
    parameters: Dict[str, Any]
    prerequisites: List[str]


@dataclass
class QualityAssessmentReport:
    """Comprehensive quality assessment report."""
    overall_score: float
    voice_suitability_score: float
    issues_detected: List[QualityIssueDetail]
    enhancement_recommendations: List[EnhancementRecommendation]
    technical_metrics: Dict[str, float]
    processing_suggestions: Dict[str, Any]


class AudioQualityAssessor:
    """
    Comprehensive audio quality assessment system that analyzes audio
    characteristics and provides intelligent enhancement recommendations.
    """
    
    def __init__(self):
        """Initialize the quality assessor."""
        self.sample_rate_target = 22050
        self.voice_freq_range = (80, 8000)
        self.fundamental_range = (80, 400)
        
        # Quality thresholds
        self.thresholds = {
            'snr_excellent': 25.0,
            'snr_good': 15.0,
            'snr_poor': 5.0,
            'dynamic_range_excellent': 40.0,
            'dynamic_range_good': 25.0,
            'dynamic_range_poor': 15.0,
            'voice_activity_good': 0.6,
            'voice_activity_poor': 0.3,
            'spectral_clarity_good': 0.7,
            'spectral_clarity_poor': 0.4
        }
    
    def assess_audio_quality(self, audio: np.ndarray, sample_rate: int, 
                           audio_path: Optional[str] = None) -> QualityAssessmentReport:
        """
        Perform comprehensive audio quality assessment.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate
            audio_path: Optional path to audio file for additional analysis
            
        Returns:
            Comprehensive quality assessment report
        """
        logger.info("Starting comprehensive audio quality assessment")
        
        # Calculate technical metrics
        technical_metrics = self._calculate_technical_metrics(audio, sample_rate)
        
        # Detect quality issues
        issues = self._detect_quality_issues(audio, sample_rate, technical_metrics)
        
        # Calculate overall scores
        overall_score = self._calculate_overall_score(technical_metrics, issues)
        voice_suitability = self._calculate_voice_suitability(technical_metrics, issues)
        
        # Generate enhancement recommendations
        recommendations = self._generate_enhancement_recommendations(issues, technical_metrics)
        
        # Generate processing suggestions
        processing_suggestions = self._generate_processing_suggestions(technical_metrics, issues)
        
        logger.info(f"Quality assessment complete. Overall score: {overall_score:.3f}")
        
        return QualityAssessmentReport(
            overall_score=overall_score,
            voice_suitability_score=voice_suitability,
            issues_detected=issues,
            enhancement_recommendations=recommendations,
            technical_metrics=technical_metrics,
            processing_suggestions=processing_suggestions
        )
    
    def _calculate_technical_metrics(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Calculate comprehensive technical metrics."""
        metrics = {}
        
        # Basic signal metrics
        metrics['duration'] = len(audio) / sample_rate
        metrics['peak_amplitude'] = float(np.max(np.abs(audio)))
        metrics['rms_level'] = float(np.sqrt(np.mean(audio**2)))
        metrics['crest_factor'] = metrics['peak_amplitude'] / metrics['rms_level'] if metrics['rms_level'] > 0 else 0
        
        # Dynamic range
        metrics['dynamic_range_db'] = 20 * np.log10(metrics['crest_factor']) if metrics['crest_factor'] > 0 else 0
        
        # Signal-to-noise ratio
        metrics['snr_db'] = self._estimate_snr_advanced(audio, sample_rate)
        
        # Spectral metrics
        spectral_metrics = self._calculate_spectral_metrics(audio, sample_rate)
        metrics.update(spectral_metrics)
        
        # Voice activity metrics
        voice_metrics = self._calculate_voice_activity_metrics(audio, sample_rate)
        metrics.update(voice_metrics)
        
        # Frequency response metrics
        freq_metrics = self._calculate_frequency_response_metrics(audio, sample_rate)
        metrics.update(freq_metrics)
        
        # Distortion metrics
        distortion_metrics = self._calculate_distortion_metrics(audio, sample_rate)
        metrics.update(distortion_metrics)
        
        return metrics
    
    def _detect_quality_issues(self, audio: np.ndarray, sample_rate: int, 
                             metrics: Dict[str, float]) -> List[QualityIssueDetail]:
        """Detect and categorize quality issues."""
        issues = []
        
        # Low SNR
        if metrics['snr_db'] < self.thresholds['snr_poor']:
            severity = 1.0 - (metrics['snr_db'] / self.thresholds['snr_poor'])
            issues.append(QualityIssueDetail(
                issue_type=QualityIssue.LOW_SNR,
                severity=min(1.0, severity),
                description=f"Low signal-to-noise ratio ({metrics['snr_db']:.1f} dB)",
                recommended_action="Apply advanced noise reduction",
                technical_details={'snr_db': metrics['snr_db'], 'threshold': self.thresholds['snr_poor']}
            ))
        
        # Clipping detection
        clipping_ratio = np.sum(np.abs(audio) >= 0.99) / len(audio)
        if clipping_ratio > 0.001:  # More than 0.1% clipped
            issues.append(QualityIssueDetail(
                issue_type=QualityIssue.CLIPPING,
                severity=min(1.0, clipping_ratio * 1000),
                description=f"Audio clipping detected ({clipping_ratio*100:.2f}% of samples)",
                recommended_action="Apply clipping repair and level adjustment",
                technical_details={'clipping_ratio': clipping_ratio}
            ))
        
        # Low dynamic range
        if metrics['dynamic_range_db'] < self.thresholds['dynamic_range_poor']:
            severity = 1.0 - (metrics['dynamic_range_db'] / self.thresholds['dynamic_range_poor'])
            issues.append(QualityIssueDetail(
                issue_type=QualityIssue.LOW_DYNAMIC_RANGE,
                severity=min(1.0, severity),
                description=f"Low dynamic range ({metrics['dynamic_range_db']:.1f} dB)",
                recommended_action="Apply dynamic range expansion",
                technical_details={'dynamic_range_db': metrics['dynamic_range_db']}
            ))
        
        # Poor frequency response
        if metrics.get('frequency_response_score', 1.0) < 0.5:
            issues.append(QualityIssueDetail(
                issue_type=QualityIssue.POOR_FREQUENCY_RESPONSE,
                severity=1.0 - metrics['frequency_response_score'],
                description="Uneven frequency response detected",
                recommended_action="Apply frequency response correction",
                technical_details={'freq_response_score': metrics['frequency_response_score']}
            ))
        
        # Compression artifacts
        if metrics.get('compression_artifacts', 0.0) > 0.3:
            issues.append(QualityIssueDetail(
                issue_type=QualityIssue.COMPRESSION_ARTIFACTS,
                severity=metrics['compression_artifacts'],
                description="Compression artifacts detected",
                recommended_action="Apply artifact removal and bandwidth extension",
                technical_details={'artifact_level': metrics['compression_artifacts']}
            ))
        
        # Insufficient voice content
        if metrics.get('voice_activity_ratio', 1.0) < self.thresholds['voice_activity_poor']:
            severity = 1.0 - (metrics['voice_activity_ratio'] / self.thresholds['voice_activity_good'])
            issues.append(QualityIssueDetail(
                issue_type=QualityIssue.INSUFFICIENT_VOICE_CONTENT,
                severity=min(1.0, severity),
                description=f"Low voice activity ({metrics['voice_activity_ratio']*100:.1f}%)",
                recommended_action="Trim silence and enhance voice segments",
                technical_details={'voice_activity_ratio': metrics['voice_activity_ratio']}
            ))
        
        # Spectral distortion
        if metrics.get('spectral_clarity', 1.0) < self.thresholds['spectral_clarity_poor']:
            severity = 1.0 - (metrics['spectral_clarity'] / self.thresholds['spectral_clarity_good'])
            issues.append(QualityIssueDetail(
                issue_type=QualityIssue.SPECTRAL_DISTORTION,
                severity=min(1.0, severity),
                description="Spectral distortion detected",
                recommended_action="Apply spectral enhancement",
                technical_details={'spectral_clarity': metrics['spectral_clarity']}
            ))
        
        return issues
    
    def _generate_enhancement_recommendations(self, issues: List[QualityIssueDetail], 
                                           metrics: Dict[str, float]) -> List[EnhancementRecommendation]:
        """Generate prioritized enhancement recommendations."""
        recommendations = []
        
        # Sort issues by severity
        sorted_issues = sorted(issues, key=lambda x: x.severity, reverse=True)
        
        for i, issue in enumerate(sorted_issues):
            priority = i + 1  # Higher severity = higher priority
            
            if issue.issue_type == QualityIssue.LOW_SNR:
                recommendations.append(EnhancementRecommendation(
                    enhancement_type="advanced_noise_reduction",
                    priority=priority,
                    expected_improvement=min(0.8, issue.severity),
                    description="Apply multi-stage noise reduction with voice preservation",
                    parameters={
                        'method': 'spectral_subtraction_wiener',
                        'aggressiveness': min(0.8, issue.severity),
                        'preserve_voice': True
                    },
                    prerequisites=[]
                ))
            
            elif issue.issue_type == QualityIssue.CLIPPING:
                recommendations.append(EnhancementRecommendation(
                    enhancement_type="clipping_repair",
                    priority=1,  # Always highest priority
                    expected_improvement=0.6,
                    description="Repair clipped audio segments",
                    parameters={
                        'method': 'cubic_spline_interpolation',
                        'threshold': 0.99
                    },
                    prerequisites=[]
                ))
            
            elif issue.issue_type == QualityIssue.LOW_DYNAMIC_RANGE:
                recommendations.append(EnhancementRecommendation(
                    enhancement_type="dynamic_range_expansion",
                    priority=priority,
                    expected_improvement=min(0.5, issue.severity * 0.7),
                    description="Expand dynamic range while preserving voice characteristics",
                    parameters={
                        'expansion_ratio': 1.5,
                        'threshold': -20,
                        'voice_preserve': True
                    },
                    prerequisites=["clipping_repair"]
                ))
            
            elif issue.issue_type == QualityIssue.COMPRESSION_ARTIFACTS:
                recommendations.append(EnhancementRecommendation(
                    enhancement_type="artifact_removal",
                    priority=priority,
                    expected_improvement=min(0.7, issue.severity * 0.8),
                    description="Remove compression artifacts and restore frequency content",
                    parameters={
                        'method': 'spectral_interpolation',
                        'bandwidth_extension': True
                    },
                    prerequisites=["advanced_noise_reduction"]
                ))
            
            elif issue.issue_type == QualityIssue.POOR_FREQUENCY_RESPONSE:
                recommendations.append(EnhancementRecommendation(
                    enhancement_type="frequency_response_correction",
                    priority=priority,
                    expected_improvement=0.4,
                    description="Correct frequency response for optimal voice analysis",
                    parameters={
                        'eq_type': 'voice_optimized',
                        'target_curve': 'flat_voice_range'
                    },
                    prerequisites=["artifact_removal"]
                ))
        
        # Add general recommendations based on metrics
        if metrics.get('duration', 0) < 3.0:
            recommendations.append(EnhancementRecommendation(
                enhancement_type="duration_warning",
                priority=5,
                expected_improvement=0.0,
                description="Audio duration is short - longer samples improve voice cloning quality",
                parameters={'minimum_recommended': 5.0},
                prerequisites=[]
            ))
        
        return recommendations
    
    def _generate_processing_suggestions(self, metrics: Dict[str, float], 
                                       issues: List[QualityIssueDetail]) -> Dict[str, Any]:
        """Generate processing pipeline suggestions."""
        suggestions = {
            'preprocessing_order': [],
            'model_selection': {},
            'synthesis_parameters': {},
            'quality_targets': {}
        }
        
        # Determine preprocessing order based on issues
        if any(issue.issue_type == QualityIssue.CLIPPING for issue in issues):
            suggestions['preprocessing_order'].append('clipping_repair')
        
        suggestions['preprocessing_order'].extend([
            'level_normalization',
            'advanced_noise_reduction',
            'quality_enhancement',
            'compression_restoration',
            'dynamic_range_optimization',
            'frequency_response_correction'
        ])
        
        # Model selection suggestions
        if metrics.get('snr_db', 20) < 10:
            suggestions['model_selection']['noise_robust'] = True
        
        if metrics.get('voice_activity_ratio', 1.0) < 0.5:
            suggestions['model_selection']['voice_enhancement'] = True
        
        # Synthesis parameter suggestions
        suggestions['synthesis_parameters'] = {
            'quality_priority': 'high' if len(issues) > 2 else 'balanced',
            'enhancement_aggressiveness': min(0.8, len(issues) * 0.2),
            'voice_preservation': True
        }
        
        # Quality targets
        suggestions['quality_targets'] = {
            'minimum_snr': 15.0,
            'minimum_dynamic_range': 25.0,
            'minimum_voice_activity': 0.6,
            'target_similarity': 0.95
        }
        
        return suggestions
    
    def _calculate_overall_score(self, metrics: Dict[str, float], 
                               issues: List[QualityIssueDetail]) -> float:
        """Calculate overall quality score."""
        # Base score from metrics
        snr_score = min(1.0, max(0.0, (metrics.get('snr_db', 0) - 5) / 20))
        dynamic_range_score = min(1.0, max(0.0, (metrics.get('dynamic_range_db', 0) - 15) / 25))
        voice_activity_score = metrics.get('voice_activity_ratio', 0.5)
        spectral_score = metrics.get('spectral_clarity', 0.5)
        
        base_score = (snr_score * 0.3 + dynamic_range_score * 0.25 + 
                     voice_activity_score * 0.25 + spectral_score * 0.2)
        
        # Penalty for severe issues
        severity_penalty = sum(issue.severity for issue in issues if issue.severity > 0.7) * 0.1
        
        overall_score = max(0.0, base_score - severity_penalty)
        
        return overall_score
    
    def _calculate_voice_suitability(self, metrics: Dict[str, float], 
                                   issues: List[QualityIssueDetail]) -> float:
        """Calculate voice cloning suitability score."""
        # Voice-specific factors
        voice_activity = metrics.get('voice_activity_ratio', 0.5)
        voice_clarity = metrics.get('spectral_clarity', 0.5)
        harmonic_content = metrics.get('harmonic_ratio', 0.5)
        duration_factor = min(1.0, metrics.get('duration', 0) / 5.0)  # 5 seconds is good
        
        # Penalties for voice-critical issues
        voice_penalties = 0.0
        for issue in issues:
            if issue.issue_type in [QualityIssue.INSUFFICIENT_VOICE_CONTENT, 
                                  QualityIssue.SPECTRAL_DISTORTION]:
                voice_penalties += issue.severity * 0.2
        
        suitability = (voice_activity * 0.3 + voice_clarity * 0.25 + 
                      harmonic_content * 0.2 + duration_factor * 0.25 - voice_penalties)
        
        return max(0.0, min(1.0, suitability))
    
    # Helper methods for metric calculation
    
    def _estimate_snr_advanced(self, audio: np.ndarray, sample_rate: int) -> float:
        """Advanced SNR estimation using multiple methods."""
        # Method 1: Energy-based VAD
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.01 * sample_rate)
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_energy = np.sum(frames**2, axis=0)
        
        # Adaptive thresholding
        energy_sorted = np.sort(frame_energy)
        noise_threshold = energy_sorted[int(len(energy_sorted) * 0.2)]  # Bottom 20%
        signal_threshold = energy_sorted[int(len(energy_sorted) * 0.8)]  # Top 20%
        
        noise_frames = frame_energy[frame_energy <= noise_threshold]
        signal_frames = frame_energy[frame_energy >= signal_threshold]
        
        if len(noise_frames) > 0 and len(signal_frames) > 0:
            noise_power = np.mean(noise_frames)
            signal_power = np.mean(signal_frames)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 40
        else:
            snr = 20.0  # Default reasonable value
        
        return float(snr)
    
    def _calculate_spectral_metrics(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Calculate spectral analysis metrics."""
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]
        
        # Spectral clarity (inverse of bandwidth variability)
        spectral_clarity = 1.0 / (1.0 + np.std(spectral_bandwidth) / np.mean(spectral_bandwidth))
        
        # Harmonic ratio estimation
        harmonic_ratio = self._estimate_harmonic_ratio(audio, sample_rate)
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroid)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_flatness_mean': float(np.mean(spectral_flatness)),
            'spectral_clarity': float(spectral_clarity),
            'harmonic_ratio': float(harmonic_ratio)
        }
    
    def _calculate_voice_activity_metrics(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Calculate voice activity metrics."""
        # Simple energy-based VAD
        frame_length = int(0.025 * sample_rate)
        hop_length = int(0.01 * sample_rate)
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        frame_energy = np.sum(frames**2, axis=0)
        
        # Voice activity detection
        energy_threshold = np.percentile(frame_energy, 30)
        voice_frames = frame_energy > energy_threshold
        voice_activity_ratio = np.sum(voice_frames) / len(voice_frames)
        
        # Voice quality indicators
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zero_crossing_rate)
        
        return {
            'voice_activity_ratio': float(voice_activity_ratio),
            'zero_crossing_rate_mean': float(zcr_mean)
        }
    
    def _calculate_frequency_response_metrics(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Calculate frequency response metrics."""
        # Power spectral density
        freqs, psd = scipy.signal.welch(audio, fs=sample_rate, nperseg=2048)
        
        # Voice frequency range analysis
        voice_mask = (freqs >= self.voice_freq_range[0]) & (freqs <= self.voice_freq_range[1])
        voice_psd = psd[voice_mask]
        
        # Frequency response flatness in voice range
        if len(voice_psd) > 0:
            freq_response_score = 1.0 - (np.std(voice_psd) / np.mean(voice_psd))
        else:
            freq_response_score = 0.0
        
        # High frequency content
        high_freq_mask = freqs > sample_rate * 0.4
        if np.any(high_freq_mask):
            high_freq_energy = np.mean(psd[high_freq_mask])
            total_energy = np.mean(psd)
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        else:
            high_freq_ratio = 0.0
        
        return {
            'frequency_response_score': float(max(0.0, min(1.0, freq_response_score))),
            'high_frequency_ratio': float(high_freq_ratio)
        }
    
    def _calculate_distortion_metrics(self, audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Calculate distortion and artifact metrics."""
        # THD estimation (simplified)
        thd = self._estimate_thd(audio, sample_rate)
        
        # Compression artifacts (high frequency rolloff)
        freqs, psd = scipy.signal.welch(audio, fs=sample_rate, nperseg=2048)
        high_freq_mask = freqs > sample_rate * 0.3
        
        if np.any(high_freq_mask):
            high_freq_energy = np.mean(psd[high_freq_mask])
            total_energy = np.mean(psd)
            compression_artifacts = 1.0 - (high_freq_energy / total_energy) if total_energy > 0 else 1.0
        else:
            compression_artifacts = 1.0
        
        return {
            'total_harmonic_distortion': float(thd),
            'compression_artifacts': float(max(0.0, min(1.0, compression_artifacts)))
        }
    
    def _estimate_harmonic_ratio(self, audio: np.ndarray, sample_rate: int) -> float:
        """Estimate harmonic-to-noise ratio."""
        # Autocorrelation-based approach
        autocorr = np.correlate(audio, audio, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if len(autocorr) > 1:
            # Normalize
            autocorr = autocorr / autocorr[0] if autocorr[0] > 0 else autocorr
            
            # Find first peak (excluding zero lag)
            peak_idx = np.argmax(autocorr[1:]) + 1
            harmonic_ratio = autocorr[peak_idx] if peak_idx < len(autocorr) else 0.0
        else:
            harmonic_ratio = 0.0
        
        return float(max(0.0, min(1.0, harmonic_ratio)))
    
    def _estimate_thd(self, audio: np.ndarray, sample_rate: int) -> float:
        """Estimate total harmonic distortion (simplified)."""
        # FFT-based approach
        fft_result = np.fft.fft(audio)
        magnitude = np.abs(fft_result[:len(fft_result)//2])
        
        # Find fundamental frequency
        freqs = np.fft.fftfreq(len(audio), 1/sample_rate)[:len(magnitude)]
        
        # Simple THD estimation
        if len(magnitude) > 10:
            fundamental_power = np.max(magnitude)
            harmonic_power = np.sum(magnitude) - fundamental_power
            thd = harmonic_power / fundamental_power if fundamental_power > 0 else 0.0
        else:
            thd = 0.0
        
        return float(max(0.0, min(1.0, thd)))


# Global instance
audio_quality_assessor = AudioQualityAssessor()