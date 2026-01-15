"""
Comprehensive Quality Metrics Engine for Perfect Voice Cloning.

This module implements a complete quality assessment system that measures
voice cloning fidelity across multiple dimensions: speaker similarity,
prosody matching, timbre matching, emotion matching, and naturalness.

The system provides real-time quality validation with automatic regeneration
triggers when quality falls below the 95% threshold.
"""

import logging
import numpy as np
import librosa
from scipy import signal, stats
from scipy.spatial.distance import cosine
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for voice cloning output."""
    # Core similarity metrics (0-1 scale)
    speaker_similarity: float  # Speaker identity match
    prosody_similarity: float  # Rhythm and intonation match
    timbre_similarity: float   # Voice texture match
    emotion_similarity: float  # Emotional expression match
    spectral_similarity: float # Frequency content match
    temporal_similarity: float # Timing alignment match
    
    # Quality metrics
    naturalness_score: float   # How natural the speech sounds
    clarity_score: float       # Audio clarity
    artifact_score: float      # Absence of artifacts (1 = no artifacts)
    
    # Overall scores
    overall_similarity: float  # Weighted combination of similarities
    overall_quality: float     # Combined quality score
    
    # Confidence and metadata
    confidence: float          # Confidence in the metrics
    meets_threshold: bool      # Whether output meets 95% threshold
    weak_areas: List[str]      # Areas needing improvement
    recommendations: List[str] # Improvement recommendations


@dataclass
class DetailedAnalysis:
    """Detailed analysis breakdown for debugging and improvement."""
    pitch_analysis: Dict[str, float]
    formant_analysis: Dict[str, float]
    mfcc_analysis: Dict[str, float]
    energy_analysis: Dict[str, float]
    rhythm_analysis: Dict[str, float]
    spectral_analysis: Dict[str, float]


class ComprehensiveQualityMetrics:
    """
    Comprehensive quality assessment engine for voice cloning.
    
    Measures quality across multiple dimensions:
    1. Speaker Similarity: Using speaker embeddings (ECAPA-TDNN)
    2. Prosody Similarity: Pitch contours, rhythm, stress patterns
    3. Timbre Similarity: MFCC, spectral envelope, formants
    4. Emotion Similarity: Energy dynamics, pitch variation
    5. Naturalness: MOS prediction, artifact detection
    
    Target: >95% overall similarity for acceptance
    """
    
    # Quality thresholds
    SIMILARITY_THRESHOLD = 0.95  # 95% similarity required
    NATURALNESS_THRESHOLD = 0.90  # 90% naturalness required
    ARTIFACT_THRESHOLD = 0.95    # 95% artifact-free required
    
    # Metric weights for overall score
    SIMILARITY_WEIGHTS = {
        'speaker': 0.30,    # Speaker identity is most important
        'timbre': 0.25,     # Voice texture
        'prosody': 0.20,    # Rhythm and intonation
        'spectral': 0.15,   # Frequency content
        'emotion': 0.10     # Emotional expression
    }
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize quality metrics engine.
        
        Args:
            sample_rate: Audio sample rate for analysis
        """
        self.sample_rate = sample_rate
        self.hop_length = 256
        self.n_fft = 2048
        self.n_mfcc = 20
        
        # Speaker encoder for similarity
        self._speaker_encoder = None
        
        logger.info("Comprehensive Quality Metrics engine initialized")
    
    def compute_quality_metrics(
        self,
        synthesized_audio: np.ndarray,
        reference_audio: np.ndarray,
        synthesized_sr: Optional[int] = None,
        reference_sr: Optional[int] = None
    ) -> QualityMetrics:
        """
        Compute comprehensive quality metrics.
        
        Args:
            synthesized_audio: Synthesized audio array
            reference_audio: Reference audio array
            synthesized_sr: Sample rate of synthesized audio
            reference_sr: Sample rate of reference audio
            
        Returns:
            QualityMetrics object with all measurements
        """
        synth_sr = synthesized_sr or self.sample_rate
        ref_sr = reference_sr or self.sample_rate
        
        # Resample if needed
        if synth_sr != self.sample_rate:
            synthesized_audio = librosa.resample(
                synthesized_audio, orig_sr=synth_sr, target_sr=self.sample_rate
            )
        if ref_sr != self.sample_rate:
            reference_audio = librosa.resample(
                reference_audio, orig_sr=ref_sr, target_sr=self.sample_rate
            )
        
        # Normalize audio
        synthesized_audio = self._normalize_audio(synthesized_audio)
        reference_audio = self._normalize_audio(reference_audio)
        
        # Compute individual metrics
        speaker_sim = self._compute_speaker_similarity(synthesized_audio, reference_audio)
        prosody_sim = self._compute_prosody_similarity(synthesized_audio, reference_audio)
        timbre_sim = self._compute_timbre_similarity(synthesized_audio, reference_audio)
        emotion_sim = self._compute_emotion_similarity(synthesized_audio, reference_audio)
        spectral_sim = self._compute_spectral_similarity(synthesized_audio, reference_audio)
        temporal_sim = self._compute_temporal_similarity(synthesized_audio, reference_audio)
        
        # Quality metrics
        naturalness = self._compute_naturalness_score(synthesized_audio)
        clarity = self._compute_clarity_score(synthesized_audio)
        artifact = self._compute_artifact_score(synthesized_audio)
        
        # Overall similarity (weighted)
        overall_similarity = (
            speaker_sim * self.SIMILARITY_WEIGHTS['speaker'] +
            timbre_sim * self.SIMILARITY_WEIGHTS['timbre'] +
            prosody_sim * self.SIMILARITY_WEIGHTS['prosody'] +
            spectral_sim * self.SIMILARITY_WEIGHTS['spectral'] +
            emotion_sim * self.SIMILARITY_WEIGHTS['emotion']
        )
        
        # Overall quality
        overall_quality = (overall_similarity * 0.7 + naturalness * 0.2 + artifact * 0.1)
        
        # Confidence based on audio quality
        confidence = self._compute_confidence(reference_audio, synthesized_audio)
        
        # Check threshold
        meets_threshold = overall_similarity >= self.SIMILARITY_THRESHOLD
        
        # Identify weak areas and recommendations
        weak_areas, recommendations = self._identify_improvements(
            speaker_sim, prosody_sim, timbre_sim, emotion_sim,
            spectral_sim, naturalness, artifact
        )
        
        return QualityMetrics(
            speaker_similarity=speaker_sim,
            prosody_similarity=prosody_sim,
            timbre_similarity=timbre_sim,
            emotion_similarity=emotion_sim,
            spectral_similarity=spectral_sim,
            temporal_similarity=temporal_sim,
            naturalness_score=naturalness,
            clarity_score=clarity,
            artifact_score=artifact,
            overall_similarity=overall_similarity,
            overall_quality=overall_quality,
            confidence=confidence,
            meets_threshold=meets_threshold,
            weak_areas=weak_areas,
            recommendations=recommendations
        )

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range."""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def _compute_speaker_similarity(
        self,
        synthesized: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """
        Compute speaker identity similarity using MFCC-based comparison.
        
        For production, this should use ECAPA-TDNN embeddings.
        """
        try:
            # Extract MFCCs
            mfcc_synth = librosa.feature.mfcc(
                y=synthesized, sr=self.sample_rate, n_mfcc=self.n_mfcc
            )
            mfcc_ref = librosa.feature.mfcc(
                y=reference, sr=self.sample_rate, n_mfcc=self.n_mfcc
            )
            
            # Mean MFCCs
            mfcc_synth_mean = np.mean(mfcc_synth, axis=1)
            mfcc_ref_mean = np.mean(mfcc_ref, axis=1)
            
            # Cosine similarity
            similarity = 1 - cosine(mfcc_synth_mean, mfcc_ref_mean)
            
            # Also compare MFCC dynamics
            mfcc_synth_std = np.std(mfcc_synth, axis=1)
            mfcc_ref_std = np.std(mfcc_ref, axis=1)
            dynamics_sim = 1 - cosine(mfcc_synth_std, mfcc_ref_std)
            
            # Combine
            return float(np.clip(similarity * 0.7 + dynamics_sim * 0.3, 0, 1))
            
        except Exception as e:
            logger.warning(f"Speaker similarity computation failed: {e}")
            return 0.5
    
    def _compute_prosody_similarity(
        self,
        synthesized: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """Compute prosody (rhythm, intonation) similarity."""
        try:
            # Extract pitch contours
            f0_synth, _, _ = librosa.pyin(
                synthesized, fmin=50, fmax=500, sr=self.sample_rate
            )
            f0_ref, _, _ = librosa.pyin(
                reference, fmin=50, fmax=500, sr=self.sample_rate
            )
            
            # Clean NaN values
            f0_synth_clean = f0_synth[~np.isnan(f0_synth)]
            f0_ref_clean = f0_ref[~np.isnan(f0_ref)]
            
            if len(f0_synth_clean) < 10 or len(f0_ref_clean) < 10:
                return 0.5
            
            # Pitch statistics similarity
            pitch_mean_sim = 1 - min(1, abs(np.mean(f0_synth_clean) - np.mean(f0_ref_clean)) / 100)
            pitch_std_sim = 1 - min(1, abs(np.std(f0_synth_clean) - np.std(f0_ref_clean)) / 50)
            pitch_range_sim = 1 - min(1, abs(
                (np.max(f0_synth_clean) - np.min(f0_synth_clean)) -
                (np.max(f0_ref_clean) - np.min(f0_ref_clean))
            ) / 200)
            
            # Speech rate similarity (using energy-based segmentation)
            rms_synth = librosa.feature.rms(y=synthesized)[0]
            rms_ref = librosa.feature.rms(y=reference)[0]
            
            # Count voiced segments
            threshold = 0.02
            voiced_synth = np.sum(rms_synth > threshold) / len(rms_synth)
            voiced_ref = np.sum(rms_ref > threshold) / len(rms_ref)
            rate_sim = 1 - min(1, abs(voiced_synth - voiced_ref) * 2)
            
            # Combine prosody metrics
            prosody_sim = (
                pitch_mean_sim * 0.3 +
                pitch_std_sim * 0.25 +
                pitch_range_sim * 0.25 +
                rate_sim * 0.2
            )
            
            return float(np.clip(prosody_sim, 0, 1))
            
        except Exception as e:
            logger.warning(f"Prosody similarity computation failed: {e}")
            return 0.5
    
    def _compute_timbre_similarity(
        self,
        synthesized: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """Compute timbre (voice texture) similarity."""
        try:
            # Spectral centroid
            centroid_synth = np.mean(librosa.feature.spectral_centroid(
                y=synthesized, sr=self.sample_rate
            ))
            centroid_ref = np.mean(librosa.feature.spectral_centroid(
                y=reference, sr=self.sample_rate
            ))
            centroid_sim = 1 - min(1, abs(centroid_synth - centroid_ref) / 2000)
            
            # Spectral rolloff
            rolloff_synth = np.mean(librosa.feature.spectral_rolloff(
                y=synthesized, sr=self.sample_rate
            ))
            rolloff_ref = np.mean(librosa.feature.spectral_rolloff(
                y=reference, sr=self.sample_rate
            ))
            rolloff_sim = 1 - min(1, abs(rolloff_synth - rolloff_ref) / 4000)
            
            # Spectral bandwidth
            bandwidth_synth = np.mean(librosa.feature.spectral_bandwidth(
                y=synthesized, sr=self.sample_rate
            ))
            bandwidth_ref = np.mean(librosa.feature.spectral_bandwidth(
                y=reference, sr=self.sample_rate
            ))
            bandwidth_sim = 1 - min(1, abs(bandwidth_synth - bandwidth_ref) / 2000)
            
            # Spectral flatness
            flatness_synth = np.mean(librosa.feature.spectral_flatness(y=synthesized))
            flatness_ref = np.mean(librosa.feature.spectral_flatness(y=reference))
            flatness_sim = 1 - min(1, abs(flatness_synth - flatness_ref) * 10)
            
            # Combine timbre metrics
            timbre_sim = (
                centroid_sim * 0.3 +
                rolloff_sim * 0.25 +
                bandwidth_sim * 0.25 +
                flatness_sim * 0.2
            )
            
            return float(np.clip(timbre_sim, 0, 1))
            
        except Exception as e:
            logger.warning(f"Timbre similarity computation failed: {e}")
            return 0.5

    def _compute_emotion_similarity(
        self,
        synthesized: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """Compute emotional expression similarity."""
        try:
            # Energy dynamics
            rms_synth = librosa.feature.rms(y=synthesized)[0]
            rms_ref = librosa.feature.rms(y=reference)[0]
            
            # Energy statistics
            energy_mean_sim = 1 - min(1, abs(np.mean(rms_synth) - np.mean(rms_ref)) * 10)
            energy_std_sim = 1 - min(1, abs(np.std(rms_synth) - np.std(rms_ref)) * 10)
            energy_range_sim = 1 - min(1, abs(
                (np.max(rms_synth) - np.min(rms_synth)) -
                (np.max(rms_ref) - np.min(rms_ref))
            ) * 5)
            
            # Pitch variation (emotional indicator)
            f0_synth, _, _ = librosa.pyin(synthesized, fmin=50, fmax=500, sr=self.sample_rate)
            f0_ref, _, _ = librosa.pyin(reference, fmin=50, fmax=500, sr=self.sample_rate)
            
            f0_synth_clean = f0_synth[~np.isnan(f0_synth)]
            f0_ref_clean = f0_ref[~np.isnan(f0_ref)]
            
            if len(f0_synth_clean) > 0 and len(f0_ref_clean) > 0:
                pitch_var_sim = 1 - min(1, abs(
                    np.std(f0_synth_clean) - np.std(f0_ref_clean)
                ) / 50)
            else:
                pitch_var_sim = 0.5
            
            # Combine emotion metrics
            emotion_sim = (
                energy_mean_sim * 0.25 +
                energy_std_sim * 0.25 +
                energy_range_sim * 0.25 +
                pitch_var_sim * 0.25
            )
            
            return float(np.clip(emotion_sim, 0, 1))
            
        except Exception as e:
            logger.warning(f"Emotion similarity computation failed: {e}")
            return 0.5
    
    def _compute_spectral_similarity(
        self,
        synthesized: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """Compute spectral content similarity."""
        try:
            # Compute spectrograms
            spec_synth = np.abs(librosa.stft(synthesized, n_fft=self.n_fft))
            spec_ref = np.abs(librosa.stft(reference, n_fft=self.n_fft))
            
            # Mean spectral envelope
            env_synth = np.mean(spec_synth, axis=1)
            env_ref = np.mean(spec_ref, axis=1)
            
            # Normalize
            env_synth = env_synth / (np.max(env_synth) + 1e-8)
            env_ref = env_ref / (np.max(env_ref) + 1e-8)
            
            # Cosine similarity of spectral envelopes
            envelope_sim = 1 - cosine(env_synth, env_ref)
            
            # Mel-spectrogram similarity
            mel_synth = librosa.feature.melspectrogram(
                y=synthesized, sr=self.sample_rate, n_mels=80
            )
            mel_ref = librosa.feature.melspectrogram(
                y=reference, sr=self.sample_rate, n_mels=80
            )
            
            mel_synth_mean = np.mean(mel_synth, axis=1)
            mel_ref_mean = np.mean(mel_ref, axis=1)
            
            mel_synth_mean = mel_synth_mean / (np.max(mel_synth_mean) + 1e-8)
            mel_ref_mean = mel_ref_mean / (np.max(mel_ref_mean) + 1e-8)
            
            mel_sim = 1 - cosine(mel_synth_mean, mel_ref_mean)
            
            # Combine
            spectral_sim = envelope_sim * 0.4 + mel_sim * 0.6
            
            return float(np.clip(spectral_sim, 0, 1))
            
        except Exception as e:
            logger.warning(f"Spectral similarity computation failed: {e}")
            return 0.5
    
    def _compute_temporal_similarity(
        self,
        synthesized: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """Compute temporal alignment similarity."""
        try:
            # Duration ratio
            duration_ratio = len(synthesized) / len(reference)
            duration_sim = 1 - min(1, abs(1 - duration_ratio))
            
            # Onset detection
            onset_synth = librosa.onset.onset_detect(
                y=synthesized, sr=self.sample_rate, units='time'
            )
            onset_ref = librosa.onset.onset_detect(
                y=reference, sr=self.sample_rate, units='time'
            )
            
            # Onset count similarity
            if len(onset_ref) > 0:
                onset_count_sim = 1 - min(1, abs(len(onset_synth) - len(onset_ref)) / len(onset_ref))
            else:
                onset_count_sim = 0.5
            
            # Combine
            temporal_sim = duration_sim * 0.5 + onset_count_sim * 0.5
            
            return float(np.clip(temporal_sim, 0, 1))
            
        except Exception as e:
            logger.warning(f"Temporal similarity computation failed: {e}")
            return 0.5
    
    def _compute_naturalness_score(self, audio: np.ndarray) -> float:
        """Compute naturalness score (MOS prediction)."""
        try:
            # Spectral flatness (lower is more natural for speech)
            flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
            flatness_score = 1 - min(1, flatness * 5)
            
            # Zero crossing rate (should be in natural range)
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
            zcr_score = 1 - min(1, abs(zcr - 0.08) * 10)
            
            # RMS energy consistency
            rms = librosa.feature.rms(y=audio)[0]
            rms_consistency = 1 - min(1, np.std(rms) / (np.mean(rms) + 1e-8))
            
            # Combine
            naturalness = flatness_score * 0.4 + zcr_score * 0.3 + rms_consistency * 0.3
            
            return float(np.clip(naturalness, 0, 1))
            
        except Exception as e:
            logger.warning(f"Naturalness computation failed: {e}")
            return 0.5
    
    def _compute_clarity_score(self, audio: np.ndarray) -> float:
        """Compute audio clarity score."""
        try:
            # Spectral centroid (higher = clearer)
            centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate))
            centroid_score = min(1, centroid / 3000)
            
            # RMS energy (adequate energy = clearer)
            rms = np.mean(librosa.feature.rms(y=audio))
            energy_score = min(1, rms * 10)
            
            # Combine
            clarity = centroid_score * 0.5 + energy_score * 0.5
            
            return float(np.clip(clarity, 0, 1))
            
        except Exception as e:
            logger.warning(f"Clarity computation failed: {e}")
            return 0.5
    
    def _compute_artifact_score(self, audio: np.ndarray) -> float:
        """Compute artifact-free score (1 = no artifacts)."""
        try:
            # Check for clipping
            clipping_ratio = np.mean(np.abs(audio) > 0.99)
            clipping_score = 1 - min(1, clipping_ratio * 100)
            
            # Check for sudden amplitude changes (glitches)
            diff = np.abs(np.diff(audio))
            glitch_ratio = np.mean(diff > 0.5)
            glitch_score = 1 - min(1, glitch_ratio * 100)
            
            # Check for silence gaps
            rms = librosa.feature.rms(y=audio)[0]
            silence_ratio = np.mean(rms < 0.001)
            silence_score = 1 - min(1, silence_ratio * 2)
            
            # Combine
            artifact_score = clipping_score * 0.4 + glitch_score * 0.4 + silence_score * 0.2
            
            return float(np.clip(artifact_score, 0, 1))
            
        except Exception as e:
            logger.warning(f"Artifact detection failed: {e}")
            return 0.5
    
    def _compute_confidence(
        self,
        reference: np.ndarray,
        synthesized: np.ndarray
    ) -> float:
        """Compute confidence in the quality metrics."""
        # Based on audio quality and duration
        ref_duration = len(reference) / self.sample_rate
        synth_duration = len(synthesized) / self.sample_rate
        
        # Duration factor
        duration_factor = min(1, min(ref_duration, synth_duration) / 5)
        
        # Audio quality factor
        ref_rms = np.mean(librosa.feature.rms(y=reference))
        synth_rms = np.mean(librosa.feature.rms(y=synthesized))
        quality_factor = min(1, (ref_rms + synth_rms) * 5)
        
        return float(duration_factor * 0.5 + quality_factor * 0.5)
    
    def _identify_improvements(
        self,
        speaker_sim: float,
        prosody_sim: float,
        timbre_sim: float,
        emotion_sim: float,
        spectral_sim: float,
        naturalness: float,
        artifact: float
    ) -> Tuple[List[str], List[str]]:
        """Identify weak areas and generate recommendations."""
        weak_areas = []
        recommendations = []
        
        threshold = 0.90  # Below this is considered weak
        
        if speaker_sim < threshold:
            weak_areas.append("speaker_identity")
            recommendations.append("Improve speaker embedding extraction or use longer reference audio")
        
        if prosody_sim < threshold:
            weak_areas.append("prosody")
            recommendations.append("Adjust speech rate and pitch contour matching")
        
        if timbre_sim < threshold:
            weak_areas.append("timbre")
            recommendations.append("Apply spectral envelope matching post-processing")
        
        if emotion_sim < threshold:
            weak_areas.append("emotion")
            recommendations.append("Enhance energy dynamics and pitch variation")
        
        if spectral_sim < threshold:
            weak_areas.append("spectral")
            recommendations.append("Apply frequency response correction")
        
        if naturalness < threshold:
            weak_areas.append("naturalness")
            recommendations.append("Use neural vocoder enhancement")
        
        if artifact < threshold:
            weak_areas.append("artifacts")
            recommendations.append("Apply artifact removal post-processing")
        
        return weak_areas, recommendations


# Global instance
_quality_metrics: Optional[ComprehensiveQualityMetrics] = None


def get_quality_metrics() -> ComprehensiveQualityMetrics:
    """Get or create global quality metrics instance."""
    global _quality_metrics
    if _quality_metrics is None:
        _quality_metrics = ComprehensiveQualityMetrics()
    return _quality_metrics
